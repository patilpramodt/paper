"""
strategies/banknifty_2candle_breakout_strategy.py

BANKNIFTY_2CANDLE_BO — 2-Candle Breakout Scalper, BankNifty option SELLING.

RULES (1-minute candles on the BankNifty index, token 260105)
─────────────────────────────────────────────────────────────
  1. Skip the 09:15 candle.
  2. From 09:16, find the first pair of CONSECUTIVE candles with OPPOSITE
     colours (one close>open, one close<open; a doji never qualifies).
       pair_mode "first_opposite_pair"  — scan forward until found (default)
       pair_mode "fixed_2nd_3rd"        — only 09:16 + 09:17; if they are
                                          not opposite colours, no trade today
  3. range_high = max(pair highs), range_low = min(pair lows).
  4. After the pair's second candle closes, a 1-minute candle CLOSING beyond
     the range is the signal (a wick through the level is not):
       close > range_high  → bullish → SELL ATM PE
       close < range_low   → bearish → SELL ATM CE
  5. Strike = ATM at signal time ± strike_offset_steps (positive = OTM).
     Expiry = nearest in the live NFO dump (+ min_dte), monthly for BankNifty.
  6. Exits, measured on the INDEX (1:1 on the range):
       SL     = opposite side of the range
       target = entry spot ± (range_high - range_low)
       exit_trigger "close" (1-min candle close, default) or "touch" (tick).
     The short option is bought back at that moment's price.
  7. One re-entry per day, only if trade 1 hit its SL. Same range, same rule.
     Never after a target. Max 2 trades/day.
  8. No new entries from a candle closing after entry_cutoff (14:30).
     Square-off at square_off_time (15:15).

ASSUMPTIONS (not covered by the spec — flag if wrong)
──────────────────────────────────────────────────────
  - An exit consumes its candle: re-entry is evaluated from the NEXT 1-min
    candle. (A close-mode SL candle is itself a close beyond the other side
    of the range; it does not double as the re-entry signal.)
  - entry spot = BankNifty LTP at the moment the option leg fills (within
    a second of the breakout close). The replay uses the breakout close.
  - fixed_2nd_3rd still requires the two candles to be opposite colours.
  - The last eligible breakout candle is the one CLOSING at entry_cutoff
    (i.e. the 14:29 candle for a 14:30 cutoff).
  - After a mid-session restart, the first live candle is partial (its open
    is wrong), so it cannot be half of a pair; it is treated like a doji.

INDEX ROUTING
─────────────
  Deliberately NO class-level INDEX_TOKEN — BankNifty (260105) is MarketHub's
  main index, and a non-None INDEX_TOKEN makes MarketHub skip this strategy
  on every BankNifty tick (see banknifty_candle_breakout_strategy.py).

TOKEN OWNERSHIP
───────────────
  Option legs are subscribed/unsubscribed via the hub refcount only.
  set_token_owner() is NOT used: it restricts a token's ticks to its owners,
  and ATM BankNifty strikes are shared (BB_STOCH, WsPCR, ...). Owning one
  would silently cut those strategies off.

RESTART
───────
  pre_market() rebuilds today's 1-min candles from Kite history and replays
  them through the same engine in "virtual" mode, so pair, range, trade count
  and re-entry eligibility are what continuous running would have produced.
  A trade the replay finds still open is NOT entered late (stale signal) —
  it is tracked on the index only, and its outcome still drives re-entry.
  A real open position is persisted to CFG["state_file"] and re-attached on
  restart (LIVE: verified against kite.positions()).

LIVE / PAPER MODE
─────────────────
  LIVE_MODE = False below. Paper never touches the live-trade slot
  (OrderRouter.acquire_slot returns True for paper without recording an owner).
"""

import json
import logging
import os
import threading
import time as _time_mod
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import CandleBuilder
from core.costs import estimate_spread, fixed_costs_rs
from core.csv_log import append_row
from core.instruments import get_atm_strike, get_nearest_expiry

log = logging.getLogger("strategy.banknifty_2candle_bo")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG — change to True only when ready to trade real money.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

BANKNIFTY_INDEX_TOKEN = 260105

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "enabled"               : True,

    # ── Session (IST) ─────────────────────────────────────────────────────────
    "session_open"          : dtime(9, 15),    # this candle is skipped
    "first_candle"          : dtime(9, 16),    # pair search starts here
    "entry_cutoff"          : dtime(14, 30),   # no breakout candle closing after this
    "square_off_time"       : dtime(15, 15),

    # ── Signal ────────────────────────────────────────────────────────────────
    "candle_minutes"        : 1,
    "pair_mode"             : "first_opposite_pair",   # or "fixed_2nd_3rd"
    "exit_trigger"          : "close",                 # or "touch"
    "max_trades_per_day"    : 2,
    "max_reentries"         : 1,                       # only after an SL_HIT

    # ── Option selection ──────────────────────────────────────────────────────
    "strike_step"           : 100,
    "strike_offset_steps"   : 0,       # +N = N strikes OTM for the sold leg
    "min_dte"               : 0,       # 0 = nearest expiry in the dump
    "lots"                  : 1,       # lot size read from the instruments dump

    # ── Execution ─────────────────────────────────────────────────────────────
    "max_price_age_sec"     : 5,       # never fill on an option price older than this
    "pending_timeout_sec"   : 10,      # drop the entry if no live option tick by then
    "margin_per_lot_rs"     : 310000,  # LIVE fallback if kite.order_margins() fails
    "emergency_retry_sec"   : 30,
    "emergency_max_attempts": 30,

    # ── Output (cwd = repo root, like the other strategies) ───────────────────
    "csv_file"              : "banknifty_2candle_bo_trades.csv",
    "signals_csv"           : "banknifty_2candle_bo_signals.csv",
    "state_file"            : "banknifty_2candle_bo_state.json",
}

BULL, BEAR = "BULL", "BEAR"
SELL_LEG = {BULL: "PE", BEAR: "CE"}


def _colour(c: dict) -> Optional[str]:
    if c["close"] > c["open"]:
        return "GREEN"
    if c["close"] < c["open"]:
        return "RED"
    return None   # doji


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE — pure index logic, shared by the live strategy and the replay
# ═════════════════════════════════════════════════════════════════════════════

class TwoCandleEngine:
    """
    Feed CLOSED 1-minute index candles ({ts, open, high, low, close}, ts = bar
    start) in order. Every call returns a list of event dicts.

    Events:  PAIR_FOUND, NO_OPPOSITE_PAIR, SIGNAL, EXIT_DUE,
             NO_BREAKOUT_BY_CUTOFF, DONE_FOR_DAY

    The caller owns execution: after a SIGNAL it calls open_position() on a
    fill or signal_failed() if the entry did not happen; after EXIT_DUE it
    calls close_position() once the leg is actually closed.
    """

    def __init__(self, cfg: dict = CFG):
        self.cfg = cfg
        self.prev          = None     # last candle eligible to start a pair
        self.pair          = None     # (c1, c2)
        self.range_high    = None
        self.range_low     = None
        self.pair_closed   = False    # pair search finished (found or failed)
        self.trades        = 0        # positions opened today
        self.results       = []       # exit reasons, in order
        self.position      = None
        self.pending       = False    # SIGNAL emitted, waiting for the caller
        self.exiting       = False    # EXIT_DUE emitted, waiting for the caller
        self.last_exit_bar = None     # exit consumes its candle
        self.done          = None     # reason no more entries today
        self.partial_bar   = None     # ts of a partial candle (restart)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _close_time(self, c: dict) -> dtime:
        return (c["ts"] + timedelta(minutes=self.cfg["candle_minutes"])).time()

    def can_enter(self) -> bool:
        if self.done or self.position or self.pending or self.range_high is None:
            return False
        if self.trades >= self.cfg["max_trades_per_day"]:
            return False
        if self.trades == 0:
            return True
        reentries = self.trades - 1
        return self.results[-1] == "SL_HIT" and reentries < self.cfg["max_reentries"]

    def _finish(self, reason: str, ts) -> list:
        if self.done:
            return []
        self.done = reason
        return [{"event": "DONE_FOR_DAY", "reason": reason, "ts": ts}]

    def _check_done(self, ts) -> list:
        if self.done or self.position or self.pending or self.exiting:
            return []
        if self.trades >= self.cfg["max_trades_per_day"]:
            return self._finish("MAX_TRADES", ts)
        if self.trades and self.results[-1] != "SL_HIT":
            return self._finish(f"NO_REENTRY_AFTER_{self.results[-1]}", ts)
        if self.trades and self.trades - 1 >= self.cfg["max_reentries"]:
            return self._finish("MAX_REENTRIES", ts)
        return []

    # ── exits ────────────────────────────────────────────────────────────────

    def exit_hit(self, price: float = None, candle: dict = None) -> Optional[tuple]:
        """
        (reason, index_exit_price) if the open position's SL/target is hit.
        Close mode looks at candle['close']; touch mode at a tick price, or at
        a candle's high/low (replay — SL checked first, the conservative order).
        """
        p = self.position
        if p is None or self.exiting:
            return None
        bull = p["side"] == BULL
        if candle is not None and self.cfg["exit_trigger"] == "close":
            px = candle["close"]
            if (px <= p["sl"]) if bull else (px >= p["sl"]):
                return "SL_HIT", px
            if (px >= p["target"]) if bull else (px <= p["target"]):
                return "TARGET", px
            return None
        if candle is not None:   # touch, replay on candle extremes
            if (candle["low"] <= p["sl"]) if bull else (candle["high"] >= p["sl"]):
                return "SL_HIT", p["sl"]
            if (candle["high"] >= p["target"]) if bull else (candle["low"] <= p["target"]):
                return "TARGET", p["target"]
            return None
        if price is None or self.cfg["exit_trigger"] != "touch":
            return None
        if (price <= p["sl"]) if bull else (price >= p["sl"]):
            return "SL_HIT", price
        if (price >= p["target"]) if bull else (price <= p["target"]):
            return "TARGET", price
        return None

    def on_price(self, price: float, ts: datetime) -> list:
        """Tick-level check — only does anything for exit_trigger='touch'."""
        hit = self.exit_hit(price=price)
        if not hit:
            return []
        self.exiting = True
        return [{"event": "EXIT_DUE", "reason": hit[0], "index_px": hit[1], "ts": ts}]

    def time_exit_due(self, ts: datetime) -> bool:
        return (self.position is not None and not self.exiting
                and ts.time() >= self.cfg["square_off_time"])

    # ── state transitions driven by the caller ───────────────────────────────

    def open_position(self, side: str, entry_spot: float, ts: datetime) -> dict:
        width = self.range_high - self.range_low
        bull  = side == BULL
        self.pending  = False
        self.trades  += 1
        self.position = {
            "side"      : side,
            "trade_no"  : self.trades,
            "entry_spot": entry_spot,
            "entry_ts"  : ts,
            "sl"        : self.range_low if bull else self.range_high,
            "target"    : entry_spot + width if bull else entry_spot - width,
        }
        return self.position

    def signal_failed(self):
        self.pending = False

    def mark_exiting(self):
        self.exiting = True

    def close_position(self, reason: str, bar_ts: datetime) -> list:
        self.position      = None
        self.exiting       = False
        self.last_exit_bar = bar_ts
        self.results.append(reason)
        return self._check_done(bar_ts)

    # ── main entry point ─────────────────────────────────────────────────────

    def on_candle(self, c: dict) -> list:
        cfg = self.cfg
        t   = c["ts"].time()
        if t < cfg["first_candle"]:
            return []    # 09:15 (and anything earlier) is skipped
        ev = []
        close_t = self._close_time(c)

        # 1. Exit check on the open position (close mode, or replay touch).
        if self.position is not None and not self.exiting:
            hit = self.exit_hit(candle=c)
            if hit:
                self.exiting = True
                return [{"event": "EXIT_DUE", "reason": hit[0], "index_px": hit[1],
                         "ts": c["ts"], "bar_ts": c["ts"], "candle": c}]
            if close_t >= cfg["square_off_time"]:
                self.exiting = True
                return [{"event": "EXIT_DUE", "reason": "TIME_EXIT", "index_px": c["close"],
                         "ts": c["ts"], "bar_ts": c["ts"], "candle": c}]
            return []

        # 2. Pair search.
        if not self.pair_closed:
            if close_t > cfg["entry_cutoff"]:
                self.pair_closed = True
                ev.append({"event": "NO_OPPOSITE_PAIR", "reason": "none_before_cutoff", "ts": c["ts"]})
                ev += self._finish("NO_OPPOSITE_PAIR", c["ts"])
                return ev
            return self._pair_step(c)

        # 3. Breakout.
        if self.done or self.pending or self.exiting or self.range_high is None:
            return []
        if self.last_exit_bar is not None and c["ts"] <= self.last_exit_bar:
            return []
        if close_t > cfg["entry_cutoff"]:
            if self.can_enter():
                ev.append({"event": "NO_BREAKOUT_BY_CUTOFF", "trades": self.trades, "ts": c["ts"]})
            return ev + self._finish("ENTRY_CUTOFF", c["ts"])
        if not self.can_enter():
            return self._check_done(c["ts"])

        if c["close"] > self.range_high:
            side = BULL
        elif c["close"] < self.range_low:
            side = BEAR
        else:
            return []
        self.pending = True
        return [{"event": "SIGNAL", "side": side, "leg": SELL_LEG[side], "candle": c,
                 "ts": c["ts"], "trade_no": self.trades + 1}]

    def _pair_step(self, c: dict) -> list:
        cfg   = self.cfg
        fixed = cfg["pair_mode"] == "fixed_2nd_3rd"
        col   = None if c["ts"] == self.partial_bar else _colour(c)

        if fixed:
            second = (datetime.combine(c["ts"].date(), cfg["first_candle"])
                      + timedelta(minutes=cfg["candle_minutes"])).time()
            if c["ts"].time() == cfg["first_candle"]:
                self.prev = c if col else None
                if col is None:
                    return self._pair_fail(c, "first_candle_doji")
                return []
            if c["ts"].time() != second or self.prev is None:
                return []   # (already failed, or a gap in the data)
            prev_col = _colour(self.prev)
            if col is None or col == prev_col:
                return self._pair_fail(c, "fixed_pair_not_opposite")
            return self._set_pair(self.prev, c)

        if col is None:
            self.prev = None   # a doji breaks the chain
            return []
        step = timedelta(minutes=cfg["candle_minutes"])
        if (self.prev is not None and c["ts"] - self.prev["ts"] == step
                and _colour(self.prev) != col):
            return self._set_pair(self.prev, c)
        self.prev = c
        return []

    def _pair_fail(self, c: dict, reason: str) -> list:
        self.pair_closed = True
        ev = [{"event": "NO_OPPOSITE_PAIR", "reason": reason, "ts": c["ts"]}]
        return ev + self._finish("NO_OPPOSITE_PAIR", c["ts"])

    def _set_pair(self, c1: dict, c2: dict) -> list:
        self.pair        = (c1, c2)
        self.pair_closed = True
        self.range_high  = max(c1["high"], c2["high"])
        self.range_low   = min(c1["low"], c2["low"])
        return [{"event": "PAIR_FOUND", "c1": c1, "c2": c2, "range_high": self.range_high,
                 "range_low": self.range_low, "ts": c2["ts"]}]


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY — execution, logging and restart around the engine
# ═════════════════════════════════════════════════════════════════════════════

class BankNifty2CandleBreakoutStrategy(BaseStrategy):

    # NOTE: deliberately NO class-level INDEX_TOKEN attribute — see
    # "INDEX ROUTING" in the module docstring.

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "BANKNIFTY_2CANDLE_BO"

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self._cb          = CandleBuilder(minutes=CFG["candle_minutes"])
        self._engine      = TwoCandleEngine(CFG)
        self._instruments = None
        self._expiry      = None
        self._ready       = False
        self._day         = None

        self._trade       = None    # open option leg (or virtual stale trade)
        self._pending     = None    # signal waiting for a live option price
        self._last_spot   = None
        self._completed   = []
        self._today_gross = 0.0
        self._today_net   = 0.0
        self._eod_done    = False

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | pair_mode={CFG['pair_mode']} "
            f"exit_trigger={CFG['exit_trigger']} cutoff={CFG['entry_cutoff']:%H:%M} "
            f"square_off={CFG['square_off_time']:%H:%M}"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        if not CFG["enabled"]:
            log.info(f"[{self.name}] disabled via CFG")
            return False

        self._instruments = instruments
        self._day = _now_ist().date()
        try:
            if CFG["min_dte"] > 0:
                self._expiry = get_nearest_expiry(
                    instruments._df, self._day + timedelta(days=CFG["min_dte"]))
            else:
                self._expiry = pm.expiry_date
        except Exception as e:
            log.error(f"[{self.name}] expiry resolution failed: {e}")
            self._expiry = None
        if self._expiry is None:
            log.error(f"[{self.name}] no expiry in the instruments dump — skipping today")
            return False

        # Seeding/restore run on the FIRST live tick (_startup), not here: t.py
        # does recorder setup and two 5-min backfills between pre_market() and
        # the WebSocket start, and a candle closing in that gap would be lost.
        self._started = False
        self._ready = True
        log.info(
            f"[{self.name}] Pre-market | expiry={self._expiry} "
            f"DTE={(self._expiry - self._day).days} "
            f"mode={'LIVE' if LIVE_MODE else 'PAPER'}"
        )
        return True

    def _startup(self, now: datetime):
        self._started = True
        self._seed_today(now)
        self._restore_state()
        self._flag_stale()

    def _seed_today(self, now: datetime = None):
        """
        Mid-session restart: replay today's completed 1-min candles through
        the engine in virtual mode (no orders). MarketHub.backfill() is
        5-minute only, so this fetches 1-minute bars itself.
        """
        now  = now or _now_ist()
        kite = getattr(self._hub, "kite", None)
        if kite is None or now.time() <= CFG["first_candle"]:
            return
        start   = datetime.combine(now.date(), CFG["session_open"])
        cur_bar = now.replace(second=0, microsecond=0)
        try:
            raw = kite.historical_data(BANKNIFTY_INDEX_TOKEN, start, now, "minute")
        except Exception as e:
            log.error(f"[{self.name}] 1-min seed FAILED: {e} — pair search starts from live candles")
            return
        bars = []
        for r in raw or []:
            ts = r["date"].replace(tzinfo=None) if hasattr(r["date"], "tzinfo") else r["date"]
            if ts >= cur_bar:
                break
            bars.append({"ts": ts, "open": float(r["open"]), "high": float(r["high"]),
                         "low": float(r["low"]), "close": float(r["close"])})
        self.seed_candles(bars, cur_bar)

    def seed_candles(self, bars: list, cur_bar: datetime):
        """Replay closed candles in virtual mode. cur_bar = the minute in progress."""
        eng = self._engine
        for c in bars:
            for e in eng.on_candle(c):
                if e["event"] == "SIGNAL":
                    eng.open_position(e["side"], c["close"], c["ts"])
                elif e["event"] == "EXIT_DUE":
                    eng.close_position(e["reason"], e["bar_ts"])
                elif e["event"] in ("PAIR_FOUND", "NO_OPPOSITE_PAIR"):
                    self._log_event(e, seeded=True)
        # The first live candle starts mid-minute — its open is not the real open.
        eng.partial_bar = cur_bar
        if bars:
            self._last_spot = bars[-1]["close"]

        if eng.position is not None:
            p = eng.position
            self._trade = {"virtual": True, "trade_no": p["trade_no"], "side": p["side"],
                           "entry_ts": p["entry_ts"], "entry_spot": p["entry_spot"]}
        if bars:
            self._signal_row("RESTART_SEEDED", bars[-1]["ts"],
                             note=f"{len(bars)} candles; trades={eng.trades} "
                                  f"results={eng.results} done={eng.done}")
            log.info(f"[{self.name}] RESTART seeded {len(bars)} 1-min candles | "
                     f"range={eng.range_low}-{eng.range_high} trades={eng.trades} "
                     f"results={eng.results} done={eng.done}")

    def _flag_stale(self):
        """A replayed trade still open at restart (and no real position) is never entered late."""
        t, p = self._trade, self._engine.position
        if not (t and t.get("virtual") and p):
            return
        log.warning(
            f"[{self.name}] RESTART: trade #{p['trade_no']} {p['side']} from "
            f"{p['entry_ts']:%H:%M} is still open on the index — NOT entered late "
            f"(stale); tracking it on the index only"
        )
        self._signal_row("STALE_SIGNAL_SKIPPED", p["entry_ts"], side=p["side"],
                         trade_no=p["trade_no"], spot=p["entry_spot"],
                         sl=p["sl"], target=p["target"])

    # ── Persisted open position (restart) ─────────────────────────────────────

    def _save_state(self):
        t = self._trade
        if not t or t.get("virtual"):
            return
        keep = {k: v for k, v in t.items() if not k.startswith("_")}
        keep["day"] = self._day.isoformat() if self._day else None
        try:
            with open(CFG["state_file"], "w") as f:
                json.dump(keep, f, default=str)
        except Exception as e:
            log.error(f"[{self.name}] state save failed: {e}")

    def _clear_state(self):
        try:
            if os.path.isfile(CFG["state_file"]):
                os.remove(CFG["state_file"])
        except Exception as e:
            log.error(f"[{self.name}] state clear failed: {e}")

    def _restore_state(self):
        path = CFG["state_file"]
        if not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                st = json.load(f)
        except Exception as e:
            log.error(f"[{self.name}] state file unreadable ({e}) — ignored")
            return
        if st.get("day") != (self._day.isoformat() if self._day else None):
            log.info(f"[{self.name}] state file is from {st.get('day')} — discarded")
            self._clear_state()
            return

        if st.get("mode") == "LIVE" or LIVE_MODE:
            qty = self._hub.order_router.net_quantity(st["symbol"])
            if qty == 0:
                log.error(f"[{self.name}] RESTART: persisted short {st['symbol']} is FLAT in "
                          f"Zerodha (closed outside the bot) — not re-attached")
                self._signal_row("RESTART_POSITION_GONE", _now_ist(), note=st["symbol"])
                self._engine.results.append("CLOSED_EXTERNALLY")
                self._engine.trades = max(self._engine.trades, int(st["trade_no"]))
                self._engine.position = None
                self._trade = None
                self._clear_state()
                return
            if qty is not None and qty != -int(st["qty"]):
                log.error(f"[{self.name}] RESTART: {st['symbol']} net qty {qty} != "
                          f"expected -{st['qty']} — CHECK ZERODHA CONSOLE")
            if not self._acquire_slot():
                log.error(f"[{self.name}] RESTART: live slot held by "
                          f"{self._hub.order_router.slot_owner()} — position managed anyway")

        for k in ("entry_ts", "pair_c1", "pair_c2"):
            if st.get(k):
                st[k] = datetime.fromisoformat(st[k])
        st["expiry"] = date.fromisoformat(st["expiry"])
        st["_exit_in_progress"] = False
        eng = self._engine
        # The real record wins over the virtual replay.
        eng.range_high, eng.range_low = st["range_high"], st["range_low"]
        eng.pair_closed, eng.pending, eng.exiting, eng.done = True, False, False, None
        eng.trades  = int(st["trade_no"])
        eng.results = ["SL_HIT"] * (eng.trades - 1)   # a later trade only exists after an SL
        eng.position = {"side": st["side"], "trade_no": eng.trades, "entry_spot": st["entry_spot"],
                        "entry_ts": st["entry_ts"], "sl": st["sl"], "target": st["target"]}
        self._trade = st
        self.subscribe_option(st["token"])
        log.warning(f"[{self.name}] RESTART: re-attached open short {st['symbol']} "
                    f"(trade #{st['trade_no']}, SL {st['sl']:.2f}, target {st['target']:.2f})")
        self._signal_row("RESTART_POSITION_RESTORED", _now_ist(), side=st["side"],
                         trade_no=st["trade_no"], note=st["symbol"])

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        if not self._ready or not price:
            return
        tick_ts = tick_ts or ts
        if not self._started:
            self._startup(tick_ts)
        self._last_spot = price

        # 1. Candle close → exits (close mode) and signals.
        if tick_ts.time() >= CFG["session_open"]:
            closed = self._cb.feed_tick(price, 0, tick_ts)
            if closed is not None:
                self._handle_events(self._engine.on_candle(closed), ts)

        # 2. Touch-mode exit on the tick.
        if self._trade and not self._trade.get("_exit_in_progress"):
            self._handle_events(self._engine.on_price(price, ts), ts)

        # 3. SQUARE-OFF. Runs on every tick at/after square_off_time, before any
        #    other gate — an early `if t > close: return` once made this dead code.
        if self._trade and self._engine.time_exit_due(ts):
            self._engine.mark_exiting()
            self._exit("TIME_EXIT", ts, price)
        if ts.time() >= CFG["square_off_time"] and self._pending:
            self._drop_pending(ts, "square_off_time")

        # 4. Pending entry timeout.
        if self._pending and (ts - self._pending["ts"]).total_seconds() > CFG["pending_timeout_sec"]:
            self._drop_pending(ts, "no_live_option_tick")

    def on_candle(self, candle: dict, ts: datetime):
        pass   # 5-min BankNifty candles — this strategy builds its own 1-min bars

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if self._pending and token == self._pending["token"] and not self._trade:
            self._try_fill(ts)

    # ── Event dispatch ────────────────────────────────────────────────────────

    def _handle_events(self, events: list, ts: datetime):
        for e in events:
            ev = e["event"]
            if ev == "SIGNAL":
                self._log_event(e)
                self._on_signal(e, ts)
            elif ev == "EXIT_DUE":
                self._exit(e["reason"], ts, e.get("index_px"), bar_ts=e.get("bar_ts"))
            else:
                self._log_event(e)

    def _log_event(self, e: dict, seeded: bool = False):
        ev = e["event"]
        note = "seeded" if seeded else ""
        if ev == "PAIR_FOUND":
            log.info(f"[{self.name}] PAIR {e['c1']['ts']:%H:%M}+{e['c2']['ts']:%H:%M} "
                     f"range {e['range_low']:.2f} - {e['range_high']:.2f} "
                     f"(width {e['range_high'] - e['range_low']:.2f})")
            self._signal_row("PAIR_FOUND", e["ts"], pair=self._pair_str(),
                             range_high=e["range_high"], range_low=e["range_low"], note=note)
        elif ev == "SIGNAL":
            c = e["candle"]
            log.info(f"[{self.name}] BREAKOUT {e['side']} — {c['ts']:%H:%M} closed {c['close']:.2f} "
                     f"beyond {self._engine.range_low:.2f}-{self._engine.range_high:.2f} "
                     f"→ SELL {e['leg']} (trade #{e['trade_no']})")
            self._signal_row("SIGNAL", e["ts"], side=e["side"], trade_no=e["trade_no"],
                             spot=c["close"], pair=self._pair_str(),
                             range_high=self._engine.range_high, range_low=self._engine.range_low)
        else:
            log.info(f"[{self.name}] {ev} {e.get('reason', '')}")
            self._signal_row(ev, e["ts"], reason=e.get("reason", ""), pair=self._pair_str(),
                             range_high=self._engine.range_high, range_low=self._engine.range_low,
                             note=note)

    def _pair_str(self) -> str:
        p = self._engine.pair
        return f"{p[0]['ts']:%H:%M}+{p[1]['ts']:%H:%M}" if p else ""

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _on_signal(self, e: dict, ts: datetime):
        spot   = self._last_spot or e["candle"]["close"]
        leg    = e["leg"]
        step   = CFG["strike_step"]
        off    = CFG["strike_offset_steps"] * step
        strike = get_atm_strike(spot, step=step) + (off if leg == "CE" else -off)
        token, sym = self._instruments.get_option_token(strike, leg, self._expiry)
        lot = self._instruments.lot_size(token) if token else 0
        if not token or not sym or not lot:
            log.error(f"[{self.name}] no {leg} contract for strike={strike} expiry={self._expiry} "
                      f"(token={token} lot={lot}) — signal SKIPPED")
            self._signal_row("ENTRY_FAILED", ts, side=e["side"], trade_no=e["trade_no"],
                             reason="no_option_contract")
            self._engine.signal_failed()
            return
        self.subscribe_option(token)
        self._pending = {"token": token, "sym": sym, "strike": strike, "leg": leg,
                         "side": e["side"], "lot": lot, "qty": lot * CFG["lots"],
                         "ts": ts, "signal": e}
        self._try_fill(ts)

    def _drop_pending(self, ts: datetime, reason: str):
        p, self._pending = self._pending, None
        self.unsubscribe_option(p["token"])
        self._engine.signal_failed()
        log.warning(f"[{self.name}] entry for {p['sym']} dropped — {reason}")
        self._signal_row("ENTRY_FAILED", ts, side=p["side"], trade_no=p["signal"]["trade_no"],
                         reason=reason, note=p["sym"])

    def _spread(self, token: int, ltp: float) -> float:
        bid, ask, _, _ = self._hub.best_bid_ask(token)
        if bid and ask and ask > bid > 0 and (ask - bid) <= max(0.5 * ltp, 1.0):
            return round(ask - bid, 2)
        return estimate_spread(ltp)

    def _try_fill(self, ts: datetime):
        p = self._pending
        tok, sym = p["token"], p["sym"]
        raw = self.get_price(tok)
        pts = self.get_price_ts(tok)
        if not raw or raw <= 0 or pts is None or (ts - pts).total_seconds() > CFG["max_price_age_sec"]:
            return   # wait for a live tick (on_option_tick) or the timeout

        if LIVE_MODE:
            ok, need, avail = self._hub.order_router.check_margin(
                self.name, sym, p["qty"], "SELL", CFG["margin_per_lot_rs"] * CFG["lots"])
            if not ok:
                self._drop_pending(ts, f"margin need={need:.0f} avail={avail:.0f}")
                return

        if not self._acquire_slot():
            self._drop_pending(ts, f"live_slot_held_by_{self._hub.order_router.slot_owner()}")
            return

        res = self._place_sell(sym, tok, p["qty"], raw)   # SELL to open
        if res is None:
            self._release_slot()
            self._drop_pending(ts, "sell_order_failed")
            return
        order_id, fill = res
        spread = self._spread(tok, raw)
        if not LIVE_MODE:
            fill = round(max(0.05, raw - spread / 2.0), 2)   # paper: you hit the bid
        self._pending = None

        spot = self._last_spot or p["signal"]["candle"]["close"]
        pos  = self._engine.open_position(p["side"], spot, ts)
        eng  = self._engine
        c1, c2 = eng.pair
        self._trade = {
            "virtual": False, "mode": "LIVE" if LIVE_MODE else "PAPER",
            "trade_no": pos["trade_no"], "side": p["side"], "leg": p["leg"],
            "symbol": sym, "token": tok, "strike": p["strike"], "expiry": self._expiry,
            "dte": (self._expiry - ts.date()).days, "lot": p["lot"], "qty": p["qty"],
            "pair_c1": c1["ts"], "pair_c2": c2["ts"],
            "range_high": eng.range_high, "range_low": eng.range_low,
            "signal_ts": p["signal"]["ts"], "signal_close": p["signal"]["candle"]["close"],
            "entry_ts": ts, "entry_spot": spot, "entry_raw": raw,
            "entry_slip": round(raw - fill, 2), "entry_fill": fill, "spread": spread,
            "sl": pos["sl"], "target": pos["target"], "order_id": order_id,
            "_exit_in_progress": False,
        }
        self._save_state()
        log.info(
            f"[{self.name}] [{self._trade['mode']}] ENTRY #{pos['trade_no']} SELL {sym} "
            f"RawTick={raw:.2f} Slip={raw - fill:.2f} Fill={fill:.2f} qty={p['qty']} | "
            f"spot {spot:.2f} SL {pos['sl']:.2f} target {pos['target']:.2f} | "
            f"expiry {self._expiry} DTE {self._trade['dte']} | order_id={order_id}"
        )
        self._signal_row("ENTRY_FILLED", ts, side=p["side"], trade_no=pos["trade_no"],
                         spot=spot, note=f"{sym} @ {fill:.2f}")

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _exit(self, reason: str, ts: datetime, index_px: float = None, bar_ts: datetime = None):
        t = self._trade
        bar_ts = bar_ts or ts.replace(second=0, microsecond=0)
        if t is None:
            # EXIT_DUE without a position on our side (should not happen) — keep the engine consistent.
            self._handle_done(self._engine.close_position(reason, bar_ts), ts)
            return
        if t.get("_exit_in_progress"):
            return
        exit_spot = self._last_spot if self._last_spot else index_px

        if t.get("virtual"):
            p = self._engine.position or {}
            pts = ((exit_spot - t["entry_spot"]) if t["side"] == BULL
                   else (t["entry_spot"] - exit_spot)) if exit_spot else None
            log.info(f"[{self.name}] STALE (virtual) trade #{t['trade_no']} exit [{reason}] "
                     f"spot {exit_spot} | index pts {pts}")
            self._signal_row("STALE_VIRTUAL_EXIT", ts, side=t["side"], trade_no=t["trade_no"],
                             reason=reason, spot=exit_spot, sl=p.get("sl"), target=p.get("target"))
            self._trade = None
            self._handle_done(self._engine.close_position(reason, bar_ts), ts)
            return

        t["_exit_in_progress"] = True
        tok = t["token"]
        raw = self.get_price(tok)
        age = self.get_price_ts(tok)
        if age is not None and (ts - age).total_seconds() > CFG["max_price_age_sec"]:
            log.warning(f"[{self.name}] {t['symbol']} exit on a "
                        f"{(ts - age).total_seconds():.0f}s-old option price")
        if not raw:
            raw = t["entry_raw"]
            log.error(f"[{self.name}] {t['symbol']} NO option price at exit — booked at entry raw")

        res = self._hub.order_router.place_buy_with_retry(   # BUY to close
            self.name, t["symbol"], tok, t["qty"], raw, LIVE_MODE, max_retries=3)
        if res is not None:
            order_id, fill = res
            self._finalize(t, raw, fill, order_id, reason, ts, exit_spot, bar_ts)
            return

        if LIVE_MODE:
            if self._hub.order_router.net_quantity(t["symbol"]) == 0:
                log.warning(f"[{self.name}] BUY failed but {t['symbol']} is FLAT at the exchange")
                self._finalize(t, raw, raw, None, f"{reason}_EXCHANGE_CLOSED", ts, exit_spot, bar_ts)
                return
            log.error(f"\n{'!'*60}\n[{self.name}] CRITICAL: BUY-to-close FAILED — SHORT "
                      f"{t['symbol']} STILL OPEN. Slot LOCKED. Emergency thread starting.\n{'!'*60}")
            self._start_emergency_exit(t, raw, reason, exit_spot, bar_ts)
            return
        # Paper buy never fails, but handle it.
        self._finalize(t, raw, raw, None, reason, ts, exit_spot, bar_ts)

    def _start_emergency_exit(self, t: dict, ref: float, reason: str, exit_spot, bar_ts):
        def _loop():
            n, wait = CFG["emergency_max_attempts"], CFG["emergency_retry_sec"]
            for attempt in range(1, n + 1):
                _time_mod.sleep(wait)
                log.error(f"[{self.name}] Emergency BUY attempt {attempt}/{n} | {t['symbol']}")
                if self._hub.order_router.net_quantity(t["symbol"]) == 0:
                    self._finalize(t, ref, ref, None, f"{reason}_EXCHANGE_CLOSED",
                                   _now_ist(), exit_spot, bar_ts)
                    return
                ltp = self.get_price(t["token"]) or ref
                res = self._hub.order_router.place_buy(
                    self.name, t["symbol"], t["token"], t["qty"], ltp, LIVE_MODE)
                if res:
                    self._finalize(t, ltp, res[1], res[0], f"{reason}_EMERGENCY",
                                   _now_ist(), exit_spot, bar_ts)
                    return
            log.error(f"\n{'!'*60}\n[{self.name}] GAVE UP closing SHORT {t['symbol']} after "
                      f"{n} attempts.\n  *** SQUARE OFF MANUALLY IN ZERODHA CONSOLE NOW ***\n{'!'*60}")

        threading.Thread(target=_loop, name="bn-2candle-bo-emergency-exit", daemon=True).start()

    def _finalize(self, t: dict, raw: float, fill: float, order_id, reason: str,
                  ts: datetime, exit_spot, bar_ts: datetime):
        spread = self._spread(t["token"], raw) if raw else t["spread"]
        if not LIVE_MODE:
            fill = round(raw + spread / 2.0, 2)   # paper: buying back lifts the offer
        qty   = t["qty"]
        gross = round((t["entry_fill"] - fill) * qty, 2)
        # fixed_costs_rs(qty, BUY px, SELL px): the short's buy leg is the exit.
        costs = fixed_costs_rs(qty, fill, t["entry_fill"])
        net   = round(gross - costs, 2)
        self._today_gross += gross
        self._today_net   += net
        t.update(exit_ts=ts, exit_spot=exit_spot, exit_raw=raw, exit_slip=round(fill - raw, 2),
                 exit_fill=fill, exit_reason=reason, gross=gross, costs=costs, net=net,
                 exit_order_id=order_id)

        self._release_slot()
        self.unsubscribe_option(t["token"])
        self._completed.append(t)
        self._trade = None
        self._clear_state()

        log.info(
            f"[{self.name}] [{t['mode']}] EXIT #{t['trade_no']} [{reason}] BUY {t['symbol']} "
            f"RawTick={raw:.2f} Slip={fill - raw:.2f} Exit={fill:.2f} | "
            f"{t['entry_fill']:.2f} → {fill:.2f} | spot {exit_spot} | "
            f"gross={gross:.0f} costs={costs:.0f} net={net:.0f} | day net={self._today_net:.0f}"
        )
        self._log_trade(t)
        base = reason.split("_EXCHANGE")[0].split("_EMERGENCY")[0]
        self._handle_done(self._engine.close_position(base, bar_ts), ts)

    def _handle_done(self, events: list, ts: datetime):
        for e in events:
            self._log_event(e)

    # ── CSV logging ───────────────────────────────────────────────────────────

    TRADE_FIELDS = [
        "date", "trade_no", "side", "pair_c1", "pair_c2", "range_high", "range_low",
        "expiry", "dte", "strike", "symbol", "qty",
        "entry_time", "entry_spot", "entry_raw_tick", "entry_slip", "entry_premium",
        "sl", "target",
        "exit_time", "exit_spot", "exit_raw_tick", "exit_slip", "exit_premium",
        "exit_reason", "gross_pnl", "costs", "net_pnl", "mode", "order_id", "exit_order_id",
    ]

    SIGNAL_FIELDS = [
        "timestamp", "event", "side", "trade_no", "pair", "range_high", "range_low",
        "spot", "sl", "target", "reason", "note", "mode",
    ]

    def _log_trade(self, t: dict):
        try:
            append_row(CFG["csv_file"], self.TRADE_FIELDS, {
                "date": t["entry_ts"].strftime("%Y-%m-%d"), "trade_no": t["trade_no"],
                "side": t["leg"], "pair_c1": t["pair_c1"].strftime("%H:%M"),
                "pair_c2": t["pair_c2"].strftime("%H:%M"),
                "range_high": t["range_high"], "range_low": t["range_low"],
                "expiry": t["expiry"], "dte": t["dte"], "strike": t["strike"],
                "symbol": t["symbol"], "qty": t["qty"],
                "entry_time": t["entry_ts"].strftime("%H:%M:%S"),
                "entry_spot": round(t["entry_spot"], 2), "entry_raw_tick": t["entry_raw"],
                "entry_slip": t["entry_slip"], "entry_premium": t["entry_fill"],
                "sl": round(t["sl"], 2), "target": round(t["target"], 2),
                "exit_time": t["exit_ts"].strftime("%H:%M:%S"),
                "exit_spot": round(t["exit_spot"], 2) if t["exit_spot"] else "",
                "exit_raw_tick": t["exit_raw"], "exit_slip": t["exit_slip"],
                "exit_premium": t["exit_fill"], "exit_reason": t["exit_reason"],
                "gross_pnl": t["gross"], "costs": t["costs"], "net_pnl": t["net"],
                "mode": t["mode"], "order_id": t["order_id"], "exit_order_id": t["exit_order_id"],
            })
        except Exception as e:
            log.error(f"[{self.name}] trade CSV write failed: {e}")

    def _signal_row(self, event: str, ts: datetime, **kw):
        row = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "event": event,
               "mode": "LIVE" if LIVE_MODE else "PAPER"}
        row.update({k: ("" if v is None else v) for k, v in kw.items()})
        try:
            append_row(CFG["signals_csv"], self.SIGNAL_FIELDS, row)
        except Exception as e:
            log.error(f"[{self.name}] signal CSV write failed: {e}")

    # ── EOD ───────────────────────────────────────────────────────────────────

    def eod_summary(self):
        # Safety net: the 15:15 square-off normally fires from on_tick.
        if self._trade and not self._trade.get("_exit_in_progress"):
            log.error(f"[{self.name}] position still open at EOD — squaring off now")
            self._engine.mark_exiting()
            self._exit("TIME_EXIT", _now_ist(), self._last_spot)

        eng = self._engine
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'} | "
                 f"expiry={self._expiry} | pair={self._pair_str() or 'none'} "
                 f"range={eng.range_low}-{eng.range_high} | done={eng.done}")
        for t in self._completed:
            log.info(f"[{self.name}]   #{t['trade_no']} SELL {t['symbol']} [{t['exit_reason']}] "
                     f"{t['entry_fill']:.2f} → {t['exit_fill']:.2f} "
                     f"gross={t['gross']:.0f} net={t['net']:.0f}")
        wins = sum(1 for t in self._completed if t["net"] > 0)
        log.info(f"[{self.name}] Trades {len(self._completed)} | W/L {wins}/{len(self._completed) - wins} "
                 f"| gross {self._today_gross:.0f} | NET {self._today_net:.0f}")
        log.info(f"[{self.name}] {'='*50}\n")
