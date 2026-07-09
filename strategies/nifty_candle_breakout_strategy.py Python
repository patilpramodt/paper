"""
strategies/nifty_candle_breakout_strategy.py

NIFTY_CANDLE_BREAKOUT — 10s marubozu + 5s confirm + tick breakout, Nifty 50 only.

REQUIREMENTS (as given)
────────────────────────
  1. Nifty only.
  2. Watch every closed 10-second index candle. A candle qualifies as a
     "trigger" (C1) if it has NO wick (open/close occupy the full
     high-low range, within a small tolerance) AND its body
     (|close - open|) is strictly greater than 10 points.
  3. After C1 closes, watch the very next 5-second candle (the first
     half of the following 10s bucket). If it closes the SAME color as
     C1 → confirmed, otherwise the setup is abandoned.
  4. Once confirmed, watch price tick-by-tick for the rest of that same
     10-second window (the "next 10 sec candle", which the confirm
     5s candle is the first half of). The moment price breaks the
     confirm candle's high (GREEN → CE) or low (RED → PE), enter
     IMMEDIATELY — no waiting for any candle to close.
  5. SL/TP are fixed at 10 points on the OPTION PREMIUM (not the index),
     matching the convention used by SPIKE / SPIKE_NIFTY.
  6. "Skip" rule: entry also requires price to have crossed the confirm
     candle's CLOSE (not just high/low) — for a clean GREEN breakout the
     close is always <= high so this is implied, but it's checked
     explicitly for symmetry and safety. If the watched 10-second window
     closes with no breakout, the setup is abandoned and the state
     machine resets to scan for a fresh C1 (per explicit confirmation —
     we do NOT keep watching subsequent windows).

CLARIFICATIONS CONFIRMED WITH USER
────────────────────────────────────
  - "last close" in the skip rule = the CONFIRM 5-second candle's close.
  - SL/TP of 10 points = option premium points (like SPIKE/SPIKE_NIFTY),
    not index points.
  - Breakout is only watched within the single 10-second candle right
    after the confirm candle. If it closes with no breakout, reset.

ASSUMPTIONS (stated — not covered by the spec, flag if wrong)
─────────────────────────────────────────────────────────────
  - This strategy is NOT limited to the 9:15 open like SPIKE/SPIKE_NIFTY.
    It scans all day (09:15–15:00 for new setups, force-exits any open
    trade by 15:15) and can take MULTIPLE trades per day, one at a time.
  - "No wick" allows a small tolerance (wick_tolerance_pts, default 0.5)
    since real tick data essentially never produces an exact-zero wick.
  - ATM strike is recomputed at the moment of each signal (not fixed at
    market open), since this strategy can fire at any time of day and
    spot may have drifted from the pre-market ATM. This mirrors the
    pattern used in nifty_directional_strategy.py.
  - One trade at a time for this strategy; the global live-trade slot
    (OrderRouter) still applies on top of that in LIVE_MODE.

INDEX ROUTING
──────────────
  INDEX_TOKEN = 256265 (NSE:NIFTY 50). MarketHub routes Nifty ticks here
  exclusively, same mechanism used by SPIKE_NIFTY / BB_STOCH_NIFTY /
  NIFTY_DIRECTIONAL. Uses the shared nifty_pm / nifty_instruments already
  wired up in t.py for any strategy with this INDEX_TOKEN.

LIVE / PAPER MODE
─────────────────
  LIVE_MODE = False below. Flip to True only when ready to trade real
  money — all other strategies are unaffected by this flag.

ORDER EXECUTION / ROBUSTNESS
──────────────────────────────
  Mirrors the bug-fixed patterns already established in spike_nifty.py /
  nifty_directional_strategy.py:
    - _exit_in_progress guard against concurrent exit attempts.
    - Slot released ONLY after position is confirmed closed.
    - Emergency exit background thread if SELL fails repeatedly in
      LIVE_MODE (position stays open, slot stays locked, retried in the
      background rather than silently abandoned).
    - SL/TP checked only after a short grace period post-fill to avoid
      acting on stale buffered ticks.
"""

import csv
import logging
import os
import threading
import time as _time_mod
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder
from core.instruments import get_atm_strike

log = logging.getLogger("strategy.nifty_candle_breakout")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG — change to True only when ready to trade real money.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

NIFTY_STRIKE_STEP = 50

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "quantity"               : 65,

    # ── Session windows ───────────────────────────────────────────────────────
    "start_time"             : dtime(9, 15),
    "last_entry_time"        : dtime(15, 0),   # stop scanning for NEW setups after this
    "close_time"             : dtime(15, 15),  # force-exit any open trade

    # ── Candle pattern parameters ─────────────────────────────────────────────
    "bucket_10s"             : 10,
    "bucket_5s"              : 5,
    "min_body_pts"           : 10.0,   # body must be strictly greater than this
    "wick_tolerance_pts"     : 0.5,    # "no wick" allowed tolerance

    # ── SL / TP (fixed, on OPTION PREMIUM) ────────────────────────────────────
    "sl_points"              : 10.0,
    "tp_points"              : 10.0,
    "sl_grace_seconds"       : 5,

    # ── Emergency exit (LIVE_MODE only) ───────────────────────────────────────
    "emergency_retry_sec"    : 30,
    "emergency_max_attempts" : 30,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"               : "nifty_candle_breakout_trades.csv",
}


class NiftyCandleBreakoutStrategy(BaseStrategy):

    # Routes Nifty 50 index ticks here exclusively (shared with other
    # Nifty strategies — MarketHub delivers to every strategy with this token).
    INDEX_TOKEN = 256265

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "NIFTY_CANDLE_BREAKOUT"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._c10 = SecondCandleBuilder(seconds=CFG["bucket_10s"])
        self._c5  = SecondCandleBuilder(seconds=CFG["bucket_5s"])

        self._market_opened = False
        self._expiry_date    = None
        self._instruments    = None

        # ── Pattern state machine ─────────────────────────────────────────────
        # states: SCAN → WAIT_CONFIRM → WATCH_BREAKOUT → (SCAN)
        self._state              = "SCAN"
        self._c1                 = None   # trigger 10s candle dict
        self._trigger_color      = None   # "GREEN" / "RED"
        self._confirm5           = None   # confirm 5s candle dict
        self._breakout_window_ts = None   # ts of the 10s bucket being watched

        self._trade         = None
        self._pending_entry  = None
        self._trades_today   = 0
        self._today_pnl      = 0.0
        self._completed      = []

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | qty={CFG['quantity']} "
            f"min_body={CFG['min_body_pts']}pts wick_tol={CFG['wick_tolerance_pts']}pts "
            f"SL=-{CFG['sl_points']} TP=+{CFG['tp_points']}"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        """
        Receives Nifty-specific PreMarketData + InstrumentStore (routed by
        t.py because INDEX_TOKEN == 256265). No pre-subscription of a fixed
        ATM pair is done here — ATM drifts through the day and this strategy
        can fire at any time, so the strike is resolved fresh at signal time.
        """
        self._instruments = instruments
        self._expiry_date = pm.expiry_date

        log.info(
            f"[{self.name}] Pre-market | expiry={pm.expiry_date} "
            f"mode={'LIVE' if LIVE_MODE else 'PAPER'}"
        )
        return True

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

        closed10 = self._c10.feed_tick(price, tick_ts)
        closed5  = self._c5.feed_tick(price, tick_ts)

        # ── Force-exit any open trade at close_time ───────────────────────────
        if (self._trade and self._trade["state"] == "OPEN"
                and not self._trade.get("_exit_in_progress")):
            if t >= CFG["close_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "EOD_CLOSE", ts)
            return

        # A trade is open — freeze pattern scanning until it closes.
        if self._trade is not None:
            return

        # No new setups too close to EOD.
        if t >= CFG["last_entry_time"]:
            return

        # ── Pattern state machine ─────────────────────────────────────────────
        if self._state == "SCAN":
            if closed10 is not None:
                self._check_trigger_candle(closed10)

        elif self._state == "WAIT_CONFIRM":
            if closed5 is not None:
                self._check_confirm_candle(closed5, ts)

        elif self._state == "WATCH_BREAKOUT":
            self._check_breakout_tick(price, ts)
            # If the watched window just closed with no breakout, reset.
            if (self._state == "WATCH_BREAKOUT" and closed10 is not None
                    and closed10["ts"] == self._breakout_window_ts):
                log.info(
                    f"[{self.name}] No breakout within watched window "
                    f"({self._breakout_window_ts.strftime('%H:%M:%S')}) — resetting scan"
                )
                self._reset_pattern_state()

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Resolves a pending entry (if the option had no valid live price at
        signal time) and manages fixed SL/TP for the open trade.
        """
        # ── Resolve pending entry ─────────────────────────────────────────────
        if (self._pending_entry and token == self._pending_entry["token"]
                and not self._trade):
            p = self._pending_entry
            self._pending_entry = None
            log.info(
                f"[{self.name}] Pending entry resolved — first live tick for "
                f"{p['sym']} @ {price:.2f}"
            )
            self._build_entry(p["sym"], p["token"], p["signal"], ts, p["reason"])
            return

        if not (self._trade and token == self._trade.get("token")):
            return

        if self._trade["state"] != "OPEN" or self._trade.get("_exit_in_progress"):
            return

        # ── SL / TP grace period ──────────────────────────────────────────────
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from is not None and ts < sl_active_from:
            return

        if price <= self._trade["sl"]:
            self._do_exit(price, "SL_HIT", ts)
        elif price >= self._trade["tp"]:
            self._do_exit(price, "TP_HIT", ts)

    # ── Pattern detection ────────────────────────────────────────────────────

    def _check_trigger_candle(self, c: dict):
        """
        Rule 2: no-wick 10s candle with body strictly > min_body_pts.
        """
        body = c["close"] - c["open"]
        if abs(body) <= CFG["min_body_pts"]:
            return

        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]
        if upper_wick > CFG["wick_tolerance_pts"] or lower_wick > CFG["wick_tolerance_pts"]:
            return

        color = "GREEN" if body > 0 else "RED"

        self._c1            = c
        self._trigger_color = color
        self._state          = "WAIT_CONFIRM"

        log.info(
            f"[{self.name}] Trigger candle (C1) {color} | "
            f"o={c['open']:.1f} h={c['high']:.1f} l={c['low']:.1f} c={c['close']:.1f} "
            f"body={abs(body):.1f} @ {c['ts'].strftime('%H:%M:%S')} — waiting for confirm 5s"
        )

    def _check_confirm_candle(self, c5: dict, ts: datetime):
        """
        Rule 3: the next 5s candle must close the same color as C1.
        """
        if c5["close"] > c5["open"]:
            color5 = "GREEN"
        elif c5["close"] < c5["open"]:
            color5 = "RED"
        else:
            color5 = "FLAT"

        if color5 != self._trigger_color:
            log.info(
                f"[{self.name}] Confirm 5s candle mismatch "
                f"(C1={self._trigger_color} confirm={color5}) — resetting scan"
            )
            self._reset_pattern_state()
            return

        self._confirm5           = c5
        self._breakout_window_ts = c5["ts"]   # also the start of the 10s bucket being watched
        self._state               = "WATCH_BREAKOUT"

        log.info(
            f"[{self.name}] Confirm 5s candle matches ({color5}) | "
            f"o={c5['open']:.1f} h={c5['high']:.1f} l={c5['low']:.1f} c={c5['close']:.1f} "
            f"@ {c5['ts'].strftime('%H:%M:%S')} — watching breakout in next 10s window"
        )

    def _check_breakout_tick(self, price: float, ts: datetime):
        """
        Rule 4 + 6: immediate entry the instant price breaks the confirm
        candle's high/low AND has crossed its close (implied for a clean
        breakout, checked explicitly for the stated skip rule).
        """
        c5 = self._confirm5
        if self._trigger_color == "GREEN":
            if price > c5["high"] and price > c5["close"]:
                self._reset_pattern_state()
                self._fire_entry("CE", price, ts)
        else:
            if price < c5["low"] and price < c5["close"]:
                self._reset_pattern_state()
                self._fire_entry("PE", price, ts)

    def _reset_pattern_state(self):
        self._state              = "SCAN"
        self._c1                 = None
        self._trigger_color      = None
        self._confirm5           = None
        self._breakout_window_ts = None

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _fire_entry(self, signal: str, index_price: float, ts: datetime):
        strike = get_atm_strike(index_price, step=NIFTY_STRIKE_STEP)
        token, sym = self._instruments.get_option_token(strike, signal, self._expiry_date)

        if not token or not sym:
            log.error(
                f"[{self.name}] No option token | signal={signal} "
                f"strike={strike} expiry={self._expiry_date} — trade SKIPPED"
            )
            return

        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason="10s_marubozu_5s_confirm_breakout")

    def _build_entry(self, sym: str, token: int, signal: str, ts: datetime, reason: str):
        opt_price = self.get_price(token)

        if not opt_price or opt_price <= 0:
            log.warning(
                f"[{self.name}] No live price yet for {sym} — storing pending entry"
            )
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy has a position")
            return

        result = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if result is None:
            self._release_slot()
            log.error(f"[{self.name}] BUY order FAILED for {sym} — entry aborted")
            return

        order_id, fill_price = result

        sl = fill_price - CFG["sl_points"]
        tp = fill_price + CFG["tp_points"]
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])

        self._trade = {
            "state"            : "OPEN",
            "symbol"           : sym,
            "token"            : token,
            "signal"           : signal,
            "entry"            : fill_price,
            "sl"               : sl,
            "tp"               : tp,
            "entry_time"       : ts,
            "sl_active_from"   : sl_active_from,
            "order_id"         : order_id,
            "qty"              : CFG["quantity"],
            "_exit_in_progress": False,
        }
        self._trades_today += 1

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] ENTRY #{self._trades_today} {sym} @ {fill_price:.2f} | "
            f"SL={sl:.2f} (-{CFG['sl_points']}) TP={tp:.2f} (+{CFG['tp_points']}) | "
            f"reason={reason} | order_id={order_id}"
        )

        self._log_csv({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"   : sym,
            "action"   : "ENTRY",
            "price"    : fill_price,
            "sl"       : round(sl, 2),
            "tp"       : round(tp, 2),
            "status"   : "OPEN",
            "pnl"      : 0,
            "reason"   : reason,
            "mode"     : mode_tag,
            "order_id" : order_id,
        })

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _do_exit(self, exit_price: float, reason: str, ts: datetime):
        t = self._trade
        if not t or t["state"] != "OPEN":
            return
        if t.get("_exit_in_progress"):
            return
        t["_exit_in_progress"] = True

        result = self._place_sell_with_retry(
            t["symbol"], t["token"], t["qty"], exit_price, max_retries=3
        )

        if result is not None:
            order_id, sell_price = result
            t["state"] = "CLOSED"
            self._release_slot()
            self._finalize_exit(t, sell_price, order_id, reason, ts)
            return

        if LIVE_MODE:
            still_open = self._hub.order_router._is_position_open(t["symbol"])

            if not still_open:
                log.warning(
                    f"[{self.name}] SELL failed after 3 retries but position confirmed "
                    f"CLOSED by exchange. Treating as closed."
                )
                t["state"] = "CLOSED"
                self._release_slot()
                self._finalize_exit(t, exit_price, None, f"{reason}_EXCHANGE_CLOSED", ts)
                return

            log.error(
                f"\n{'!'*60}\n"
                f"[{self.name}] CRITICAL: SELL failed after 3 retries — "
                f"position STILL OPEN for {t['symbol']}!\n"
                f"  Slot is LOCKED. Emergency exit thread starting.\n"
                f"{'!'*60}"
            )
            self._start_emergency_exit(t, exit_price, reason, ts)
            return

        # Paper mode: paper sell never fails, but handle gracefully.
        t["state"] = "CLOSED"
        self._release_slot()
        self._finalize_exit(t, exit_price, None, reason, ts)

    def _start_emergency_exit(self, t: dict, ref_price: float, reason: str, ts: datetime):
        def _loop():
            max_attempts = CFG["emergency_max_attempts"]
            retry_sec    = CFG["emergency_retry_sec"]

            for attempt in range(1, max_attempts + 1):
                _time_mod.sleep(retry_sec)

                log.error(
                    f"[{self.name}] Emergency exit attempt {attempt}/{max_attempts} | "
                    f"{t['symbol']} | slot LOCKED"
                )

                still_open = self._hub.order_router._is_position_open(t["symbol"])
                if not still_open:
                    log.info(
                        f"[{self.name}] Emergency exit: {t['symbol']} confirmed CLOSED "
                        f"by exchange on attempt {attempt}"
                    )
                    t["state"] = "CLOSED"
                    self._release_slot()
                    now = _now_ist()
                    self._finalize_exit(t, ref_price, None, f"{reason}_EXCHANGE_CLOSED", now)
                    return

                ltp = self.get_price(t["token"]) or ref_price
                result = self._hub.order_router.place_sell(
                    self.name, t["symbol"], t["token"], t["qty"], ltp, LIVE_MODE
                )

                if result:
                    order_id, sell_price = result
                    t["state"] = "CLOSED"
                    self._release_slot()
                    log.info(
                        f"[{self.name}] Emergency exit SUCCESS on attempt {attempt} @ {sell_price:.2f}"
                    )
                    now = _now_ist()
                    self._finalize_exit(t, sell_price, order_id, f"{reason}_EMERGENCY", now)
                    return

                log.error(
                    f"[{self.name}] Emergency exit attempt {attempt}/{max_attempts} FAILED | "
                    f"{t['symbol']} | ltp={ltp:.2f}"
                )

            log.error(
                f"\n{'!'*60}\n"
                f"[{self.name}] GAVE UP emergency exit for {t['symbol']} after "
                f"{max_attempts} attempts ({max_attempts * retry_sec // 60} min).\n"
                f"  *** SQUARE OFF MANUALLY IN ZERODHA CONSOLE IMMEDIATELY! ***\n"
                f"{'!'*60}"
            )
            t["state"] = "CLOSED"
            self._release_slot()

        thread = threading.Thread(
            target=_loop, name="nifty-candle-breakout-emergency-exit", daemon=True,
        )
        thread.start()
        log.info(f"[{self.name}] Emergency exit thread started for {t['symbol']}")

    def _finalize_exit(self, t: dict, sell_price: float, order_id: Optional[str], reason: str, ts: datetime):
        pnl = (sell_price - t["entry"]) * t["qty"]
        self._today_pnl += pnl

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] EXIT [{reason}] {t['symbol']} @ {sell_price:.2f} "
            f"| PnL={pnl:.0f} ({pnl / t['qty']:.1f}/unit) | Today={self._today_pnl:.0f} | "
            f"order_id={order_id}"
        )

        self._log_csv({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"   : t["symbol"],
            "action"   : "EXIT",
            "price"    : sell_price,
            "sl"       : round(t["sl"], 2),
            "tp"       : round(t["tp"], 2),
            "status"   : "CLOSED",
            "pnl"      : round(pnl, 2),
            "reason"   : reason,
            "mode"     : mode_tag,
            "order_id" : order_id,
        })
        self._completed.append({**t, "exit_price": sell_price, "exit_reason": reason, "pnl": pnl})

        # Trade fully resolved — free up scanning for the next setup.
        self._trade = None
        self._reset_pattern_state()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "symbol", "action", "price",
            "sl", "tp", "status", "pnl", "reason", "mode", "order_id",
        ]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Trades taken   : {self._trades_today}")
        for t in self._completed:
            log.info(
                f"[{self.name}]   {t['symbol']} [{t['exit_reason']}] "
                f"entry={t['entry']:.2f} exit={t['exit_price']:.2f} "
                f"PnL={t['pnl']:.0f} ({t['pnl'] / t['qty']:.1f}/unit)"
            )
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'='*50}\n")
