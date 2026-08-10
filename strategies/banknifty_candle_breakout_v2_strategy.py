"""
strategies/banknifty_candle_breakout_strategy.py

BANKNIFTY_CANDLE_BREAKOUT_V2 — C1 point-move trigger + C2 point-move entry, BankNifty only.

REQUIREMENTS (as given — rewritten 2026-07-20, replaces the old
10s-body / 5s-confirm-color / breakout-watch design AND the 30pt fast
path entirely; the fast path is superseded because C1 now early-closes
on its own point-move threshold, which is lower than the old 30pt
fast-path threshold, so the fast path could never fire ahead of it)
──────────────────────────────────────────────────────────────
  1. BankNifty only.
  2. C1: watch every tick within the current 10-second window against
     THAT window's own open. If price moves >= c1_move_pts (20 points)
     from the window's open at any point before 10 seconds elapse,
     close C1 immediately at that tick (early close). Direction is
     GREEN if the move is up, RED if down.
     - If 10 seconds elapse WITHOUT the move threshold being hit, there
       is NO trigger — that window is discarded and the next 10-second
       window starts scanning fresh. The threshold is now required,
       not optional (no fallback to a plain body-size check at 10s).
  3. C2: the moment C1 closes, start a new window with open = C1's
     closing price. Watch price tick-by-tick against C2's own open.
     Close C2 immediately if price moves >= c2_move_pts (10 points) from
     C2's open, OR after 5 seconds elapse, whichever comes first.
  4. When C2 closes, compare its close vs its open to get its color
     (GREEN if close > open, RED if close < open). It must match C1's
     direction — if it doesn't, the setup is abandoned and the state
     machine resets to scan for a fresh C1.
  5. If C2's color matches, enter IMMEDIATELY (no further breakout
     watch): CE at C2's HIGH for a GREEN/CE setup, PE at C2's LOW for a
     RED/PE setup (high/low mirrored by direction).
  6. SL/TP are fixed at 20 / 30 points on the OPTION PREMIUM (not the
     index), unchanged from before — only the trigger/entry mechanics
     changed, per user request.

CLARIFICATIONS CONFIRMED WITH USER
────────────────────────────────────
  - C1 no longer waits for a candle to close at a fixed boundary before
    checking its trigger condition — the point-move check runs live,
    tick-by-tick, and can close C1 early. If no trigger within the 10s
    window, that window is skipped (no trigger), not treated as a
    fallback body check.
  - C2 replaces the old fixed 5s confirm candle: it's a floating window
    starting the instant C1 closes (not aligned to any clock boundary),
    closing early on its own point-move threshold or after 5s, whichever
    is first.
  - The old breakout-watch stage (tick-watching for a break of the
    confirm candle's high/low after it closed) is REMOVED. Entry now
    fires immediately when C2 closes and its color matches C1 — using
    C2's high (CE) / low (PE) as the entry reference price.
  - The old 30pt fast path (skip confirm+breakout entirely) is removed:
    C1's own threshold (20pts) is now the live tick-based fast trigger,
    so a separate higher-threshold fast path had nothing left to do.
  - SL/TP of 20/30 points = option premium points, unchanged.

ASSUMPTIONS (stated — not covered by the spec, flag if wrong)
─────────────────────────────────────────────────────────────
  - This strategy is NOT limited to the 9:15 open like SPIKE/SPIKE_NIFTY.
    It scans all day (09:15–15:00 for new setups, force-exits any open
    trade by 15:15) and can take MULTIPLE trades per day, one at a time.
  - ATM strike is recomputed at the moment of each signal (not fixed at
    market open), since this strategy can fire at any time of day and
    spot may have drifted from the pre-market ATM. This mirrors the
    pattern used in nifty_directional_strategy.py.
  - One trade at a time for this strategy; the global live-trade slot
    (OrderRouter) still applies on top of that in LIVE_MODE.

INDEX ROUTING
──────────────
  NO class-level INDEX_TOKEN attribute (deliberately — see Bug 16 note in
  bb_stoch_strategy.py). BankNifty (260105) is MarketHub's MAIN index, so
  strategies tracking it must leave INDEX_TOKEN unset/None to receive the
  main on_tick()/on_candle() broadcast. Setting INDEX_TOKEN = 260105 here
  previously caused MarketHub to treat this strategy as tracking a
  *different* index and silently skip it on every tick — zero candles,
  zero triggers, all session, with no error logged. Fixed 2026-07-14.
  Same mechanism/shared banknifty_pm / banknifty_instruments used by
  BANKNIFTY_EXPIRY_MOMENTUM / BB_STOCH.

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
from core.fast_indicators import compute_fast_indicators, CANDLE_WINDOW, INDICATOR_FIELDS
from core.costs import entry_fill, exit_fill, net_pnl_rs, round_trip_cost_pts

log = logging.getLogger("strategy.banknifty_candle_breakout_v2")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


def strike_from_symbol(sym: str) -> str:
    """Best-effort strike extraction from an option tradingsymbol for CSV logging."""
    digits = "".join(ch for ch in sym if ch.isdigit())
    # tradingsymbols embed the expiry date digits too; strike is the trailing
    # numeric run right before the CE/PE suffix, so pull it from the raw string.
    for suffix in ("CE", "PE"):
        if sym.endswith(suffix):
            tail = sym[:-2]
            num = ""
            for ch in reversed(tail):
                if ch.isdigit():
                    num = ch + num
                else:
                    break
            return num
    return digits


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG — change to True only when ready to trade real money.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

BANKNIFTY_STRIKE_STEP = 100

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "quantity"               : 30,

    # ── Session windows ───────────────────────────────────────────────────────
    "start_time"             : dtime(9, 15),
    "last_entry_time"        : dtime(15, 0),   # stop scanning for NEW setups after this
    "close_time"             : dtime(15, 15),  # force-exit any open trade

    # ── Candle pattern parameters ─────────────────────────────────────────────
    # ── Candle pattern parameters ─────────────────────────────────────────────
    "c1_window_sec"          : 10,    # C1 times out (no trigger) after this many seconds
    "c1_move_pts"            : 20.0,  # price move from C1 window's open that closes C1 early
    "c2_window_sec"          : 5,     # C2 times out (closes as-is) after this many seconds
    "c2_move_pts"            : 10.0,  # price move from C2's open that closes C2 early

    # ── SL / trailing lock (on OPTION PREMIUM) — min 10pt gain, unlimited upside ──
    # ── FIX A: risk / reward on OPTION PREMIUM points ─────────────────────────
    # The old geometry locked profit at exactly the same distance it armed at,
    # with an equally tight trail — a fraction of the risk rather than a
    # multiple of it. Reward must exceed risk AND exceed the round-trip cost
    # priced in core/costs.py, or no entry signal can rescue the strategy.
    "sl_points"              : 30.0,
    "trail_activate_pts"     : 34.0,   # open profit before trail arms
    "lock_pts"               : 10.0,   # profit locked the moment it arms
    "trail_distance_pts"     : 22.0,   # thereafter trail this far behind peak
    "sl_grace_seconds"       : 5,

    # ── FIX F: time stop ──────────────────────────────────────────────────────
    "max_hold_seconds"       : 180,
    "time_stop_min_profit"   : 12.0,

    # ── FIX C: risk caps (all NEW — there were none) ──────────────────────────
    # TEST-PHASE SWITCH (2026-08): both master switches below are OFF while you
    # collect a clean, unfiltered month of paper data. All the tracking, halt
    # bookkeeping, and CSV logging still runs underneath so you can see exactly
    # what each gate WOULD have blocked — nothing is deleted, it just isn't
    # enforced yet. Flip both to True once you've reviewed a month of data and
    # decided on thresholds; no code changes needed, just these two flags.
    "enforce_risk_caps"      : False,
    "enforce_direction_gate" : False,
    "max_trades_day"         : 6,
    "max_daily_loss_rs"      : 6000.0,
    "max_consec_losses"      : 3,
    "post_loss_cooldown_sec" : 180,

    # ── FIX E: directional gates (were computed but unused) ───────────────────
    "require_vwap_side"      : True,   # only takes effect if enforce_direction_gate=True
    "require_supertrend"     : True,   # only takes effect if enforce_direction_gate=True

    # ── Emergency exit (LIVE_MODE only) ───────────────────────────────────────
    "emergency_retry_sec"    : 30,
    "emergency_max_attempts" : 30,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"               : "banknifty_candle_breakout_v2_trades.csv",
}


class BankNiftyCandleBreakoutV2Strategy(BaseStrategy):

    # NOTE: deliberately NO class-level INDEX_TOKEN attribute here.
    # MarketHub's _handle_index_tick() treats any strategy with a non-None
    # INDEX_TOKEN as tracking a DIFFERENT index and skips it on the main
    # BankNifty broadcast (see bb_stoch_strategy.py's "Bug 16" comment for
    # the same historical mistake). Setting INDEX_TOKEN = 260105 here
    # silently prevented on_tick() from ever being called — zero ticks,
    # zero candles, zero trigger detections, all session.
    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "BANKNIFTY_CANDLE_BREAKOUT_V2"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._c10 = SecondCandleBuilder(seconds=CFG["c1_window_sec"])

        self._market_opened = False
        self._expiry_date    = None
        self._instruments    = None

        # ── Pattern state machine ─────────────────────────────────────────────
        # states: SCAN → WAIT_C2 → (SCAN)
        self._state              = "SCAN"
        self._c1                 = None   # trigger window dict (open/high/low/close/ts)
        self._trigger_color      = None   # "GREEN" / "RED"
        self._c2                 = None   # floating C2 window dict (open/high/low/close/start_ts)

        self._signal_meta    = None   # metrics collected from C1/C2 for the next entry

        # Live PreMarketData reference (set in pre_market()) — read
        # self._pm.pcr at signal time so it reflects the live-refreshed
        # value, not a snapshot taken at market open. Log-only.
        self._pm = None

        self._trade         = None
        self._pending_entry  = None
        self._trades_today   = 0
        self._today_pnl      = 0.0
        self._completed      = []

        # FIX C: risk state — this strategy previously had NO daily loss
        # limit, NO cap on trades per day and NO cooldown after a loss.
        self._consec_losses = 0
        self._halted        = False
        self._halt_reason   = ""
        self._last_loss_ts  = 0.0

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | qty={CFG['quantity']} "
            f"c1_move={CFG['c1_move_pts']}pts c2_move={CFG['c2_move_pts']}pts "
            f"SL=-{CFG['sl_points']} | trail: lock +{CFG['trail_activate_pts']}, "
            f"trail {CFG['trail_distance_pts']} behind peak (unlimited upside)"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        """
        Receives BankNifty-specific PreMarketData + InstrumentStore (routed by
        t.py's default/else branch, same as any strategy with no INDEX_TOKEN
        set). No pre-subscription of a fixed
        ATM pair is done here — ATM drifts through the day and this strategy
        can fire at any time, so the strike is resolved fresh at signal time.
        """
        self._instruments = instruments
        self._expiry_date = pm.expiry_date
        self._pm = pm

        # Reset daily risk state (matters when the process is long-lived).
        self._trades_today  = 0
        self._today_pnl     = 0.0
        self._consec_losses = 0
        self._halted        = False
        self._halt_reason   = ""
        self._last_loss_ts  = 0.0
        self._completed     = []

        log.info(
            f"[{self.name}] Pre-market | expiry={pm.expiry_date} "
            f"mode={'LIVE' if LIVE_MODE else 'PAPER'}"
        )
        return True

    # ── Fast/leading indicator snapshot (LOG-ONLY — see core/fast_indicators.py) ──

    def _indicator_snapshot(self, spot: float) -> dict:
        """
        Computes VWAP / PCR / EMA slope / RSI slope / MACD histogram / ATR% /
        Supertrend from the strategy's own rolling 10s candle buffer plus the
        live hub VWAP and live PCR. Called once per signal event and merged
        into both the signals CSV and (via _signal_meta) the trades CSV.
        Entry/exit decisions never read this — log-only, mirrors v1.
        """
        candles = self._c10.last_n_closed(CANDLE_WINDOW)
        vwap    = self._hub.session_vwap.value
        pcr     = self._pm.pcr if self._pm is not None else None
        return compute_fast_indicators(candles, spot=spot, vwap=vwap, pcr=pcr)

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

        # Feed C1's 10s window builder. Used only while state == SCAN; while
        # WAIT_C2 it keeps accumulating in the background but is ignored, and
        # is force-reset to a fresh window whenever we return to SCAN (see
        # _reset_pattern_state) so a stale open never leaks into a new C1 check.
        closed10 = self._c10.feed_tick(price, tick_ts)

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
            self._check_c1_tick(price, tick_ts, closed10)

        elif self._state == "WAIT_C2":
            self._check_c2_tick(price, tick_ts)

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Resolves a pending entry (if the option had no valid live price at
        signal time) and manages the trailing stop for the open trade.
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

        # ── SL grace period ────────────────────────────────────────────────────
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from is not None and ts < sl_active_from:
            return

        # ── Trailing stop — NO fixed TP, profit is unlimited ──────────────────
        # Once the trade is up by trail_activate_pts, SL locks to at least
        # entry+trail_activate_pts (guarantees the minimum 10pt gain) and then
        # trails trail_distance_pts behind the running peak. SL only ever
        # ratchets UP, never down. The only exits are this trailing SL (or the
        # original risk SL before activation) and EOD force-close.
        # ── FIX A: trail arms late, locks a real profit, trails wide ────────
        # Old: armed at +trail_activate and locked EXACTLY that same level with
        # an equally tight trail, so every winner was stopped out on ordinary
        # premium noise at the lock level while every loser paid the full SL.
        # See the module docstring for the breakeven-win-rate arithmetic.
        t = self._trade
        t["peak"] = max(t["peak"], price)
        gain = t["peak"] - t["entry"]
        if gain >= CFG["trail_activate_pts"]:
            locked_sl = max(
                t["entry"] + CFG["lock_pts"],
                t["peak"]  - CFG["trail_distance_pts"],
            )
            if locked_sl > t["sl"]:
                t["sl"] = locked_sl
                t["trail_armed"] = True

        if price <= t["sl"]:
            reason = "TRAIL_SL_HIT" if t.get("trail_armed") else "SL_HIT"
            self._do_exit(price, reason, ts)
            return

        # ── FIX F: time stop for trades going nowhere ───────────────────────
        held = (ts - t["entry_time"]).total_seconds()
        if held >= CFG["max_hold_seconds"]:
            if (price - t["entry"]) < CFG["time_stop_min_profit"]:
                log.info(
                    f"[{self.name}] Time stop | held={held:.0f}s "
                    f"pnl={(price - t['entry']):+.2f}pts — exiting"
                )
                self._do_exit(price, "TIME_STOP", ts)

    # ── Pattern detection ────────────────────────────────────────────────────

    def _check_c1_tick(self, price: float, ts: datetime, closed10: Optional[dict]):
        """
        C1: live tick-by-tick check against the current 10s window's open.
        Closes early (triggers) the instant the move threshold is hit.
        If the window times out (closed10 fires) with no trigger, that
        window is simply discarded — no fallback body check.
        """
        cur = self._c10.get_current()
        if cur is None:
            return

        move = price - cur["open"]
        if abs(move) >= CFG["c1_move_pts"]:
            color = "GREEN" if move > 0 else "RED"
            c1 = {
                "open": cur["open"], "high": max(cur["high"], price),
                "low": min(cur["low"], price), "close": price, "ts": ts,
            }
            self._c1            = c1
            self._trigger_color = color
            self._state          = "WAIT_C2"

            log.info(
                f"[{self.name}] C1 trigger {color} | move={abs(move):.1f}pts | "
                f"o={c1['open']:.1f} h={c1['high']:.1f} l={c1['low']:.1f} c={c1['close']:.1f} "
                f"@ {ts.strftime('%H:%M:%S')} — starting C2"
            )
            ind = self._indicator_snapshot(c1["close"])
            self._log_signal_csv({
                "timestamp"  : ts.strftime("%Y-%m-%d %H:%M:%S"),
                "event"      : "C1_TRIGGER",
                "color"      : color,
                "c1_open"    : c1["open"], "c1_high": c1["high"],
                "c1_low"     : c1["low"],  "c1_close": c1["close"],
                "c1_body"    : round(abs(move), 2),
                "confirm_open": "", "confirm_high": "", "confirm_low": "", "confirm_close": "",
                "breakout_price": "",
                **ind,
            })

            self._start_c2(price, ts)
            # Force a fresh window on the very next tick — avoids a stale
            # open leaking into the next C1 check once we return to SCAN.
            self._c10.current_candle = None
            return

        if closed10 is not None:
            # This window (closed10) timed out at 10s with no trigger —
            # no-op, next window has already started (cur reflects it).
            pass

    def _start_c2(self, price: float, ts: datetime):
        self._c2 = {"open": price, "high": price, "low": price, "close": price, "start_ts": ts}

    def _check_c2_tick(self, price: float, ts: datetime):
        """
        C2: floating window starting at C1's close. Closes early on its
        own point-move threshold, or after c2_window_sec, whichever first.
        """
        c2 = self._c2
        c2["high"]  = max(c2["high"], price)
        c2["low"]   = min(c2["low"], price)
        c2["close"] = price

        elapsed = (ts - c2["start_ts"]).total_seconds()
        move    = price - c2["open"]

        if abs(move) >= CFG["c2_move_pts"] or elapsed >= CFG["c2_window_sec"]:
            self._finish_c2(ts)

    def _finish_c2(self, ts: datetime):
        """
        Rule 4+5: C2's color must match C1's direction to proceed. On a
        match, enter IMMEDIATELY at C2's high (CE) / low (PE) — no further
        breakout watch. On a mismatch, reset and resume scanning.
        """
        c1, c2 = self._c1, self._c2

        if c2["close"] > c2["open"]:
            color2 = "GREEN"
        elif c2["close"] < c2["open"]:
            color2 = "RED"
        else:
            color2 = "FLAT"

        if color2 != self._trigger_color:
            log.info(
                f"[{self.name}] C2 mismatch (C1={self._trigger_color} C2={color2}) — resetting scan"
            )
            ind = self._indicator_snapshot(c2["close"])
            self._log_signal_csv({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "event"    : "C2_MISMATCH",
                "color"    : self._trigger_color,
                "c1_open"  : c1["open"], "c1_high": c1["high"],
                "c1_low"   : c1["low"],  "c1_close": c1["close"],
                "c1_body"  : round(abs(c1["close"] - c1["open"]), 2),
                "confirm_open": c2["open"], "confirm_high": c2["high"],
                "confirm_low" : c2["low"],  "confirm_close": c2["close"],
                "breakout_price": "",
                **ind,
            })
            self._reset_pattern_state()
            return

        entry_price = c2["high"] if self._trigger_color == "GREEN" else c2["low"]
        signal      = "CE" if self._trigger_color == "GREEN" else "PE"

        log.info(
            f"[{self.name}] C2 match ({color2}) | "
            f"o={c2['open']:.1f} h={c2['high']:.1f} l={c2['low']:.1f} c={c2['close']:.1f} "
            f"@ {ts.strftime('%H:%M:%S')} — entering {signal} @ {entry_price:.1f}"
        )
        ind = self._indicator_snapshot(entry_price)
        self._log_signal_csv({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "event"    : "C2_ENTER",
            "color"    : color2,
            "c1_open"  : c1["open"], "c1_high": c1["high"],
            "c1_low"   : c1["low"],  "c1_close": c1["close"],
            "c1_body"  : round(abs(c1["close"] - c1["open"]), 2),
            "confirm_open": c2["open"], "confirm_high": c2["high"],
            "confirm_low" : c2["low"],  "confirm_close": c2["close"],
            "breakout_price": round(entry_price, 2),
            **ind,
        })

        self._signal_meta = self._build_signal_meta(entry_price, ts, ind)
        self._reset_pattern_state()
        self._fire_entry(signal, entry_price, ts)

    def _build_signal_meta(self, entry_price: float, ts: datetime, ind: dict = None) -> dict:
        c1, c2 = self._c1, self._c2
        return {
            "index_price"   : entry_price,
            "weekday"       : ts.strftime("%A"),
            "c1_body"       : round(abs(c1["close"] - c1["open"]), 2),
            "c1_open"       : c1["open"], "c1_high": c1["high"],
            "c1_low"        : c1["low"],  "c1_close": c1["close"],
            "confirm_open"  : c2["open"], "confirm_high": c2["high"],
            "confirm_low"   : c2["low"],  "confirm_close": c2["close"],
            "breakout_price": round(entry_price, 2),
            **(ind or {}),
        }

    def _reset_pattern_state(self):
        self._state         = "SCAN"
        self._c1            = None
        self._trigger_color = None
        self._c2            = None
        # Force a fresh C1 window baseline on the next tick.
        self._c10.current_candle = None

    # ── Entry ─────────────────────────────────────────────────────────────────

    # ── FIX E: directional gates (indicators were computed then discarded) ──

    def _direction_allowed(self, signal: str, ind: dict) -> tuple:
        """
        Returns (allowed: bool, reason: str).

        compute_fast_indicators() already ran on every signal event and was
        written to CSV without ever influencing a decision. Taking a 10-second
        breakout against the session trend is the most common way this pattern
        loses, so VWAP side and fast Supertrend now gate direction.

        Both gates FAIL OPEN when the indicator is blank/not ready, so
        warm-up behaviour is unchanged. Set the CFG flags False to revert.
        TEST PHASE: also fails open entirely while enforce_direction_gate=False —
        see the CFG comment above.
        """
        if not CFG["enforce_direction_gate"]:
            return True, ""
        if CFG["require_vwap_side"]:
            side = ind.get("spot_vs_vwap", "")
            if side == "BELOW" and signal == "CE":
                return False, "vwap_side_below_for_CE"
            if side == "ABOVE" and signal == "PE":
                return False, "vwap_side_above_for_PE"

        if CFG["require_supertrend"] and ind.get("indicators_ready"):
            st = ind.get("supertrend_dir", "")
            if st == "DOWN" and signal == "CE":
                return False, "supertrend_down_for_CE"
            if st == "UP" and signal == "PE":
                return False, "supertrend_up_for_PE"

        return True, ""

    # ── FIX C: risk caps ────────────────────────────────────────────────────

    def _risk_block_reason(self) -> "Optional[str]":
        """
        Returns a reason string if new entries WOULD BE blocked, else None.
        TEST PHASE: when enforce_risk_caps=False (default now) this never
        actually blocks an entry — the caller only logs the reason. The
        underlying counters (_halted, _consec_losses, _today_pnl) keep
        updating exactly as before so the month-end CSV shows what each
        cap would have done.
        """
        if not CFG["enforce_risk_caps"]:
            return None
        if self._halted:
            return self._halt_reason or "halted"
        if self._trades_today >= CFG["max_trades_day"]:
            return f"max_trades_day({CFG['max_trades_day']})"
        if self._today_pnl <= -abs(CFG["max_daily_loss_rs"]):
            return f"max_daily_loss(Rs{self._today_pnl:.0f})"
        if self._consec_losses >= CFG["max_consec_losses"]:
            return f"max_consec_losses({self._consec_losses})"
        if self._last_loss_ts > 0:
            elapsed = _time_mod.time() - self._last_loss_ts
            if elapsed < CFG["post_loss_cooldown_sec"]:
                return f"post_loss_cooldown({CFG['post_loss_cooldown_sec'] - elapsed:.0f}s)"
        return None

    def _fire_entry(self, signal: str, index_price: float, ts: datetime,
                     reason: str = "c1_c2_points_entry"):
        # ── FIX C + FIX E: risk caps and directional gates ──────────────────
        # Placed here rather than at each pattern call-site so every entry
        # path (breakout, fast-tick, C2 points entry) is covered.
        block = self._risk_block_reason()
        if block:
            log.info(f"[{self.name}] Entry blocked by risk gate: {block} — skipping")
            self._signal_meta = None
            return

        _ind = self._indicator_snapshot(index_price)
        _ok, _why = self._direction_allowed(signal, _ind)
        if not _ok:
            log.info(
                f"[{self.name}] Entry blocked by direction gate: {_why} "
                f"(signal={signal} vwap_side={_ind.get('spot_vs_vwap', '?')} "
                f"st={_ind.get('supertrend_dir', '?')}) — skipping"
            )
            self._signal_meta = None
            return

        strike = get_atm_strike(index_price, step=BANKNIFTY_STRIKE_STEP)
        token, sym = self._instruments.get_option_token(strike, signal, self._expiry_date)

        if not token or not sym:
            log.error(
                f"[{self.name}] No option token | signal={signal} "
                f"strike={strike} expiry={self._expiry_date} — trade SKIPPED"
            )
            return

        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason=reason)

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
            self.unsubscribe_option(token)   # FIX D: don't leak the refcount
            return

        result = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if result is None:
            self._release_slot()
            self.unsubscribe_option(token)   # FIX D
            log.error(f"[{self.name}] BUY order FAILED for {sym} — entry aborted")
            return

        order_id, raw_fill = result

        # ── FIX B: pay the bid-ask on entry in paper mode ────────────────────
        # OrderRouter._paper_fill() returns the raw LTP for both legs, so paper
        # P&L had zero spread and zero brokerage. Live mode already returns a
        # real exchange average_price, so only paper fills are adjusted.
        fill_price = raw_fill if LIVE_MODE else entry_fill(raw_fill)

        sl = fill_price - CFG["sl_points"]
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])

        meta = self._signal_meta or {}
        self._signal_meta = None
        self._trade = {
            "state"            : "OPEN",
            "symbol"           : sym,
            "token"            : token,
            "signal"           : signal,
            "entry"            : fill_price,
            "sl"               : sl,
            "tp"               : None,   # no fixed TP — trailing stop only, profit unlimited
            "peak"             : fill_price,
            "entry_time"       : ts,
            "sl_active_from"   : sl_active_from,
            "order_id"         : order_id,
            "qty"              : CFG["quantity"],
            "_exit_in_progress": False,
            "meta"             : meta,
        }
        self._trades_today += 1

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        # Round-trip cost is logged on every entry so the drag is never
        # invisible again. If this number approaches CFG["lock_pts"], the
        # locked profit is being eaten entirely by spread + charges.
        rt_cost = round_trip_cost_pts(fill_price, CFG["quantity"])
        self._trade["rt_cost_pts"] = rt_cost
        log.info(
            f"[{self.name}] [{mode_tag}] ENTRY #{self._trades_today} {sym} "
            f"@ {fill_price:.2f} (ltp={raw_fill:.2f}) | SL={sl:.2f} (-{CFG['sl_points']}) | "
            f"trail arms +{CFG['trail_activate_pts']} locks +{CFG['lock_pts']} then "
            f"trails {CFG['trail_distance_pts']} behind peak | "
            f"est round-trip cost={rt_cost:.2f}pts | reason={reason} | order_id={order_id}"
        )

        meta = self._trade.get("meta", {})
        self._log_csv({
            "timestamp"      : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"         : sym,
            "action"         : "ENTRY",
            "price"          : fill_price,
            "sl"             : round(sl, 2),
            "tp"             : "",
            "status"         : "OPEN",
            "pnl"            : 0,
            "reason"         : reason,
            "mode"           : mode_tag,
            "order_id"       : order_id,
            "weekday"        : meta.get("weekday", ts.strftime("%A")),
            "strike"         : strike_from_symbol(sym),
            "c1_body"        : meta.get("c1_body", ""),
            "c1_open"        : meta.get("c1_open", ""),
            "c1_high"        : meta.get("c1_high", ""),
            "c1_low"         : meta.get("c1_low", ""),
            "c1_close"       : meta.get("c1_close", ""),
            "confirm_open"   : meta.get("confirm_open", ""),
            "confirm_high"   : meta.get("confirm_high", ""),
            "confirm_low"    : meta.get("confirm_low", ""),
            "confirm_close"  : meta.get("confirm_close", ""),
            "breakout_price" : meta.get("breakout_price", ""),
            "time_in_trade_s": "",
            "sl_tp_slippage" : "",
            **{k: meta.get(k, "") for k in INDICATOR_FIELDS},
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
            target=_loop, name="banknifty-candle-breakout-v2-emergency-exit", daemon=True,
        )
        thread.start()
        log.info(f"[{self.name}] Emergency exit thread started for {t['symbol']}")

    def _finalize_exit(self, t: dict, sell_price: float, order_id: Optional[str], reason: str, ts: datetime):
        # ── FIX B: pay the bid-ask on exit, then subtract brokerage/STT/GST ──
        raw_sell   = sell_price
        sell_price = raw_sell if LIVE_MODE else exit_fill(raw_sell)
        gross_pnl  = round((sell_price - t["entry"]) * t["qty"], 2)
        pnl        = net_pnl_rs(t["entry"], sell_price, t["qty"])
        self._today_pnl += pnl

        # ── FIX C: consecutive-loss / cooldown / halt bookkeeping ───────────
        if pnl < 0:
            self._consec_losses += 1
            self._last_loss_ts   = _time_mod.time()
        else:
            self._consec_losses = 0

        if self._today_pnl <= -abs(CFG["max_daily_loss_rs"]):
            self._halted      = True
            self._halt_reason = f"max_daily_loss(Rs{self._today_pnl:.0f})"
            log.warning(f"[{self.name}] HALTED for the day — {self._halt_reason}")
        elif self._consec_losses >= CFG["max_consec_losses"]:
            self._halted      = True
            self._halt_reason = f"max_consec_losses({self._consec_losses})"
            log.warning(f"[{self.name}] HALTED for the day — {self._halt_reason}")

        time_in_trade_s = round((ts - t["entry_time"]).total_seconds(), 1)
        # Positive slippage = filled worse than the intended SL level (gap-through);
        # negative/zero = filled at or better than the level. No TP_HIT anymore —
        # profit is unlimited via the trailing stop, so only SL_HIT/TRAIL_SL_HIT
        # (both contain "SL_HIT") carry a slippage figure.
        if "SL_HIT" in reason:
            sl_tp_slippage = round(t["sl"] - sell_price, 2)
        else:
            sl_tp_slippage = ""

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        slip_tag = f" | slip={sl_tp_slippage}" if sl_tp_slippage != "" else ""
        log.info(
            f"[{self.name}] [{mode_tag}] EXIT [{reason}] {t['symbol']} @ {sell_price:.2f} "
            f"(ltp={raw_sell:.2f}) | gross={gross_pnl:.0f} net={pnl:.0f} "
            f"({pnl / t['qty']:.1f}/unit) | cost_drag={gross_pnl - pnl:.0f} | "
            f"Today={self._today_pnl:.0f} | held={time_in_trade_s:.0f}s{slip_tag} | "
            f"consec_L={self._consec_losses} | order_id={order_id}"
        )

        meta = t.get("meta", {})
        self._log_csv({
            "timestamp"      : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"         : t["symbol"],
            "action"         : "EXIT",
            "price"          : sell_price,
            "sl"             : round(t["sl"], 2),
            "tp"             : "",
            "status"         : "CLOSED",
            "pnl"            : round(pnl, 2),
            "reason"         : reason,
            "mode"           : mode_tag,
            "order_id"       : order_id,
            "weekday"        : meta.get("weekday", ts.strftime("%A")),
            "strike"         : strike_from_symbol(t["symbol"]),
            "c1_body"        : meta.get("c1_body", ""),
            "c1_open"        : meta.get("c1_open", ""),
            "c1_high"        : meta.get("c1_high", ""),
            "c1_low"         : meta.get("c1_low", ""),
            "c1_close"       : meta.get("c1_close", ""),
            "confirm_open"   : meta.get("confirm_open", ""),
            "confirm_high"   : meta.get("confirm_high", ""),
            "confirm_low"    : meta.get("confirm_low", ""),
            "confirm_close"  : meta.get("confirm_close", ""),
            "breakout_price" : meta.get("breakout_price", ""),
            "time_in_trade_s": time_in_trade_s,
            "sl_tp_slippage" : sl_tp_slippage,
            **{k: meta.get(k, "") for k in INDICATOR_FIELDS},
        })
        self._completed.append({
            **t, "exit_price": sell_price, "exit_reason": reason,
            "pnl": pnl, "gross_pnl": gross_pnl,
        })

        # FIX D: release the WebSocket subscription this trade acquired.
        # _fire_entry() called subscribe_option() on every entry and nothing
        # ever released it, so MarketHub's refcount for every strike traded
        # stayed permanently above zero for the whole session.
        try:
            self.unsubscribe_option(t["token"])
        except Exception as e:
            log.warning(f"[{self.name}] unsubscribe error for {t['token']}: {e}")

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
            "weekday", "strike",
            "c1_body", "c1_open", "c1_high", "c1_low", "c1_close",
            "confirm_open", "confirm_high", "confirm_low", "confirm_close",
            "breakout_price", "time_in_trade_s", "sl_tp_slippage",
        ] + INDICATOR_FIELDS
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def _log_signal_csv(self, row: dict):
        """
        Logs EVERY trigger candle regardless of outcome (mismatched, timed out,
        or entered) so the C1/confirm thresholds can be tuned from the full
        population of setups, not just the ones that became trades.

        Also carries the log-only fast/leading indicator snapshot (VWAP, PCR,
        EMA slope, RSI slope, MACD histogram, ATR%, Supertrend — see
        core/fast_indicators.py) so those can be studied against outcomes
        after the fact. None of it feeds back into the state machine above.
        """
        fname  = CFG["csv_file"].replace("_trades.csv", "_signals.csv")
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "event", "color",
            "c1_open", "c1_high", "c1_low", "c1_close", "c1_body",
            "confirm_open", "confirm_high", "confirm_low", "confirm_close",
            "breakout_price",
        ] + INDICATOR_FIELDS
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Trades taken   : {self._trades_today}")
        wins   = [x for x in self._completed if x["pnl"] > 0]
        losses = [x for x in self._completed if x["pnl"] <= 0]
        gross  = sum(x.get("gross_pnl", 0) for x in self._completed)

        for t in self._completed:
            log.info(
                f"[{self.name}]   {t['symbol']} [{t['exit_reason']}] "
                f"entry={t['entry']:.2f} exit={t['exit_price']:.2f} "
                f"gross={t.get('gross_pnl', 0):.0f} net={t['pnl']:.0f} "
                f"({t['pnl'] / t['qty']:.1f}/unit)"
            )

        if self._completed:
            wr = len(wins) / len(self._completed) * 100
            log.info(
                f"[{self.name}] W/L            : {len(wins)}/{len(losses)}  "
                f"(win rate {wr:.1f}%)"
            )
        log.info(f"[{self.name}] Gross PnL      : {gross:.0f}")
        log.info(f"[{self.name}] Cost drag      : {gross - self._today_pnl:.0f}")
        log.info(f"[{self.name}] NET PnL        : {self._today_pnl:.0f}")
        if self._halted:
            log.info(f"[{self.name}] Halted         : {self._halt_reason}")
        log.info(f"[{self.name}] {'='*50}\n")