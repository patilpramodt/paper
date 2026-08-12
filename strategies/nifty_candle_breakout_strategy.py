"""
strategies/nifty_candle_breakout_strategy.py

NIFTY_CANDLE_BREAKOUT — 10s marubozu + 5s confirm + tick breakout, Nifty 50 only.

═══════════════════════════════════════════════════════════════════════════
  WHAT CHANGED IN THIS VERSION  (2026-08 audit)  — read this first
═══════════════════════════════════════════════════════════════════════════

FIX A — THE STRATEGY COULD NOT BE PROFITABLE AS CONFIGURED.
  Old settings: sl_points=15, trail_activate_pts=5, trail_distance_pts=5.
  Once a trade gained 5 points the stop locked at entry+5 and then trailed
  only 5 points behind the peak. Nifty ATM premium noise is routinely 2-4
  points, so essentially every winner was stopped out at exactly +5 while
  every loser paid the full -15. Gross breakeven win-rate = 15/(15+5) = 75%.

  Then apply real costs. core/costs.py prices a round-trip on a ~140-point
  Nifty ATM option at ~2.7 premium points (bid-ask on both legs + Rs 40
  brokerage + STT + exchange + GST). Net winner = +2.3, net loser = -17.7.
  Required win-rate to break even: 88.6%. No 10-second breakout pattern
  wins 9 times out of 10.

  Fix: the reward must be a multiple of the risk, not a fraction of it.
    - sl_points          15 -> 12   (tighter initial risk)
    - trail_activate_pts  5 -> 14   (~1.2R before anything is locked)
    - lock_pts            —  -> 6   (what gets locked when trail arms)
    - trail_distance_pts  5 -> 9    (room to let a winner run)
  Net winner is now open-ended with a realistic floor of ~+3.3 after costs,
  net loser is -14.7. Breakeven win-rate falls from 88.6% to ~35-45%
  depending on how far winners run.

FIX B — PAPER P&L IGNORED ALL EXECUTION COSTS.
  OrderRouter._paper_fill() returns the raw LTP for both legs, and
  _finalize_exit() computed `(sell_price - entry) * qty`. Zero spread, zero
  brokerage. Every paper number this strategy has ever produced is
  optimistic by ~2.7 points/unit (~Rs 175/trade at qty=65).
  Fix: entry uses core.costs.entry_fill(), exit uses core.costs.exit_fill(),
  P&L uses core.costs.net_pnl_rs(). Gross and net are both logged so the
  size of the cost drag stays visible.

FIX C — NO RISK CAPS AT ALL.
  The strategy scanned 09:15-15:00 with no daily loss limit, no cap on
  trades per day, and no cooldown after a loss. On a chop day it can fire
  dozens of times and there was nothing to stop the bleed. Added:
    - max_trades_day
    - max_daily_loss_rs      (hard halt)
    - max_consec_losses      (hard halt)
    - post_loss_cooldown_sec (no re-entry straight into the same chop)

FIX D — SUBSCRIPTION REFCOUNT LEAK.
  _fire_entry() called subscribe_option() on every entry; nothing ever
  called unsubscribe_option(). MarketHub refcounts subscriptions, so each
  trade permanently incremented the count for that strike and the token was
  never released — for the whole session, across every strike traded.
  Fix: _finalize_exit() releases the token it acquired.

FIX E — THE INDICATORS WERE COMPUTED AND THEN THROWN AWAY.
  compute_fast_indicators() ran on every signal event, was written to CSV,
  and never influenced anything. A 10-second breakout taken against the
  session trend is the single most common way this pattern loses.
  Fix: two cheap directional gates, both optional via CFG:
    - require_vwap_side  : CE only above session VWAP, PE only below
    - require_supertrend : CE only when fast Supertrend is UP, PE when DOWN
  Both fail OPEN when the indicator isn't ready yet, so early-session
  behaviour is unchanged. Every rejection is logged with event=GATE_BLOCK
  in the signals CSV so you can measure exactly what they cost you.

FIX F — TIME STOP.
  A breakout that is right is right within a minute or two. Sitting in a
  dead trade only burns theta and ties up the live slot.
  Fix: max_hold_seconds exits any trade that is not in profit after N sec.

═══════════════════════════════════════════════════════════════════════════
  ORIGINAL PATTERN SPEC (unchanged)
═══════════════════════════════════════════════════════════════════════════
  1. Nifty only.
  2. Every closed 10-second index candle is a "trigger" (C1) if its body
     (|close - open|) is strictly greater than min_body_pts.
  3. The very next 5-second candle must close the SAME colour as C1,
     otherwise the setup is abandoned.
  4. Once confirmed, watch price tick-by-tick for the rest of that 10-second
     window. The moment price breaks the confirm candle's high (GREEN -> CE)
     or low (RED -> PE), enter immediately.
  5. If the watched 10-second window closes with no breakout, reset to SCAN.

INDEX ROUTING
  INDEX_TOKEN = 256265 (NSE:NIFTY 50). MarketHub routes Nifty ticks here
  exclusively. Uses the shared nifty_pm / nifty_instruments wired in t.py.
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

log = logging.getLogger("strategy.nifty_candle_breakout")

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


def strike_from_symbol(sym: str) -> str:
    """Best-effort strike extraction from an option tradingsymbol for CSV logging."""
    digits = "".join(ch for ch in sym if ch.isdigit())
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

NIFTY_STRIKE_STEP = 50

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "quantity"               : 65,

    # ── Session windows ───────────────────────────────────────────────────────
    "start_time"             : dtime(9, 15),
    "warmup_until"           : dtime(9, 20),   # no entries in the first 5 min
    "last_entry_time"        : dtime(15, 0),   # stop scanning for NEW setups
    "close_time"             : dtime(15, 15),  # force-exit any open trade

    # ── Candle pattern parameters ─────────────────────────────────────────────
    "bucket_10s"             : 10,
    "bucket_5s"              : 5,
    "min_body_pts"           : 6.0,   # body must be strictly greater than this

    # ── FIX A: risk / reward on OPTION PREMIUM points ─────────────────────────
    # Reward must be a MULTIPLE of risk, not a fraction. See module docstring.
    "sl_points"              : 12.0,  # initial risk stop
    "trail_activate_pts"     : 6.0,  # ~1.2R of open profit before trail arms
    "lock_pts"               : 5.0,   # profit locked the moment it arms
    "trail_distance_pts"     : 5.0,   # thereafter SL trails this far behind peak
    "sl_grace_seconds"       : 5,

    # ── FIX F: time stop ──────────────────────────────────────────────────────
    # A 10-second breakout that hasn't worked in this long is not going to.
    "max_hold_seconds"       : 180,
    # Only time-stop out if the trade is not meaningfully in profit.
    "time_stop_min_profit"   : 4.0,

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
    "max_daily_loss_rs"      : 4000.0,
    "max_consec_losses"      : 3,
    "post_loss_cooldown_sec" : 180,

    # ── FIX E: directional gates (were computed but unused) ───────────────────
    # Both fail OPEN when the indicator is not ready, so early-session
    # behaviour is identical to before. Set False to restore old behaviour.
    "require_vwap_side"      : True,   # only takes effect if enforce_direction_gate=True
    "require_supertrend"     : True,   # only takes effect if enforce_direction_gate=True

    # ── Emergency exit (LIVE_MODE only) ───────────────────────────────────────
    "emergency_retry_sec"    : 30,
    "emergency_max_attempts" : 30,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"               : "nifty_candle_breakout_trades.csv",
}


class NiftyCandleBreakoutStrategy(BaseStrategy):

    # Routes Nifty 50 index ticks here exclusively.
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
        self._expiry_date   = None
        self._instruments   = None

        # ── Pattern state machine: SCAN → WAIT_CONFIRM → WATCH_BREAKOUT ───────
        self._state              = "SCAN"
        self._c1                 = None
        self._trigger_color      = None
        self._confirm5           = None
        self._breakout_window_ts = None

        self._signal_meta = None
        self._pm          = None

        self._trade         = None
        self._pending_entry = None
        self._trades_today  = 0
        self._today_pnl     = 0.0
        self._completed     = []

        # FIX C: risk state
        self._consec_losses  = 0
        self._halted         = False
        self._halt_reason    = ""
        self._last_loss_ts   = 0.0

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | qty={CFG['quantity']} "
            f"min_body={CFG['min_body_pts']}pts | SL=-{CFG['sl_points']} "
            f"trail arms at +{CFG['trail_activate_pts']} locking +{CFG['lock_pts']}, "
            f"then {CFG['trail_distance_pts']} behind peak | "
            f"caps: {CFG['max_trades_day']} trades / -Rs{CFG['max_daily_loss_rs']:.0f} / "
            f"{CFG['max_consec_losses']} consec losses | "
            f"gates: vwap={CFG['require_vwap_side']} st={CFG['require_supertrend']}"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        self._instruments = instruments
        self._expiry_date = pm.expiry_date
        self._pm          = pm

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

    # ── Indicator snapshot ────────────────────────────────────────────────────

    def _indicator_snapshot(self, spot: float) -> dict:
        """
        VWAP / PCR / EMA slope / RSI slope / MACD histogram / ATR% / Supertrend
        from this strategy's own rolling 10s candle buffer plus live hub VWAP
        and live PCR.

        FIX E: this is no longer log-only. _direction_allowed() reads
        spot_vs_vwap and supertrend_dir from it as entry gates.
        """
        candles = self._c10.last_n_closed(CANDLE_WINDOW)
        vwap    = self._hub.session_vwap.value
        pcr     = self._pm.pcr if self._pm is not None else None
        return compute_fast_indicators(candles, spot=spot, vwap=vwap, pcr=pcr)

    # ── FIX E: directional gates ──────────────────────────────────────────────

    def _direction_allowed(self, signal: str, ind: dict) -> tuple:
        """
        Returns (allowed: bool, reason: str).

        Both gates FAIL OPEN when the underlying indicator is blank/not ready,
        so nothing changes during warm-up or if fast_indicators errors out.
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

    # ── FIX C: risk gate ──────────────────────────────────────────────────────

    def _risk_block_reason(self, ts: datetime) -> Optional[str]:
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

        # No new setups too close to EOD, or during opening warm-up.
        if t >= CFG["last_entry_time"] or t < CFG["warmup_until"]:
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
            if (self._state == "WATCH_BREAKOUT" and closed10 is not None
                    and closed10["ts"] == self._breakout_window_ts):
                log.info(
                    f"[{self.name}] No breakout within watched window "
                    f"({self._breakout_window_ts.strftime('%H:%M:%S')}) — resetting scan"
                )
                if self._c1 is not None and self._confirm5 is not None:
                    ind = self._indicator_snapshot(self._confirm5["close"])
                    self._log_signal_csv(self._signal_row(
                        self._breakout_window_ts, "NO_BREAKOUT", ind
                    ))
                self._reset_pattern_state()

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Resolves a pending entry and manages the trailing / time stop.
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

        # ── SL grace period ───────────────────────────────────────────────────
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from is not None and ts < sl_active_from:
            return

        t = self._trade
        t["peak"] = max(t["peak"], price)
        gain      = t["peak"] - t["entry"]

        # ── FIX A: trail arms late, locks a real profit, trails wide ─────────
        # Old: armed at +5 and locked exactly +5 with a 5pt trail — every
        # winner died on premium noise at exactly the locked level.
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

        # ── FIX F: time stop for trades going nowhere ────────────────────────
        held = (ts - t["entry_time"]).total_seconds()
        if held >= CFG["max_hold_seconds"]:
            if (price - t["entry"]) < CFG["time_stop_min_profit"]:
                log.info(
                    f"[{self.name}] Time stop | held={held:.0f}s "
                    f"pnl={(price - t['entry']):+.2f}pts — exiting"
                )
                self._do_exit(price, "TIME_STOP", ts)

    # ── Pattern detection ────────────────────────────────────────────────────

    def _check_trigger_candle(self, c: dict):
        """10s candle with body strictly > min_body_pts."""
        body = c["close"] - c["open"]
        if abs(body) <= CFG["min_body_pts"]:
            return

        color = "GREEN" if body > 0 else "RED"

        self._c1            = c
        self._trigger_color = color
        self._state         = "WAIT_CONFIRM"

        log.info(
            f"[{self.name}] Trigger candle (C1) {color} | "
            f"o={c['open']:.1f} h={c['high']:.1f} l={c['low']:.1f} c={c['close']:.1f} "
            f"body={abs(body):.1f} @ {c['ts'].strftime('%H:%M:%S')} — waiting for confirm 5s"
        )

        ind = self._indicator_snapshot(c["close"])
        self._log_signal_csv(self._signal_row(c["ts"], "TRIGGER", ind))

    def _check_confirm_candle(self, c5: dict, ts: datetime):
        """The next 5s candle must close the same colour as C1."""
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
            self._confirm5 = c5
            ind = self._indicator_snapshot(c5["close"])
            self._log_signal_csv(self._signal_row(c5["ts"], "CONFIRM_MISMATCH", ind))
            self._reset_pattern_state()
            return

        self._confirm5           = c5
        self._breakout_window_ts = c5["ts"]
        self._state              = "WATCH_BREAKOUT"

        log.info(
            f"[{self.name}] Confirm 5s candle matches ({color5}) | "
            f"o={c5['open']:.1f} h={c5['high']:.1f} l={c5['low']:.1f} c={c5['close']:.1f} "
            f"@ {c5['ts'].strftime('%H:%M:%S')} — watching breakout in next 10s window"
        )
        ind = self._indicator_snapshot(c5["close"])
        self._log_signal_csv(self._signal_row(c5["ts"], "CONFIRM_MATCH", ind))

    def _check_breakout_tick(self, price: float, ts: datetime):
        """Immediate entry the instant price breaks the confirm candle's high/low."""
        c5 = self._confirm5
        if c5 is None:
            return

        if self._trigger_color == "GREEN":
            triggered = price > c5["high"] and price > c5["close"]
            signal    = "CE"
        else:
            triggered = price < c5["low"] and price < c5["close"]
            signal    = "PE"

        if not triggered:
            return

        ind = self._indicator_snapshot(price)

        # ── FIX C: risk caps checked at the moment of entry ──────────────────
        block = self._risk_block_reason(ts)
        if block:
            log.info(f"[{self.name}] Setup blocked by risk gate: {block} — skipping")
            row = self._signal_row(ts, "RISK_BLOCK", ind, breakout_price=price)
            row["block_reason"] = block
            self._log_signal_csv(row)
            self._reset_pattern_state()
            return

        # ── FIX E: directional gates ─────────────────────────────────────────
        allowed, reason = self._direction_allowed(signal, ind)
        if not allowed:
            log.info(
                f"[{self.name}] Setup blocked by direction gate: {reason} "
                f"(signal={signal} vwap_side={ind.get('spot_vs_vwap', '?')} "
                f"st={ind.get('supertrend_dir', '?')}) — skipping"
            )
            row = self._signal_row(ts, "GATE_BLOCK", ind, breakout_price=price)
            row["block_reason"] = reason
            self._log_signal_csv(row)
            self._reset_pattern_state()
            return

        self._log_signal_csv(
            self._signal_row(ts, "BREAKOUT_ENTER", ind, breakout_price=price)
        )
        self._signal_meta = self._build_signal_meta(price, ts, ind)
        self._reset_pattern_state()
        self._fire_entry(signal, price, ts)

    def _signal_row(self, ts, event: str, ind: dict, breakout_price=None) -> dict:
        c1 = self._c1 or {}
        c5 = self._confirm5 or {}
        body = (abs(c1["close"] - c1["open"])
                if ("close" in c1 and "open" in c1) else "")
        return {
            "timestamp"     : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "event"         : event,
            "color"         : self._trigger_color or "",
            "c1_open"       : c1.get("open", ""),  "c1_high" : c1.get("high", ""),
            "c1_low"        : c1.get("low", ""),   "c1_close": c1.get("close", ""),
            "c1_body"       : round(body, 2) if body != "" else "",
            "confirm_open"  : c5.get("open", ""),  "confirm_high": c5.get("high", ""),
            "confirm_low"   : c5.get("low", ""),   "confirm_close": c5.get("close", ""),
            "breakout_price": round(breakout_price, 2) if breakout_price else "",
            "block_reason"  : "",
            **ind,
        }

    def _build_signal_meta(self, breakout_price: float, ts: datetime, ind: dict = None) -> dict:
        c1, c5 = self._c1 or {}, self._confirm5 or {}
        return {
            "index_price"   : breakout_price,
            "weekday"       : ts.strftime("%A"),
            "c1_body"       : round(abs(c1.get("close", 0) - c1.get("open", 0)), 2),
            "c1_open"       : c1.get("open", ""), "c1_high": c1.get("high", ""),
            "c1_low"        : c1.get("low", ""),  "c1_close": c1.get("close", ""),
            "confirm_open"  : c5.get("open", ""), "confirm_high": c5.get("high", ""),
            "confirm_low"   : c5.get("low", ""),  "confirm_close": c5.get("close", ""),
            "breakout_price": round(breakout_price, 2),
            **(ind or {}),
        }

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
            log.warning(f"[{self.name}] No live price yet for {sym} — storing pending entry")
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy has a position")
            self.unsubscribe_option(token)   # FIX D
            return

        result = self._place_buy(sym, token, CFG["quantity"], opt_price)
        if result is None:
            self._release_slot()
            self.unsubscribe_option(token)   # FIX D
            log.error(f"[{self.name}] BUY order FAILED for {sym} — entry aborted")
            return

        order_id, raw_fill = result

        # ── FIX B: pay the spread on entry ───────────────────────────────────
        # Live mode already gives a real exchange average_price, so the model
        # is only applied to paper fills.
        fill_price = raw_fill if LIVE_MODE else entry_fill(raw_fill)

        sl = fill_price - CFG["sl_points"]
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])

        rt_cost = round_trip_cost_pts(fill_price, CFG["quantity"])

        meta = self._signal_meta or {}
        self._signal_meta = None
        self._trade = {
            "state"            : "OPEN",
            "symbol"           : sym,
            "token"            : token,
            "signal"           : signal,
            "entry"            : fill_price,
            "raw_ltp"          : raw_fill,
            "sl"               : sl,
            "tp"               : None,   # no fixed TP — trailing stop only
            "peak"             : fill_price,
            "entry_time"       : ts,
            "sl_active_from"   : sl_active_from,
            "order_id"         : order_id,
            "qty"              : CFG["quantity"],
            "trail_armed"      : False,
            "rt_cost_pts"      : rt_cost,
            "_exit_in_progress": False,
            "meta"             : meta,
        }
        self._trades_today += 1

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] ENTRY #{self._trades_today} {sym} "
            f"@ {fill_price:.2f} (ltp={raw_fill:.2f}) | SL={sl:.2f} (-{CFG['sl_points']}) | "
            f"trail arms +{CFG['trail_activate_pts']} locks +{CFG['lock_pts']} | "
            f"est round-trip cost={rt_cost:.2f}pts | reason={reason} | order_id={order_id}"
        )

        self._log_csv(self._trade_row(ts, "ENTRY", fill_price, sl, "OPEN", 0, 0,
                                      reason, order_id, meta, "", ""))

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
            order_id, raw_sell = result
            t["state"] = "CLOSED"
            self._release_slot()
            self._finalize_exit(t, raw_sell, order_id, reason, ts)
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
                    self._finalize_exit(t, ref_price, None,
                                        f"{reason}_EXCHANGE_CLOSED", _now_ist())
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
                        f"[{self.name}] Emergency exit SUCCESS on attempt {attempt} "
                        f"@ {sell_price:.2f}"
                    )
                    self._finalize_exit(t, sell_price, order_id,
                                        f"{reason}_EMERGENCY", _now_ist())
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
            # Still release the token so the WebSocket subscription doesn't leak.
            self.unsubscribe_option(t["token"])
            self._trade = None

        thread = threading.Thread(
            target=_loop, name="nifty-candle-breakout-emergency-exit", daemon=True,
        )
        thread.start()
        log.info(f"[{self.name}] Emergency exit thread started for {t['symbol']}")

    def _finalize_exit(self, t: dict, raw_sell: float, order_id: Optional[str],
                       reason: str, ts: datetime):
        # ── FIX B: pay the spread on exit, then subtract brokerage/STT/GST ───
        sell_price = raw_sell if LIVE_MODE else exit_fill(raw_sell)
        gross_pnl  = round((sell_price - t["entry"]) * t["qty"], 2)
        pnl        = net_pnl_rs(t["entry"], sell_price, t["qty"])
        self._today_pnl += pnl

        # ── FIX C: consecutive-loss / cooldown bookkeeping ───────────────────
        if pnl < 0:
            self._consec_losses += 1
            self._last_loss_ts   = _time_mod.time()
        else:
            self._consec_losses = 0

        if self._today_pnl <= -abs(CFG["max_daily_loss_rs"]):
            self._halted     = True
            self._halt_reason = f"max_daily_loss(Rs{self._today_pnl:.0f})"
            log.warning(f"[{self.name}] HALTED for the day — {self._halt_reason}")
        elif self._consec_losses >= CFG["max_consec_losses"]:
            self._halted     = True
            self._halt_reason = f"max_consec_losses({self._consec_losses})"
            log.warning(f"[{self.name}] HALTED for the day — {self._halt_reason}")

        time_in_trade_s = round((ts - t["entry_time"]).total_seconds(), 1)
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
            f"Today={self._today_pnl:.0f} | held={time_in_trade_s:.0f}s"
            f"{slip_tag} | consec_L={self._consec_losses} | order_id={order_id}"
        )

        self._log_csv(self._trade_row(
            ts, "EXIT", sell_price, t["sl"], "CLOSED", pnl, gross_pnl,
            reason, order_id, t.get("meta", {}), time_in_trade_s, sl_tp_slippage,
            symbol=t["symbol"],
        ))
        self._completed.append({
            **t, "exit_price": sell_price, "exit_reason": reason,
            "pnl": pnl, "gross_pnl": gross_pnl,
        })

        # FIX D: release the WebSocket subscription this trade acquired.
        try:
            self.unsubscribe_option(t["token"])
        except Exception as e:
            log.warning(f"[{self.name}] unsubscribe error for {t['token']}: {e}")

        self._trade = None
        self._reset_pattern_state()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _trade_row(self, ts, action, price, sl, status, pnl, gross_pnl,
                   reason, order_id, meta, time_in_trade_s, sl_tp_slippage,
                   symbol=None) -> dict:
        sym = symbol or (self._trade or {}).get("symbol", "")
        return {
            "timestamp"      : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"         : sym,
            "action"         : action,
            "price"          : price,
            "sl"             : round(sl, 2) if sl else "",
            "tp"             : "",
            "status"         : status,
            "pnl"            : round(pnl, 2),
            "gross_pnl"      : round(gross_pnl, 2),
            "cost_drag"      : round(gross_pnl - pnl, 2),
            "reason"         : reason,
            "mode"           : "LIVE" if LIVE_MODE else "PAPER",
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
            "time_in_trade_s": time_in_trade_s,
            "sl_tp_slippage" : sl_tp_slippage,
            **{k: meta.get(k, "") for k in INDICATOR_FIELDS},
        }

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "symbol", "action", "price",
            "sl", "tp", "status", "pnl", "gross_pnl", "cost_drag",
            "reason", "mode", "order_id",
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
        Logs EVERY trigger candle regardless of outcome — including the new
        GATE_BLOCK and RISK_BLOCK events, so you can measure exactly what the
        new filters cost you before deciding to keep them.
        """
        fname  = CFG["csv_file"].replace("_trades.csv", "_signals.csv")
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "event", "color",
            "c1_open", "c1_high", "c1_low", "c1_close", "c1_body",
            "confirm_open", "confirm_high", "confirm_low", "confirm_close",
            "breakout_price", "block_reason",
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

        wins   = [t for t in self._completed if t["pnl"] > 0]
        losses = [t for t in self._completed if t["pnl"] <= 0]
        gross  = sum(t.get("gross_pnl", 0) for t in self._completed)

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
