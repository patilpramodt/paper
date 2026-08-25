"""
strategies/banknifty_candle_breakout_strategy.py

BANKNIFTY_CANDLE_BREAKOUT — 10s marubozu + 5s confirm + tick breakout, BankNifty only.

REQUIREMENTS (as given)
────────────────────────
  1. BankNifty only.
  2. Watch every closed 10-second index candle. A candle qualifies as a
     "trigger" (C1) if its body (|close - open|) is strictly greater
     than 20 points. (Wick length is NOT checked — removed per user
     request; a candle with a normal wick can still trigger as long as
     the body threshold is met.)
  3. After C1 closes, watch the very next 5-second candle (the first
     half of the following 10s bucket). If it closes the SAME color as
     C1 → confirmed, otherwise the setup is abandoned.
  4. Once confirmed, watch price tick-by-tick for the rest of that same
     10-second window (the "next 10 sec candle", which the confirm
     5s candle is the first half of). The moment price breaks the
     confirm candle's high (GREEN → CE) or low (RED → PE), enter
     IMMEDIATELY — no waiting for any candle to close.

  4a. FAST PATH (added per user request, BankNifty only): while scanning
      for C1 (state SCAN), watch every tick of the currently-forming 10s
      candle against ITS OWN open. If price moves >= fast_entry_pts
      (30 points) from that candle's open at any point before it closes,
      enter IMMEDIATELY — skip the 5s confirm candle and the breakout
      watch entirely. This runs ahead of the normal flow: if 30 points
      is never hit intra-candle, the candle still closes and is evaluated
      by the existing >20pt body + confirm + breakout-watch logic exactly
      as before. Only ever active in SCAN state (i.e. not while already
      watching a confirm/breakout for a prior C1).
  5. SL/TP are fixed at 10 points on the OPTION PREMIUM (not the index),
     using the same fixed-points-on-premium convention as SPIKE / SPIKE_NIFTY
     (kept at 10 points on premium — only the trigger body threshold
     changes for BankNifty, per user request).
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

log = logging.getLogger("strategy.banknifty_candle_breakout")

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
    "bucket_10s"             : 10,
    "bucket_5s"              : 5,
    "min_body_pts"           : 20.0,   # body must be strictly greater than this
    "fast_entry_pts"         : 30.0,   # intra-candle tick move that triggers immediate entry (fast path, skips confirm)

    # ── SL / trailing lock (on OPTION PREMIUM) — min 10pt gain, unlimited upside ──
    # ── FIX A: risk / reward on OPTION PREMIUM points ─────────────────────────
    # The old geometry locked profit at exactly the same distance it armed at,
    # with an equally tight trail — a fraction of the risk rather than a
    # multiple of it. Reward must exceed risk AND exceed the round-trip cost
    # priced in core/costs.py, or no entry signal can rescue the strategy.
    "sl_points"              : 30.0,
    "trail_activate_pts"     : 20.0,   # open profit before trail arms
    "lock_pts"               : 10.0,   # profit locked the moment it arms
    "trail_distance_pts"     : 10.0,   # thereafter trail this far behind peak
    "sl_grace_seconds"       : 5,

    # ── FIX F: time stop ──────────────────────────────────────────────────────
    "max_hold_seconds"       : 300,
    "time_stop_min_profit"   : 12.0,

    # ── Risk caps removed (2026-08-12) ─────────────────────────────────────────
    # Pure paper-data collection phase: no max daily loss, no max trades/day,
    # no consecutive-loss halt, no post-loss cooldown. Every setup that fires
    # takes a trade so the CSV reflects the strategy's raw, unfiltered
    # behaviour.
    "enforce_direction_gate" : False,

    # ── FIX E: directional gates (were computed but unused) ───────────────────
    "require_vwap_side"      : True,   # only takes effect if enforce_direction_gate=True
    "require_supertrend"     : True,   # only takes effect if enforce_direction_gate=True

    # ── Indicator filter (added 2026-08-18, retuned 2026-08-18 for August-only) ─
    # Best-performing filter found for THIS strategy on AUGUST-2026 trades
    # ONLY (08-03 to 08-18, ~66 indicator-ready trades): PCR neutral
    # ([pcr_min, pcr_max]) AND ATR% above its August sample median AND MACD
    # histogram slope pointing the same way as the trade direction.
    # August sample: 66 trades / 60.6% WR / -Rs 3,127 baseline ->
    # 22 trades / 59.1% WR / +Rs 1,534 (net flipped positive; this beat the
    # earlier 2-filter version, which reached +Rs 1,312 on 26 trades).
    #
    # CAVEAT (also logged at runtime — see _indicator_filter_allowed): tuned
    # specifically on August data at the user's request — NOT re-checked
    # against the pre-August (Jul24-Aug10) sample, where an earlier, looser
    # version of this filter cut mostly WINNING trades. Treat as August-
    # specific, not a portable edge. Defaults to OFF (fails open), same as
    # enforce_direction_gate above — flip True only after re-validating.
    # REVERTED 2026-08-25: cross-strategy backtest (2026-07-14 to 2026-08-25,
    # 269 trades) shows this filter flipped the strategy from +30/trade
    # pre-activation to -325/trade post-activation (n=27). It was tuned on
    # an August-only sample and the code's own original caveat said as much
    # ("NOT re-checked against pre-August data, where an earlier looser
    # version cut mostly WINNING trades"). Data confirmed the caveat was
    # right. Leave OFF until re-tuned on a walk-forward sample, not a
    # single fixed window. See /areas/paper-main.md.
    "enforce_indicator_filter": False,
    "min_atr_pct"             : 0.0222,  # ATR% must exceed this (August sample median)
    "require_macd_slope"      : True,    # macd_slope must agree with trade direction
    "require_pcr_neutral"     : True,    # pcr must be inside [pcr_min, pcr_max]
    "pcr_min"                  : 0.7,
    "pcr_max"                  : 1.3,

    # ── Weakest-hour guard (added 2026-08-25) ─────────────────────────────────
    # Backtest across 27 sessions shows hour 10 (10:00-10:59) is the one
    # consistently negative hour for ALL FOUR candle-breakout variants
    # (avg -185 to -253/trade) while every other hour is flat-to-positive on
    # average. Skips NEW setups only — an already-open trade is still
    # managed normally through SL/trail/time-stop.
    "avoid_hour"              : 10,

    # ── Fast-tick entry path (DISABLED 2026-08-25) ────────────────────────────
    # Full-log backtest (2026-07-14 to 2026-08-25, 131 fast-tick trades)
    # shows this no-confirmation path is negative in BOTH months, while the
    # confirm-based path on the same strategy is flat-to-positive:
    #     fast-tick : Jul  69 trades  -561  (-8/trade)
    #                 Aug  62 trades -3717 (-60/trade)
    #     confirm   : Jul 100 trades  -246  (-2/trade)
    #                 Aug  38 trades +3016 (+79/trade)
    # Disabling it takes the strategy from -1508 to +2770 over the same
    # history. This is a mechanism-level result (entering with zero
    # confirmation chases spikes into reversals), consistent across both
    # months — not a threshold tuned to one window. Set True to re-enable.
    "enable_fast_tick_entry"  : False,

    # ── Emergency exit (LIVE_MODE only) ───────────────────────────────────────
    "emergency_retry_sec"    : 30,
    "emergency_max_attempts" : 30,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"               : "banknifty_candle_breakout_trades.csv",
}


class BankNiftyCandleBreakoutStrategy(BaseStrategy):

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
        return "BANKNIFTY_CANDLE_BREAKOUT"

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

        self._signal_meta    = None   # metrics collected from C1/confirm/breakout for the next entry

        # Live PreMarketData reference (set in pre_market()) — read
        # self._pm.pcr at signal time so it reflects the live-refreshed
        # value, not a 9:08 AM snapshot. Log-only, mirrors orb_v2.py pattern.
        self._pm = None

        self._trade         = None
        self._pending_entry  = None
        self._trades_today   = 0
        self._today_pnl      = 0.0
        self._completed      = []

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | qty={CFG['quantity']} "
            f"min_body={CFG['min_body_pts']}pts "
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
        # Kept as a LIVE reference (not just pm.pcr snapshotted here) so
        # _indicator_snapshot() always reads the current PCR — the
        # background refresh thread in premarket.py keeps updating pm.pcr
        # every pcr_interval seconds all session long.
        self._pm = pm

        # Reset daily counters (matters when the process is long-lived).
        self._trades_today  = 0
        self._today_pnl     = 0.0
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
        Entry/exit decisions never read this — log-only.
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

        # No new setups during the historically weakest hour — see CFG note.
        if CFG.get("avoid_hour") is not None and t.hour == CFG["avoid_hour"]:
            return

        # ── Pattern state machine ─────────────────────────────────────────────
        if self._state == "SCAN":
            # Fast path: 30pt intra-candle tick move → enter immediately,
            # no confirm/breakout watch. DISABLED by default since 2026-08-25
            # (see enable_fast_tick_entry in CFG) — negative in both months
            # of the backtest. When off, every candle falls through to the
            # normal >20pt body + confirm + breakout-watch flow below.
            if CFG.get("enable_fast_tick_entry", True) and self._pending_entry is None:
                self._check_fast_tick_trigger(price, ts)
                if self._trade is not None or self._pending_entry is not None:
                    return
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
                if self._c1 is not None and self._confirm5 is not None:
                    ind = self._indicator_snapshot(self._confirm5["close"])
                    self._log_signal_csv({
                        "timestamp": self._breakout_window_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "event"    : "NO_BREAKOUT",
                        "color"    : self._trigger_color,
                        "c1_open"  : self._c1["open"], "c1_high": self._c1["high"],
                        "c1_low"   : self._c1["low"],  "c1_close": self._c1["close"],
                        "c1_body"  : round(abs(self._c1["close"] - self._c1["open"]), 2),
                        "confirm_open": self._confirm5["open"], "confirm_high": self._confirm5["high"],
                        "confirm_low" : self._confirm5["low"],  "confirm_close": self._confirm5["close"],
                        "breakout_price": "",
                        **ind,
                    })
                self._reset_pattern_state()

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

    def _check_fast_tick_trigger(self, price: float, ts: datetime):
        """
        Rule 4a (fast path): watch every tick of the currently-forming 10s
        candle against its own open. If price has moved >= fast_entry_pts
        from that open, enter IMMEDIATELY — no confirm 5s candle, no
        breakout watch, no waiting for the candle to close. Only called
        while state == SCAN. If this doesn't fire, the candle proceeds to
        close normally and is evaluated by _check_trigger_candle() as before.
        """
        cur = self._c10.get_current()
        if cur is None:
            return

        move = price - cur["open"]
        if abs(move) < CFG["fast_entry_pts"]:
            return

        color  = "GREEN" if move > 0 else "RED"
        signal = "CE" if color == "GREEN" else "PE"

        log.info(
            f"[{self.name}] FAST TICK TRIGGER {color} | candle_open={cur['open']:.1f} "
            f"price={price:.1f} move={move:+.1f} (>= {CFG['fast_entry_pts']}pts) "
            f"@ {ts.strftime('%H:%M:%S')} — entering immediately, skipping confirm"
        )

        ind = self._indicator_snapshot(price)

        self._log_signal_csv({
            "timestamp"     : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "event"         : "FAST_TICK_ENTER",
            "color"         : color,
            "c1_open"       : cur["open"], "c1_high": cur["high"],
            "c1_low"        : cur["low"],  "c1_close": price,
            "c1_body"       : round(abs(move), 2),
            "confirm_open"  : "", "confirm_high": "", "confirm_low": "", "confirm_close": "",
            "breakout_price": round(price, 2),
            **ind,
        })

        self._signal_meta = {
            "index_price"   : price,
            "weekday"       : ts.strftime("%A"),
            "c1_body"       : round(abs(move), 2),
            "c1_open"       : cur["open"], "c1_high": cur["high"],
            "c1_low"        : cur["low"],  "c1_close": price,
            "confirm_open"  : "", "confirm_high": "",
            "confirm_low"   : "", "confirm_close": "",
            "breakout_price": round(price, 2),
            **ind,
        }
        self._reset_pattern_state()
        self._fire_entry(signal, price, ts, reason="tick_30pt_fast_entry")

    def _check_trigger_candle(self, c: dict):
        """
        Rule 2: 10s candle with body strictly > min_body_pts.
        (No-wick condition removed per user request — wick length is no
        longer checked; only the body-size threshold matters now.)
        """
        body = c["close"] - c["open"]
        if abs(body) <= CFG["min_body_pts"]:
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

        ind = self._indicator_snapshot(c["close"])

        self._log_signal_csv({
            "timestamp"  : c["ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "event"      : "TRIGGER",
            "color"      : color,
            "c1_open"    : c["open"], "c1_high": c["high"],
            "c1_low"     : c["low"],  "c1_close": c["close"],
            "c1_body"    : round(abs(body), 2),
            "confirm_open": "", "confirm_high": "", "confirm_low": "", "confirm_close": "",
            "breakout_price": "",
            **ind,
        })

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
            ind = self._indicator_snapshot(c5["close"])
            self._log_signal_csv({
                "timestamp": c5["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                "event"    : "CONFIRM_MISMATCH",
                "color"    : self._trigger_color,
                "c1_open"  : self._c1["open"], "c1_high": self._c1["high"],
                "c1_low"   : self._c1["low"],  "c1_close": self._c1["close"],
                "c1_body"  : round(abs(self._c1["close"] - self._c1["open"]), 2),
                "confirm_open": c5["open"], "confirm_high": c5["high"],
                "confirm_low" : c5["low"],  "confirm_close": c5["close"],
                "breakout_price": "",
                **ind,
            })
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
        ind = self._indicator_snapshot(c5["close"])
        self._log_signal_csv({
            "timestamp": c5["ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "event"    : "CONFIRM_MATCH",
            "color"    : color5,
            "c1_open"  : self._c1["open"], "c1_high": self._c1["high"],
            "c1_low"   : self._c1["low"],  "c1_close": self._c1["close"],
            "c1_body"  : round(abs(self._c1["close"] - self._c1["open"]), 2),
            "confirm_open": c5["open"], "confirm_high": c5["high"],
            "confirm_low" : c5["low"],  "confirm_close": c5["close"],
            "breakout_price": "",
            **ind,
        })

    def _check_breakout_tick(self, price: float, ts: datetime):
        """
        Rule 4 + 6: immediate entry the instant price breaks the confirm
        candle's high/low AND has crossed its close (implied for a clean
        breakout, checked explicitly for the stated skip rule).
        """
        c5 = self._confirm5
        c1 = self._c1
        if self._trigger_color == "GREEN":
            if price > c5["high"] and price > c5["close"]:
                ind = self._indicator_snapshot(price)
                self._log_signal_csv({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "event"    : "BREAKOUT_ENTER",
                    "color"    : self._trigger_color,
                    "c1_open"  : c1["open"], "c1_high": c1["high"],
                    "c1_low"   : c1["low"],  "c1_close": c1["close"],
                    "c1_body"  : round(abs(c1["close"] - c1["open"]), 2),
                    "confirm_open": c5["open"], "confirm_high": c5["high"],
                    "confirm_low" : c5["low"],  "confirm_close": c5["close"],
                    "breakout_price": round(price, 2),
                    **ind,
                })
                self._signal_meta = self._build_signal_meta(price, ts, ind)
                self._reset_pattern_state()
                self._fire_entry("CE", price, ts)
        else:
            if price < c5["low"] and price < c5["close"]:
                ind = self._indicator_snapshot(price)
                self._log_signal_csv({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "event"    : "BREAKOUT_ENTER",
                    "color"    : self._trigger_color,
                    "c1_open"  : c1["open"], "c1_high": c1["high"],
                    "c1_low"   : c1["low"],  "c1_close": c1["close"],
                    "c1_body"  : round(abs(c1["close"] - c1["open"]), 2),
                    "confirm_open": c5["open"], "confirm_high": c5["high"],
                    "confirm_low" : c5["low"],  "confirm_close": c5["close"],
                    "breakout_price": round(price, 2),
                    **ind,
                })
                self._signal_meta = self._build_signal_meta(price, ts, ind)
                self._reset_pattern_state()
                self._fire_entry("PE", price, ts)

    def _build_signal_meta(self, breakout_price: float, ts: datetime, ind: dict = None) -> dict:
        c1, c5 = self._c1, self._confirm5
        return {
            "index_price"   : breakout_price,
            "weekday"       : ts.strftime("%A"),
            "c1_body"       : round(abs(c1["close"] - c1["open"]), 2),
            "c1_open"       : c1["open"], "c1_high": c1["high"],
            "c1_low"        : c1["low"],  "c1_close": c1["close"],
            "confirm_open"  : c5["open"], "confirm_high": c5["high"],
            "confirm_low"   : c5["low"],  "confirm_close": c5["close"],
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

    # ── Indicator filter (added 2026-08-18) ────────────────────────────────────

    def _indicator_filter_allowed(self, signal: str, ind: dict) -> tuple:
        """
        Returns (allowed: bool, reason: str).

        Best combo found for BANKNIFTY_CANDLE_BREAKOUT on AUGUST-2026 data
        specifically: PCR neutral AND ATR% > August sample median AND MACD
        histogram slope aligned with trade direction. See CFG comment for
        the before/after numbers and the generalization caveat.

        Fails open when indicators aren't ready yet or CFG flag is off —
        same convention as _direction_allowed above.
        """
        if not CFG["enforce_indicator_filter"]:
            return True, ""
        if not ind.get("indicators_ready"):
            return True, ""

        if CFG["require_pcr_neutral"]:
            pcr = ind.get("pcr")
            if pcr is not None and pcr != "" and not (CFG["pcr_min"] <= pcr <= CFG["pcr_max"]):
                return False, "pcr_outside_neutral_band"

        atr_pct = ind.get("atr_pct")
        if atr_pct is not None and atr_pct != "" and atr_pct < CFG["min_atr_pct"]:
            return False, "atr_pct_below_min"

        if CFG["require_macd_slope"]:
            macd_slope = ind.get("macd_slope")
            if macd_slope is not None and macd_slope != "":
                if signal == "CE" and macd_slope <= 0:
                    return False, "macd_slope_not_bullish_for_CE"
                if signal == "PE" and macd_slope >= 0:
                    return False, "macd_slope_not_bearish_for_PE"

        return True, ""

    def _fire_entry(self, signal: str, index_price: float, ts: datetime,
                     reason: str = "10s_marubozu_5s_confirm_breakout"):
        # ── FIX E: directional gate ───────────────────────────────────────────
        # Placed here rather than at each pattern call-site so every entry
        # path (breakout, fast-tick, C2 points entry) is covered.
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

        # ── Indicator filter (ATR% + MACD slope) ─────────────────────────────
        _iok, _iwhy = self._indicator_filter_allowed(signal, _ind)
        if not _iok:
            log.info(
                f"[{self.name}] Entry blocked by indicator filter: {_iwhy} "
                f"(signal={signal} atr_pct={_ind.get('atr_pct', '?')} "
                f"macd_slope={_ind.get('macd_slope', '?')}) — skipping"
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
            target=_loop, name="banknifty-candle-breakout-emergency-exit", daemon=True,
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
            f"order_id={order_id}"
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
        log.info(f"[{self.name}] {'='*50}\n")
