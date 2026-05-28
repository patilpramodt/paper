"""
strategies/nifty_directional_strategy.py

NIFTY_DIRECTIONAL Strategy
─────────────────────────────────────────────────────────────────────────────

PHILOSOPHY
  "Less but perfect entry." Wait for market to prove direction, then buy
  the retest — not the breakout.  Two entry modes handle different market
  conditions so the framework never completely misses a big day.

INDEX
  Index       : NSE:NIFTY 50
  INDEX_TOKEN : 256265  (Zerodha fixed token)
  Expiry      : Weekly  (Nifty 50 weekly; get_nearest_expiry auto-picks)
  Strike step : 50 pts
  Lot size    : 75  (verify before going live — SEBI revises periodically)

MODE DETECTION  (finalised at first 5-min candle close after 9:20)
  Mode B triggers if ANY of:
    • Gap > 0.4 %  vs prev close at 9:15 open
    • First 15-min OR range > MODE_B_RANGE_TRIGGER (80) pts
    • Expiry day (Thu) AND gap > 0.2 %
  Else: Mode A.

MODE A — NORMAL TRENDING DAY
  Entry windows : 9:30–11:00 AM  and  1:15–2:15 PM
  Hard cutoff   : no new entries after 2:30 PM
  Filters (all must pass):
    1. EMA stack  : EMA9 > EMA20 > EMA50 (CE)  or  EMA9 < EMA20 < EMA50 (PE)
    2. VWAP side  : price above VWAP (CE) or below VWAP (PE)
    3. RSI(14)    : 55–68 (CE) or 32–45 (PE)   — avoids overbought/oversold entries
    4. Pullback   : at least one of the last 2 candle lows ≤ EMA9 (CE)
                    or one of the last 2 candle highs ≥ EMA9 (PE)
                    AND current candle resumed direction (close > EMA9 for CE)
    5. PCR gate   : if available, CE needs PCR ≥ PCR_MIN (0.8),
                                  PE needs PCR ≤ PCR_MAX (1.2)
  On valid signal: enter ATM call/put, current weekly expiry.

MODE B — MOMENTUM / EXPIRY / GAP-AND-GO DAY
  Entry window  : 9:15–11:00 AM
  Hard cutoff   : no new entries after 11:00 AM in Mode B
  Logic:
    After initial spike direction is detected (gap or strong first bar),
    wait for a micro-consolidation: 2–4 consecutive 5-min candles whose
    H-L range stays below MODE_B_CONSOL_RANGE (80) pts.
    Entry trigger: next candle closes above consolidation high (CE) or
                   below consolidation low (PE).
    IV guard: if VIX > VIX_SKIP_THRESH (18), skip Mode B entries
              (IV is too expensive at spike peak).
  On valid signal: enter ATM call/put.

STRIKE SELECTION
  Default: ATM strike (rounded to nearest 50-pt multiple).
  1 OTM only when MODE_B_CONSOL entry and score is marginal — currently
  always ATM for simplicity (easier to review in paper mode).

TRADE MANAGEMENT  (option premium based, tick-by-tick via on_option_tick)
  SL      : fill_price × (1 − SL_PCT)  i.e. 30 % of premium paid
  Trail 1 : when premium ≥ fill × 1.50  → move SL to breakeven (fill_price)
  Trail 2 : track peak premium; SL = peak × (1 − TRAIL_PCT)  i.e. 20 % below peak
  Hard exit: 3:00 PM regardless of position
  Max trades: 2 per day

CANONICAL PATTERNS (t.py integration)
  • INDEX_TOKEN = 256265  → t.py routes nifty_pm + nifty_instruments via
                            the `if strat_index == 256265` branch.
  • hub.add_index_token(256265) already called in t.py — no change needed.
  • hub.backfill(hub.kite, index_token=256265) already called — warms up
    _buf_5m so EMA/RSI are seeded from historical candles at start.
  • Logger name "strategy.nifty_directional" must be added to t.py _STRAT dict.

ISOLATION GUARANTEE
  This strategy shares zero mutable state with any other strategy.
  All reads from PreMarketData are via self._pm (live reference — same
  PCR-caching fix applied to ORB_v2). No BankNifty state is touched.
  Other strategies are completely unaffected.
"""

import csv
import logging
import os
import threading
from collections import deque
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.candle import CandleBuilder
from core.instruments import get_atm_strike
from core.vwap import SessionVWAP

log = logging.getLogger("strategy.nifty_directional")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
#  Change to True only when you want real orders.
#  All other strategies remain in paper mode until their own flag is changed.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  NIFTY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_STRIKE_STEP = 50
NIFTY_INDEX_TOKEN = 256265

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Lot size ──────────────────────────────────────────────────────────────
    "quantity"                : 75,

    # ── Session windows ───────────────────────────────────────────────────────
    "market_open"             : dtime(9, 15),
    "mode_a_start"            : dtime(9, 30),
    "mode_a_window1_end"      : dtime(11, 0),
    "mode_a_window2_start"    : dtime(13, 15),
    "mode_a_window2_end"      : dtime(14, 15),
    "mode_b_start"            : dtime(9, 15),
    "mode_b_cutoff"           : dtime(11, 0),
    "entry_cutoff"            : dtime(14, 30),   # hard: no new entries after this
    "hard_exit_time"          : dtime(15, 0),

    # ── Mode detection thresholds ─────────────────────────────────────────────
    "mode_b_gap_pct"          : 0.004,    # > 0.4 % gap triggers Mode B
    "mode_b_gap_pct_expiry"   : 0.002,    # > 0.2 % gap on expiry day
    "mode_b_range_trigger"    : 80,       # first-15-min OR range > 80 pts → Mode B
    "mode_b_consol_range"     : 80,       # consolidation must be tighter than this
    "mode_b_consol_min_bars"  : 2,        # min bars in consolidation
    "mode_b_consol_max_bars"  : 4,        # max bars before consolidation expires

    # ── Indicator settings ────────────────────────────────────────────────────
    "ema_fast"                : 9,
    "ema_mid"                 : 20,
    "ema_slow"                : 50,
    "rsi_period"              : 14,
    "buf_size"                : 100,      # rolling 5-min candle buffer

    # ── Mode A signal filters ─────────────────────────────────────────────────
    "rsi_ce_min"              : 55,       # RSI must be in this range for CE
    "rsi_ce_max"              : 68,       # avoid overbought entries
    "rsi_pe_min"              : 32,       # avoid oversold entries for PE
    "rsi_pe_max"              : 45,
    "pcr_min_ce"              : 0.80,     # PCR must be ≥ this for CE (bullish OI skew)
    "pcr_max_pe"              : 1.20,     # PCR must be ≤ this for PE (bearish OI skew)

    # ── Mode B IV guard ───────────────────────────────────────────────────────
    "vix_skip_thresh"         : 18.0,     # skip Mode B entries when VIX > this

    # ── Trade management ──────────────────────────────────────────────────────
    "sl_pct"                  : 0.30,     # SL = fill × (1 − 0.30)
    "trail1_trigger_mult"     : 1.50,     # trail1: when premium ≥ fill × 1.50
    "trail2_distance_pct"     : 0.20,     # trail2: SL = peak × (1 − 0.20)
    "max_trades_per_day"      : 2,

    # ── Output ────────────────────────────────────────────────────────────────
    "csv_file"                : "nifty_directional_trades.csv",

    # ── SL grace period after BUY confirm ────────────────────────────────────
    # Prevents stale WebSocket ticks (buffered during _confirm_order poll)
    # from immediately triggering a false SL exit right after entry.
    "sl_grace_seconds"        : 10,
}


class NiftyDirectionalStrategy(BaseStrategy):

    # MarketHub routes Nifty 50 ticks (256265) exclusively to strategies with
    # this class attribute.  BankNifty ticks are never delivered here.
    INDEX_TOKEN = NIFTY_INDEX_TOKEN

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "NIFTY_DIRECTIONAL"

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # ── Internal 5-min candle builder (Nifty strategies build their own) ──
        # MarketHub does NOT call on_candle() for extra-index strategies.
        # on_tick() feeds price into this builder; when a bar closes it calls
        # _process_candle() — mirrors the BBStochNifty pattern exactly.
        self._candle_builder = CandleBuilder(minutes=5)

        # ── Candle buffer for indicator computation ───────────────────────────
        self._buf: deque = deque(maxlen=CFG["buf_size"])

        # ── Internal Nifty VWAP ───────────────────────────────────────────────
        # MarketHub's session_vwap tracks BankNifty prices — useless here.
        self._vwap = SessionVWAP()

        # ── Pre-market data ───────────────────────────────────────────────────
        self._pm            = None       # live reference (for live PCR reads)
        self._vix           = None
        self._prev_close    = None
        self._ema200        = None
        self._expiry        = None
        self._dte_days      = None
        self._instruments   = None

        # ── Pre-subscribed option tokens ──────────────────────────────────────
        self._pre_ce_token  : Optional[int] = None
        self._pre_pe_token  : Optional[int] = None
        self._pre_ce_sym    : Optional[str] = None
        self._pre_pe_sym    : Optional[str] = None

        # ── Mode detection state ──────────────────────────────────────────────
        self._mode               : Optional[str] = None   # "A" or "B"
        self._open_price         : Optional[float] = None
        self._or_high            : Optional[float] = None
        self._or_low             : Optional[float] = None
        self._mode_finalised     : bool = False
        self._spike_direction    : Optional[str] = None   # "CE" or "PE" from gap

        # ── Mode B consolidation tracking ─────────────────────────────────────
        # State machine: WATCHING → CONSOLIDATING → triggered (entry) / EXPIRED
        self._mb_state           : str = "WATCHING"   # "WATCHING", "CONSOL"
        self._mb_consol_high     : Optional[float] = None
        self._mb_consol_low      : Optional[float] = None
        self._mb_consol_bars     : int = 0

        # ── Trade state ───────────────────────────────────────────────────────
        self._trade              = None
        self._today_pnl          : float = 0.0
        self._trades_taken       : int = 0
        self._day_paused         : bool = False
        self._completed          : list = []

        self._lock = threading.Lock()

        mode_tag = "[LIVE]" if LIVE_MODE else "[PAPER]"
        log.info(
            f"[{self.name}] Initialized {mode_tag} | "
            f"qty={CFG['quantity']} SL={CFG['sl_pct']*100:.0f}% "
            f"trail1=+{(CFG['trail1_trigger_mult']-1)*100:.0f}% "
            f"trail2=peak−{CFG['trail2_distance_pct']*100:.0f}%"
        )

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        """
        Called from t.py with nifty_pm and nifty_instruments (INDEX_TOKEN=256265 branch).
        Stores live pm reference so _process_candle() always reads current PCR.
        Pre-subscribes ATM CE+PE so option prices are warm before 9:15.
        """
        self._instruments = instruments
        self._pm          = pm            # live reference — NOT a snapshot
        self._vix         = pm.vix
        self._prev_close  = pm.prev_close
        self._ema200      = pm.ema200_daily
        self._expiry      = pm.expiry_date
        self._dte_days    = pm.dte_days

        log.info(
            f"[{self.name}] Pre-market | VIX={self._vix} PCR={pm.pcr} "
            f"prev_close={self._prev_close} EMA200={self._ema200} "
            f"expiry={self._expiry} DTE={self._dte_days}"
        )

        # Pre-subscribe ATM CE + PE so option prices arrive before 9:15 AM
        ref = self._prev_close or (pm.prev_last5m_close if hasattr(pm, "prev_last5m_close") else None)
        if ref:
            strike = get_atm_strike(ref, step=NIFTY_STRIKE_STEP)
            ce_tok, ce_sym = instruments.get_option_token(strike, "CE", self._expiry)
            pe_tok, pe_sym = instruments.get_option_token(strike, "PE", self._expiry)

            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym

            if ce_tok:
                self.subscribe_option(ce_tok)
                log.info(f"[{self.name}] Pre-subscribed CE: {ce_sym} ({ce_tok})")
            if pe_tok:
                self.subscribe_option(pe_tok)
                log.info(f"[{self.name}] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.warning(
                f"[{self.name}] No prev_close — cannot pre-subscribe options. "
                f"Will subscribe on first 9:15 tick."
            )

        return True

    # ── on_tick  (every Nifty index tick) ────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        # Hard exit: must be checked BEFORE the market-hours guard below,
        # otherwise the guard returns early and this block is never reached.
        if self._trade and self._trade["state"] == "OPEN" and t >= CFG["hard_exit_time"]:
            opt_ltp = self.get_price(self._trade["token"]) or self._trade["entry"]
            self._do_exit(opt_ltp, "HARD_EXIT_EOD", ts)
            return

        # Ignore outside trading hours
        if t < CFG["market_open"] or t >= CFG["hard_exit_time"]:
            return

        # Update internal Nifty VWAP (proxy_weight=1 — price-only, no volume)
        self._vwap.update(price, price, price, volume=0, proxy_weight=1)

        # Capture true open price at first 9:15 tick (for gap calculation)
        if self._open_price is None and t >= CFG["market_open"]:
            self._open_price = price
            log.info(f"[{self.name}] Open price captured: {price:.2f}")

            # Late-subscribe ATM options if pre-market subscription failed
            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_now(price)

        # Track OR range for the first 15 min (9:15–9:30)
        if CFG["market_open"] <= t < dtime(9, 30):
            if self._or_high is None or price > self._or_high:
                self._or_high = price
            if self._or_low is None or price < self._or_low:
                self._or_low = price

        # Build 5-min candle from live ticks
        closed = self._candle_builder.feed_tick(price, 1, ts)
        if closed is not None:
            self._process_candle(closed, ts)

    # ── on_candle  (backfill from hub.backfill) ───────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        """
        During backfill, MarketHub replays historical Nifty candles here.
        In live trading, on_tick() builds candles internally via CandleBuilder.
        Both paths funnel into _process_candle() — identical indicator warmup.
        """
        self._process_candle(candle, ts)

    # ── on_option_tick  (live premium management) ────────────────────────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        """
        Runs on every WebSocket option tick.
        Handles: SL check, Trail-1 (BE move), Trail-2 (peak trail).
        """
        if not (self._trade and token == self._trade.get("token")):
            return
        if self._trade["state"] != "OPEN":
            return
        if price <= 0:
            return

        # SL grace period — suppress check for N seconds after entry fill
        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from and ts < sl_active_from:
            return

        entry  = self._trade["entry"]
        sl     = self._trade["sl"]
        peak   = self._trade.get("peak", entry)
        trail1 = self._trade.get("trail1_active", False)

        # Update peak
        if price > peak:
            self._trade["peak"] = price
            peak = price

        # Trail 1: when premium ≥ fill × 1.50 → move SL to breakeven
        if not trail1 and price >= entry * CFG["trail1_trigger_mult"]:
            new_sl = entry   # breakeven
            if new_sl > sl:
                self._trade["sl"] = new_sl
                self._trade["trail1_active"] = True
                log.info(
                    f"[{self.name}] Trail-1 activated: SL → BE={new_sl:.2f} "
                    f"(premium={price:.2f} ≥ {entry * CFG['trail1_trigger_mult']:.2f})"
                )
                sl = new_sl

        # Trail 2: SL = peak × (1 − TRAIL_PCT) — only after trail1 active
        if trail1:
            trail2_sl = peak * (1 - CFG["trail2_distance_pct"])
            if trail2_sl > sl:
                self._trade["sl"] = trail2_sl
                log.debug(
                    f"[{self.name}] Trail-2: SL updated {sl:.2f} → {trail2_sl:.2f} "
                    f"(peak={peak:.2f})"
                )
                sl = trail2_sl

        # SL check
        if price <= sl:
            self._do_exit(price, "SL_HIT", ts)

    # ── Core candle processor ─────────────────────────────────────────────────

    def _process_candle(self, candle: dict, ts: datetime):
        """
        Called for every 5-min candle (backfill or live).
        1. Append to indicator buffer.
        2. Finalise mode detection once after 9:20 (first candle closes after OR).
        3. Run Mode A or Mode B entry check when in the right time window.
        4. Check Mode B consolidation state machine.
        """
        self._buf.append(candle)

        t = ts.time()

        # ── Finalise mode after first candle at/after 9:20 ────────────────────
        if not self._mode_finalised and t >= dtime(9, 20):
            self._finalise_mode(ts)

        # Backfill guard — skip entry logic for stale historical candles
        stale = (_now_ist() - ts).total_seconds() > 600
        if stale:
            return

        if self._day_paused:
            return

        if self._trades_taken >= CFG["max_trades_per_day"]:
            return

        if self._trade and self._trade["state"] == "OPEN":
            return

        if t >= CFG["entry_cutoff"]:
            return

        if not self._mode:
            return

        if self._mode == "A":
            self._check_mode_a_entry(candle, ts)
        elif self._mode == "B":
            self._check_mode_b_entry(candle, ts)

    # ── Mode detection ────────────────────────────────────────────────────────

    def _finalise_mode(self, ts: datetime):
        """
        Classify today as Mode A (normal) or Mode B (momentum/gap/expiry).
        Called once after the first 5-min candle closes at/after 9:20 AM.
        """
        prev = self._prev_close
        open_p = self._open_price

        gap_pct = abs(open_p - prev) / prev if (prev and open_p) else 0.0

        # Is today expiry?
        today = _now_ist().date()
        is_expiry = (self._expiry == today)

        # OR range from 9:15–9:30 ticks
        or_range = (self._or_high - self._or_low) if (
            self._or_high is not None and self._or_low is not None
        ) else 0.0

        mode_b = (
            gap_pct > CFG["mode_b_gap_pct"] or
            or_range > CFG["mode_b_range_trigger"] or
            (is_expiry and gap_pct > CFG["mode_b_gap_pct_expiry"])
        )

        self._mode = "B" if mode_b else "A"
        self._mode_finalised = True

        # Determine spike direction for Mode B
        if mode_b and prev and open_p:
            self._spike_direction = "CE" if open_p > prev else "PE"

        log.info(
            f"[{self.name}] Mode finalised: {self._mode} | "
            f"gap={gap_pct:.2%} OR_range={or_range:.0f}pts "
            f"is_expiry={is_expiry} spike_dir={self._spike_direction}"
        )

    # ── Mode A entry check ────────────────────────────────────────────────────

    def _check_mode_a_entry(self, candle: dict, ts: datetime):
        """
        Mode A: EMA stack + VWAP + RSI + pullback to 9 EMA.
        Two time windows: 9:30–11:00 AM and 1:15–2:15 PM.
        """
        t = ts.time()
        in_window = (
            (CFG["mode_a_start"] <= t < CFG["mode_a_window1_end"]) or
            (CFG["mode_a_window2_start"] <= t < CFG["mode_a_window2_end"])
        )
        if not in_window:
            return

        if len(self._buf) < CFG["ema_slow"] + 5:
            return

        ind = self._compute_indicators()
        if not ind:
            return

        direction = self._mode_a_signal(ind, candle)
        if not direction:
            return

        # PCR gate — live reference via self._pm
        live_pcr = self._pm.pcr if self._pm else None
        if live_pcr:
            if direction == "CE" and live_pcr < CFG["pcr_min_ce"]:
                log.info(
                    f"[{self.name}] [ModeA] CE skipped — PCR={live_pcr:.2f} "
                    f"< {CFG['pcr_min_ce']}"
                )
                return
            if direction == "PE" and live_pcr > CFG["pcr_max_pe"]:
                log.info(
                    f"[{self.name}] [ModeA] PE skipped — PCR={live_pcr:.2f} "
                    f"> {CFG['pcr_max_pe']}"
                )
                return

        log.info(
            f"[{self.name}] [ModeA] Signal: {direction} at {ts.strftime('%H:%M')} | "
            f"EMA9={ind['ema9']:.0f} EMA20={ind['ema20']:.0f} EMA50={ind['ema50']:.0f} "
            f"RSI={ind['rsi']:.1f} VWAP={ind['vwap'] or 'N/A'} "
            f"close={candle['close']:.2f}"
        )

        self._enter(direction, candle["close"], ts, reason="mode_a_pullback")

    def _mode_a_signal(self, ind: dict, candle: dict) -> Optional[str]:
        """
        Returns "CE", "PE", or None.
        All 4 Mode A conditions must pass simultaneously.
        """
        e9, e20, e50 = ind["ema9"], ind["ema20"], ind["ema50"]
        rsi  = ind["rsi"]
        vwap = ind["vwap"]
        close = candle["close"]
        low   = candle["low"]
        high  = candle["high"]

        # ── CE path ───────────────────────────────────────────────────────────
        ema_stack_ce  = e9 > e20 > e50
        rsi_ok_ce     = CFG["rsi_ce_min"] <= rsi <= CFG["rsi_ce_max"]
        vwap_ok_ce    = (vwap is None) or (close > vwap)
        # Pullback: last 2 candle lows touched or crossed below EMA9,
        # current close resumed above EMA9
        prev_two = list(self._buf)[-3:-1] if len(self._buf) >= 3 else []
        pullback_ce = (
            any(c["low"] <= e9 for c in prev_two) and
            close > e9
        )

        if ema_stack_ce and rsi_ok_ce and vwap_ok_ce and pullback_ce:
            return "CE"

        # ── PE path ───────────────────────────────────────────────────────────
        ema_stack_pe  = e9 < e20 < e50
        rsi_ok_pe     = CFG["rsi_pe_min"] <= rsi <= CFG["rsi_pe_max"]
        vwap_ok_pe    = (vwap is None) or (close < vwap)
        pullback_pe = (
            any(c["high"] >= e9 for c in prev_two) and
            close < e9
        )

        if ema_stack_pe and rsi_ok_pe and vwap_ok_pe and pullback_pe:
            return "PE"

        return None

    # ── Mode B entry check ────────────────────────────────────────────────────

    def _check_mode_b_entry(self, candle: dict, ts: datetime):
        """
        Mode B: wait for a tight consolidation after the initial spike, then
        enter the breakout of that consolidation.
        Window: 9:15–11:00 AM.
        """
        t = ts.time()
        if not (CFG["mode_b_start"] <= t < CFG["mode_b_cutoff"]):
            return

        # IV guard — skip Mode B if VIX too high
        if self._vix and self._vix > CFG["vix_skip_thresh"]:
            log.info(
                f"[{self.name}] [ModeB] Entry skipped — VIX={self._vix:.1f} "
                f"> {CFG['vix_skip_thresh']}"
            )
            return

        bar_range = candle["high"] - candle["low"]
        direction = self._spike_direction  # already set in _finalise_mode

        if direction is None:
            return

        # ── Consolidation state machine ───────────────────────────────────────
        if self._mb_state == "WATCHING":
            # Transition to CONSOLIDATING when a candle is sufficiently tight
            if bar_range < CFG["mode_b_consol_range"]:
                self._mb_state       = "CONSOL"
                self._mb_consol_high = candle["high"]
                self._mb_consol_low  = candle["low"]
                self._mb_consol_bars = 1
                log.info(
                    f"[{self.name}] [ModeB] Consolidation started at {ts.strftime('%H:%M')} | "
                    f"range={bar_range:.0f}pts "
                    f"H={candle['high']:.0f} L={candle['low']:.0f}"
                )
            return

        if self._mb_state == "CONSOL":
            # Expand consolidation zone if bar is still tight
            if bar_range < CFG["mode_b_consol_range"]:
                self._mb_consol_high = max(self._mb_consol_high, candle["high"])
                self._mb_consol_low  = min(self._mb_consol_low, candle["low"])
                self._mb_consol_bars += 1
                log.info(
                    f"[{self.name}] [ModeB] Consolidation bar {self._mb_consol_bars} | "
                    f"zone=[{self._mb_consol_low:.0f}–{self._mb_consol_high:.0f}] "
                    f"range={self._mb_consol_high - self._mb_consol_low:.0f}pts"
                )

                # Check for breakout from consolidation (need min bars first)
                if self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    if direction == "CE" and candle["close"] > self._mb_consol_high:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout CE at {ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} > zone_high={self._mb_consol_high:.0f}"
                        )
                        self._mb_state = "WATCHING"   # reset for potential next setup
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_consol_breakout_ce")
                        return

                    if direction == "PE" and candle["close"] < self._mb_consol_low:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout PE at {ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} < zone_low={self._mb_consol_low:.0f}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_consol_breakout_pe")
                        return

                # Tight bars exceeded max — zone too wide, reset and re-seek
                if self._mb_consol_bars >= CFG["mode_b_consol_max_bars"]:
                    log.info(
                        f"[{self.name}] [ModeB] Consolidation expired (tight) after "
                        f"{self._mb_consol_bars} bars — resetting"
                    )
                    self._mb_state = "WATCHING"

            # Bar is too wide — consolidation expires, start watching again
            elif self._mb_consol_bars >= CFG["mode_b_consol_max_bars"]:
                log.info(
                    f"[{self.name}] [ModeB] Consolidation expired after "
                    f"{self._mb_consol_bars} bars — resetting"
                )
                self._mb_state = "WATCHING"
            else:
                # Wide bar mid-consolidation — breakout check before expiry
                if self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    if direction == "CE" and candle["close"] > self._mb_consol_high:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout CE (wide bar) at "
                            f"{ts.strftime('%H:%M')}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_breakout_wide_ce")
                        return
                    if direction == "PE" and candle["close"] < self._mb_consol_low:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout PE (wide bar) at "
                            f"{ts.strftime('%H:%M')}"
                        )
                        self._mb_state = "WATCHING"
                        self._enter(direction, candle["close"], ts,
                                    reason="mode_b_breakout_wide_pe")
                        return
                # Wide bar, not enough bars, not a breakout — reset
                self._mb_state = "WATCHING"

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _enter(self, direction: str, index_price: float, ts: datetime, reason: str):
        """
        Resolve ATM option token, place BUY, record trade.
        SL is set from confirmed fill price (not pre-order LTP).
        """
        with self._lock:
            if self._trade and self._trade["state"] == "OPEN":
                return
            if self._trades_taken >= CFG["max_trades_per_day"]:
                return
            if self._day_paused:
                return

        strike = get_atm_strike(index_price, step=NIFTY_STRIKE_STEP)

        # Use pre-subscribed token if strike matches, else look up fresh
        if direction == "CE" and self._pre_ce_token:
            # Verify strike still matches (may drift if market moved a lot)
            pre_strike = self._get_strike_from_sym(self._pre_ce_sym)
            if pre_strike == strike:
                token, sym = self._pre_ce_token, self._pre_ce_sym
            else:
                token, sym = self._instruments.get_option_token(
                    strike, "CE", self._expiry)
        elif direction == "PE" and self._pre_pe_token:
            pre_strike = self._get_strike_from_sym(self._pre_pe_sym)
            if pre_strike == strike:
                token, sym = self._pre_pe_token, self._pre_pe_sym
            else:
                token, sym = self._instruments.get_option_token(
                    strike, "PE", self._expiry)
        else:
            token, sym = self._instruments.get_option_token(
                strike, direction, self._expiry)

        if not token or not sym:
            log.error(
                f"[{self.name}] No option token | direction={direction} "
                f"strike={strike} expiry={self._expiry}"
            )
            return

        # Subscribe if not already
        self.subscribe_option(token)

        # Get current option LTP
        opt_ltp = self.get_price(token)
        if not opt_ltp or opt_ltp <= 0:
            log.warning(
                f"[{self.name}] Option LTP unavailable for {sym} — "
                f"skipping entry (no stale-price risk)"
            )
            return

        # Acquire live trade slot
        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy active")
            return

        # Place BUY order
        result = self._place_buy(sym, token, CFG["quantity"], opt_ltp)
        if result is None:
            self._release_slot()
            log.error(f"[{self.name}] BUY FAILED for {sym}")
            return

        order_id, fill_price = result

        sl = fill_price * (1 - CFG["sl_pct"])
        sl_active_from = ts + timedelta(seconds=CFG["sl_grace_seconds"])
        mode_tag = "LIVE" if LIVE_MODE else "PAPER"

        with self._lock:
            self._trade = {
                "state"          : "OPEN",
                "direction"      : direction,
                "symbol"         : sym,
                "token"          : token,
                "entry"          : fill_price,
                "sl"             : sl,
                "peak"           : fill_price,
                "trail1_active"  : False,
                "entry_time"     : ts,
                "sl_active_from" : sl_active_from,
                "reason"         : reason,
                "mode"           : self._mode,
                "index_price"    : index_price,
                "strike"         : strike,
                "order_id"       : order_id,
                "qty"            : CFG["quantity"],
            }
            self._trades_taken += 1

        log.info(
            f"[{self.name}] [{mode_tag}] ENTRY #{self._trades_taken} | "
            f"{direction} {sym} @ {fill_price:.2f} | "
            f"SL={sl:.2f} (−{CFG['sl_pct']*100:.0f}%) | "
            f"mode={self._mode} reason={reason} | order_id={order_id}"
        )

        self._log_csv({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"      : sym,
            "direction"   : direction,
            "action"      : "ENTRY",
            "price"       : fill_price,
            "sl"          : round(sl, 2),
            "status"      : "OPEN",
            "pnl"         : 0,
            "reason"      : reason,
            "mode"        : self._mode,
            "vix"         : self._vix,
            "pcr"         : self._pm.pcr if self._pm else None,
            "order_id"    : order_id,
        })

    # ── Exit ──────────────────────────────────────────────────────────────────

    def _do_exit(self, exit_opt_price: float, reason: str, ts: datetime):
        t = self._trade
        if not t or t["state"] != "OPEN":
            return

        with self._lock:
            t["state"] = "CLOSED"

        result = self._place_sell_with_retry(
            t["symbol"], t["token"], t["qty"], exit_opt_price, max_retries=3
        )
        if result is None:
            if LIVE_MODE:
                log.error(
                    f"[{self.name}] SELL FAILED for {t['symbol']} — "
                    f"may still be open in Zerodha! SQUARE OFF MANUALLY."
                )
            sell_price = exit_opt_price
            order_id   = None
        else:
            order_id, sell_price = result

        self._release_slot()

        pnl = (sell_price - t["entry"]) * t["qty"]
        self._today_pnl += pnl

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[{self.name}] [{mode_tag}] EXIT #{self._trades_taken} | "
            f"[{reason}] {t['symbol']} @ {sell_price:.2f} | "
            f"entry={t['entry']:.2f} PnL={pnl:.0f} "
            f"({pnl / t['qty']:.1f}/unit) | Today={self._today_pnl:.0f}"
        )

        self._log_csv({
            "timestamp"   : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol"      : t["symbol"],
            "direction"   : t["direction"],
            "action"      : "EXIT",
            "price"       : sell_price,
            "sl"          : round(t["sl"], 2),
            "status"      : "CLOSED",
            "pnl"         : round(pnl, 2),
            "reason"      : reason,
            "mode"        : t["mode"],
            "vix"         : self._vix,
            "pcr"         : self._pm.pcr if self._pm else None,
            "order_id"    : order_id,
        })
        self._completed.append({**t, "exit_price": sell_price, "exit_reason": reason, "pnl": pnl})

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_indicators(self) -> Optional[dict]:
        """
        Compute EMA9/20/50, RSI14, and current VWAP from the 5-min candle buffer.
        Returns None if not enough bars.
        """
        if len(self._buf) < CFG["ema_slow"] + 5:
            return None

        candles = list(self._buf)
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = df.get("close", 0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        close = df["close"]

        ema9  = float(close.ewm(span=CFG["ema_fast"], adjust=False).mean().iloc[-1])
        ema20 = float(close.ewm(span=CFG["ema_mid"],  adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=CFG["ema_slow"], adjust=False).mean().iloc[-1])

        # RSI(14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(CFG["rsi_period"], min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(CFG["rsi_period"], min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = float((100 - 100 / (1 + rs)).fillna(50).iloc[-1])

        vwap = self._vwap.value

        return {
            "ema9" : ema9,
            "ema20": ema20,
            "ema50": ema50,
            "rsi"  : rsi,
            "vwap" : vwap,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _subscribe_atm_now(self, price: float):
        """Subscribe ATM CE+PE when pre-market subscription was skipped."""
        from core.instruments import get_atm_strike
        strike = get_atm_strike(price, step=NIFTY_STRIKE_STEP)
        log.info(
            f"[{self.name}] Late-subscribing ATM options | "
            f"strike={strike} expiry={self._expiry} spot={price:.2f}"
        )
        ce_tok, ce_sym = self._instruments.get_option_token(strike, "CE", self._expiry)
        pe_tok, pe_sym = self._instruments.get_option_token(strike, "PE", self._expiry)

        if ce_tok and self._pre_ce_token is None:
            self.subscribe_option(ce_tok)
            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            log.info(f"[{self.name}] Late-subscribed CE: {ce_sym} ({ce_tok})")
        if pe_tok and self._pre_pe_token is None:
            self.subscribe_option(pe_tok)
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym
            log.info(f"[{self.name}] Late-subscribed PE: {pe_sym} ({pe_tok})")

    @staticmethod
    def _get_strike_from_sym(sym: Optional[str]) -> Optional[int]:
        """
        Extract strike from option symbol like 'NIFTY2561824000CE'.
        Returns None if parsing fails.
        """
        if not sym:
            return None
        try:
            # Symbol format: NIFTY + YYMMDD + STRIKE + CE/PE
            # e.g. NIFTY2561824000CE → strip NIFTY (5) + date (6) = 11 chars, then strip last 2
            core = sym[11:-2]
            return int(float(core))
        except Exception:
            return None

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = [
            "timestamp", "symbol", "direction", "action", "price",
            "sl", "status", "pnl", "reason", "mode", "vix", "pcr", "order_id",
        ]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        # ── Safety net: force-close any position still open at EOD ───────────
        # Normally HARD_EXIT_EOD fires at 15:00 via on_tick(). This handles
        # the edge case where no index tick arrives at exactly 15:00 (e.g.
        # WebSocket lag, last tick at 14:59:59).
        if self._trade and self._trade["state"] == "OPEN":
            token   = self._trade.get("token")
            opt_ltp = (self.get_price(token) if token else None) or self._trade["entry"]
            log.warning(
                f"[{self.name}] EOD: trade still OPEN — force-closing at ltp={opt_ltp:.2f}"
            )
            self._do_exit(opt_ltp, "EOD_FORCE_CLOSE", _now_ist())

        log.info(f"\n[{self.name}] {'='*55}")
        log.info(f"[{self.name}] END OF DAY | mode={'LIVE' if LIVE_MODE else 'PAPER'}")
        log.info(f"[{self.name}] Market mode    : {self._mode or 'not set'}")
        log.info(f"[{self.name}] Trades taken   : {self._trades_taken}")
        log.info(f"[{self.name}] VIX            : {self._vix}")
        log.info(f"[{self.name}] PCR (EOD)      : {self._pm.pcr if self._pm else 'N/A'}")
        for i, t in enumerate(self._completed, 1):
            log.info(
                f"[{self.name}]   #{i} {t['direction']} {t['symbol']} "
                f"[{t['exit_reason']}] "
                f"entry={t['entry']:.2f} exit={t['exit_price']:.2f} "
                f"PnL={t['pnl']:.0f} ({t['pnl'] / t['qty']:.1f}/unit)"
            )
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'='*55}\n")

