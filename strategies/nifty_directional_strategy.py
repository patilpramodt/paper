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
    3. RSI(14)    : 50–72 (CE) or 28–50 (PE)   — widened from 55-68/32-45
    4. Pullback   : at least one of the last 3 candle lows ≤ EMA9*1.001 (CE)
                    or one of the last 3 candle highs ≥ EMA9*0.999 (PE)
                    AND current candle resumed direction (close > EMA9 for CE)
    5. PCR gate   : if available, CE needs PCR ≥ PCR_MIN (0.70),
                                  PE needs PCR ≤ PCR_MAX (1.30)
  On valid signal: enter ATM call/put, current weekly expiry.

MODE B — MOMENTUM / EXPIRY / GAP-AND-GO DAY
  Entry window  : 9:15–11:30 AM  (extended from 11:00)
  Hard cutoff   : no new entries after 11:30 AM in Mode B
  Logic:
    After initial spike direction is detected (gap or strong first bar),
    wait for a micro-consolidation: 2–6 consecutive 5-min candles whose
    H-L range stays below MODE_B_CONSOL_RANGE (80) pts.
    Entry trigger: NEXT candle (after consolidation is complete) closes
                   above consolidation high (CE) or below consolidation
                   low (PE). Breakout is checked on the bar AFTER the
                   consolidation window is formed, not during it.
    IV guard: if VIX > VIX_SKIP_THRESH (18), skip Mode B entries
              (IV is too expensive at spike peak)
  On valid signal: enter ATM call/put.

STRIKE SELECTION
  Default: ATM strike (rounded to nearest 50-pt multiple).

TRADE MANAGEMENT  (option premium based, tick-by-tick via on_option_tick)
  SL      : fill_price × (1 − SL_PCT)  i.e. 30 % of premium paid
  Trail 1 : when premium ≥ fill × 1.50  → move SL to breakeven (fill_price)
  Trail 2 : track peak premium; SL = peak × (1 − TRAIL_PCT)  i.e. 20 % below peak
  Hard exit: 3:00 PM regardless of position
  Max trades: 2 per day

CHANGES vs original
  Fix 1 : Mode B breakout checked on next bar after consolidation, not during
  Fix 3 : Per-candle DEBUG logging of each Mode A condition result
  Fix 4 : RSI range widened to 50-72 (CE) and 28-50 (PE)
  Fix 5 : PCR CE gate lowered to 0.70, PE gate raised to 1.30
  Fix 6 : Mode B cutoff extended from 11:00 to 11:30 AM
  Fix 7 : Mode B max consolidation bars raised from 4 to 6
  Fix 8 : Pullback check relaxed: 3-bar lookback + EMA9 ±0.1% tolerance
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
    "mode_b_cutoff"           : dtime(11, 30),   # FIX 6: extended from 11:00
    "entry_cutoff"            : dtime(14, 30),   # hard: no new entries after this
    "hard_exit_time"          : dtime(15, 0),

    # ── Mode detection thresholds ─────────────────────────────────────────────
    "mode_b_gap_pct"          : 0.004,    # > 0.4 % gap triggers Mode B
    "mode_b_gap_pct_expiry"   : 0.002,    # > 0.2 % gap on expiry day
    "mode_b_range_trigger"    : 80,       # first-15-min OR range > 80 pts → Mode B
    "mode_b_consol_range"     : 80,       # consolidation must be tighter than this
    "mode_b_consol_min_bars"  : 2,        # min bars to form a valid consolidation
    "mode_b_consol_max_bars"  : 6,        # FIX 7: raised from 4 — more time to form

    # ── Indicator settings ────────────────────────────────────────────────────
    "ema_fast"                : 9,
    "ema_mid"                 : 20,
    "ema_slow"                : 50,
    "rsi_period"              : 14,
    "buf_size"                : 100,      # rolling 5-min candle buffer

    # ── Mode A signal filters ─────────────────────────────────────────────────
    # FIX 4: widened RSI ranges (was 55-68 / 32-45)
    "rsi_ce_min"              : 50,       # RSI must be in this range for CE
    "rsi_ce_max"              : 72,
    "rsi_pe_min"              : 28,
    "rsi_pe_max"              : 50,
    # FIX 5: relaxed PCR gate (was 0.80 / 1.20)
    "pcr_min_ce"              : 0.70,     # PCR must be ≥ this for CE
    "pcr_max_pe"              : 1.30,     # PCR must be ≤ this for PE

    # ── Mode A pullback tolerance ─────────────────────────────────────────────
    # FIX 8: 0.1% EMA tolerance so near-touches count
    "pullback_ema_tolerance"  : 0.001,    # low <= EMA9 * (1 + tol) counts as pullback

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

        self._candle_builder = CandleBuilder(minutes=5)
        self._buf: deque = deque(maxlen=CFG["buf_size"])
        self._vwap = SessionVWAP()

        # ── Pre-market data ───────────────────────────────────────────────────
        self._pm            = None
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
        self._mode               : Optional[str] = None
        self._open_price         : Optional[float] = None
        self._or_high            : Optional[float] = None
        self._or_low             : Optional[float] = None
        self._mode_finalised     : bool = False
        self._spike_direction    : Optional[str] = None

        # ── Mode B consolidation tracking ─────────────────────────────────────
        # FIX 1: added _mb_consol_ready flag.
        # When True, the consolidation is complete and the NEXT candle is the
        # breakout candidate. This prevents checking close > zone_high on the
        # same bar that just expanded the zone.
        self._mb_state           : str = "WATCHING"
        self._mb_consol_high     : Optional[float] = None
        self._mb_consol_low      : Optional[float] = None
        self._mb_consol_bars     : int = 0
        self._mb_consol_ready    : bool = False   # FIX 1: breakout arm flag

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
        self._instruments = instruments
        self._pm          = pm
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

        # Bug 14 fix: seed _buf with the prior session's 5-min candles so
        # EMA9/20/50 + RSI are already warm at market open instead of
        # needing min_bars=ema_slow+2 (~260 live minutes from 9:30) to
        # fill up from empty every single day.
        self._seed_historical_buffer()

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

    # ── Historical candle seeding (Bug 14 fix, prior-session warmup) ──────────

    def _seed_historical_buffer(self):
        """
        Fetch the prior trading session's last N 5-min Nifty candles via
        hub.kite.historical_data() and load them into _buf, so EMA9/20/50
        and RSI(14) already have enough bars to be valid the moment the
        market opens — instead of needing min_bars=ema_slow+2 (52 bars,
        ~260 live minutes from 9:30) to accumulate from empty every day.

        This is separate from hub.backfill(), which only replays TODAY's
        already-elapsed candles for late starts (no-op for an on-time
        9:00 AM start, since there are no elapsed candles yet). This seeds
        yesterday's close-of-session data so today's first live candle
        isn't starting cold.

        There is a genuine overnight gap between the last seeded bar and
        the first live 9:15 bar — EMA/RSI self-correct within a few live
        bars, an accepted tradeoff (mirrors the BB_STOCH_NIFTY historical
        seeding fix already used elsewhere).

        Fails safe: if kite is unavailable, the fetch errors, or no data
        comes back, this silently falls back to the old live-only warmup
        path (no regression — the strategy just trades a bit later that
        day, exactly as it did before this fix).
        """
        kite = getattr(self._hub, "kite", None)
        if kite is None:
            log.warning(
                f"[{self.name}] No kite handle on hub — cannot seed "
                f"historical candles, falling back to live-only warmup "
                f"(~{(CFG['ema_slow'] + 2) * 5}min)"
            )
            return

        needed = self._buf.maxlen or (CFG["ema_slow"] + 10)
        today  = _now_ist().date()

        try:
            raw = kite.historical_data(
                instrument_token = self.INDEX_TOKEN,
                from_date        = datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                to_date          = datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                interval         = "5minute",
            )
        except Exception as e:
            log.warning(
                f"[{self.name}] Historical seed fetch failed: {e} — "
                f"falling back to live-only warmup "
                f"(~{(CFG['ema_slow'] + 2) * 5}min)"
            )
            return

        if not raw:
            log.warning(
                f"[{self.name}] Historical seed returned no candles — "
                f"falling back to live-only warmup "
                f"(~{(CFG['ema_slow'] + 2) * 5}min)"
            )
            return

        tail = raw[-needed:]
        with self._lock:
            self._buf.clear()
            for bar in tail:
                try:
                    self._buf.append({
                        "ts"     : bar["date"],
                        "open"   : float(bar["open"]),
                        "high"   : float(bar["high"]),
                        "low"    : float(bar["low"]),
                        "close"  : float(bar["close"]),
                        "volume" : float(bar.get("volume", 0)),
                    })
                except (KeyError, TypeError, ValueError) as e:
                    log.warning(f"[{self.name}] Skipping malformed seed bar {bar}: {e}")

        if self._buf:
            log.info(
                f"[{self.name}] Seeded {len(self._buf)} prior-session 5-min "
                f"candles ({tail[0].get('date', '?')} -> {tail[-1].get('date', '?')}) "
                f"— EMA9/20/50 + RSI warm from first live tick, "
                f"no {(CFG['ema_slow'] + 2) * 5}min wait"
            )
        else:
            log.warning(
                f"[{self.name}] Seed buffer empty after load — "
                f"falling back to live-only warmup"
            )

    # ── on_tick ───────────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if self._trade and self._trade["state"] == "OPEN" and t >= CFG["hard_exit_time"]:
            opt_ltp = self.get_price(self._trade["token"]) or self._trade["entry"]
            self._do_exit(opt_ltp, "HARD_EXIT_EOD", ts)
            return

        if t < CFG["market_open"] or t >= CFG["hard_exit_time"]:
            return

        self._vwap.update(price, price, price, volume=0, proxy_weight=1)

        if self._open_price is None and t >= CFG["market_open"]:
            self._open_price = price
            log.info(f"[{self.name}] Open price captured: {price:.2f}")

            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_now(price)

        if CFG["market_open"] <= t < dtime(9, 30):
            if self._or_high is None or price > self._or_high:
                self._or_high = price
            if self._or_low is None or price < self._or_low:
                self._or_low = price

        closed = self._candle_builder.feed_tick(price, 1, ts)
        if closed is not None:
            self._process_candle(closed, ts)

    # ── on_candle (backfill) ──────────────────────────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        self._process_candle(candle, ts)

    # ── on_option_tick (live premium SL/trail) ────────────────────────────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        if not (self._trade and token == self._trade.get("token")):
            return
        if self._trade["state"] != "OPEN":
            return
        if price <= 0:
            return

        sl_active_from = self._trade.get("sl_active_from")
        if sl_active_from and ts < sl_active_from:
            return

        entry  = self._trade["entry"]
        sl     = self._trade["sl"]
        peak   = self._trade.get("peak", entry)
        trail1 = self._trade.get("trail1_active", False)

        if price > peak:
            self._trade["peak"] = price
            peak = price

        if not trail1 and price >= entry * CFG["trail1_trigger_mult"]:
            new_sl = entry
            if new_sl > sl:
                self._trade["sl"] = new_sl
                self._trade["trail1_active"] = True
                log.info(
                    f"[{self.name}] Trail-1 activated: SL → BE={new_sl:.2f} "
                    f"(premium={price:.2f} ≥ {entry * CFG['trail1_trigger_mult']:.2f})"
                )
                sl = new_sl

        if trail1:
            trail2_sl = peak * (1 - CFG["trail2_distance_pct"])
            if trail2_sl > sl:
                self._trade["sl"] = trail2_sl
                log.debug(
                    f"[{self.name}] Trail-2: SL updated {sl:.2f} → {trail2_sl:.2f} "
                    f"(peak={peak:.2f})"
                )
                sl = trail2_sl

        if price <= sl:
            self._do_exit(price, "SL_HIT", ts)

    # ── Core candle processor ─────────────────────────────────────────────────

    def _process_candle(self, candle: dict, ts: datetime):
        self._buf.append(candle)

        t = ts.time()

        if not self._mode_finalised and t >= dtime(9, 20):
            self._finalise_mode(ts)

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
        prev   = self._prev_close
        open_p = self._open_price

        gap_pct = abs(open_p - prev) / prev if (prev and open_p) else 0.0

        today     = _now_ist().date()
        is_expiry = (self._expiry == today)

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

        if mode_b and prev and open_p:
            self._spike_direction = "CE" if open_p > prev else "PE"

        log.info(
            f"[{self.name}] Mode finalised: {self._mode} | "
            f"gap={gap_pct:.2%} OR_range={or_range:.0f}pts "
            f"is_expiry={is_expiry} spike_dir={self._spike_direction}"
        )

    # ── Mode A entry check ────────────────────────────────────────────────────

    def _check_mode_a_entry(self, candle: dict, ts: datetime):
        t = ts.time()
        in_window = (
            (CFG["mode_a_start"] <= t < CFG["mode_a_window1_end"]) or
            (CFG["mode_a_window2_start"] <= t < CFG["mode_a_window2_end"])
        )
        if not in_window:
            return

        # FIX 3 + buffer guard with logging
        min_bars = CFG["ema_slow"] + 2   # lowered from +5 to +2 (FIX 8 adjacent)
        if len(self._buf) < min_bars:
            log.debug(
                f"[{self.name}] [ModeA] {ts.strftime('%H:%M')} buffer not warm: "
                f"{len(self._buf)}/{min_bars} bars"
            )
            return

        ind = self._compute_indicators()
        if not ind:
            return

        direction = self._mode_a_signal(ind, candle, ts)  # FIX 3: pass ts for logging
        if not direction:
            return

        # PCR gate — live reference
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

    def _mode_a_signal(self, ind: dict, candle: dict, ts: datetime) -> Optional[str]:
        """
        Returns "CE", "PE", or None.
        All 4 Mode A conditions must pass simultaneously.
        FIX 3: logs each condition result at DEBUG level for diagnosability.
        FIX 4: RSI range widened.
        FIX 8: pullback uses 3-bar lookback and EMA9 ±0.1% tolerance.
        """
        e9, e20, e50 = ind["ema9"], ind["ema20"], ind["ema50"]
        rsi  = ind["rsi"]
        vwap = ind["vwap"]
        close = candle["close"]

        # FIX 8: 3-bar lookback (was 2) + tolerance band
        tol = CFG["pullback_ema_tolerance"]
        prev_three = list(self._buf)[-4:-1] if len(self._buf) >= 4 else list(self._buf)[:-1]

        # ── CE path ───────────────────────────────────────────────────────────
        ema_stack_ce  = e9 > e20 > e50
        rsi_ok_ce     = CFG["rsi_ce_min"] <= rsi <= CFG["rsi_ce_max"]
        vwap_ok_ce    = (vwap is None) or (close > vwap)
        # FIX 8: low <= EMA9 * (1 + tol) counts — catches near-touches
        pullback_ce   = (
            any(c["low"] <= e9 * (1 + tol) for c in prev_three) and
            close > e9
        )

        # FIX 3: per-condition DEBUG log
        # BUG FIX: ternary inside a format spec ({vwap:.0f if vwap else 'N/A'})
        # is invalid syntax — Python treats everything after ':' in an f-string
        # as a literal format spec, not an expression. This raised ValueError
        # on every call, silently killing on_tick() for every Mode A candle
        # (caught by MarketHub's per-tick try/except and logged only to
        # core.log, invisible in this strategy's own log file). Pre-format
        # the value as a plain string before interpolating instead.
        vwap_disp = f"{vwap:.0f}" if vwap else "N/A"
        log.debug(
            f"[{self.name}] [ModeA] {ts.strftime('%H:%M')} CE check | "
            f"EMA_stack={'✓' if ema_stack_ce else '✗'}({e9:.0f}>{e20:.0f}>{e50:.0f}) "
            f"RSI={'✓' if rsi_ok_ce else '✗'}({rsi:.1f} in [{CFG['rsi_ce_min']}-{CFG['rsi_ce_max']}]) "
            f"VWAP={'✓' if vwap_ok_ce else '✗'}(close={close:.0f} vwap={vwap_disp}) "
            f"Pullback={'✓' if pullback_ce else '✗'}"
        )

        if ema_stack_ce and rsi_ok_ce and vwap_ok_ce and pullback_ce:
            return "CE"

        # ── PE path ───────────────────────────────────────────────────────────
        ema_stack_pe  = e9 < e20 < e50
        rsi_ok_pe     = CFG["rsi_pe_min"] <= rsi <= CFG["rsi_pe_max"]
        vwap_ok_pe    = (vwap is None) or (close < vwap)
        # FIX 8: high >= EMA9 * (1 - tol) counts
        pullback_pe   = (
            any(c["high"] >= e9 * (1 - tol) for c in prev_three) and
            close < e9
        )

        log.debug(
            f"[{self.name}] [ModeA] {ts.strftime('%H:%M')} PE check | "
            f"EMA_stack={'✓' if ema_stack_pe else '✗'}({e9:.0f}<{e20:.0f}<{e50:.0f}) "
            f"RSI={'✓' if rsi_ok_pe else '✗'}({rsi:.1f} in [{CFG['rsi_pe_min']}-{CFG['rsi_pe_max']}]) "
            f"VWAP={'✓' if vwap_ok_pe else '✗'} "
            f"Pullback={'✓' if pullback_pe else '✗'}"
        )

        if ema_stack_pe and rsi_ok_pe and vwap_ok_pe and pullback_pe:
            return "PE"

        return None

    # ── Mode B entry check ────────────────────────────────────────────────────

    def _check_mode_b_entry(self, candle: dict, ts: datetime):
        """
        Mode B: wait for a tight consolidation after the initial spike, then
        enter the breakout of that consolidation.
        Window: 9:15–11:30 AM (FIX 6).

        FIX 1: Breakout is checked on the bar AFTER the consolidation is
        complete (_mb_consol_ready=True), not on the bar that expands the zone.
        This prevents the impossible condition of close > zone_high while the
        zone is still being built from the current bar.
        """
        t = ts.time()
        if not (CFG["mode_b_start"] <= t < CFG["mode_b_cutoff"]):
            return

        if self._vix and self._vix > CFG["vix_skip_thresh"]:
            log.info(
                f"[{self.name}] [ModeB] Entry skipped — VIX={self._vix:.1f} "
                f"> {CFG['vix_skip_thresh']}"
            )
            return

        direction = self._spike_direction
        if direction is None:
            return

        bar_range = candle["high"] - candle["low"]

        # ── FIX 1: Check breakout FIRST if consolidation is armed ─────────────
        # _mb_consol_ready is set to True once min_bars are formed.
        # The current candle is the first candle AFTER the consolidation window.
        if self._mb_consol_ready:
            triggered = False
            if direction == "CE" and candle["close"] > self._mb_consol_high:
                log.info(
                    f"[{self.name}] [ModeB] Breakout CE at {ts.strftime('%H:%M')} | "
                    f"close={candle['close']:.0f} > zone_high={self._mb_consol_high:.0f} "
                    f"(consol_bars={self._mb_consol_bars})"
                )
                triggered = True
            elif direction == "PE" and candle["close"] < self._mb_consol_low:
                log.info(
                    f"[{self.name}] [ModeB] Breakout PE at {ts.strftime('%H:%M')} | "
                    f"close={candle['close']:.0f} < zone_low={self._mb_consol_low:.0f} "
                    f"(consol_bars={self._mb_consol_bars})"
                )
                triggered = True

            # Always reset ready flag — we only get one breakout attempt per
            # consolidation window. If it failed, start looking again.
            self._mb_consol_ready = False
            self._mb_state = "WATCHING"

            if triggered:
                self._enter(direction, candle["close"], ts,
                            reason=f"mode_b_consol_breakout_{direction.lower()}")
            else:
                log.info(
                    f"[{self.name}] [ModeB] Breakout attempt failed at "
                    f"{ts.strftime('%H:%M')} | "
                    f"close={candle['close']:.0f} zone=[{self._mb_consol_low:.0f}"
                    f"–{self._mb_consol_high:.0f}] — re-watching"
                )
            return

        # ── Consolidation state machine ───────────────────────────────────────
        if self._mb_state == "WATCHING":
            if bar_range < CFG["mode_b_consol_range"]:
                self._mb_state       = "CONSOL"
                self._mb_consol_high = candle["high"]
                self._mb_consol_low  = candle["low"]
                self._mb_consol_bars = 1
                self._mb_consol_ready = False
                log.info(
                    f"[{self.name}] [ModeB] Consolidation started at "
                    f"{ts.strftime('%H:%M')} | "
                    f"range={bar_range:.0f}pts "
                    f"H={candle['high']:.0f} L={candle['low']:.0f}"
                )
            return

        if self._mb_state == "CONSOL":
            if bar_range < CFG["mode_b_consol_range"]:
                # Expand zone
                self._mb_consol_high = max(self._mb_consol_high, candle["high"])
                self._mb_consol_low  = min(self._mb_consol_low,  candle["low"])
                self._mb_consol_bars += 1
                log.info(
                    f"[{self.name}] [ModeB] Consolidation bar "
                    f"{self._mb_consol_bars} | "
                    f"zone=[{self._mb_consol_low:.0f}–{self._mb_consol_high:.0f}] "
                    f"range={self._mb_consol_high - self._mb_consol_low:.0f}pts"
                )

                if self._mb_consol_bars >= CFG["mode_b_consol_max_bars"]:
                    # Zone built to max — arm the breakout check for next bar
                    log.info(
                        f"[{self.name}] [ModeB] Consolidation complete "
                        f"({self._mb_consol_bars} bars) — "
                        f"armed for breakout on next candle"
                    )
                    self._mb_consol_ready = True
                    self._mb_state = "WATCHING"
                elif self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    # Minimum bars met — arm breakout for next candle
                    # (we don't wait for max; arm as soon as min is met so
                    # we don't miss a fast breakout, but the check fires on
                    # the NEXT bar, not this one)
                    log.info(
                        f"[{self.name}] [ModeB] Min bars met "
                        f"({self._mb_consol_bars}/{CFG['mode_b_consol_min_bars']}) — "
                        f"armed for breakout on next candle"
                    )
                    self._mb_consol_ready = True
                    # Stay in CONSOL so we keep expanding zone if next bar is tight
                    # but breakout check fires first in next call

            else:
                # Wide bar breaks consolidation
                if self._mb_consol_bars >= CFG["mode_b_consol_min_bars"]:
                    # Wide bar itself could be the breakout candle —
                    # check immediately (this bar is AFTER sufficient consol)
                    triggered = False
                    if direction == "CE" and candle["close"] > self._mb_consol_high:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout CE (wide bar) at "
                            f"{ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} > "
                            f"zone_high={self._mb_consol_high:.0f}"
                        )
                        triggered = True
                    elif direction == "PE" and candle["close"] < self._mb_consol_low:
                        log.info(
                            f"[{self.name}] [ModeB] Breakout PE (wide bar) at "
                            f"{ts.strftime('%H:%M')} | "
                            f"close={candle['close']:.0f} < "
                            f"zone_low={self._mb_consol_low:.0f}"
                        )
                        triggered = True

                    self._mb_state = "WATCHING"
                    self._mb_consol_ready = False

                    if triggered:
                        self._enter(direction, candle["close"], ts,
                                    reason=f"mode_b_wide_breakout_{direction.lower()}")
                    else:
                        log.info(
                            f"[{self.name}] [ModeB] Wide bar broke consolidation "
                            f"without breakout at {ts.strftime('%H:%M')} — resetting"
                        )
                else:
                    # Not enough bars — reset and start fresh
                    log.info(
                        f"[{self.name}] [ModeB] Wide bar broke early consolidation "
                        f"({self._mb_consol_bars} bars < "
                        f"{CFG['mode_b_consol_min_bars']} min) — resetting"
                    )
                    self._mb_state = "WATCHING"
                    self._mb_consol_ready = False

    # ── Entry ─────────────────────────────────────────────────────────────────

    def _enter(self, direction: str, index_price: float, ts: datetime, reason: str):
        with self._lock:
            if self._trade and self._trade["state"] == "OPEN":
                return
            if self._trades_taken >= CFG["max_trades_per_day"]:
                return
            if self._day_paused:
                return

        strike = get_atm_strike(index_price, step=NIFTY_STRIKE_STEP)

        if direction == "CE" and self._pre_ce_token:
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

        self.subscribe_option(token)

        opt_ltp = self.get_price(token)
        if not opt_ltp or opt_ltp <= 0:
            log.warning(
                f"[{self.name}] Option LTP unavailable for {sym} — "
                f"skipping entry (no stale-price risk)"
            )
            return

        if not self._acquire_slot():
            log.warning(f"[{self.name}] Trade slot blocked — another live strategy active")
            return

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
        min_bars = CFG["ema_slow"] + 2
        if len(self._buf) < min_bars:
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
        if not sym:
            return None
        try:
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

