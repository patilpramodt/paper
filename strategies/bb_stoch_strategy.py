"""
strategies/bb_stoch_strategy.py
-------------------------------------------------------------
BBStochStrategy -- BankNifty options scalper using:
  * Bollinger Bands  (BB)   -- identifies breakout / volatility expansion
  * Stochastic Oscillator   -- momentum confirmation + overbought/oversold
  * Volume Filter           -- confirms genuine institutional participation

SIGNAL LOGIC (all three must agree before entry):
--------------------------------------------------------------
  CE entry (buy call):
    1. BB Breakout : last close breaks ABOVE upper BB band
       OR BB Bounce : last close crosses back above lower BB band
          from below (mean-reversion mode, Stoch oversold)
    2. Stoch      : %K > %D (positive crossover in last 2 bars)
                    AND %K > 50  (momentum in bull territory)
                    OR  %K < 20  (oversold bounce)
    3. Volume     : current bar volume >= VOL_MULT * rolling avg vol
    4. VWAP       : close above session VWAP (hub provides tick-accurate VWAP)
    5. Session    : only between SESSION_START and ENTRY_CUTOFF

  PE entry (buy put):
    Mirror of the above -- BB break below lower band, Stoch bear cross, etc.

TRADE MANAGEMENT (on every option WebSocket tick):
--------------------------------------------------------------
  * SL    = entry_price - ATR_SL_MULT * atr
  * TP    = entry_price + ATR_TP_MULT * atr
  * Trail = moves SL to breakeven when profit >= TRAIL_ARM pts,
            then trails every TRAIL_STEP pts after that

FIXES APPLIED:
  Bug 1 — time.sleep(0.4) was called inside _enter_trade(), which is
           invoked from _evaluate_entry() → on_candle() → MarketHub's
           WebSocket callback thread. Sleeping here blocked ALL tick
           processing for every strategy for 400ms on every entry attempt.
           During that sleep, no WebSocket ticks could arrive, so the
           option we just subscribed had no price — get_price() returned
           None → all entries silently skipped.
           Fix: removed time.sleep(). At entry time, read LTP immediately.
           If unavailable, log a warning and store a pending entry (see Bug 7).

  Bug 2 — _persist_buf cleared in _evaluate_entry() BEFORE calling
           _enter_trade(). When _enter_trade() returned early (LTP unavailable),
           the persistence context was already wiped.
           Fix: _persist_buf is now cleared inside _enter_trade() only after
           a genuine fill is made. NOTE: with persistence=1 this has limited
           effect — Bug 7 fix is the correct solution for LTP misses.

  Bug 3 — _close_trade() read self._trade outside the lock, creating a
           race window between the WS thread (auto-squareoff at 15:15 via
           on_tick) and the main thread (eod_summary at 15:31). Both could
           pass the `if not trade` guard simultaneously → double P&L, double
           CSV row, double unsubscribe.
           Fix: _close_trade() now atomically reads AND clears self._trade
           inside a single lock acquisition at the very top.

  Bug 4 — ATM options not pre-subscribed in pre_market(). Even without the
           sleep bug, subscribing at the moment of signal and immediately
           reading LTP is a race — first tick may take 200–800ms.
           Fix: pre_market() subscribes ATM CE + PE at 9:08 AM based on
           prev_close reference price.

  Bug 5 — pre_market() only subscribed the opening ATM strike. When
           BankNifty moves 300+ pts intraday, the signal fires at a strike
           that was never pre-warmed → LTP miss on first attempt.
           Fix: pre_market() now subscribes ATM±200pt strikes (6 tokens
           total) to cover the typical intraday range.

  Bug 6 (NEW) — pre_market() pre-subscribes strikes based on prev_close.
           On large gap days (BankNifty gapped 1850pts down on 19-Mar-2026),
           the actual intraday ATM was 1100–2000pts away from all pre-subscribed
           strikes — every pre-warmed token was deep OTM and never used.
           Both BB_STOCH signals (12:15 BUY_CE and 14:10 BUY_PE) fired on
           completely cold tokens → LTP unavailable → trades permanently missed.
           Fix: on_candle() checks if this is the first candle of the session
           and if so calls _subscribe_spot_atm(spot) to subscribe ATM±400
           based on the ACTUAL live market price. This fires ~9:20 AM —
           75+ minutes before the earliest possible BB_STOCH signal (min_bars=14
           candles = 70min from 9:15). Token warms up long before any signal.

  Bug 7 (NEW) — When get_price() returns None (token subscribed but first
           tick hasn't arrived yet), _enter_trade() bails early with a warning.
           With persistence=1 the next 5-min candle overwrites _persist_buf
           with HOLD (BB crossover already happened, no fresh setup), so the
           signal never re-fires. Bug 2 fix (retain persist_buf) helps when
           persistence≥2 but is irrelevant for persistence=1.
           Fix: when LTP unavailable, store self._pending_entry with the full
           signal context. In on_option_tick(), when that token's FIRST live
           tick arrives (200–800ms, same 5-min window as original signal),
           execute the fill immediately via _fill_pending(). No blocking,
           no sleeping, no waiting for the next candle — the position opens
           within seconds of the signal, exactly as intended.

  Bug 10 — on_tick() called _force_close() on EVERY index tick after 15:15
            (potentially dozens per second). Second call was harmless but
            wasteful. Fix: added _squareoff_done flag — force-close fires
            once only.
"""

import csv
import logging
import os
import threading
import time as time_module
from collections import deque
from datetime import datetime, date, time as dtime, timezone, timedelta

# IST FIX: GitHub Actions runners are UTC — timestamps must be IST
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return datetime.now(tz=_IST).replace(tzinfo=None)
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

log = logging.getLogger("strategy.bb_stoch")

# ============================================================
# CONFIG  (all tunables in one place)
# ============================================================
CFG = {
    # Bollinger Bands
    "bb_period"         : 20,       # rolling window for BB mean/std
    "bb_std"            : 2.0,      # number of std deviations for bands
    "bb_squeeze_pct"    : 0.002,    # loosened: was 0.003 → 0.002 allows entries in tighter markets
                                    # band-width / close < this = squeeze (skip)

    # Stochastic Oscillator
    "stoch_k"           : 9,        # %K lookback (lowered: was 14 → need=19 bars; 9 → need=14 bars matching min_bars)
    "stoch_d"           : 3,        # %D smoothing period (SMA of %K)
    "stoch_signal"      : 3,        # signal line smoothing
    "stoch_ob"          : 75,       # overbought level for PE reversal
    "stoch_os"          : 25,       # oversold level for CE reversal

    # Volume filter
    "vol_avg_period"    : 10,       # bars to compute average volume
    "vol_mult"          : 1.0,      # loosened: was 1.2 (1.2 blocked valid signals on average-vol days)

    # VWAP
    "vwap_buffer"       : 10.0,     # allow entry within N pts of VWAP line

    # Session windows (HH, MM)
    "session_start"     : (9, 45),  # no entries before this (opening chaos)
    "session_cutoff"    : (14, 30), # no NEW entries after this
    "auto_squareoff"    : (15, 15), # force-close all open positions

    # Trade management
    "atr_period"        : 14,       # ATR lookback for SL/TP scaling
    "atr_sl_mult"       : 0.9,      # SL = entry - atr * mult
    "atr_tp_mult"       : 1.6,      # TP = entry + atr * mult
    "sl_min"            : 5.0,      # hard floor on SL distance (option pts)
    "sl_max"            : 15.0,     # hard ceiling
    "tp_min"            : 8.0,      # hard floor on TP distance
    "tp_max"            : 25.0,     # hard ceiling
    "trail_arm"         : 6.0,      # move SL to BE when profit >= this
    "trail_step"        : 2.0,      # then trail every N pts
    "slippage"          : 1.5,      # simulated fill slippage (per side)
    "exit_cooldown"     : 2.0,      # seconds between exit attempts

    # Risk
    "max_trades_day"    : 4,
    "post_sl_cooldown"  : 120,      # 2 min wait after any SL hit
    "max_daily_loss"    : 6000,     # Rs circuit breaker
    "quantity"          : 35,       # lots

    # Signal persistence (candle bars)
    "persistence"       : 1,        # lowered: was 2 (10-min confirmation window misses fast 5m moves)

    # Minimum 5-min bars needed before first signal
    # With stoch_k=9: need = 9+3+2 = 14 bars → 14×5min = 70min from 9:15 = first signal ~10:25 AM
    # BB now uses min_periods=bb_period//3 so it computes validly before 20 bars.
    # Was 20 (100min from 9:15 → 10:55 AM, left only 3.5hrs trading window).
    "min_bars"          : 14,       # was 20

    # FIX (Bug 6): strike offsets to pre-subscribe based on actual open price
    # ±400 covers intraday moves of up to 400pts in either direction from open
    # (typical BankNifty daily range is 300–600pts; ±400 covers ~80% of days)
    "spot_atm_offsets"  : [0, 200, -200, 400, -400],

    # CSV output
    "entry_csv"         : "logs/bb_stoch_entry.csv",
    "exit_csv"          : "logs/bb_stoch_exit.csv",
    "signal_csv"        : "logs/bb_stoch_signals.csv",
}


# ============================================================
# INDICATOR COMPUTATION
# ============================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR from a OHLCV DataFrame."""
    if len(df) < period + 1:
        return 0.0
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(round(atr.iloc[-1], 2))


def compute_bb(df: pd.DataFrame, period: int, nstd: float) -> dict:
    """
    Bollinger Bands on 'close'.
    Returns: mid, upper, lower, bw_pct (bandwidth as % of mid), squeeze (bool)
    """
    if len(df) < 2:
        return {"mid": 0, "upper": 0, "lower": 0, "bw_pct": 0, "squeeze": True}
    close = df["close"].astype(float)
    min_p = max(2, period // 3)   # allow partial-history computation
    mid   = close.rolling(period, min_periods=min_p).mean()
    std   = close.rolling(period, min_periods=min_p).std().fillna(0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    m     = float(mid.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    bw    = (u - l) / m if m > 0 else 0
    return {
        "mid"     : round(m, 2),
        "upper"   : round(u, 2),
        "lower"   : round(l, 2),
        "bw_pct"  : round(bw, 5),
        "squeeze" : bw < CFG["bb_squeeze_pct"],
        # Previous bar values (for crossover detection)
        "prev_upper": round(float(upper.iloc[-2]), 2) if len(df) >= 2 else u,
        "prev_lower": round(float(lower.iloc[-2]), 2) if len(df) >= 2 else l,
        "prev_close": round(float(close.iloc[-2]), 2) if len(df) >= 2 else float(close.iloc[-1]),
    }


def compute_stoch(df: pd.DataFrame, k: int, d: int) -> dict:
    """
    Full Stochastic Oscillator.
    %K = (close - lowest_low_k) / (highest_high_k - lowest_low_k) * 100
    %D = SMA(%K, d)
    Returns last and prev values for crossover detection.
    """
    need = k + d + 2
    if len(df) < need:
        return {"k": 50, "d": 50, "prev_k": 50, "prev_d": 50}

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_low   = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    denom        = highest_high - lowest_low
    denom        = denom.replace(0, np.nan)
    raw_k        = (close - lowest_low) / denom * 100
    # Smooth %K with SMA(d)
    smooth_k     = raw_k.rolling(d).mean()
    # Signal = SMA(smooth_k, d)
    signal_k     = smooth_k.rolling(d).mean()

    return {
        "k"      : round(float(smooth_k.iloc[-1]),  2),
        "d"      : round(float(signal_k.iloc[-1]),  2),
        "prev_k" : round(float(smooth_k.iloc[-2]),  2),
        "prev_d" : round(float(signal_k.iloc[-2]),  2),
    }


def compute_vol_ratio(df: pd.DataFrame, avg_period: int) -> float:
    """
    Ratio of last bar's volume vs rolling N-bar average.
    > 1.0 means above average. > vol_mult = confirmed participation.

    Returns -1.0 (sentinel) when volume data is unavailable so the caller
    can distinguish "data missing" from a genuine low-volume bar and skip
    the volume filter rather than silently blocking every trade.
    """
    if "volume" not in df.columns:
        log.warning("[BB_STOCH] No 'volume' column in 5m candles -- "
                    "volume filter will be bypassed (check MarketHub candle aggregation)")
        return -1.0  # sentinel: data unavailable

    vol = df["volume"].astype(float)

    # All zeros / NaN = another common data gap symptom
    if vol.sum() == 0 or vol.isna().all():
        log.warning("[BB_STOCH] Volume column is all zeros/NaN -- "
                    "volume filter bypassed (check feed)")
        return -1.0  # sentinel

    if len(df) < avg_period + 1:
        return -1.0  # not enough history yet -- bypass rather than block

    avg     = float(vol.iloc[-(avg_period + 1):-1].mean())
    current = float(vol.iloc[-1])
    if avg <= 0:
        return -1.0
    return round(current / avg, 3)


# ============================================================
# SIGNAL BUILDER
# ============================================================

def evaluate_signal(df: pd.DataFrame, vwap: Optional[float]) -> dict:
    """
    Run all three filters. Returns a signal dict:
        action      : "BUY_CE" | "BUY_PE" | "HOLD"
        blocked_by  : reason string if HOLD
        bb, stoch, vol_ratio, atr: indicator snapshots for logging
    """
    empty = {"action": "HOLD", "blocked_by": "no_data",
             "bb": {}, "stoch": {}, "vol_ratio": 0, "atr": 0}

    if df is None or len(df) < CFG["min_bars"]:
        return {**empty, "blocked_by": "insufficient_bars"}

    # ---- Indicators ----
    bb       = compute_bb(df, CFG["bb_period"], CFG["bb_std"])
    stoch    = compute_stoch(df, CFG["stoch_k"], CFG["stoch_d"])
    vol_ratio = compute_vol_ratio(df, CFG["vol_avg_period"])
    atr      = _compute_atr(df, CFG["atr_period"])
    close    = float(df["close"].iloc[-1])

    base = {"bb": bb, "stoch": stoch, "vol_ratio": vol_ratio, "atr": atr,
            "close": close}

    # ---- Filter 1: BB squeeze → skip (no trend) ----
    if bb["squeeze"]:
        return {"action": "HOLD", "blocked_by": "bb_squeeze", **base}

    # ---- Filter 2: Volume ----
    # vol_ratio == -1.0 is the sentinel meaning "no volume data available".
    # In that case we bypass the filter (don't block) and log a warning.
    # When real volume data is flowing, vol_ratio must exceed vol_mult (1.0x avg).
    if vol_ratio == -1.0:
        vol_ok = True   # data unavailable -- bypass, do not block
        log.debug("[BB_STOCH] Volume data unavailable -- vol filter bypassed")
    else:
        vol_ok = vol_ratio >= CFG["vol_mult"]

    if not vol_ok:
        return {"action": "HOLD", "blocked_by": "volume_low", **base}

    # ---- Stochastic crossovers ----
    k_cross_up   = stoch["prev_k"] < stoch["prev_d"] and stoch["k"] >= stoch["d"]
    k_cross_down = stoch["prev_k"] > stoch["prev_d"] and stoch["k"] <= stoch["d"]

    # ---- VWAP bias ----
    if vwap and vwap > 0:
        above_vwap = close >= (vwap - CFG["vwap_buffer"])
        below_vwap = close <= (vwap + CFG["vwap_buffer"])
    else:
        above_vwap = True  # data unavailable -- don't block
        below_vwap = True

    # ================================================================
    # CE CONDITIONS  (breakout above upper band OR oversold bounce)
    # ================================================================
    bb_breakout_up = close > bb["upper"] and bb["prev_close"] <= bb["prev_upper"]
    bb_bounce_up   = (close > bb["lower"] and bb["prev_close"] <= bb["prev_lower"]
                      and stoch["k"] < CFG["stoch_os"])

    stoch_bull     = (stoch["k"] > 50 or stoch["k"] < CFG["stoch_os"]) and (k_cross_up or stoch["k"] > stoch["d"])

    if (bb_breakout_up or bb_bounce_up) and stoch_bull and above_vwap:
        return {"action": "BUY_CE", "blocked_by": "", **base,
                "mode": "breakout" if bb_breakout_up else "bounce"}

    # ================================================================
    # PE CONDITIONS  (breakdown below lower band OR overbought reversal)
    # ================================================================
    bb_breakout_dn = close < bb["lower"] and bb["prev_close"] >= bb["prev_lower"]
    bb_bounce_dn   = (close < bb["upper"] and bb["prev_close"] >= bb["prev_upper"]
                      and stoch["k"] > CFG["stoch_ob"])

    stoch_bear     = (stoch["k"] < 50 or stoch["k"] > CFG["stoch_ob"]) and (k_cross_down or stoch["k"] < stoch["d"])

    if (bb_breakout_dn or bb_bounce_dn) and stoch_bear and below_vwap:
        return {"action": "BUY_PE", "blocked_by": "", **base,
                "mode": "breakout" if bb_breakout_dn else "bounce"}

    # ---- Granular block reasons for analysis ----
    if bb_breakout_up or bb_bounce_up:
        if not stoch_bull:
            return {"action": "HOLD", "blocked_by": "stoch_not_bull", **base}
        if not above_vwap:
            return {"action": "HOLD", "blocked_by": "below_vwap", **base}
    if bb_breakout_dn or bb_bounce_dn:
        if not stoch_bear:
            return {"action": "HOLD", "blocked_by": "stoch_not_bear", **base}
        if not below_vwap:
            return {"action": "HOLD", "blocked_by": "above_vwap", **base}

    return {"action": "HOLD", "blocked_by": "no_setup", **base}


# ============================================================
# PAPER TRADE STATE
# ============================================================

class BBTrade:
    """Minimal trade record (no external dependency on scalper_v7_core)."""
    __slots__ = (
        "symbol", "token", "option_type", "qty",
        "entry", "sl", "target", "sl_pts", "tp_pts",
        "trail_stage", "spot", "atr", "timestamp",
        "exit_pending", "last_exit_ts",
    )

    def __init__(self, symbol, token, opt_type, ltp, qty,
                 spot, sl_price, tp_price, sl_pts, tp_pts, atr):
        slip          = CFG["slippage"]
        self.symbol   = symbol
        self.token    = token
        self.option_type = opt_type
        self.qty      = qty
        self.entry    = round(ltp + slip, 2)        # simulated fill
        # Anchor SL/TP to actual fill (not pre-slippage LTP)
        off           = self.entry - ltp
        self.sl       = round(sl_price - off, 2)
        self.target   = round(tp_price + off, 2)
        self.sl_pts   = sl_pts
        self.tp_pts   = tp_pts
        self.trail_stage = 0
        self.spot     = spot
        self.atr      = atr
        self.timestamp = _now_ist().isoformat()  # FIX: was UTC on GitHub Actions
        self.exit_pending  = False
        self.last_exit_ts  = 0.0


# ============================================================
# CSV HELPERS
# ============================================================

def _csv_append(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# STRATEGY
# ============================================================

class BBStochStrategy(BaseStrategy):
    """
    BankNifty options strategy using Bollinger Bands + Stochastic + Volume.
    Plugs into the shared MarketHub framework -- zero extra API calls.

    Data consumed:
      on_candle()       -- 5-min index candles (broadcast by MarketHub)
      on_option_tick()  -- live option LTP for SL/TP management
      hub.session_vwap  -- tick-accurate intraday VWAP
    """

    @property
    def name(self) -> str:
        return "BB_STOCH"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # Private 5-min candle buffer (fed from MarketHub's on_candle broadcast)
        # Size: bb_period + stoch_k + stoch_d + headroom
        buf_size = CFG["bb_period"] + CFG["stoch_k"] + CFG["stoch_d"] + 20
        self._buf_5m: deque = deque(maxlen=buf_size)

        # Trade state
        self._trade: Optional[BBTrade] = None
        self._active_token: Optional[int] = None
        self._lock = threading.Lock()

        # Risk state
        self._daily_pnl   : float = 0.0
        self._trades_today: int   = 0
        self._last_sl_time: float = 0.0
        self._is_halted   : bool  = False
        self._results     : list  = []
        self._blocked_log : dict  = {}

        # Signal persistence (must fire on N consecutive 5-min bars)
        self._persist_buf: deque = deque(maxlen=CFG["persistence"])

        # Pre-market data
        self._instruments   = None
        self._expiry_date   = None
        self._dte           = 0
        self._session_date  = None

        # FIX (Bug 10): guard flag so _force_close is called at most once per day
        self._squareoff_done: bool = False

        # FIX (Bug 4): pre-subscribed tokens (populated in pre_market)
        self._pre_ce_token: Optional[int] = None
        self._pre_pe_token: Optional[int] = None

        # FIX (Bug 6): flag to trigger dynamic spot-based subscription on first candle
        # pre_market() subscribes based on prev_close which fails on large gap days.
        # on_candle() will subscribe ATM±400 based on actual live spot on first bar.
        self._open_atm_subscribed: bool = False

        # FIX (Bug 7): pending entry storage.
        # When _enter_trade() bails because LTP is unavailable (token subscribed
        # but first WebSocket tick hasn't arrived yet), we store the full signal
        # context here. on_option_tick() watches for this token and fills the
        # trade the moment the first live price arrives — typically 200–800ms later,
        # well within the same 5-minute window. No sleeping, no blocking.
        self._pending_entry: Optional[dict] = None

        log.info("[BB_STOCH] Strategy initialized")

    # ----------------------------------------------------------
    # PRE-MARKET
    # ----------------------------------------------------------

    def pre_market(self, premarket_data, instruments) -> bool:
        self._instruments  = instruments
        self._expiry_date  = premarket_data.expiry_date
        self._dte          = premarket_data.dte_days
        self._session_date = date.today()
        self._reset_day()
        log.info(
            f"[BB_STOCH] Pre-market | VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr} Expiry={self._expiry_date} DTE={self._dte}"
        )
        if premarket_data.pcr is None:
            log.warning("[BB_STOCH] PCR is None -- VWAP bias will be unrestricted. "
                        "Check if PCR fetch failed in core.premarket")

        # FIX (Bug 4 + 5): pre-subscribe ATM CE + PE + ATM±200 at 9:08 AM based
        # on prev_close. This warms up the typical daily range for normal sessions.
        #
        # FIX (Bug 6): this is NOT sufficient for large gap days (e.g. 1850pt gap).
        # _subscribe_spot_atm() is called on the first 5-min candle with the actual
        # live spot to cover the real intraday range regardless of gap size.
        ref_price = premarket_data.prev_close or premarket_data.prev_last5m_close
        if ref_price and self._expiry_date:
            atm = get_atm_strike(ref_price)

            # Strikes to pre-warm: ATM, ATM±200 covers ~1 std-dev intraday move
            strike_offsets = [0, 200, -200]

            for offset in strike_offsets:
                for opt_type in ("CE", "PE"):
                    adj = atm + offset
                    # On expiry day use 1-strike ITM at the primary ATM only
                    if offset == 0 and self._dte == 0:
                        adj = atm - 100 if opt_type == "CE" else atm + 100
                    tok, sym = instruments.get_option_token(adj, opt_type, self._expiry_date)
                    if tok:
                        self.subscribe_option(tok)
                        # Store primary ATM tokens for quick lookup at entry time
                        if offset == 0:
                            if opt_type == "CE":
                                self._pre_ce_token = tok
                            else:
                                self._pre_pe_token = tok
                        log.info(f"[BB_STOCH] Pre-subscribed {sym} ({tok})"
                                 f"{' [ATM]' if offset == 0 else f' [ATM{offset:+d}]'}")
                    else:
                        if offset == 0:
                            log.warning(f"[BB_STOCH] Pre-subscribe failed for ATM {adj} {opt_type}")
                        else:
                            log.debug(f"[BB_STOCH] Pre-subscribe skipped for {adj} {opt_type} "
                                      f"(token not found — likely near expiry)")
        else:
            log.warning("[BB_STOCH] No ref price for pre-subscription — "
                        "spot-based subscription will fire on first candle only")

        return True

    def _reset_day(self):
        self._daily_pnl    = 0.0
        self._trades_today = 0
        self._last_sl_time = 0.0
        self._is_halted    = False
        self._results.clear()
        self._blocked_log.clear()
        self._persist_buf.clear()
        self._squareoff_done     = False   # FIX (Bug 10): reset guard each day
        self._open_atm_subscribed = False  # FIX (Bug 6): reset so first candle re-subscribes
        self._pending_entry       = None   # FIX (Bug 7): discard any stale pending entry

    # ----------------------------------------------------------
    # FIX (Bug 6): Dynamic spot-based subscription on first candle
    # ----------------------------------------------------------

    def _subscribe_spot_atm(self, spot: float):
        """
        FIX (Bug 6): Subscribe ATM±400 strikes based on actual live spot price.

        Called on the first 5-min candle of the session (~9:20 AM).
        This covers large gap days where pre_market()'s prev_close-based
        subscriptions land on completely wrong strikes.

        On 19-Mar-2026 BankNifty gapped 1850pts down:
          prev_close = 55326 → pre-subscribed 55100/55300/55500
          actual open = 53474 → real ATM range = 53200–54200
          → all pre-subscribed tokens were deep OTM, LTP unavailable on both signals

        With ±400 offsets from actual spot this method subscribes 10 tokens
        covering 53000–54800 on that day — every strike that could realistically
        get a signal during the session.
        """
        if not spot or spot <= 0:
            return
        if not self._expiry_date or not self._instruments:
            return

        atm = get_atm_strike(spot)
        subscribed = 0

        for offset in CFG["spot_atm_offsets"]:
            for opt_type in ("CE", "PE"):
                strike = atm + offset
                tok, sym = self._instruments.get_option_token(strike, opt_type, self._expiry_date)
                if tok:
                    self.subscribe_option(tok)
                    subscribed += 1
                    log.info(
                        f"[BB_STOCH] Spot-subscribe {sym} ({tok}) "
                        f"[spot-ATM{offset:+d}]"
                    )
                else:
                    log.debug(
                        f"[BB_STOCH] Spot-subscribe skipped {strike}{opt_type} "
                        f"(token not found)"
                    )

        log.info(
            f"[BB_STOCH] Spot-based subscription complete: "
            f"{subscribed} tokens around ATM={atm} (spot={spot:.0f})"
        )

    # ----------------------------------------------------------
    # TICK CALLBACKS
    # ----------------------------------------------------------

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """Index tick -- not used directly (we work on 5-min candles).

        FIX (Bug 10): _force_close() was called on every index tick after 15:15.
        This could be dozens of calls per second. Added _squareoff_done guard
        so it fires exactly once.
        """
        # FIX (Bug 10): guard with flag — auto square-off fires exactly once
        if ts.time() >= dtime(*CFG["auto_squareoff"]) and not self._squareoff_done:
            self._squareoff_done = True
            self._force_close("AUTO-SQUAREOFF")

    def on_candle(self, candle: dict, ts: datetime):
        """
        Called by MarketHub when a 5-min index candle closes.
        This is the main signal evaluation trigger.

        FIX (Bug 6): on the first candle of the session, subscribe ATM±400
        based on the actual spot price. This happens ~9:20 AM — 65+ minutes
        before the earliest possible BB_STOCH signal (min_bars=14 × 5min =
        70min from 9:15). All tokens are warm before any signal can fire.
        """
        with self._lock:
            self._buf_5m.append(candle)

        # FIX (Bug 6): subscribe around actual open spot on first candle.
        # pre_market() uses prev_close which fails on large gap days.
        # This fires once per session, covers the true intraday ATM range.
        if not self._open_atm_subscribed:
            spot = candle.get("close", 0)
            if spot and spot > 0:
                self._subscribe_spot_atm(spot)
                self._open_atm_subscribed = True

        # Don't look for entries if a trade is already open
        if self._trade:
            return

        self._evaluate_entry(ts)

    def on_option_tick(self, token: int, price: float, ts: datetime,
                       tick_ts: datetime = None):
        """
        Live option price -- drives two things:
          1. SL/TP/trail management for open trades (existing behaviour)
          2. FIX (Bug 7): pending entry fill on first live tick

        FIX (Bug 7) explanation:
          When _enter_trade() bails because get_price() returns None
          (token subscribed but first WS tick hasn't arrived yet), it stores
          self._pending_entry with the full signal context.

          This method is called by MarketHub for EVERY tick on EVERY subscribed
          option token. The first tick on the newly subscribed token lands here
          200–800ms after subscription — still within the same 5-minute window
          as the original signal.

          We check: if this tick's token matches _pending_entry["token"], and
          no trade is open, and risk gates pass → fill immediately via
          _fill_pending(). The position opens within seconds of the signal
          as originally intended.

          This does NOT block or sleep — it runs in the WebSocket tick thread
          identically to _manage_trade() which is already called here.
        """
        # FIX (Bug 7): check for pending entry on first tick of subscribed token
        pending = self._pending_entry
        if pending is not None and token == pending["token"] and not self._trade:
            # Validate the pending entry is still actionable
            now_time = ts.time()
            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(
                    f"[BB_STOCH] Pending entry expired (past cutoff) for "
                    f"{pending['symbol']} — discarding"
                )
                self._pending_entry = None
            elif self._is_halted or self._trades_today >= CFG["max_trades_day"]:
                log.info(
                    f"[BB_STOCH] Pending entry cancelled (risk gate) for "
                    f"{pending['symbol']}"
                )
                self._pending_entry = None
            elif price and price >= 5:
                # Valid LTP arrived — fill the trade now
                self._fill_pending(pending, price, ts)
                return
            # else price still invalid — keep pending, wait for next tick
            return

        # Normal trade management for open position
        if token != self._active_token:
            return
        if not self._trade:
            return
        self._manage_trade(price, ts)

    # ----------------------------------------------------------
    # SIGNAL EVALUATION (on every 5-min candle close)
    # ----------------------------------------------------------

    def _evaluate_entry(self, ts: datetime):
        now_time = ts.time()

        # Session gate
        if now_time < dtime(*CFG["session_start"]):
            return
        if now_time >= dtime(*CFG["session_cutoff"]):
            return

        # Risk gate
        if self._is_halted:
            return
        if self._trades_today >= CFG["max_trades_day"]:
            return
        elapsed_since_sl = time_module.time() - self._last_sl_time
        if self._last_sl_time > 0 and elapsed_since_sl < CFG["post_sl_cooldown"]:
            remaining = CFG["post_sl_cooldown"] - elapsed_since_sl
            log.info(f"[BB_STOCH] Post-SL cooldown: {remaining:.0f}s remaining")
            return

        with self._lock:
            candles = list(self._buf_5m)

        df = self._to_df(candles)
        if df.empty:
            return

        n_bars = len(df)
        if n_bars < CFG["min_bars"]:
            log.debug(f"[BB_STOCH] Warming up: {n_bars}/{CFG['min_bars']} bars "
                      f"(ready in ~{(CFG['min_bars']-n_bars)*5} mins)")
            return

        vwap = self._hub.session_vwap.value

        # Run signal logic
        sig    = evaluate_signal(df, vwap)
        action = sig["action"]

        # Log signal bar
        self._log_signal(ts, sig)

        # Persistence check
        self._persist_buf.append(action)
        confirmed = (
            len(self._persist_buf) >= CFG["persistence"]
            and all(a == action and a != "HOLD" for a in self._persist_buf)
        )

        bb    = sig.get("bb", {})
        stoch = sig.get("stoch", {})
        vwap_str = f"{vwap:.2f}" if vwap is not None else "N/A"
        vol_ratio_val = sig.get("vol_ratio", -1)
        vol_str = f"{vol_ratio_val:.2f}x" if vol_ratio_val >= 0 else "N/A(no-data)"
        log.info(
            f"[BB_STOCH] {action:8s} | "
            f"Close={sig.get('close', 0):.2f} | "
            f"BB=[{bb.get('lower', 0):.1f}~{bb.get('upper', 0):.1f}] "
            f"BW={bb.get('bw_pct', 0)*100:.2f}% | "
            f"Stoch K={stoch.get('k', 0):.1f} D={stoch.get('d', 0):.1f} | "
            f"Vol={vol_str} | "
            f"VWAP={vwap_str} | "
            f"Block={sig.get('blocked_by') or 'none'} | Persist={confirmed} | "
            f"PnL={self._daily_pnl:+.1f}Rs T={self._trades_today}/{CFG['max_trades_day']}"
        )

        if not confirmed or action == "HOLD":
            if action == "HOLD":
                bl = sig.get("blocked_by", "")
                self._blocked_log[bl] = self._blocked_log.get(bl, 0) + 1
            return

        # FIX (Bug 2): _persist_buf is NOT cleared here.
        # It is cleared inside _enter_trade() only after a genuine fill.
        # If _enter_trade() bails early (LTP unavailable, sanity check fail, etc.),
        # the buffer retains the current signal so the next candle is immediately
        # re-confirmed without needing to rebuild from scratch.
        # NOTE: with persistence=1 this has limited effect since the next candle's
        # HOLD signal overwrites the deque(maxlen=1) anyway. Bug 7's _pending_entry
        # mechanism is the correct solution for the LTP miss case.
        self._enter_trade(action, sig, ts)

    # ----------------------------------------------------------
    # ENTRY
    # ----------------------------------------------------------

    def _enter_trade(self, action: str, sig: dict, ts: datetime):
        """
        Open a new options position.

        FIX (Bug 1): removed time.sleep(0.4).
        FIX (Bug 2): _persist_buf cleared only on genuine fill (not on LTP miss).
        FIX (Bug 7): on LTP miss, store _pending_entry for fill on first tick.
        """
        opt    = "CE" if action == "BUY_CE" else "PE"
        spot   = sig.get("close", 0)
        atr    = sig.get("atr", 0)
        expiry = self._expiry_date

        if expiry is None:
            log.warning("[BB_STOCH] No expiry date -- skipping entry")
            return

        atm = get_atm_strike(spot)

        # On expiry day use 1-strike ITM for better liquidity
        if self._dte == 0:
            atm = atm - 100 if opt == "CE" else atm + 100

        token, tsym = self._instruments.get_option_token(atm, opt, expiry)
        if not token:
            log.warning(f"[BB_STOCH] Option not found: ATM={atm} {opt} expiry={expiry}")
            return

        # Subscribe the token (deduplicates — safe if already subscribed from pre_market
        # or _subscribe_spot_atm). With Bug 6 fix, most entry strikes will already be
        # subscribed and warm. If spot has moved beyond ±400 from open (rare), this
        # is a fresh subscribe and LTP may not be available yet — Bug 7 handles that.
        self.subscribe_option(token)

        # FIX (Bug 1): no sleep here. Read LTP directly from the WebSocket price cache.
        ltp = self.get_price(token)

        if not ltp or ltp < 5:
            # FIX (Bug 7): store pending entry instead of silently dropping the trade.
            # on_option_tick() will fill this the moment the first tick arrives on
            # this token — typically 200–800ms, within the same 5-minute window.
            # Do NOT unsubscribe — keep token live so the tick arrives promptly.
            # Do NOT clear _persist_buf — keep persistence context (Bug 2 fix).
            log.warning(
                f"[BB_STOCH] LTP unavailable for {tsym} (token={token}) — "
                f"option subscribed, pending entry stored, will fill on first tick"
            )
            self._pending_entry = {
                "action"    : action,
                "opt"       : opt,
                "sig"       : sig,
                "token"     : token,
                "symbol"    : tsym,
                "ts"        : ts,           # signal timestamp (for delay logging)
                "atm"       : atm,
            }
            return

        # LTP is available — proceed with immediate fill
        self._execute_fill(opt, sig, token, tsym, ltp, ts)

    def _execute_fill(self, opt: str, sig: dict, token: int,
                      tsym: str, ltp: float, ts: datetime,
                      pending_delay_ms: Optional[float] = None):
        """
        Shared fill logic used by both _enter_trade() (immediate) and
        _fill_pending() (deferred on first tick).

        Constructs the BBTrade, registers it, logs, and writes entry CSV.
        """
        atr  = sig.get("atr", 0)
        spot = sig.get("close", 0)

        # Compute SL/TP
        sl_pts, tp_pts, sl_price, tp_price = self._compute_sl_tp(ltp, atr)

        # Option sanity: reject if spread eats too much of target
        est_spread = ltp * 0.03
        if tp_pts > 0 and (est_spread / tp_pts) > 0.35:
            log.warning(f"[BB_STOCH] Spread {est_spread:.1f} too large vs TP {tp_pts:.1f} -- skipping")
            self.unsubscribe_option(token)
            return

        trade = BBTrade(
            symbol   = tsym,
            token    = token,
            opt_type = opt,
            ltp      = ltp,
            qty      = CFG["quantity"],
            spot     = spot,
            sl_price = sl_price,
            tp_price = tp_price,
            sl_pts   = sl_pts,
            tp_pts   = tp_pts,
            atr      = atr,
        )

        with self._lock:
            self._trade        = trade
            self._active_token = token

        self._trades_today += 1

        # FIX (Bug 2): clear persistence buffer only here, after a confirmed fill.
        self._persist_buf.clear()

        rr = round(tp_pts / sl_pts, 2) if sl_pts else 0

        delay_str = f" [pending_delay={pending_delay_ms:.0f}ms]" if pending_delay_ms is not None else ""
        log.info(
            f"[BB_STOCH] ENTRY {opt} | {tsym} | "
            f"Fill={trade.entry:.2f} LTP={ltp:.2f}{delay_str} | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) TP={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} ATR={atr:.1f} | "
            f"Mode={sig.get('mode', '?')} BB-BW={sig['bb'].get('bw_pct', 0)*100:.2f}%"
        )

        bb = sig.get("bb", {})
        st = sig.get("stoch", {})
        _csv_append(CFG["entry_csv"], {
            "timestamp"  : trade.timestamp,
            "symbol"     : tsym,
            "opt_type"   : opt,
            "qty"        : CFG["quantity"],
            "ltp"        : ltp,
            "fill"       : trade.entry,
            "sl"         : trade.sl,
            "target"     : trade.target,
            "sl_pts"     : sl_pts,
            "tp_pts"     : tp_pts,
            "rr"         : rr,
            "spot"       : spot,
            "atr"        : atr,
            "mode"       : sig.get("mode", "pending" if pending_delay_ms else ""),
            "bb_upper"   : bb.get("upper", ""),
            "bb_lower"   : bb.get("lower", ""),
            "bb_bw_pct"  : bb.get("bw_pct", ""),
            "stoch_k"    : st.get("k", ""),
            "stoch_d"    : st.get("d", ""),
            "vol_ratio"  : sig.get("vol_ratio", ""),
            "vwap"       : self._hub.session_vwap.value if self._hub.session_vwap.value is not None else "N/A",
            "pending_delay_ms": round(pending_delay_ms, 0) if pending_delay_ms is not None else "",
        })

    # ----------------------------------------------------------
    # FIX (Bug 7): Pending entry fill on first WebSocket tick
    # ----------------------------------------------------------

    def _fill_pending(self, pending: dict, ltp: float, ts: datetime):
        """
        FIX (Bug 7): Execute fill for a pending entry on its first live tick.

        Called from on_option_tick() when the pending token's FIRST price arrives.
        This runs in the WebSocket tick thread — no blocking, no sleeping.

        Timeline example (today's missed 14:10 BUY_PE):
          14:10:00.177 — BUY_PE confirmed, _enter_trade() called
          14:10:00.178 — subscribe BANKNIFTY26MAR53700PE (token 13419266)
          14:10:00.178 — get_price() returns None (no tick yet)
          14:10:00.178 — _pending_entry stored (OLD BEHAVIOUR: trade missed forever)
          14:10:00.4   — First WS tick arrives: price=~190 → _fill_pending() executes
          14:10:00.4   — Trade OPEN, delay ~220ms (same 5-min candle)
        """
        # Clear pending atomically — only the first valid tick fills the trade
        self._pending_entry = None

        if ltp < 5:
            log.warning(f"[BB_STOCH] Pending fill: LTP {ltp:.2f} too low for {pending['symbol']} — discarding")
            return

        # Re-check all risk gates (time elapsed since signal)
        now_time = ts.time()
        if now_time >= dtime(*CFG["session_cutoff"]):
            log.info(f"[BB_STOCH] Pending fill aborted — past session cutoff ({pending['symbol']})")
            return
        if self._is_halted:
            log.info(f"[BB_STOCH] Pending fill aborted — strategy halted ({pending['symbol']})")
            return
        if self._trades_today >= CFG["max_trades_day"]:
            log.info(f"[BB_STOCH] Pending fill aborted — max trades reached ({pending['symbol']})")
            return
        if self._trade:
            log.info(f"[BB_STOCH] Pending fill aborted — trade already open ({pending['symbol']})")
            return

        # Compute delay from signal to fill for logging/analysis
        try:
            delay_ms = (ts - pending["ts"]).total_seconds() * 1000
        except Exception:
            delay_ms = None

        log.info(
            f"[BB_STOCH] PENDING FILL triggered for {pending['symbol']} "
            f"token={pending['token']} LTP={ltp:.2f} "
            f"delay={delay_ms:.0f}ms" if delay_ms is not None else
            f"[BB_STOCH] PENDING FILL triggered for {pending['symbol']}"
        )

        self._execute_fill(
            opt              = pending["opt"],
            sig              = pending["sig"],
            token            = pending["token"],
            tsym             = pending["symbol"],
            ltp              = ltp,
            ts               = ts,
            pending_delay_ms = delay_ms,
        )

    # ----------------------------------------------------------
    # TRADE MANAGEMENT (called on every option WebSocket tick)
    # ----------------------------------------------------------

    def _manage_trade(self, ltp: float, ts: datetime):
        trade = self._trade
        if not trade:
            return

        now_ts = time_module.time()
        if trade.exit_pending and (now_ts - trade.last_exit_ts) < CFG["exit_cooldown"]:
            return

        pnl_pts = ltp - trade.entry
        self._update_trail(trade, ltp, pnl_pts)

        if ltp >= trade.target:
            self._close_trade(ltp, "TARGET")
        elif ltp <= trade.sl:
            self._close_trade(ltp, "SL")

    def _update_trail(self, trade: BBTrade, ltp: float, pnl_pts: float):
        if pnl_pts >= CFG["trail_arm"] and trade.trail_stage == 0:
            new_sl = round(trade.entry, 2)
            if new_sl > trade.sl:
                trade.sl          = new_sl
                trade.trail_stage = 1
                log.info(f"[BB_STOCH] Trail -> BE | SL={trade.sl:.2f}")
        elif pnl_pts > CFG["trail_arm"] and trade.trail_stage >= 1:
            new_stage = int((pnl_pts - CFG["trail_arm"]) / CFG["trail_step"]) + 1
            if new_stage > trade.trail_stage:
                inc               = (new_stage - trade.trail_stage) * CFG["trail_step"]
                trade.sl          = round(trade.sl + inc, 2)
                trade.trail_stage = new_stage
                log.info(f"[BB_STOCH] Trail stage {new_stage} | SL={trade.sl:.2f}")

    def _close_trade(self, ltp: float, reason: str):
        """
        FIX (Bug 3): atomically claim ownership of self._trade at the top.

        Previous code read self._trade outside the lock — two threads could
        both pass the `if not trade` guard simultaneously:
          - WS thread: _force_close("AUTO-SQUAREOFF") at 15:15 via on_tick
          - Main thread: eod_summary → _force_close("EOD-SQUAREOFF") at 15:31

        Both would write to _daily_pnl, append to _results CSV, and call
        unsubscribe — double P&L, duplicate CSV row, double unsubscribe.

        Fix: read AND clear self._trade inside a single lock acquisition at
        the very top. The second caller sees None and returns immediately.
        """
        # FIX (Bug 3): atomically take ownership of the trade under the lock.
        # Only the first caller gets the trade object; the second sees None.
        with self._lock:
            trade = self._trade
            if not trade:
                return
            self._trade = None   # claim it — no other thread can close it now

        trade.exit_pending = True
        trade.last_exit_ts = time_module.time()

        slip     = CFG["slippage"]
        exit_px  = round(ltp - slip, 2)
        pnl_pts  = round(exit_px - trade.entry, 2)
        pnl_rs   = round(pnl_pts * trade.qty, 2)
        rr_act   = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl += pnl_rs

        if reason == "SL":
            self._last_sl_time = time_module.time()

        tag = "[TARGET]" if reason == "TARGET" else ("[SL]" if reason == "SL" else "[EXIT]")
        log.info(
            f"[BB_STOCH] {tag} {trade.option_type} | {trade.symbol} | {reason} | "
            f"Exit={exit_px:.2f} | PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}Rs) | "
            f"RR={rr_act:+.2f} | DayPnL={self._daily_pnl:+.2f}Rs"
        )

        result = {
            "timestamp"  : _now_ist().isoformat(),  # FIX: was UTC on GitHub Actions
            "symbol"     : trade.symbol,
            "opt_type"   : trade.option_type,
            "qty"        : trade.qty,
            "entry"      : trade.entry,
            "exit"       : exit_px,
            "pnl_pts"    : pnl_pts,
            "pnl_rs"     : pnl_rs,
            "rr_actual"  : rr_act,
            "sl_pts"     : trade.sl_pts,
            "tp_pts"     : trade.tp_pts,
            "reason"     : reason,
            "trail_stage": trade.trail_stage,
            "day_pnl_rs" : round(self._daily_pnl, 2),
        }
        _csv_append(CFG["exit_csv"], result)
        self._results.append(result)

        if self._daily_pnl <= -abs(CFG["max_daily_loss"]):
            self._is_halted = True
            log.warning(f"[BB_STOCH] HALTED -- max daily loss breached ({self._daily_pnl:.0f}Rs)")

        # self._trade is already None (cleared at the top of this method).
        # Only need to unsubscribe the option token now.
        with self._lock:
            self._unsubscribe_active()

    def _force_close(self, reason: str):
        """Force-close any open trade (EOD / auto square-off)."""
        # FIX (Bug 7): also discard any pending entry on force-close
        if self._pending_entry:
            log.info(
                f"[BB_STOCH] Discarding pending entry for "
                f"{self._pending_entry.get('symbol', '?')} on {reason}"
            )
            self._pending_entry = None

        with self._lock:
            trade = self._trade
        if not trade:
            return
        ltp = self.get_price(trade.token) or trade.entry
        self._close_trade(ltp, reason)

    def _unsubscribe_active(self):
        """Must be called with self._lock held or after trade cleared."""
        if self._active_token:
            self.unsubscribe_option(self._active_token)
            log.info(f"[BB_STOCH] Unsubscribed token {self._active_token}")
            self._active_token = None

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _compute_sl_tp(self, ltp: float, atr: float
                       ) -> Tuple[float, float, float, float]:
        """Returns (sl_pts, tp_pts, sl_price, tp_price)."""
        sl_pts = float(atr * CFG["atr_sl_mult"]) if atr > 0 else CFG["sl_min"]
        tp_pts = float(atr * CFG["atr_tp_mult"]) if atr > 0 else CFG["tp_min"]
        sl_pts = max(CFG["sl_min"], min(sl_pts, CFG["sl_max"]))
        tp_pts = max(CFG["tp_min"], min(tp_pts, CFG["tp_max"]))
        # On expiry day tighten TP (theta erodes options fast)
        if self._dte == 0:
            tp_pts = max(CFG["tp_min"], tp_pts * 0.75)
        sl_price = round(ltp - sl_pts, 2)
        tp_price = round(ltp + tp_pts, 2)
        log.info(
            f"[BB_STOCH] SL/TP | ATR={atr:.1f} | "
            f"SL={sl_price:.2f}(-{sl_pts:.1f}) TP={tp_price:.2f}(+{tp_pts:.1f})"
        )
        return sl_pts, tp_pts, sl_price, tp_price

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    def _log_signal(self, ts: datetime, sig: dict):
        bb = sig.get("bb", {})
        st = sig.get("stoch", {})
        _csv_append(CFG["signal_csv"], {
            "timestamp"  : ts.isoformat(),
            "action"     : sig.get("action", "HOLD"),
            "blocked_by" : sig.get("blocked_by", ""),
            "close"      : sig.get("close", ""),
            "bb_upper"   : bb.get("upper", ""),
            "bb_lower"   : bb.get("lower", ""),
            "bb_bw_pct"  : bb.get("bw_pct", ""),
            "bb_squeeze" : bb.get("squeeze", ""),
            "stoch_k"    : st.get("k", ""),
            "stoch_d"    : st.get("d", ""),
            "stoch_pk"   : st.get("prev_k", ""),
            "stoch_pd"   : st.get("prev_d", ""),
            "vol_ratio"  : sig.get("vol_ratio", ""),
            "atr"        : sig.get("atr", ""),
            "vwap"       : self._hub.session_vwap.value if self._hub.session_vwap.value is not None else "N/A",
        })

    # ----------------------------------------------------------
    # EOD
    # ----------------------------------------------------------

    def eod_summary(self):
        """Called by MarketHub at 3:30 PM."""
        self._force_close("EOD-SQUAREOFF")

        results = self._results
        total   = len(results)
        wins    = [r for r in results if r["pnl_pts"] > 0]
        losses  = [r for r in results if r["pnl_pts"] <= 0]
        win_pct = len(wins) / total if total else 0
        avg_win  = sum(r["pnl_pts"] for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0
        exp      = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        log.info(
            f"[BB_STOCH] EOD | Trades={total} W={len(wins)} L={len(losses)} "
            f"WinRate={win_pct*100:.1f}% Expect={exp:+.3f}pts "
            f"DayPnL={self._daily_pnl:+.2f}Rs | "
            f"5m-bars={len(self._buf_5m)}"
        )
        if self._blocked_log:
            top = sorted(self._blocked_log.items(), key=lambda x: -x[1])[:5]
            log.info(f"[BB_STOCH] Top blocks: {top}")
