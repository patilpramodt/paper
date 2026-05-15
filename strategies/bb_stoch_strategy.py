"""
strategies/bb_stoch_strategy.py
-------------------------------------------------------------
BBStochStrategy -- BankNifty options scalper using:
  * Bollinger Bands  (BB)   -- identifies breakout / volatility expansion
  * Volume Filter           -- confirms genuine institutional participation
  * EMA trend filter        -- prevents counter-trend entries
  * EMA Seeding             -- previous 10 trading days of 5-min candles
                               loaded at pre-market so EMA20 / EMA50 are
                               accurate from bar 1

SIGNAL LOGIC (all filters must agree before entry):
--------------------------------------------------------------
  CE entry (buy call):
    1. BB Breakout : last close breaks ABOVE upper BB band
       OR BB Bounce : last close crosses back above lower BB band from below
       OR BB Middle : last close crosses ABOVE the BB middle band
    2. Volume     : current bar volume >= VOL_MULT * rolling avg vol
    3. VWAP       : close above session VWAP
    4. EMA        : EMA20 > EMA50 (uptrend context) -- accurate from bar 1
                    because seeds are fetched from Zerodha historical API
                    (previous 10 trading days of 5-min candles) in pre_market()
    5. Session    : only between SESSION_START and ENTRY_CUTOFF

  PE entry (buy put):
    Mirror of the above -- BB break below lower band, bounce below upper
    band, or middle band cross DOWN.

EMA SEEDING:
--------------------------------------------------------------
  pre_market() calls _fetch_ema_seeds() which fetches the previous
  EMA_SEED_DAYS (default 10) trading days of 5-min candles from
  Zerodha's historical_data API in a single call:

      kite.historical_data(
          instrument_token = 260105,     # NSE:NIFTY BANK index
          from_date        = "YYYY-MM-DD 09:15:00",   # 10 trading days ago
          to_date          = "YYYY-MM-DD 15:30:00",   # yesterday
          interval         = "5minute"
      )

  10 trading days × ~75 candles/day ≈ 750 candles total.
  This gives EMA50 a fully-converged seed with zero warmup
  period, compared to the old single-day (75-candle) approach
  which still required ~5 days of warmup for EMA50.

  EMA20 and EMA50 are computed on those candles. The last values
  are stored as self._ema20_seed / self._ema50_seed.

  In evaluate_signal(), compute_ema() uses these seeds as the
  starting point and applies today's candles incrementally:

      result = seed
      for each today_candle_close:
          result = alpha * close + (1 - alpha) * result

  This is mathematically exact -- EMA is recursive so a correct
  seed produces correct values on every subsequent bar without
  warmup.

  Fallback: if the API call fails (holiday, auth error, etc.)
  seeds default to None and EMA falls back to the old
  candle-only behaviour.

TRADE MANAGEMENT (on every option WebSocket tick):
--------------------------------------------------------------
  * SL    = entry_price - ATR_SL_MULT * atr   (anchored to fill price)
  * TP    = entry_price + ATR_TP_MULT * atr   (anchored to fill price)
  * Trail = moves SL to breakeven when profit >= TRAIL_ARM pts,
            then trails every TRAIL_STEP pts after that

FIXES APPLIED:
  Bug 1  -- time.sleep() inside WebSocket callback blocked all tick
             processing. Removed entirely.
  Bug 2  -- _persist_buf cleared before fill; now cleared only on
             genuine fill inside _execute_fill().
  Bug 3  -- _close_trade() race condition between WS thread and main
             thread. Fixed with atomic lock at top.
  Bug 4  -- ATM options not pre-subscribed. Fixed in pre_market().
  Bug 5  -- Only opening ATM pre-subscribed. Now ATM+-200 covered.
  Bug 6  -- Large gap days: spot-based ATM+-400 subscription on
             first candle (~9:20 AM).
  Bug 7  -- LTP unavailable on entry: stored as _pending_entry,
             filled on first tick via on_option_tick().
  Bug 8  -- SL/TP anchored to LTP not fill price. Fixed in BBTrade.
  Bug 9  -- Trail breakeven SL at entry, not entry+slippage. Fixed.
  Bug 10 -- _force_close fired on every tick after 15:15. Fixed with
             _squareoff_done flag.
  Bug 11 -- BBTrade __slots__ missing "order_id". Fixed.
"""

import csv
import logging
import os
import threading
import time as time_module
from collections import deque
from datetime import datetime, date, time as dtime, timezone, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

# IST FIX: GitHub Actions runners are UTC.
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


log = logging.getLogger("strategy.bb_stoch")

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False


# ============================================================
# CONFIG
# ============================================================
CFG = {
    # Bollinger Bands
    "bb_period"          : 20,
    "bb_std"             : 2.0,
    "bb_squeeze_pct"     : 0.002,   # band-width/close < this → squeeze, skip all entries
    "bb_mid_squeeze_pct" : 0.003,   # stricter threshold for middle-band cross only

    # Volume filter
    "vol_avg_period"     : 10,
    "vol_mult"           : 0.85,

    # VWAP
    "vwap_buffer"        : 10.0,    # allow entry within N pts of VWAP

    # Session windows (HH, MM)
    "session_start"      : (9, 45),
    "session_cutoff"     : (15, 0),
    "auto_squareoff"     : (15, 15),

    # Trade management
    "atr_period"         : 14,
    "atr_sl_mult"        : 0.8,
    "atr_tp_mult"        : 2.0,
    "sl_min"             : 20.0,
    "sl_max"             : 50.0,
    "tp_min"             : 30.0,
    "tp_max"             : 120.0,
    "trail_arm"          : 25.0,    # move SL to BE when profit >= this
    "trail_step"         : 12.0,    # then trail every N pts
    "slippage"           : 1.5,
    "exit_cooldown"      : 2.0,

    # Risk
    "max_trades_day"     : 999999,
    "post_sl_cooldown"   : 120,
    "max_daily_loss"     : 6000,
    "quantity"           : 30,

    # Signal persistence
    "persistence"        : 1,

    # Minimum 5-min bars before first signal.
    # With EMA seeding active, EMA20/50 are accurate from bar 1.
    # Remaining constraints: BB needs 7 bars, ATR needs 15 (falls
    # back to sl_min if not ready), Volume needs 11 (bypasses if
    # not ready). 14 bars = 70 min from 9:15 → first signal ~10:25 AM.
    "min_bars"           : 7,

    # Spot-based strike offsets for Bug 6 subscription
    "spot_atm_offsets"   : [0, 200, -200, 400, -400],

    # EMA Seeding -- Zerodha historical API
    # NSE:NIFTY BANK index token (constant, not expiry-specific)
    "ema_seed_token"     : 260105,

    # Number of previous trading days to fetch for EMA seeding.
    # 10 days × ~75 candles/day ≈ 750 candles -- gives EMA50 a
    # fully-converged starting value with zero intraday warmup.
    # Previously this was 1 day (75 candles); increase to 10 for
    # a significantly more accurate seed, especially for EMA50.
    "ema_seed_days"      : 10,

    # CSV paths
    "entry_csv"          : "logs/bb_stoch_entry.csv",
    "exit_csv"           : "logs/bb_stoch_exit.csv",
    "signal_csv"         : "logs/bb_stoch_signals.csv",
}


# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([
                high - low,
                (high - prev).abs(),
                (low  - prev).abs()
            ], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(round(atr.iloc[-1], 2))


def compute_bb(df: pd.DataFrame, period: int, nstd: float) -> dict:
    """
    Bollinger Bands on 'close'.
    Returns current + previous bar values for crossover detection.
    """
    if len(df) < 2:
        return {
            "mid": 0, "upper": 0, "lower": 0,
            "bw_pct": 0, "squeeze": True,
            "prev_mid": 0, "prev_upper": 0,
            "prev_lower": 0, "prev_close": 0,
        }
    close = df["close"].astype(float)
    min_p = max(2, period // 3)
    mid   = close.rolling(period, min_periods=min_p).mean()
    std   = close.rolling(period, min_periods=min_p).std().fillna(0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    m     = float(mid.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    bw    = (u - l) / m if m > 0 else 0
    return {
        "mid"        : round(m, 2),
        "upper"      : round(u, 2),
        "lower"      : round(l, 2),
        "bw_pct"     : round(bw, 5),
        "squeeze"    : bw < CFG["bb_squeeze_pct"],
        "prev_mid"   : round(float(mid.iloc[-2]),   2),
        "prev_upper" : round(float(upper.iloc[-2]), 2),
        "prev_lower" : round(float(lower.iloc[-2]), 2),
        "prev_close" : round(float(close.iloc[-2]), 2),
    }


def compute_vol_ratio(df: pd.DataFrame, avg_period: int) -> float:
    """
    Ratio of last bar's volume vs rolling N-bar average.
    Returns -1.0 sentinel when data unavailable -- caller bypasses filter.
    """
    if "volume" not in df.columns:
        return -1.0
    vol = df["volume"].astype(float)
    if vol.sum() == 0 or vol.isna().all():
        return -1.0
    if len(df) < avg_period + 1:
        return -1.0
    avg     = float(vol.iloc[-(avg_period + 1):-1].mean())
    current = float(vol.iloc[-1])
    if avg <= 0:
        return -1.0
    return round(current / avg, 3)


def compute_ema(df: pd.DataFrame, period: int,
                seed: Optional[float] = None) -> float:
    """
    Exponential Moving Average on 'close'.

    With seed (EMA seeding from previous 10 trading days of 5-min candles):
        Starts from yesterday's closing EMA value and applies today's
        candles one by one using:
            result = alpha * price + (1 - alpha) * result
        where alpha = 2 / (period + 1).
        Mathematically exact from bar 1 -- zero warmup needed.

    Without seed (fallback -- today's candles only):
        Standard pandas ewm(). Needs ~30-50 bars to converge.
    """
    if len(df) < 1:
        return seed if seed is not None else 0.0

    close = df["close"].astype(float)

    if seed is not None:
        alpha  = 2.0 / (period + 1)
        result = float(seed)
        for price in close:
            result = alpha * float(price) + (1.0 - alpha) * result
        return round(result, 2)
    else:
        if len(df) < 2:
            return 0.0
        ema = close.ewm(span=period, adjust=False).mean()
        return float(round(ema.iloc[-1], 2))


# ============================================================
# SIGNAL BUILDER
# ============================================================

def evaluate_signal(df: pd.DataFrame,
                    vwap: Optional[float],
                    ema20_seed: Optional[float] = None,
                    ema50_seed: Optional[float] = None) -> dict:
    """
    Run all filters. Returns:
        action     : "BUY_CE" | "BUY_PE" | "HOLD"
        blocked_by : reason string if HOLD
        bb, vol_ratio, atr, ema20, ema50, close : snapshots for logging

    ema20_seed / ema50_seed: closing EMA values computed from the previous
    10 trading days of 5-min candles, loaded by _fetch_ema_seeds().
    When provided, EMA is exact from bar 1.
    When None, falls back to today's candles only (less accurate early).
    """
    empty = {"action": "HOLD", "blocked_by": "no_data",
             "bb": {}, "vol_ratio": 0, "atr": 0}

    if df is None or len(df) < CFG["min_bars"]:
        return {**empty, "blocked_by": "insufficient_bars"}

    # ---- Indicators ----
    bb        = compute_bb(df, CFG["bb_period"], CFG["bb_std"])
    vol_ratio = compute_vol_ratio(df, CFG["vol_avg_period"])
    atr       = _compute_atr(df, CFG["atr_period"])
    close     = float(df["close"].iloc[-1])

    # EMA trend filter -- seeded so accurate from bar 1
    ema20    = compute_ema(df, 20, seed=ema20_seed)
    ema50    = compute_ema(df, 50, seed=ema50_seed)
    ema_bull = ema20 > ema50
    ema_bear = ema20 < ema50

    base = {
        "bb": bb, "vol_ratio": vol_ratio, "atr": atr,
        "close": close, "ema20": ema20, "ema50": ema50,
    }

    # Gate 1: BB squeeze
    if bb["squeeze"]:
        return {"action": "HOLD", "blocked_by": "bb_squeeze", **base}

    # Gate 2: Volume
    if vol_ratio == -1.0:
        vol_ok = True   # data unavailable -- bypass
        log.debug("[BB_STOCH] Volume data unavailable -- vol filter bypassed")
    else:
        vol_ok = round(vol_ratio, 4) >= CFG["vol_mult"]

    if not vol_ok:
        return {"action": "HOLD", "blocked_by": "volume_low", **base}

    # Gate 3: VWAP bias
    if vwap and vwap > 0:
        above_vwap = close >= (vwap - CFG["vwap_buffer"])
        below_vwap = close <= (vwap + CFG["vwap_buffer"])
    else:
        above_vwap = True
        below_vwap = True

    # ================================================================
    # CE CONDITIONS
    #   Breakout : close crosses ABOVE upper band (this bar vs prev bar)
    #   Bounce   : close crosses back ABOVE lower band
    #   Middle   : close crosses ABOVE middle band
    # ================================================================
    bb_breakout_up  = close > bb["upper"] and bb["prev_close"] <= bb["prev_upper"]
    bb_bounce_up    = close > bb["lower"] and bb["prev_close"] <= bb["prev_lower"]
    bb_mid_cross_up = close > bb["mid"]   and bb["prev_close"] <= bb["prev_mid"]

    ce_bb_trigger = bb_breakout_up or bb_bounce_up or bb_mid_cross_up

    if ce_bb_trigger and above_vwap:
        if not ema_bull:
            return {"action": "HOLD", "blocked_by": "ema_trend_ce", **base}
        if bb_breakout_up:
            mode = "breakout"
        elif bb_bounce_up:
            mode = "bounce"
        else:
            if bb["bw_pct"] < CFG["bb_mid_squeeze_pct"]:
                return {"action": "HOLD", "blocked_by": "bb_mid_squeeze", **base}
            mode = "middle"
        return {"action": "BUY_CE", "blocked_by": "", **base, "mode": mode}

    # ================================================================
    # PE CONDITIONS
    #   Breakout : close crosses BELOW lower band
    #   Bounce   : close crosses back BELOW upper band
    #   Middle   : close crosses BELOW middle band
    # ================================================================
    bb_breakout_dn  = close < bb["lower"] and bb["prev_close"] >= bb["prev_lower"]
    bb_bounce_dn    = close < bb["upper"] and bb["prev_close"] >= bb["prev_upper"]
    bb_mid_cross_dn = close < bb["mid"]   and bb["prev_close"] >= bb["prev_mid"]

    pe_bb_trigger = bb_breakout_dn or bb_bounce_dn or bb_mid_cross_dn

    if pe_bb_trigger and below_vwap:
        if not ema_bear:
            return {"action": "HOLD", "blocked_by": "ema_trend_pe", **base}
        if bb_breakout_dn:
            mode = "breakout"
        elif bb_bounce_dn:
            mode = "bounce"
        else:
            if bb["bw_pct"] < CFG["bb_mid_squeeze_pct"]:
                return {"action": "HOLD", "blocked_by": "bb_mid_squeeze", **base}
            mode = "middle"
        return {"action": "BUY_PE", "blocked_by": "", **base, "mode": mode}

    # Granular block reasons for log analysis
    if ce_bb_trigger:
        if not above_vwap:
            return {"action": "HOLD", "blocked_by": "below_vwap", **base}
        if not ema_bull:
            return {"action": "HOLD", "blocked_by": "ema_trend_ce", **base}
    if pe_bb_trigger:
        if not below_vwap:
            return {"action": "HOLD", "blocked_by": "above_vwap", **base}
        if not ema_bear:
            return {"action": "HOLD", "blocked_by": "ema_trend_pe", **base}

    return {"action": "HOLD", "blocked_by": "no_setup", **base}


# ============================================================
# PAPER TRADE STATE
# ============================================================

class BBTrade:
    __slots__ = (
        "symbol", "token", "option_type", "qty",
        "entry", "sl", "target", "sl_pts", "tp_pts",
        "trail_stage", "spot", "atr", "timestamp",
        "exit_pending", "last_exit_ts", "order_id",
    )

    def __init__(self, symbol, token, opt_type, ltp, qty,
                 spot, sl_pts, tp_pts, atr):
        slip             = CFG["slippage"]
        self.symbol      = symbol
        self.token       = token
        self.option_type = opt_type
        self.qty         = qty
        self.entry       = round(ltp + slip, 2)
        # Bug 8 fix: anchor SL/TP to actual fill price
        self.sl          = round(self.entry - sl_pts, 2)
        self.target      = round(self.entry + tp_pts, 2)
        self.sl_pts      = sl_pts
        self.tp_pts      = tp_pts
        self.trail_stage = 0
        self.spot        = spot
        self.atr         = atr
        self.timestamp   = _now_ist().isoformat()
        self.exit_pending  = False
        self.last_exit_ts  = 0.0
        self.order_id      = ""   # Bug 11 fix


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
    BankNifty options strategy: Bollinger Bands + Volume + EMA trend filter.

    EMA is seeded from the previous 10 trading days of 5-min candles
    (Zerodha historical API) in pre_market(), making EMA20 and EMA50
    fully converged and accurate from bar 1.
    All filters are live and working from the very first candle.
    """

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "BB_STOCH"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # 5-min candle buffer -- size covers EMA50 lookback + headroom
        buf_size = max(CFG["bb_period"], 50) + 20
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

        # Signal persistence
        self._persist_buf: deque = deque(maxlen=CFG["persistence"])

        # Pre-market data
        self._instruments  = None
        self._expiry_date  = None
        self._dte          = 0
        self._session_date = None

        # EMA seeds -- set by _fetch_ema_seeds() in pre_market().
        # None = seeds unavailable, compute_ema() falls back to
        # today's candles only.
        self._ema20_seed: Optional[float] = None
        self._ema50_seed: Optional[float] = None

        # Bug 10 fix
        self._squareoff_done: bool = False

        # Bug 4/5 fix
        self._pre_ce_token: Optional[int] = None
        self._pre_pe_token: Optional[int] = None

        # Bug 6 fix
        self._open_atm_subscribed: bool = False

        # Bug 7 fix
        self._pending_entry: Optional[dict] = None

        # Change B: entry delay (next-candle breakout confirmation)
        self._entry_delay_pending: Optional[dict] = None

        log.info("[BB_STOCH] Strategy initialized")

    # ----------------------------------------------------------
    # PRE-MARKET
    # ----------------------------------------------------------

    def pre_market(self, premarket_data, instruments) -> bool:
        self._instruments  = instruments
        self._expiry_date  = premarket_data.expiry_date
        self._dte          = premarket_data.dte_days
        self._session_date = _now_ist().date()
        self._reset_day()

        log.info(
            f"[BB_STOCH] Pre-market | VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr} Expiry={self._expiry_date} DTE={self._dte}"
        )

        # ── EMA SEEDING ──────────────────────────────────────────────────────
        # Fetch previous 10 trading days of 5-min candles from Zerodha
        # historical API and compute EMA20/50 seeds. Makes EMA accurate
        # from bar 1 of today with full EMA50 convergence (no warmup).
        self._ema20_seed, self._ema50_seed = self._fetch_ema_seeds()

        # ── PRE-SUBSCRIBE STRIKES ─────────────────────────────────────────────
        # Bug 4+5 fix: warm up ATM and ATM+-200 tokens before signal fires.
        ref_price = premarket_data.prev_close or premarket_data.prev_last5m_close
        if ref_price and self._expiry_date:
            atm = get_atm_strike(ref_price)
            for offset in [0, 200, -200]:
                for opt_type in ("CE", "PE"):
                    strike = atm + offset
                    if offset == 0 and self._dte == 0:
                        strike = atm - 100 if opt_type == "CE" else atm + 100
                    tok, sym = instruments.get_option_token(strike, opt_type, self._expiry_date)
                    if tok:
                        self.subscribe_option(tok)
                        if offset == 0:
                            if opt_type == "CE":
                                self._pre_ce_token = tok
                            else:
                                self._pre_pe_token = tok
                        log.info(
                            f"[BB_STOCH] Pre-subscribed {sym} ({tok})"
                            f"{' [ATM]' if offset == 0 else f' [ATM{offset:+d}]'}"
                        )
        else:
            log.warning("[BB_STOCH] No ref price for pre-subscription")

        return True

    # ----------------------------------------------------------
    # EMA SEEDING  (10 trading days)
    # ----------------------------------------------------------

    def _fetch_ema_seeds(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch the previous EMA_SEED_DAYS (default 10) trading days of
        5-min candles from Zerodha historical_data API in a single call,
        then compute EMA20 / EMA50 seeds.

        API call used:
            kite.historical_data(
                instrument_token = 260105,              # NSE:NIFTY BANK index
                from_date        = "YYYY-MM-DD 09:15:00",  # 10 trading days ago
                to_date          = "YYYY-MM-DD 15:30:00",  # yesterday
                interval         = "5minute"
            )

        10 days × ~75 candles/day ≈ 750 candles.
        This gives EMA50 a fully-converged seed value, eliminating the
        ~30-bar intraday warmup that the old single-day fetch required.

        Returns (ema20, ema50) on success, (None, None) on any failure.
        On failure the strategy logs a warning and falls back to computing
        EMA from today's candles only.
        """
        try:
            today      = _now_ist().date()
            seed_days  = CFG["ema_seed_days"]   # default 10

            # Collect the last `seed_days` weekdays going backwards from
            # today. Zerodha handles NSE holidays gracefully (returns fewer
            # candles on partial/holiday days; we only need >= 50 total).
            trading_days = []
            d = today - timedelta(days=1)
            while len(trading_days) < seed_days:
                if d.weekday() < 5:          # 0=Mon … 4=Fri
                    trading_days.append(d)
                d -= timedelta(days=1)

            from_day = trading_days[-1]      # oldest day in the window
            to_day   = trading_days[0]       # most recent (= yesterday)

            from_dt  = f"{from_day} 09:15:00"
            to_dt    = f"{to_day} 15:30:00"

            log.info(
                f"[BB_STOCH] EMA seed: fetching 5-min candles "
                f"from {from_day} to {to_day} "
                f"({seed_days} trading days, ~{seed_days * 75} candles expected, "
                f"token={CFG['ema_seed_token']}) ..."
            )

            candles = self._hub.kite.historical_data(
                instrument_token = CFG["ema_seed_token"],
                from_date        = from_dt,
                to_date          = to_dt,
                interval         = "5minute"
            )

            # Require at least 50 candles (covers EMA50 minimum).
            # Full 10-day window normally yields ~750 but holidays reduce it.
            min_required = 50
            if not candles or len(candles) < min_required:
                log.warning(
                    f"[BB_STOCH] EMA seed: only "
                    f"{len(candles) if candles else 0} candles returned "
                    f"for {from_day} → {to_day} (need >= {min_required}). "
                    f"Possible holidays / auth issue. "
                    f"Falling back to today-candles-only EMA."
                )
                return None, None

            closes = pd.Series([float(c["close"]) for c in candles])
            ema20  = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50  = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])

            log.info(
                f"[BB_STOCH] EMA seeds ready | "
                f"{from_day} → {to_day} ({seed_days} days) | "
                f"bars={len(candles)} | "
                f"EMA20={ema20:.2f}  EMA50={ema50:.2f} | "
                f"Trend={'BULL (EMA20>EMA50)' if ema20 > ema50 else 'BEAR (EMA20<EMA50)'} | "
                f"EMA filter is LIVE and fully converged from bar 1 today"
            )
            return ema20, ema50

        except Exception as e:
            log.warning(
                f"[BB_STOCH] EMA seed fetch failed: {e}. "
                f"EMA will warm up from today's candles only "
                f"(less accurate for first ~30 bars)."
            )
            return None, None

    def _reset_day(self):
        self._daily_pnl           = 0.0
        self._trades_today        = 0
        self._last_sl_time        = 0.0
        self._is_halted           = False
        self._results.clear()
        self._blocked_log.clear()
        self._persist_buf.clear()
        self._squareoff_done      = False
        self._open_atm_subscribed = False
        self._pending_entry       = None
        self._entry_delay_pending = None
        # Reset seeds so stale previous-day values are never used if
        # _fetch_ema_seeds() is not called (e.g. on a reconnect reset).
        # They are set again immediately after this by pre_market().
        self._ema20_seed = None
        self._ema50_seed = None

    # ----------------------------------------------------------
    # Bug 6: Dynamic spot-based subscription on first candle
    # ----------------------------------------------------------

    def _subscribe_spot_atm(self, spot: float):
        """
        Subscribe ATM+-400 based on actual live spot price.
        Called on first 5-min candle (~9:20 AM) to cover large gap days
        where pre_market()'s prev_close-based subscriptions are off-market.
        """
        if not spot or spot <= 0:
            return
        if not self._expiry_date or not self._instruments:
            return

        atm        = get_atm_strike(spot)
        subscribed = 0
        for offset in CFG["spot_atm_offsets"]:
            for opt_type in ("CE", "PE"):
                tok, sym = self._instruments.get_option_token(
                    atm + offset, opt_type, self._expiry_date
                )
                if tok:
                    self.subscribe_option(tok)
                    subscribed += 1
                    log.info(
                        f"[BB_STOCH] Spot-sub {sym} ({tok}) "
                        f"[spot-ATM{offset:+d}]"
                    )
        log.info(
            f"[BB_STOCH] Spot subscription done: "
            f"{subscribed} tokens around ATM={atm} (spot={spot:.0f})"
        )

    # ----------------------------------------------------------
    # TICK CALLBACKS
    # ----------------------------------------------------------

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Index price tick.
        Bug 10 fix: _force_close fires exactly once via _squareoff_done.
        Change B: confirms or expires the entry delay trigger level.
        """
        # Bug 10 fix
        if ts.time() >= dtime(*CFG["auto_squareoff"]) and not self._squareoff_done:
            self._squareoff_done = True
            self._force_close("AUTO-SQUAREOFF")
            return

        # Change B: entry delay confirmation
        confirm = self._entry_delay_pending
        if confirm and not self._trade and not self._is_halted:
            now_time = ts.time()
            action   = confirm["action"]
            trigger  = confirm["trigger_level"]

            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(
                    f"[BB_STOCH] Entry delay expired (past cutoff) "
                    f"| {action} trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
            elif action == "BUY_CE" and price > trigger:
                log.info(
                    f"[BB_STOCH] Entry delay CONFIRMED | BUY_CE "
                    f"| index={price:.2f} > trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])
            elif action == "BUY_PE" and price < trigger:
                log.info(
                    f"[BB_STOCH] Entry delay CONFIRMED | BUY_PE "
                    f"| index={price:.2f} < trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])

    def on_candle(self, candle: dict, ts: datetime):
        """
        Called by MarketHub when a 5-min index candle closes.
        Bug 6 fix: on first candle subscribe ATM+-400 from actual spot.
        Change B: expire unconfirmed entry delay (fake breakout).
        """
        with self._lock:
            self._buf_5m.append(candle)

        # Bug 6 fix: spot-based subscription on first candle of session
        if not self._open_atm_subscribed:
            spot = candle.get("close", 0)
            if spot and spot > 0:
                self._subscribe_spot_atm(spot)
                self._open_atm_subscribed = True

        # Change B: expire any entry delay that survived a full candle
        if self._entry_delay_pending:
            edp = self._entry_delay_pending
            log.info(
                f"[BB_STOCH] Entry delay expired (no index confirmation "
                f"in previous candle) | {edp['action']} trigger={edp['trigger_level']:.2f}"
            )
            self._entry_delay_pending = None

        if self._trade:
            return

        self._evaluate_entry(ts)

    def on_option_tick(self, token: int, price: float, ts: datetime,
                       tick_ts: datetime = None):
        """
        Live option price tick.
        Bug 7 fix: fills pending entry on first tick of subscribed token.
        """
        # Bug 7: pending entry fill
        pending = self._pending_entry
        if pending is not None and token == pending["token"] and not self._trade:
            now_time = ts.time()
            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(f"[BB_STOCH] Pending entry expired (past cutoff) for "
                         f"{pending['symbol']} -- discarding")
                self._pending_entry = None
            elif self._is_halted or self._trades_today >= CFG["max_trades_day"]:
                log.info(f"[BB_STOCH] Pending entry cancelled (risk gate) for "
                         f"{pending['symbol']}")
                self._pending_entry = None
            elif price and price >= 5:
                self._fill_pending(pending, price, ts)
            return

        if token != self._active_token or not self._trade:
            return
        self._manage_trade(price, ts)

    # ----------------------------------------------------------
    # SIGNAL EVALUATION
    # ----------------------------------------------------------

    def _evaluate_entry(self, ts: datetime):
        now_time = ts.time()

        if now_time < dtime(*CFG["session_start"]):
            return
        if now_time >= dtime(*CFG["session_cutoff"]):
            return
        if self._is_halted:
            return
        if self._trades_today >= CFG["max_trades_day"]:
            return
        elapsed = time_module.time() - self._last_sl_time
        if self._last_sl_time > 0 and elapsed < CFG["post_sl_cooldown"]:
            log.info(f"[BB_STOCH] Post-SL cooldown: {CFG['post_sl_cooldown'] - elapsed:.0f}s remaining")
            return

        with self._lock:
            candles = list(self._buf_5m)
        df = self._to_df(candles)
        if df.empty:
            return

        n_bars = len(df)
        if n_bars < CFG["min_bars"]:
            log.debug(
                f"[BB_STOCH] Warming up: {n_bars}/{CFG['min_bars']} bars "
                f"(ready in ~{(CFG['min_bars'] - n_bars) * 5} mins)"
            )
            return

        vwap = self._hub.session_vwap.value

        # Pass EMA seeds -- compute_ema() produces accurate EMA from bar 1
        # when seeds are available, or falls back to today-only EMA.
        sig    = evaluate_signal(
            df, vwap,
            ema20_seed=self._ema20_seed,
            ema50_seed=self._ema50_seed,
        )
        action = sig["action"]

        self._log_signal(ts, sig)

        # Persistence check
        self._persist_buf.append(action)
        confirmed = (
            len(self._persist_buf) >= CFG["persistence"]
            and all(a == action and a != "HOLD" for a in self._persist_buf)
        )

        bb            = sig.get("bb", {})
        vwap_str      = f"{vwap:.2f}" if vwap is not None else "N/A"
        vol_ratio_val = sig.get("vol_ratio", -1)
        vol_str       = f"{vol_ratio_val:.2f}x" if vol_ratio_val >= 0 else "N/A(no-data)"
        seed_tag      = "seeded" if self._ema20_seed is not None else "warming-up"

        log.info(
            f"[BB_STOCH] {action:8s} | "
            f"Close={sig.get('close', 0):.2f} | "
            f"BB=[{bb.get('lower', 0):.1f}~{bb.get('upper', 0):.1f}] "
            f"BW={bb.get('bw_pct', 0)*100:.2f}% | "
            f"Vol={vol_str} | VWAP={vwap_str} | "
            f"EMA20={sig.get('ema20', 0):.2f} EMA50={sig.get('ema50', 0):.2f} [{seed_tag}] | "
            f"Block={sig.get('blocked_by') or 'none'} | Persist={confirmed} | "
            f"PnL={self._daily_pnl:+.1f}Rs T={self._trades_today}/{CFG['max_trades_day']}"
        )

        if not confirmed or action == "HOLD":
            if action == "HOLD":
                bl = sig.get("blocked_by", "")
                self._blocked_log[bl] = self._blocked_log.get(bl, 0) + 1
            return

        # Change B: set trigger level, wait for next-candle index confirmation.
        # on_tick() fires entry the moment index price crosses signal-candle's
        # high (CE) or low (PE). Full next candle without confirmation = expired.
        last_candle   = candles[-1] if candles else {}
        trigger_level = (
            float(last_candle.get("high", 0)) if action == "BUY_CE"
            else float(last_candle.get("low", 0))
        )

        if trigger_level <= 0:
            log.warning(
                f"[BB_STOCH] Entry delay: trigger level unavailable for {action} "
                f"-- entering immediately"
            )
            self._enter_trade(action, sig, ts)
            return

        log.info(
            f"[BB_STOCH] Entry delay SET | {action} | "
            f"trigger={'>' if action == 'BUY_CE' else '<'}{trigger_level:.2f} | "
            f"waiting for next-candle index break via on_tick()"
        )
        self._entry_delay_pending = {
            "action"        : action,
            "trigger_level" : trigger_level,
            "sig"           : sig,
            "ts"            : ts,
        }

    # ----------------------------------------------------------
    # ENTRY
    # ----------------------------------------------------------

    def _enter_trade(self, action: str, sig: dict, ts: datetime):
        """
        Open a new options position.
        Bug 1 fix: no sleep.
        Bug 2 fix: persist_buf cleared only on genuine fill.
        Bug 7 fix: on LTP miss store _pending_entry for fill on first tick.
        """
        opt    = "CE" if action == "BUY_CE" else "PE"
        spot   = sig.get("close", 0)
        atr    = sig.get("atr", 0)
        expiry = self._expiry_date

        if expiry is None:
            log.warning("[BB_STOCH] No expiry date -- skipping entry")
            return

        atm = get_atm_strike(spot)
        if self._dte == 0:
            atm = atm - 100 if opt == "CE" else atm + 100

        token, tsym = self._instruments.get_option_token(atm, opt, expiry)
        if not token:
            log.warning(f"[BB_STOCH] Option not found: ATM={atm} {opt} expiry={expiry}")
            return

        self.subscribe_option(token)
        ltp = self.get_price(token)

        if not ltp or ltp < 5:
            log.warning(
                f"[BB_STOCH] LTP unavailable for {tsym} (token={token}) -- "
                f"storing pending entry, will fill on first tick"
            )
            self._pending_entry = {
                "action" : action,
                "opt"    : opt,
                "sig"    : sig,
                "token"  : token,
                "symbol" : tsym,
                "ts"     : ts,
                "atm"    : atm,
            }
            return

        self._execute_fill(opt, sig, token, tsym, ltp, ts)

    def _execute_fill(self, opt: str, sig: dict, token: int,
                      tsym: str, ltp: float, ts: datetime,
                      pending_delay_ms: Optional[float] = None):
        """Shared fill logic for _enter_trade() and _fill_pending()."""
        atr  = sig.get("atr", 0)
        spot = sig.get("close", 0)

        sl_pts, tp_pts = self._compute_sl_tp(ltp, atr)

        est_spread = ltp * 0.03
        if tp_pts > 0 and (est_spread / tp_pts) > 0.40:
            log.warning(
                f"[BB_STOCH] Spread {est_spread:.1f} too large vs TP {tp_pts:.1f} -- skipping"
            )
            self.unsubscribe_option(token)
            return

        if not self._acquire_slot():
            log.warning("[BB_STOCH] Trade slot blocked -- another live strategy has a position")
            self.unsubscribe_option(token)
            return

        buy_order_id = self._place_buy(tsym, token, CFG["quantity"], ltp)
        if LIVE_MODE and buy_order_id is None:
            self._release_slot()
            log.error(f"[BB_STOCH] BUY order FAILED for {tsym} -- entry aborted")
            self.unsubscribe_option(token)
            return

        trade = BBTrade(
            symbol   = tsym,
            token    = token,
            opt_type = opt,
            ltp      = ltp,
            qty      = CFG["quantity"],
            spot     = spot,
            sl_pts   = sl_pts,
            tp_pts   = tp_pts,
            atr      = atr,
        )
        trade.order_id = buy_order_id

        with self._lock:
            self._trade        = trade
            self._active_token = token

        self._trades_today += 1
        self._persist_buf.clear()   # Bug 2 fix

        rr        = round(tp_pts / sl_pts, 2) if sl_pts else 0
        mode_tag  = "LIVE" if LIVE_MODE else "PAPER"
        delay_str = (f" [pending_delay={pending_delay_ms:.0f}ms]"
                     if pending_delay_ms is not None else "")

        log.info(
            f"[BB_STOCH] [{mode_tag}] ENTRY {opt} | {tsym} | "
            f"Fill={trade.entry:.2f} LTP={ltp:.2f}{delay_str} | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) TP={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} ATR={atr:.1f} | buy_order={buy_order_id} | "
            f"Mode={sig.get('mode', '?')} BB-BW={sig['bb'].get('bw_pct', 0)*100:.2f}%"
        )

        bb = sig.get("bb", {})
        _csv_append(CFG["entry_csv"], {
            "timestamp"        : trade.timestamp,
            "symbol"           : tsym,
            "opt_type"         : opt,
            "qty"              : CFG["quantity"],
            "ltp"              : ltp,
            "fill"             : trade.entry,
            "sl"               : trade.sl,
            "target"           : trade.target,
            "sl_pts"           : sl_pts,
            "tp_pts"           : tp_pts,
            "rr"               : rr,
            "spot"             : spot,
            "atr"              : atr,
            "exec_mode"        : mode_tag,
            "order_id"         : buy_order_id,
            "mode"             : sig.get("mode", "pending" if pending_delay_ms else ""),
            "bb_upper"         : bb.get("upper", ""),
            "bb_mid"           : bb.get("mid", ""),
            "bb_lower"         : bb.get("lower", ""),
            "bb_bw_pct"        : bb.get("bw_pct", ""),
            "ema20"            : sig.get("ema20", ""),
            "ema50"            : sig.get("ema50", ""),
            "ema_seeded"       : self._ema20_seed is not None,
            "vol_ratio"        : sig.get("vol_ratio", ""),
            "vwap"             : (self._hub.session_vwap.value
                                  if self._hub.session_vwap.value is not None else "N/A"),
            "pending_delay_ms" : (round(pending_delay_ms, 0)
                                  if pending_delay_ms is not None else ""),
        })

    # ----------------------------------------------------------
    # Bug 7: Pending entry fill on first WebSocket tick
    # ----------------------------------------------------------

    def _fill_pending(self, pending: dict, ltp: float, ts: datetime):
        """Fill a pending entry on its first live tick. No blocking."""
        self._pending_entry = None

        if ltp < 5:
            log.warning(
                f"[BB_STOCH] Pending fill: LTP {ltp:.2f} too low for "
                f"{pending['symbol']} -- discarding"
            )
            return

        now_time = ts.time()
        if now_time >= dtime(*CFG["session_cutoff"]):
            log.info(f"[BB_STOCH] Pending fill aborted -- past session cutoff ({pending['symbol']})")
            return
        if self._is_halted:
            log.info(f"[BB_STOCH] Pending fill aborted -- strategy halted ({pending['symbol']})")
            return
        if self._trades_today >= CFG["max_trades_day"]:
            log.info(f"[BB_STOCH] Pending fill aborted -- max trades reached ({pending['symbol']})")
            return
        if self._trade:
            log.info(f"[BB_STOCH] Pending fill aborted -- trade already open ({pending['symbol']})")
            return

        try:
            delay_ms = (ts - pending["ts"]).total_seconds() * 1000
        except Exception:
            delay_ms = None

        log.info(
            f"[BB_STOCH] PENDING FILL | {pending['symbol']} "
            f"token={pending['token']} LTP={ltp:.2f}"
            + (f" delay={delay_ms:.0f}ms" if delay_ms is not None else "")
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
    # TRADE MANAGEMENT
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
        """
        Bug 9 fix: trail SL to entry + slippage so exit at trail SL
        yields net P&L = 0 (true breakeven), not -slippage.
        """
        if pnl_pts >= CFG["trail_arm"] and trade.trail_stage == 0:
            new_sl = round(trade.entry + CFG["slippage"], 2)
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
        """Bug 3 fix: atomically claim self._trade at the top."""
        with self._lock:
            trade = self._trade
            if not trade:
                return
            self._trade = None

        trade.exit_pending = True
        trade.last_exit_ts = time_module.time()

        slip    = CFG["slippage"]
        exit_px = round(ltp - slip, 2)
        pnl_pts = round(exit_px - trade.entry, 2)
        pnl_rs  = round(pnl_pts * trade.qty, 2)
        rr_act  = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl += pnl_rs
        if reason == "SL":
            self._last_sl_time = time_module.time()

        # Use retry-aware sell: checks position status before each retry so we
        # never fire a phantom sell against an already-closed position, and never
        # silently drop a real open position after a transient exchange rejection.
        sell_order_id = self._place_sell_with_retry(
            trade.symbol, trade.token, trade.qty, ltp,
            max_retries=3,
        )
        # Release slot AFTER retry loop so no other strategy can enter while we
        # are still trying to close this position.
        self._release_slot()

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        tag      = "[TARGET]" if reason == "TARGET" else ("[SL]" if reason == "SL" else "[EXIT]")
        log.info(
            f"[BB_STOCH] [{mode_tag}] {tag} {trade.option_type} | {trade.symbol} | {reason} | "
            f"Exit={exit_px:.2f} | PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}Rs) | "
            f"RR={rr_act:+.2f} | DayPnL={self._daily_pnl:+.2f}Rs | "
            f"sell_order={sell_order_id}"
        )

        result = {
            "timestamp"   : _now_ist().isoformat(),
            "symbol"      : trade.symbol,
            "opt_type"    : trade.option_type,
            "qty"         : trade.qty,
            "entry"       : trade.entry,
            "exit"        : exit_px,
            "pnl_pts"     : pnl_pts,
            "pnl_rs"      : pnl_rs,
            "rr_actual"   : rr_act,
            "sl_pts"      : trade.sl_pts,
            "tp_pts"      : trade.tp_pts,
            "reason"      : reason,
            "trail_stage" : trade.trail_stage,
            "day_pnl_rs"  : round(self._daily_pnl, 2),
            "exec_mode"   : mode_tag,
            "sell_order"  : sell_order_id,
        }
        _csv_append(CFG["exit_csv"], result)
        self._results.append(result)

        if self._daily_pnl <= -abs(CFG["max_daily_loss"]):
            self._is_halted = True
            log.warning(
                f"[BB_STOCH] HALTED -- max daily loss breached "
                f"({self._daily_pnl:.0f}Rs)"
            )

        with self._lock:
            self._unsubscribe_active()

    def _force_close(self, reason: str):
        if self._pending_entry:
            log.info(
                f"[BB_STOCH] Discarding pending entry for "
                f"{self._pending_entry.get('symbol', '?')} on {reason}"
            )
            self._pending_entry = None
        if self._entry_delay_pending:
            log.info(
                f"[BB_STOCH] Discarding entry delay "
                f"({self._entry_delay_pending.get('action', '?')}) on {reason}"
            )
            self._entry_delay_pending = None
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

    def _compute_sl_tp(self, ltp: float, atr: float) -> Tuple[float, float]:
        sl_pts = float(atr * CFG["atr_sl_mult"]) if atr > 0 else CFG["sl_min"]
        tp_pts = float(atr * CFG["atr_tp_mult"]) if atr > 0 else CFG["tp_min"]
        sl_pts = max(CFG["sl_min"], min(sl_pts, CFG["sl_max"]))
        tp_pts = max(CFG["tp_min"], min(tp_pts, CFG["tp_max"]))
        if self._dte == 0:
            tp_pts = max(CFG["tp_min"], tp_pts * 0.75)
        rr = round(tp_pts / sl_pts, 2) if sl_pts else 0
        log.info(
            f"[BB_STOCH] SL/TP | ATR={atr:.1f} "
            f"SL_pts={sl_pts:.1f} TP_pts={tp_pts:.1f} RR=1:{rr}"
        )
        return sl_pts, tp_pts

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
        _csv_append(CFG["signal_csv"], {
            "timestamp"  : ts.isoformat(),
            "action"     : sig.get("action", "HOLD"),
            "mode"       : sig.get("mode", ""),
            "blocked_by" : sig.get("blocked_by", ""),
            "close"      : sig.get("close", ""),
            "bb_upper"   : bb.get("upper", ""),
            "bb_mid"     : bb.get("mid", ""),
            "bb_lower"   : bb.get("lower", ""),
            "bb_bw_pct"  : bb.get("bw_pct", ""),
            "bb_squeeze" : bb.get("squeeze", ""),
            "ema20"      : sig.get("ema20", ""),
            "ema50"      : sig.get("ema50", ""),
            "ema_seeded" : self._ema20_seed is not None,
            "vol_ratio"  : sig.get("vol_ratio", ""),
            "atr"        : sig.get("atr", ""),
            "vwap"       : (self._hub.session_vwap.value
                            if self._hub.session_vwap.value is not None else "N/A"),
        })

    # ----------------------------------------------------------
    # EOD
    # ----------------------------------------------------------

    def eod_summary(self):
        self._force_close("EOD-SQUAREOFF")

        results  = self._results
        total    = len(results)
        wins     = [r for r in results if r["pnl_pts"] > 0]
        losses   = [r for r in results if r["pnl_pts"] <= 0]
        win_pct  = len(wins) / total if total else 0
        avg_win  = sum(r["pnl_pts"] for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0
        exp      = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        log.info(
            f"[BB_STOCH] EOD | "
            f"Trades={total} W={len(wins)} L={len(losses)} "
            f"WinRate={win_pct*100:.1f}% Expect={exp:+.3f}pts "
            f"DayPnL={self._daily_pnl:+.2f}Rs | "
            f"5m-bars={len(self._buf_5m)} | "
            f"EMA-seeded={self._ema20_seed is not None} | "
            f"EMA-seed-days={CFG['ema_seed_days']}"
        )
        if self._blocked_log:
            top = sorted(self._blocked_log.items(), key=lambda x: -x[1])[:5]
            log.info(f"[BB_STOCH] Top blocks today: {top}")
