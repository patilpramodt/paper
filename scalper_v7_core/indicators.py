# ==============================================================================
# indicators.py  V6 Enhanced Indicator Suite
#
# New in V6:
#   [OK] RSI Z-score (adaptive to rolling mean/std)
#   [OK] RSI slope (3-bar momentum direction)
#   [OK] MACD histogram series (for expansion check)
#   [OK] ATR % of spot (regime gate)
#   [OK] 5-candle structure high/low (for breakout filter)
#   [OK] Regime classification embedded in output
#   [OK] All Wilder's smoothing (RSI, ATR)  same as V5
# ==============================================================================

import pandas as pd
import numpy as np
from scalper_v7_core.logger_setup import log
from scalper_v7_core.config import (
    EMA_FAST_PERIOD, EMA_SLOW_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD,
    RSI_ZSCORE_WINDOW, RSI_SLOPE_LOOKBACK,
    STRUCTURE_LOOKBACK,
    ATR_MIN_PCT, ATR_STRONG_PCT,
    REGIME_EMA_ATR_RATIO, REGIME_EMA_ATR_RATIO_1M,
    ATR_VOL_RATIO_WINDOW, EMA_GAP_5M_FIXED,
)


def compute_indicators(df: pd.DataFrame, timeframe: str = "5m") -> dict:
    """
    Compute all indicators for V6.

    timeframe: "5m" (default) or "1m" — selects which EMA-gap/ATR ratio is
    used for the regime/is_trending classification, so this function agrees
    with _get_5m_regime() in signal_logic.py instead of always using the
    5-minute ratio. Previously this always used REGIME_EMA_ATR_RATIO (0.25)
    even when called on 1-min candles, which both mislabeled the "Regime="
    log field for 1-min and (via is_trending) silently doubled the RSI
    Z-score threshold / blocked entries on quiet days where trend_bias was
    still correctly BULL/BEAR.

    Returns dict with:
        Core:
            ema20, ema50, ema_gap
            rsi14, rsi_z, rsi_slope
            macd_hist, macd_slope, macd_hist_series (last 3)
            atr14, atr_pct, volume_trend
        Structure:
            structure_high (max high last N candles)
            structure_low  (min low  last N candles)
            last_close, last_high, last_low
        Regime:
            is_trending    (bool)
            regime         ("TRENDING" | "SIDEWAYS" | "WEAK")
        Meta:
            bar_count, zscore_ready
    """
    empty = {
        "ema20": 0.0, "ema50": 0.0, "ema_gap": 0.0,
        "rsi14": 50.0, "rsi_z": 0.0, "rsi_slope": 0.0, "rsi_acc": 0.0,
        "macd_hist": 0.0, "macd_slope": 0.0,
        "macd_hist_series": [0.0, 0.0, 0.0],
        "atr14": 0.0, "atr_pct": 0.0, "atr_vol_ratio": 0.8,
        "volume_trend": "flat",
        "structure_high": 0.0, "structure_low": 0.0,
        "last_close": 0.0, "last_high": 0.0, "last_low": 0.0,
        "vwap": 0.0,
        "is_trending": False, "regime": "WEAK",
        "bar_count": 0, "zscore_ready": False,
    }

    if df is None or df.empty or "close" not in df.columns:
        return empty

    df = df.copy().dropna(subset=["close"])
    if len(df) < 10:
        return empty

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)  if "high"   in df.columns else close
    low    = df["low"].astype(float)   if "low"    in df.columns else close
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(
        [0.0] * len(close), index=close.index
    )

    out = dict(empty)
    out["bar_count"]  = len(close)
    out["last_close"] = float(close.iloc[-1])
    out["last_high"]  = float(high.iloc[-1])
    out["last_low"]   = float(low.iloc[-1])

    # 
    # EMA 20 / 50
    # 
    ema20_s = close.ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema50_s = close.ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    out["ema20"]   = float(ema20_s.iloc[-1])
    out["ema50"]   = float(ema50_s.iloc[-1])
    out["ema_gap"] = float(ema20_s.iloc[-1] - ema50_s.iloc[-1])

    # 
    # ATR 14  Wilder's
    # 
    if len(df) >= 15:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1/14, adjust=False).mean()
        out["atr14"] = float(round(atr_series.iloc[-1], 2))

    spot = float(close.iloc[-1])
    if spot > 0 and out["atr14"] > 0:
        out["atr_pct"] = round(out["atr14"] / spot * 100, 4)

    # 
    # ATR Vol-Ratio (ATR vs rolling mean ATR)
    # Checks if volatility is EXPANDING vs its own baseline.
    # Different from atr_pct (absolute level check).
    # vol_ratio > 1.0 means current ATR > average ATR = expanding vol.
    # 
    if len(df) >= ATR_VOL_RATIO_WINDOW + 14:
        prev_close_vr = close.shift(1)
        tr_all = pd.concat([
            high - low,
            (high - prev_close_vr).abs(),
            (low  - prev_close_vr).abs(),
        ], axis=1).max(axis=1)
        atr_full = tr_all.ewm(alpha=1/14, adjust=False).mean()
        rolling_mean_atr = atr_full.iloc[-(ATR_VOL_RATIO_WINDOW):].mean()
        if rolling_mean_atr > 0:
            out["atr_vol_ratio"] = round(out["atr14"] / rolling_mean_atr, 3)

    # 
    # VWAP (Intraday Volume-Weighted Average Price)
    # The only non-price-derivative signal in the system.
    # Addresses collinearity: EMA/MACD/RSI all derive from price
    # smoothing. VWAP is volume-weighted = orthogonal information.
    # Computed fresh from available candles (intraday approximation).
    # 
    if "volume" in df.columns and len(df) >= 5:
        typical_price = (high + low + close) / 3.0
        cum_tp_vol    = (typical_price * volume).cumsum()
        cum_vol       = volume.cumsum()
        vwap_series   = cum_tp_vol / (cum_vol + 1e-9)
        out["vwap"]   = float(round(vwap_series.iloc[-1], 2))

    # 
    # RSI 14  Wilder's smoothing
    # 
    rsi_series = pd.Series([50.0] * len(close), index=close.index)
    if len(close) >= 15:
        delta    = close.diff()
        gain     = delta.clip(lower=0.0)
        loss     = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        out["rsi14"] = float(round(rsi_series.iloc[-1], 2))

    # 
    # RSI Z-Score (adaptive threshold)
    # 
    zscore_ready = len(rsi_series.dropna()) >= RSI_ZSCORE_WINDOW
    out["zscore_ready"] = zscore_ready
    if zscore_ready:
        rsi_window = rsi_series.iloc[-RSI_ZSCORE_WINDOW:]
        rsi_mean   = rsi_window.mean()
        rsi_std    = rsi_window.std()
        if rsi_std > 0.5:   # avoid division by near-zero std on flat days
            out["rsi_z"] = float(round((out["rsi14"] - rsi_mean) / rsi_std, 3))

    # 
    # RSI Slope  3-bar momentum direction
    # 
    if len(rsi_series) >= RSI_SLOPE_LOOKBACK + 1:
        out["rsi_slope"] = float(round(
            rsi_series.iloc[-1] - rsi_series.iloc[-(RSI_SLOPE_LOOKBACK + 1)],
            2,
        ))

    # 
    # RSI Acceleration (2nd derivative)
    # slope_now  = RSI[-1] - RSI[-2]
    # slope_prev = RSI[-2] - RSI[-3]
    # acc        = slope_now - slope_prev
    # Positive acc = momentum speeding up (impulse start signal)
    # 
    if len(rsi_series) >= 4:
        slope_now  = float(rsi_series.iloc[-1] - rsi_series.iloc[-2])
        slope_prev = float(rsi_series.iloc[-2] - rsi_series.iloc[-3])
        out["rsi_acc"] = round(slope_now - slope_prev, 3)

    # 
    # MACD (12, 26, 9)  Standard
    # 
    ema_fast    = close.ewm(span=MACD_FAST,          adjust=False).mean()
    ema_slow    = close.ewm(span=MACD_SLOW,          adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    hist_series = macd_line - signal_line

    out["macd_hist"] = float(round(hist_series.iloc[-1], 4))

    # Last 3 histogram values for expansion check
    if len(hist_series) >= 3:
        out["macd_hist_series"] = [
            float(hist_series.iloc[-3]),
            float(hist_series.iloc[-2]),
            float(hist_series.iloc[-1]),
        ]

    if len(hist_series) >= 2:
        out["macd_slope"] = float(round(
            hist_series.iloc[-1] - hist_series.iloc[-2], 4
        ))

    # 
    # Structure: High/Low of last N candles (excluding current)
    # 
    lookback = min(STRUCTURE_LOOKBACK, len(df) - 1)
    if lookback > 0:
        past = df.iloc[-(lookback + 1):-1]
        out["structure_high"] = float(past["high"].max()) if "high" in df.columns else float(close.iloc[-2])
        out["structure_low"]  = float(past["low"].min())  if "low"  in df.columns else float(close.iloc[-2])

    # 
    # Volume Trend
    # Labels: "bullish", "bearish", "weak", "flat"
    # 
    if len(volume) >= 10 and volume.sum() > 0:
        avg_vol   = volume.iloc[-10:].mean()
        last_vol  = float(volume.iloc[-1])
        vol_ratio = last_vol / (avg_vol + 1e-9)
        if vol_ratio >= 1.10:
            out["volume_trend"] = "bullish" if float(close.iloc[-1]) >= float(close.iloc[-2]) else "bearish"
        elif vol_ratio <= 0.85:
            out["volume_trend"] = "weak"
        else:
            out["volume_trend"] = "flat"

    # 
    # Regime Classification
    #
    # FIX: previously always used REGIME_EMA_ATR_RATIO (the 5-min constant)
    # regardless of which timeframe's candles were passed in, and additionally
    # required atr_pct >= ATR_MIN_PCT before calling anything "TRENDING" even
    # when the EMA-gap/ATR ratio alone (the same test _get_5m_regime uses for
    # trend_bias) said otherwise. That second condition is a DEAD-MARKET check,
    # not a trend-agreement check — bundling it into is_trending caused
    # ind5m['is_trending'] to read False on quiet-but-trending days even while
    # trend_bias correctly read BULL/BEAR, which (a) forced the stricter RSI
    # Z-score threshold and (b) directly blocked entries via the "5m_not_trending"
    # gate in signal_logic.py.
    #
    # Now: ratio matches the timeframe this was computed for, and "TRENDING"
    # is driven purely by the EMA-gap/ATR ratio test — same test as
    # _get_5m_regime() — so is_trending agrees with trend_bias by construction.
    # WEAK (dead market / no ATR) is reported separately.
    #
    # BUG 15 FIX: the "atr <= 0" branch above still disagreed with
    # _get_5m_regime() in one case the previous fix missed — atr14 (5m)
    # requires len(df) >= 15 (75 min of 5-min candles) before it's nonzero,
    # so for ~25 min every single morning (confirmed identical clock window
    # 10:01-10:25 across 4 days of live logs, once len(df_5m) is 10-14)
    # atr14 == 0 here, forcing is_trending=False unconditionally — while
    # _get_5m_regime() falls back to a FIXED gap threshold (EMA_GAP_5M_FIXED,
    # or 3.0 pts on 1m) instead of refusing to call a trend. trend_bias could
    # therefore read BULL/BEAR off a real, large EMA gap while is_trending
    # stayed hard-wired False, blocking every entry via "5m_not_trending" —
    # this single-handedly ate the cleanest trend of the whole 4-day sample
    # (June 24, 10:01-10:18: RSI 67->77, Rz up to 1.35, MACD hist up to +22).
    # Fix: when atr<=0, use the SAME fixed-gap fallback as _get_5m_regime()
    # to decide is_trending, instead of forcing False.
    ratio       = REGIME_EMA_ATR_RATIO if timeframe == "5m" else REGIME_EMA_ATR_RATIO_1M
    fixed_gap   = EMA_GAP_5M_FIXED if timeframe == "5m" else 3.0
    atr = out["atr14"]
    gap = abs(out["ema_gap"])

    if atr > 0:
        trending = (gap / atr) >= ratio
    else:
        trending = gap >= fixed_gap  # mirrors _get_5m_regime()'s own atr<=0 fallback

    if trending:
        out["regime"]      = "TRENDING"
        out["is_trending"] = True
    elif atr > 0 and out["atr_pct"] < ATR_MIN_PCT:
        out["regime"]      = "WEAK"
        out["is_trending"] = False
    elif atr <= 0:
        out["regime"]      = "WEAK"
        out["is_trending"] = False
    else:
        out["regime"]      = "SIDEWAYS"
        out["is_trending"] = False

    return out

