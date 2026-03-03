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
    REGIME_EMA_ATR_RATIO,
    ATR_VOL_RATIO_WINDOW,
)


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all indicators for V6.

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
    atr = out["atr14"]
    gap = abs(out["ema_gap"])

    if atr > 0:
        if out["atr_pct"] < ATR_MIN_PCT:
            out["regime"]      = "WEAK"
            out["is_trending"] = False
        elif gap / atr >= REGIME_EMA_ATR_RATIO:
            out["regime"]      = "TRENDING"
            out["is_trending"] = True
        else:
            out["regime"]      = "SIDEWAYS"
            out["is_trending"] = False

    return out
