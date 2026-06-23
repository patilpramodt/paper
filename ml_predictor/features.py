"""
ml_predictor/features.py
──────────────────────────
Feature engineering for the rebuilt ML pipeline.

Computes 23 features from raw 5-min OHLC data (volume and OI removed —
see rationale below). Does NOT compute the target; call targets.add_target()
separately and join.

Removed vs. old build, and why:
    OI features (pcr, ce_oi_delta, pe_oi_delta)
        → Always 0.0 in historical training (no candle-level OI in the
          CSVs), real-valued only at live inference. Model never learned
          what these mean. Dropped entirely.
    Volume features (volume_ratio, volume_trend)
        → Source 'volume' column is 0 for every row in the index CSVs
          (Kite returns 0 for NIFTY50/BANKNIFTY tokens). The old (high-low)*close
          proxy is just a rescaled ATR and duplicates atr14/candle_range.
          Dropped entirely.

VIX and cross-index inputs:
    This module's build_features() always expects REAL multi-row VIX/cross
    DataFrames (or None). It does not know or care whether the caller is
    training (full history) or live (last ~30 candles) — that distinction
    lives in predictor.py, not here. Passing a single-row / scalar-broadcast
    DataFrame here will silently produce vix_change == 0 / cross_ret_1 == 0,
    so callers MUST pass real historical series.

Usage:
    from ml_predictor.features import build_features, FEATURE_COLS
    df_feat = build_features(df_ohlc, df_vix=df_vix, df_cross=df_cross)
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    # Candle shape (4)
    "candle_body", "candle_range", "upper_wick_ratio", "lower_wick_ratio",
    # Trend (5)
    "ema9_slope", "price_vs_ema9", "price_vs_ema21", "ema9_vs_ema21", "ema21_vs_ema50",
    # Momentum (3)
    "rsi14", "macd_line", "macd_hist",
    # Volatility (3)
    "bb_width", "bb_pos", "atr14",
    # VWAP (1)
    "price_vs_vwap",
    # VIX (2)
    "vix_level", "vix_change",
    # Time (2)
    "time_sin", "time_cos",
    # Lookback returns (4)
    "ret_1", "ret_2", "ret_3", "ret_5",
    # Cross-index (1)
    "cross_ret_1",
]
# 25 features total


def build_features(
    df: pd.DataFrame,
    df_vix: pd.DataFrame = None,
    df_cross: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build the full feature set from raw OHLC data. Does NOT add a target —
    call targets.add_target() on the result (or before/after, order doesn't
    matter since they touch disjoint columns) and dropna on 'target' yourself.

    Args:
        df:        Raw 5-min OHLC (DatetimeIndex, cols include open/high/low/close).
                   A 'volume' column is tolerated but ignored.
        df_vix:    Real VIX OHLC history, same-ish index range as df. If None,
                   vix_level/vix_change are filled 0 (degraded mode, log this).
        df_cross:  Real cross-index OHLC history (Nifty for BankNifty model,
                   and vice versa). If None, cross_ret_1 filled 0.

    Returns:
        DataFrame with FEATURE_COLS. Rows with NaN (indicator warm-up period
        at the start of history) are dropped.
    """
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]

    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)

    _add_candle_features(d)
    _add_ema_features(d)
    _add_rsi(d)
    _add_macd(d)
    _add_bollinger(d)
    _add_atr(d)
    _add_vwap(d)
    _add_vix_features(d, df_vix)
    _add_time_features(d)
    _add_lookback_returns(d)
    _add_cross_index(d, df_cross)

    _drop_internals(d)
    d.dropna(inplace=True)

    return d


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_candle_features(d):
    body = d["close"] - d["open"]
    rng  = (d["high"] - d["low"]).replace(0, np.nan)
    upper_wick = d["high"] - d[["open", "close"]].max(axis=1)
    lower_wick = d[["open", "close"]].min(axis=1) - d["low"]

    d["candle_body"]      = body / d["open"]
    d["candle_range"]     = (d["high"] - d["low"]) / d["open"]
    d["upper_wick_ratio"] = (upper_wick / rng).clip(-5, 5)
    d["lower_wick_ratio"] = (lower_wick / rng).clip(-5, 5)


def _add_ema_features(d):
    d["_ema9"]  = d["close"].ewm(span=9,  adjust=False).mean()
    d["_ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["_ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    d["ema9_slope"]     = (d["_ema9"] - d["_ema9"].shift(1)) / d["_ema9"].shift(1)
    d["price_vs_ema9"]  = (d["close"] - d["_ema9"])  / d["_ema9"]
    d["price_vs_ema21"] = (d["close"] - d["_ema21"]) / d["_ema21"]
    d["ema9_vs_ema21"]  = (d["_ema9"]  - d["_ema21"]) / d["_ema21"]
    d["ema21_vs_ema50"] = (d["_ema21"] - d["_ema50"]) / d["_ema50"]


def _add_rsi(d, period: int = 14):
    delta    = d["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi14"] = (100 - (100 / (1 + rs))) / 100   # normalised to [0,1]


def _add_macd(d):
    ema12          = d["close"].ewm(span=12, adjust=False).mean()
    ema26          = d["close"].ewm(span=26, adjust=False).mean()
    d["macd_line"] = (ema12 - ema26) / d["close"]
    macd_signal    = d["macd_line"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd_line"] - macd_signal


def _add_bollinger(d, period: int = 20, std_dev: float = 2.0):
    bb_mid      = d["close"].rolling(period).mean()
    rolling_std = d["close"].rolling(period).std()
    bb_upper    = bb_mid + std_dev * rolling_std
    bb_lower    = bb_mid - std_dev * rolling_std
    bb_range    = (bb_upper - bb_lower).replace(0, np.nan)

    d["bb_width"] = bb_range / bb_mid
    d["bb_pos"]   = ((d["close"] - bb_lower) / bb_range).clip(0, 1)


def _add_atr(d, period: int = 14):
    hl = d["high"] - d["low"]
    hc = (d["high"] - d["close"].shift(1)).abs()
    lc = (d["low"]  - d["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr14"] = (tr.ewm(com=period - 1, min_periods=period).mean() / d["close"])


def _add_vwap(d):
    """
    Session VWAP using true price-range as weight (NOT volume — volume is
    always 0 for these instruments, see module docstring). This is a
    deliberate change from the old build: rather than disguise a fake
    "volume proxy" as VWAP weighting, we use typical-price equal-weighting
    accumulated per session, which is the standard fallback when no real
    volume exists.
    """
    date_key = d.index.date
    tp = (d["high"] + d["low"] + d["close"]) / 3
    cum_tp  = tp.groupby(date_key).cumsum()
    cum_n   = pd.Series(1, index=d.index).groupby(date_key).cumsum()
    vwap = cum_tp / cum_n
    d["price_vs_vwap"] = (d["close"] - vwap) / vwap.replace(0, np.nan)


def _add_vix_features(d, df_vix: pd.DataFrame = None):
    if df_vix is not None and not df_vix.empty:
        df_v = df_vix.copy()
        df_v.columns = [c.lower() for c in df_v.columns]
        if df_v.index.tz is not None:
            df_v.index = df_v.index.tz_localize(None)
        vix_close = df_v["close"].reindex(d.index, method="ffill")
        d["vix_level"]  = vix_close / 100
        d["vix_change"] = vix_close.pct_change().clip(-0.5, 0.5)
    else:
        d["vix_level"]  = 0.0
        d["vix_change"] = 0.0


def _add_time_features(d):
    """
    Cyclical (sin/cos) encoding of time-of-day, normalized over the trading
    session (09:15-15:30, 375 minutes wide).

    Bug fixed here: dividing by 375 (the session WIDTH) means the 15:30
    candle (minutes=375) maps to phase=2*pi, which is mathematically
    identical to phase=0 (minutes=0, i.e. 09:15) — sin=0, cos=1 for both.
    The model would see session close and session open as the same moment.
    Confirmed: in the historical CSVs no day actually reaches a 15:30
    candle (NSE's last 5-min candle is 15:25-15:30, sourced under the
    15:25 timestamp; 2,083/2,087 days end at 15:25), so this hasn't been
    triggered by training data so far. But predictor.py fetches live data
    directly from Kite rather than these CSVs, and relying on "the bug
    happens to not be reachable today" is fragile — a single off-by-one
    candle from Kite, a session-hours change, or a different instrument's
    quirk would silently corrupt this feature with no visible error.
    Dividing by 376 instead of 375 means minutes=375 maps to phase
    slightly less than 2*pi, distinct from minutes=0 — closing the gap
    entirely rather than relying on it never being hit.
    """
    minutes = (d.index.hour - 9) * 60 + (d.index.minute - 15)
    phase   = 2 * np.pi * minutes / 376
    d["time_sin"] = np.sin(phase)
    d["time_cos"] = np.cos(phase)


def _add_lookback_returns(d):
    for n in [1, 2, 3, 5]:
        d[f"ret_{n}"] = d["close"].pct_change(n).clip(-0.05, 0.05)


def _add_cross_index(d, df_cross: pd.DataFrame = None):
    if df_cross is not None and not df_cross.empty:
        df_c = df_cross.copy()
        df_c.columns = [c.lower() for c in df_c.columns]
        if df_c.index.tz is not None:
            df_c.index = df_c.index.tz_localize(None)
        cross = df_c["close"].reindex(d.index, method="ffill")
        d["cross_ret_1"] = cross.pct_change(1).clip(-0.05, 0.05)
    else:
        d["cross_ret_1"] = 0.0


def _drop_internals(d):
    drop = ["open", "high", "low", "close", "volume", "_ema9", "_ema21", "_ema50"]
    d.drop(columns=[c for c in drop if c in d.columns], inplace=True)
