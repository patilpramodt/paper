"""
ml_predictor/feature_engineering.py
─────────────────────────────────────
Computes all 31 features from raw 5-min OHLCV data.
Also supports merging VIX data and live OI features from hub.

Features:
  Candle shape   (4): body, range, upper_wick, lower_wick
  Trend EMA      (5): ema9_slope, price_vs_ema9/21, ema9_vs_21, ema21_vs_50
  Momentum       (3): rsi14, macd_line, macd_hist
  Volatility     (3): bb_width, bb_pos, atr14
  Volume         (2): volume_ratio, volume_trend
  VWAP           (1): price_vs_vwap
  OI / PCR       (3): pcr, ce_oi_delta, pe_oi_delta  (filled 0 if unavailable)
  VIX            (2): vix_level, vix_change
  Time encoding  (2): time_sin, time_cos
  Lookback ret   (4): ret_1, ret_2, ret_3, ret_5
  Cross index    (1): cross_ret_1  (Nifty↔BankNifty last return)
  Target         (1): target (next candle green=1 / red=0)

Usage:
    from ml_predictor.feature_engineering import build_features, add_live_oi
    df_feat = build_features(df_ohlcv, df_vix=df_vix)
    X = df_feat.drop("target", axis=1)
    y = df_feat["target"]
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Candle shape
    "candle_body", "candle_range", "upper_wick_ratio", "lower_wick_ratio",
    # Trend
    "ema9_slope", "price_vs_ema9", "price_vs_ema21", "ema9_vs_ema21", "ema21_vs_ema50",
    # Momentum
    "rsi14", "macd_line", "macd_hist",
    # Volatility
    "bb_width", "bb_pos", "atr14",
    # Volume
    "volume_ratio", "volume_trend",
    # VWAP
    "price_vs_vwap",
    # OI/PCR (filled 0 when not available in historical training)
    "pcr", "ce_oi_delta", "pe_oi_delta",
    # VIX
    "vix_level", "vix_change",
    # Time
    "time_sin", "time_cos",
    # Returns
    "ret_1", "ret_2", "ret_3", "ret_5",
    # Cross-index
    "cross_ret_1",
]


def build_features(
    df: pd.DataFrame,
    df_vix: pd.DataFrame = None,
    df_cross: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build full feature set from raw OHLCV data.

    Args:
        df:        Raw 5-min OHLCV (DatetimeIndex, cols=open/high/low/close/volume)
        df_vix:    Optional VIX 5-min data (same index structure). If None → zeros.
        df_cross:  Optional other index OHLCV for cross-index correlation. If None → zeros.

    Returns:
        DataFrame with FEATURE_COLS + 'target'. NaN rows dropped.
    """
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]

    # Strip timezone from index — ensures VIX/cross reindex aligns correctly
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)

    _add_candle_features(d)
    _add_ema_features(d)
    _add_rsi(d)
    _add_macd(d)
    _add_bollinger(d)
    _add_atr(d)
    _add_volume_features(d)
    _add_vwap(d)
    _add_oi_placeholders(d)
    _add_vix_features(d, df_vix)
    _add_time_features(d)
    _add_lookback_returns(d)
    _add_cross_index(d, df_cross)
    _add_target(d)

    # Drop intermediate columns
    _drop_internals(d)
    d.dropna(inplace=True)

    return d


def add_live_oi(features_row: dict, hub, atm_ce_token: int, atm_pe_token: int,
                prev_ce_oi: int = 0, prev_pe_oi: int = 0) -> dict:
    """
    Inject live OI features into a single-row feature dict at prediction time.
    Called from predictor.py every 5-min candle close.

    Args:
        features_row:   Dict of features (from build_live_row())
        hub:            MarketHub instance (has hub.last_oi())
        atm_ce_token:   Current ATM CE Zerodha token
        atm_pe_token:   Current ATM PE Zerodha token
        prev_ce_oi:     CE OI from previous candle (for delta)
        prev_pe_oi:     PE OI from previous candle (for delta)

    Returns:
        Updated features_row with pcr, ce_oi_delta, pe_oi_delta filled.
    """
    ce_oi = hub.last_oi(atm_ce_token)
    pe_oi = hub.last_oi(atm_pe_token)

    features_row["pcr"]         = (pe_oi / ce_oi) if ce_oi > 0 else 1.0
    features_row["ce_oi_delta"] = (ce_oi - prev_ce_oi) / max(prev_ce_oi, 1)
    features_row["pe_oi_delta"] = (pe_oi - prev_pe_oi) / max(prev_pe_oi, 1)

    return features_row


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _add_candle_features(d):
    body       = d["close"] - d["open"]
    rng        = (d["high"] - d["low"]).replace(0, np.nan)
    upper_wick = d["high"] - d[["open", "close"]].max(axis=1)
    lower_wick = d[["open", "close"]].min(axis=1) - d["low"]

    d["candle_body"]      = body / d["open"]
    d["candle_range"]     = (d["high"] - d["low"]) / d["open"]
    d["upper_wick_ratio"] = (upper_wick / rng).clip(-5, 5)
    d["lower_wick_ratio"] = (lower_wick / rng).clip(-5, 5)
    d["is_green"]         = (d["close"] >= d["open"]).astype(int)


def _add_ema_features(d):
    d["_ema9"]  = d["close"].ewm(span=9,  adjust=False).mean()
    d["_ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["_ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    d["ema9_slope"]    = (d["_ema9"] - d["_ema9"].shift(1)) / d["_ema9"].shift(1)
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
    ema12           = d["close"].ewm(span=12, adjust=False).mean()
    ema26           = d["close"].ewm(span=26, adjust=False).mean()
    d["macd_line"]  = (ema12 - ema26) / d["close"]
    macd_signal     = d["macd_line"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]  = d["macd_line"] - macd_signal


def _add_bollinger(d, period: int = 20, std_dev: float = 2.0):
    bb_mid         = d["close"].rolling(period).mean()
    rolling_std    = d["close"].rolling(period).std()
    bb_upper       = bb_mid + std_dev * rolling_std
    bb_lower       = bb_mid - std_dev * rolling_std
    bb_range       = (bb_upper - bb_lower).replace(0, np.nan)

    d["bb_width"]  = bb_range / bb_mid
    d["bb_pos"]    = ((d["close"] - bb_lower) / bb_range).clip(0, 1)
    d["_bb_mid"]   = bb_mid


def _add_atr(d, period: int = 14):
    hl  = d["high"] - d["low"]
    hc  = (d["high"] - d["close"].shift(1)).abs()
    lc  = (d["low"]  - d["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr14"] = (tr.ewm(com=period - 1, min_periods=period).mean() / d["close"])


def _add_volume_features(d, period: int = 20):
    # Nifty/BankNifty index instruments return zero volume from Kite.
    # Use price-range proxy as volume surrogate (captures volatility activity).
    vol = d["volume"].copy()
    if vol.sum() == 0:
        vol = ((d["high"] - d["low"]) * d["close"]).clip(lower=1)
    vol_ma            = vol.rolling(period).mean()
    d["volume_ratio"] = (vol / vol_ma.replace(0, np.nan)).clip(0, 10)
    d["volume_trend"] = vol.pct_change().clip(-5, 5)
    d["_vol_proxy"]   = vol   # store for VWAP


def _add_vwap(d):
    d["_date"] = d.index.date
    tp         = (d["high"] + d["low"] + d["close"]) / 3
    # Use proxy volume if real volume is zero
    vol = d["_vol_proxy"] if "_vol_proxy" in d.columns else d["volume"]
    if vol.sum() == 0:
        vol = ((d["high"] - d["low"]) * d["close"]).clip(lower=1)
    cum_tp_vol         = (tp * vol).groupby(d["_date"]).cumsum()
    cum_vol            = vol.groupby(d["_date"]).cumsum()
    vwap               = cum_tp_vol / cum_vol.replace(0, np.nan)
    d["price_vs_vwap"] = (d["close"] - vwap) / vwap.replace(0, np.nan)


def _add_oi_placeholders(d):
    """In historical training these are zero — filled at live prediction time."""
    d["pcr"]         = 0.0
    d["ce_oi_delta"] = 0.0
    d["pe_oi_delta"] = 0.0


def _add_vix_features(d, df_vix: pd.DataFrame = None):
    if df_vix is not None and not df_vix.empty:
        df_v = df_vix.copy()
        df_v.columns = [c.lower() for c in df_v.columns]
        vix_close = df_v["close"].reindex(d.index, method="ffill")
        d["vix_level"]  = vix_close / 100          # normalise: VIX 15 → 0.15
        d["vix_change"] = vix_close.pct_change().clip(-0.5, 0.5)
    else:
        # Fallback: zeros (model still works, OI/time features carry the load)
        d["vix_level"]  = 0.0
        d["vix_change"] = 0.0


def _add_time_features(d):
    minutes = (d.index.hour - 9) * 60 + (d.index.minute - 15)
    phase   = 2 * np.pi * minutes / 375   # 375 min = 09:15 → 15:30
    d["time_sin"] = np.sin(phase)
    d["time_cos"] = np.cos(phase)


def _add_lookback_returns(d):
    for n in [1, 2, 3, 5]:
        d[f"ret_{n}"] = d["close"].pct_change(n).clip(-0.05, 0.05)


def _add_cross_index(d, df_cross: pd.DataFrame = None):
    if df_cross is not None and not df_cross.empty:
        df_c = df_cross.copy()
        df_c.columns = [c.lower() for c in df_c.columns]
        cross = df_c["close"].reindex(d.index, method="ffill")
        d["cross_ret_1"] = cross.pct_change(1).clip(-0.05, 0.05)
    else:
        d["cross_ret_1"] = 0.0


def _add_target(d):
    """Target = is next candle green? Shift is_green backward by 1."""
    d["target"] = d["is_green"].shift(-1)


def _drop_internals(d):
    drop = ["open", "high", "low", "close", "volume",
            "_ema9", "_ema21", "_ema50", "_bb_mid", "_date", "_vol_proxy", "is_green"]
    d.drop(columns=[c for c in drop if c in d.columns], inplace=True)
