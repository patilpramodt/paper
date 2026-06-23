"""
ml_predictor/targets.py
─────────────────────────
Target construction for the 4-candle-forward, ±0.15% deadband classifier.

Target definition (locked):
    fwd_ret = (close[t+N] - close[t]) / close[t]   where N = FORWARD_CANDLES (4)
    label   = UP    if fwd_ret >  DEADBAND   (+0.15%)
              DOWN  if fwd_ret < -DEADBAND
              FLAT  otherwise

Session safety (fixes the old bug):
    The shift is computed PER TRADING DAY via groupby(date).shift(-N).
    The last N candles of every day get NaN and are dropped — they are
    NEVER allowed to read into the next day's open.

Usage:
    from ml_predictor.targets import add_target, TARGET_CLASSES

    df = add_target(df_ohlcv)          # adds 'fwd_ret' and 'target' columns
    df = df.dropna(subset=["target"])  # drop session-boundary + tail rows
"""

import numpy as np
import pandas as pd

# ── Locked config ────────────────────────────────────────────────────────────
FORWARD_CANDLES = 4        # 4 candles × 5 min = 20 minutes ahead
DEADBAND        = 0.0015   # ±0.15%

# Class labels (also the encoding used for XGBoost / LSTM training)
DOWN, FLAT, UP = 0, 1, 2
TARGET_CLASSES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}
CLASS_NAMES_TO_INT = {v: k for k, v in TARGET_CLASSES.items()}


def add_target(
    df: pd.DataFrame,
    forward_candles: int = FORWARD_CANDLES,
    deadband: float = DEADBAND,
) -> pd.DataFrame:
    """
    Adds 'fwd_ret' (raw forward return) and 'target' (0/1/2 class) columns.

    Rows where the forward window crosses a session boundary, or falls in
    the last `forward_candles` rows of the dataset, get NaN in both columns
    and must be dropped by the caller before training.

    Args:
        df:               DataFrame with DatetimeIndex and a 'close' column.
        forward_candles:  How many candles ahead to measure return (default 4).
        deadband:         ± threshold for UP/DOWN vs FLAT (default 0.0015 = 0.15%).

    Returns:
        Copy of df with 'fwd_ret' and 'target' columns added.
    """
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]

    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)

    date_key = d.index.date

    # groupby(date) ensures shift(-N) never reads across a session boundary —
    # any row whose +N window would land on the next day gets NaN here.
    fwd_close = d.groupby(date_key)["close"].shift(-forward_candles)

    d["fwd_ret"] = (fwd_close - d["close"]) / d["close"]

    d["target"] = pd.Series(np.nan, index=d.index)
    d.loc[d["fwd_ret"] > deadband, "target"]  = UP
    d.loc[d["fwd_ret"] < -deadband, "target"] = DOWN
    d.loc[(d["fwd_ret"] >= -deadband) & (d["fwd_ret"] <= deadband), "target"] = FLAT

    return d


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Quick diagnostic: counts + % for each target class. Drops NaN first."""
    valid = df["target"].dropna().astype(int)
    counts = valid.value_counts().reindex([DOWN, FLAT, UP]).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(1)
    out = pd.DataFrame({"count": counts, "pct": pct})
    out.index = out.index.map(TARGET_CLASSES)
    return out


def class_weights(df: pd.DataFrame) -> dict:
    """
    Inverse-frequency class weights for the 3-class target.
    Used as sample_weight multipliers so UP/DOWN (the minority, tradeable
    classes) aren't drowned out by FLAT during training.

    Returns: {0: w_down, 1: w_flat, 2: w_up}

    A class with ZERO samples in this slice gets weight 0.0, not an
    inflated phantom weight. The previous version used counts.get(cls, 1)
    — defaulting a missing class to a count of 1 — which produced a huge,
    arbitrary weight (e.g. ~167x a real minority class's weight on a
    500-row slice) for a class that has no actual rows to apply it to.
    That weight is harmless in the SAME slice it's computed on (nothing
    to multiply it against), but it becomes dangerous the moment a nearly-
    identical slice (e.g. next week's fine-tune window) has even 1-2 real
    samples of that class — they'd then receive a weight derived from an
    arbitrary floor, not from any real frequency information. Weight 0.0
    is the honest answer: "this class wasn't observed in this window, so
    weighting it is meaningless," rather than fabricating a number that
    happens to look large.

    A subtler bug introduced by that fix: dividing by a hardcoded
    n_classes=3 always — even when only 2 classes are actually present —
    underweights the present minority class. Example: DOWN=10, FLAT=300,
    UP=0. Dividing by 3 gives DOWN a weight of 10.3; but with UP
    contributing nothing real to this slice, DOWN's weight relative to
    FLAT should be computed as if there were only 2 classes splitting the
    "inverse frequency" mass, giving 15.5 — about 50% higher. Using a
    hardcoded 3 silently underweights DOWN by treating UP's absence as if
    it still occupied a third of the weighting mass. Fixed by dividing by
    n_present (the count of classes with at least 1 real sample), not a
    constant 3.
    """
    valid = df["target"].dropna().astype(int)
    counts = valid.value_counts()
    n = len(valid)
    n_present = (counts > 0).sum()  # only classes actually observed in this slice
    weights = {}
    for cls in [DOWN, FLAT, UP]:
        c = counts.get(cls, 0)
        weights[cls] = (n / (n_present * c)) if c > 0 else 0.0
    return weights
