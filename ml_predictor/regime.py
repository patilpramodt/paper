"""
ml_predictor/regime.py
─────────────────────────
Single source of truth for time-of-day buckets and VIX regime.

Old bug fixed here: train.py defined bucket A as 09:15–10:30, while
regime_detector.py's live "open" bucket only covered 09:15–09:45 — so
candles between 09:45 and 10:30 were trained under model A but served
live by model B. There is now exactly ONE function (`get_bucket`) that
both training and live prediction call, so they cannot disagree.

Bucket cutoffs (aligned, as decided):
    A  "open"   09:15 – 10:30
    B  "midday" 10:30 – 14:00
    C  "close"  14:00 – 15:30

VIX regime thresholds are kept separate from buckets (orthogonal axis) and
are intentionally left at provisional defaults — see NOTE at bottom. They
must be re-tuned against the NEW 3-class target's actual backtested
precision/recall before being trusted, not copied from the old binary-target
thresholds.
"""

from datetime import time
import pandas as pd

# ── Bucket cutoffs (single source of truth) ──────────────────────────────────
BUCKET_BOUNDS = {
    "A": (time(9, 15), time(10, 30)),   # open
    "B": (time(10, 30), time(14, 0)),   # midday
    "C": (time(14, 0), time(15, 30)),   # close
}
BUCKET_NAMES = {"A": "open", "B": "midday", "C": "close"}


def get_bucket(ts: pd.Timestamp) -> str:
    """
    Returns 'A', 'B', or 'C' for a given timestamp, using the single
    aligned cutoff table above. Used identically by train.py (vectorized
    via get_bucket_series) and predictor.py (single timestamp at live time).
    """
    t = ts.time()
    for bucket, (start, end) in BUCKET_BOUNDS.items():
        if start <= t < end:
            return bucket
    # Outside 09:15–15:30 (e.g. exactly 15:30 close print) -> fold into C
    return "C"


def get_bucket_series(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized bucket assignment for a full DataFrame index (training)."""
    return pd.Series([get_bucket(ts) for ts in index], index=index, name="bucket")


# ── VIX regime (provisional — re-tune after new-target backtest) ────────────
# NOTE: these thresholds were tuned for the OLD binary next-candle target.
# They must NOT be assumed valid for the new 4-candle ±0.15% 3-class target.
# Treat QUIET/NORMAL/VOLATILE as labels only until regime-level precision/
# recall is measured on the new target and thresholds are re-set deliberately.
VIX_REGIME_BOUNDS = {
    "QUIET":    (0,    13),
    "NORMAL":   (13,   20),
    "VOLATILE": (20, 9999),
}


def get_vix_regime(vix_level: float) -> str:
    for regime, (lo, hi) in VIX_REGIME_BOUNDS.items():
        if lo <= vix_level < hi:
            return regime
    return "VOLATILE"
