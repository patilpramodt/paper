"""
ml_predictor/targets.py
─────────────────────────
Target construction — ATR-scaled TRIPLE-BARRIER labeling.

Replaces the old fixed-deadband, endpoint-only target
(fwd_ret = close[t+N] vs close[t], compared to a flat ±0.15%).
That method only ever looked at where price landed AFTER exactly N
candles — it ignored the whole path in between, and used a single
fixed % band regardless of how volatile that particular session was.
Both of those are among the drivers of the FLAT-prediction dominance
problem (see the model redesign notes).

Target definition (new):
    From each candle t, draw three barriers:
        upper = close[t] + ATR_MULTIPLIER * atr[t]
        lower = close[t] - ATR_MULTIPLIER * atr[t]
        time  = t + FORWARD_CANDLES              (unchanged: 4 candles / 20 min)

    Walk forward candle-by-candle from t+1 .. t+FORWARD_CANDLES and label
    whichever barrier is touched FIRST:
        UP    if a future candle's HIGH reaches `upper` before any future
              candle's LOW reaches `lower`
        DOWN  if a future candle's LOW reaches `lower` first
        FLAT  if neither barrier is touched before the time barrier

    `atr[t]` is a CAUSAL (past-only) ATR — it only uses candles up to and
    including t, so there is no lookahead leak into the label it's used
    to build.

    Same-candle tie-break: if a single future candle's range touches BOTH
    barriers (a wide-range candle), we don't have tick-level ordering to
    know which was touched first. We break the tie using that candle's
    own open->close direction: bullish candle (close >= open) -> UP,
    bearish candle (close < open) -> DOWN. This is a documented heuristic,
    not exact — 5-min OHLC bars don't carry intra-candle sequencing.

Why this fixes what the old method got wrong:
    1. Uses the WHOLE path, not just the t+N endpoint. A candle that
       spikes 0.3% at t+2 and drifts back to +0.05% by t+4 was labeled
       FLAT before (endpoint only). It's labeled UP now — correctly,
       since a real tradeable move did happen along the way.
    2. Barriers SCALE with recent volatility (ATR) instead of a fixed
       0.15%. Calm sessions get tighter barriers (more UP/DOWN labels
       survive), volatile sessions get wider ones (noise doesn't get
       mislabeled as a "move") — this directly targets class-balance
       across different volatility regimes, which a flat % band cannot.

Session safety (unchanged from the old file):
    A row is only labeled if it has AT LEAST `forward_candles` more
    candles remaining in the SAME trading day. Rows near the end of the
    day (not enough future candles left in-session) get NaN and must be
    dropped by the caller — exactly like before. The barrier walk NEVER
    reads into the next day's open.

Config change note (read before touching train.py / live_tracker.py):
    DEADBAND is kept only as a legacy constant (for any code that still
    imports it for logging / display) — it is NOT the live decision
    boundary anymore; ATR_MULTIPLIER is. `live_tracker.py` currently
    recomputes the OLD fixed-deadband rule independently to score live
    predictions (see its DEADBAND import) — that file needs a matching
    update, or it will silently score predictions against a different
    label definition than the one the model was actually trained on.
    Flagging this explicitly so it doesn't slip through unnoticed.

Usage:
    from ml_predictor.targets import add_target, TARGET_CLASSES

    df = add_target(df_ohlcv)          # adds 'atr', 'upper_barrier',
                                        # 'lower_barrier', 'fwd_ret', and
                                        # 'target' columns
    df = df.dropna(subset=["target"])  # drop session-boundary + tail rows
"""

import numpy as np
import pandas as pd

# ── Locked config ────────────────────────────────────────────────────────────
FORWARD_CANDLES = 4        # time barrier: 4 candles x 5 min = 20 minutes ahead
ATR_PERIOD      = 14       # candles used for the causal ATR
ATR_MULTIPLIER  = 1.5      # upper/lower barrier = close +/- ATR_MULTIPLIER * atr

# Legacy constant only — see "Config change note" above. No longer used to
# decide labels; kept so anything still importing it for logging doesn't
# break, and as a documented reference point for the band the old model used.
DEADBAND = 0.0015   # +/-0.15%

# Class labels (also the encoding used for XGBoost / LSTM training)
DOWN, FLAT, UP = 0, 1, 2
TARGET_CLASSES = {DOWN: "DOWN", FLAT: "FLAT", UP: "UP"}
CLASS_NAMES_TO_INT = {v: k for k, v in TARGET_CLASSES.items()}


def _causal_atr(d: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Standard True-Range-based ATR (EWM smoothed), computed using ONLY past
    data up to and including each row — never a centered or forward-
    looking window. This is what makes it safe to use as the barrier
    width for a label built from the SAME row.

    Returns a Series aligned to d.index, in absolute price units (points),
    not normalized by close — because the barriers it feeds are absolute
    price levels (close +/- multiplier * atr).
    """
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def add_target(
    df: pd.DataFrame,
    forward_candles: int = FORWARD_CANDLES,
    atr_multiplier: float = ATR_MULTIPLIER,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """
    Adds 'atr', 'upper_barrier', 'lower_barrier', 'fwd_ret', and 'target'
    (0/1/2 class) columns using ATR-scaled triple-barrier labeling.

    Rows where the forward window crosses a session boundary, falls in
    the last `forward_candles` rows of the dataset, or land in the ATR
    warm-up period (first `atr_period` candles of the whole series) get
    NaN in 'target' and must be dropped by the caller before training.

    Args:
        df:               DataFrame with DatetimeIndex and
                           open/high/low/close columns.
        forward_candles:  Time barrier — max candles ahead to check
                           (default 4 = 20 min).
        atr_multiplier:   Upper/lower barrier distance as a multiple of
                           ATR (default 1.5).
        atr_period:       Candles used for the causal ATR (default 14).

    Returns:
        Copy of df with the columns described above added.
    """
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]

    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)

    date_key = pd.Series(d.index.date, index=d.index)

    atr = _causal_atr(d, atr_period)
    d["atr"] = atr

    close = d["close"].to_numpy()
    high = d["high"].to_numpy()
    low = d["low"].to_numpy()
    open_ = d["open"].to_numpy()
    atr_vals = atr.to_numpy()

    upper = close + atr_multiplier * atr_vals
    lower = close - atr_multiplier * atr_vals
    d["upper_barrier"] = upper
    d["lower_barrier"] = lower

    n = len(d)

    # A row is valid only if it has `forward_candles` MORE rows after it
    # in the SAME trading day (never read across a session boundary —
    # same safety contract as the old file's groupby(date).shift(-N)).
    remaining_in_day = date_key.groupby(date_key).cumcount(ascending=False).to_numpy()
    valid = remaining_in_day >= forward_candles
    valid &= ~np.isnan(atr_vals)  # exclude ATR warm-up rows too

    # target defaults to FLAT for every valid row; overwritten below the
    # moment an UP/DOWN barrier touch is found. Invalid rows -> NaN.
    target = np.full(n, FLAT, dtype=float)
    target[~valid] = np.nan

    # fwd_ret is a diagnostic column: the return realized at whichever
    # candle actually decided the label (or at the full time-barrier
    # horizon for rows that stayed FLAT). It is NOT used to derive the
    # label itself — the barrier walk below is.
    fwd_ret = np.full(n, np.nan)
    last_idx = np.arange(n) + forward_candles
    last_idx_clipped = np.clip(last_idx, 0, n - 1)
    fwd_ret[valid] = (close[last_idx_clipped[valid]] - close[valid]) / close[valid]

    decided = ~valid  # invalid rows are already "settled" (NaN), skip them

    for k in range(1, forward_candles + 1):
        fut = np.arange(n) + k
        in_bounds = fut < n
        active = valid & (~decided) & in_bounds
        if not active.any():
            continue

        fut_idx = fut[active]
        hit_up = high[fut_idx] >= upper[active]
        hit_down = low[fut_idx] <= lower[active]

        both = hit_up & hit_down
        # tie-break on the future candle's own open->close direction
        tie_up = both & (close[fut_idx] >= open_[fut_idx])
        tie_down = both & ~tie_up

        up_mask = (hit_up & ~hit_down) | tie_up
        down_mask = (hit_down & ~hit_up) | tie_down

        active_positions = np.where(active)[0]
        pos_up = active_positions[up_mask]
        pos_down = active_positions[down_mask]

        if pos_up.size:
            target[pos_up] = UP
            fwd_ret[pos_up] = (close[fut[pos_up]] - close[pos_up]) / close[pos_up]
            decided[pos_up] = True
        if pos_down.size:
            target[pos_down] = DOWN
            fwd_ret[pos_down] = (close[fut[pos_down]] - close[pos_down]) / close[pos_down]
            decided[pos_down] = True

    d["fwd_ret"] = fwd_ret
    d["target"] = target

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
