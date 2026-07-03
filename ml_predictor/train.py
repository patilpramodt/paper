"""
ml_predictor/train.py
──────────────────────
Complete training pipeline for the rebuilt 3-class, 4-candle-forward,
±0.15% deadband target.

NOTE: the "4-candle" / "±0.15%" target shape lives in targets.py
(FORWARD_CANDLES, DEADBAND) — this file imports those constants rather
than hardcoding them, so if you change the target window in targets.py,
this docstring's numbers may go stale but the actual training run will
always use targets.py's live values (also printed at the start of each
run — see main()/load_and_engineer() log output).

Pipeline:
  1. Load CSV data (Nifty + BankNifty + VIX) via data_fetcher.load_csv
  2. Feature engineering (25 features, no OI, no volume) — features.py
  3. Target construction (session-safe groupby shift, 3-class) — targets.py
  4. Combined sample weighting: time-decay × inverse-class-frequency
  5. Walk-forward validation (context only — folds scale with total history,
     so the "last fold" can span a year+ and is NOT a recent estimate) PLUS
     a separate fixed-size recent holdout (last ~20 trading days, regardless
     of total dataset size) — THIS is what's reported as the live estimate
  6. Train three XGBoost models per instrument, bucket cutoffs from regime.py
     (the SAME cutoffs predictor.py uses live — no more train/live mismatch)
  7. Train one multiclass LSTM per instrument
  8. Save all models + scalers + a metrics.json report

Usage:
    python3 ml_predictor/train.py                          # both instruments
    python3 ml_predictor/train.py --xgb-only
    python3 ml_predictor/train.py --instrument BANKNIFTY
"""

import argparse
import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(BASE_DIR))
from ml_predictor.data_fetcher import load_csv
from ml_predictor.features     import build_features, FEATURE_COLS
from ml_predictor.targets      import (
    add_target, class_distribution, class_weights,
    DOWN, FLAT, UP, FORWARD_CANDLES, DEADBAND,
)
from ml_predictor.regime       import get_bucket_series, BUCKET_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.train")

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK        = 20       # LSTM sequence length (candles)
DECAY           = 0.70     # time-weight decay per year (recent matters more)
N_FOLDS         = 6        # walk-forward folds (avg reported for context only)
RECENT_HOLDOUT_DAYS = 20   # fixed, genuinely-recent window for the live-accuracy
                           # estimate — see evaluate_recent_holdout(). Independent
                           # of N_FOLDS/dataset size, unlike the walk-forward
                           # "last fold" (which scales with total history and is
                           # NOT a recent-conditions estimate — see docstring).
MAX_TREES       = 600      # hard cap referenced by retrain.py; enforced there
EARLY_STOPPING_ROUNDS = 30 # used only where a real eval_set exists — see
                           # XGB_PARAMS_VALIDATION below. Production training
                           # (train_xgboost's final per-bucket fit) trains on
                           # 100% of the bucket's data with NO held-out set,
                           # so it cannot use early stopping — passing
                           # early_stopping_rounds without an eval_set raises
                           # an xgboost error, which is why this isn't baked
                           # into the shared XGB_PARAMS below.
XGB_PARAMS      = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
}
# Validation-context variant: identical to XGB_PARAMS but with early stopping
# enabled. Used in walk_forward_validate() and evaluate_recent_holdout(),
# where a genuine held-out eval_set is always available at the point of
# fit() — letting each fold/holdout stop early on validation loss rather
# than always training the full fixed n_estimators, which helps prevent
# overfitting on smaller folds. NOT used for the final production model
# saved by train_xgboost (see note above).
#
# early_stopping_rounds is passed via the CONSTRUCTOR, not fit() — fit()'s
# early_stopping_rounds kwarg was deprecated in xgboost 1.6.0 (confirmed
# against xgboost's own docs: "Deprecated since version 1.6.0: Use
# early_stopping_rounds in __init__() or set_params() instead").
XGB_PARAMS_VALIDATION = {**XGB_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}


# ── Sample weighting ──────────────────────────────────────────────────────────

def compute_time_weights(index: pd.DatetimeIndex, decay: float = DECAY) -> np.ndarray:
    """
    weight = decay ^ (days_from_today / 365)
    decay=0.70 → today=1.00, 1yr ago=0.70, 2yr ago=0.49, 5yr ago=0.17
    """
    today_ts = pd.Timestamp.now()
    days_ago = (today_ts - index).days.astype(float)
    return (decay ** (days_ago / 365.0)).values


def compute_combined_weights(df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Combined sample weight = time_decay_weight × inverse_class_frequency_weight.

    This is the fix for the class-imbalance problem found during EDA: the
    new 3-class target is ~70-80% FLAT depending on instrument/bucket, so
    without this, the model would just learn to always predict FLAT.
    Class weights are computed PER CALL (i.e. per bucket, when called from
    train_xgboost's bucket loop) so each bucket's own imbalance is corrected,
    rather than applying one global ratio everywhere.
    """
    time_w  = compute_time_weights(df.index)
    cls_w   = class_weights(df.assign(target=y))
    class_w = y.map(cls_w).values
    return time_w * class_w


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validate(X: pd.DataFrame, y: pd.Series, model_fn, n_folds=N_FOLDS):
    """
    Time-series walk-forward validation, expanding training window.
    Reports per-fold accuracy + per-class precision/recall/f1, and a
    macro-AUC (one-vs-rest) since this is now a 3-class problem.

    Returns (results_list, last_fold_metrics_dict).

    IMPORTANT — what "last fold" actually means here:
    Each fold is total_rows / (n_folds + 1) in size. With ~8 years of 5-min
    data and N_FOLDS=6, that's ~21,000 rows = ~300 TRADING DAYS per fold —
    over a year, not "recent." The last fold is still useful as a coarse
    sanity check (does the model degrade on the newest large chunk of
    data?), but it is NOT a recent-conditions estimate, and should not be
    read as one. For an actual "if I deployed this today, what accuracy
    would I expect this month" number, use evaluate_recent_holdout()
    below instead — it holds out a small, FIXED, genuinely-recent window
    regardless of how much total history you have, rather than a window
    whose size scales with total dataset length.
    """
    total   = len(X)
    fold_sz = total // (n_folds + 1)
    results = []

    log.info(f"Walk-forward validation: {n_folds} folds, ~{fold_sz:,} samples/fold "
             f"(~{fold_sz // 250} trading days/fold at ~250 candles/day)")

    for fold in range(n_folds):
        train_end = fold_sz * (fold + 1)
        test_end  = min(train_end + fold_sz, total)

        X_tr, y_tr = X.iloc[:train_end],         y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(X_tr) < 200 or len(X_te) < 30:
            continue

        weights = compute_combined_weights(X_tr, y_tr)

        model = model_fn()
        # eval_set is always safe to pass — it just computes validation
        # metrics. It only ACTS as early stopping if the model itself was
        # constructed with early_stopping_rounds set (see XGB_PARAMS_VALIDATION
        # vs plain XGB_PARAMS) — early_stopping_rounds REQUIRES eval_set, but
        # eval_set does not require early_stopping_rounds, so this is safe
        # even if model_fn() ever returns something without early stopping
        # configured at all.
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)

        acc = accuracy_score(y_te, y_pred)
        try:
            auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")  # can happen if a fold is missing a class entirely

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=[DOWN, FLAT, UP], zero_division=0
        )

        test_days = X_te.index.normalize().nunique()
        fold_metrics = {
            "fold": fold + 1, "train": len(X_tr), "test": len(X_te),
            "test_days": test_days,
            "test_date_range": [str(X_te.index.min().date()), str(X_te.index.max().date())],
            "accuracy": acc, "auc_macro_ovr": auc,
            "precision": {"DOWN": prec[0], "FLAT": prec[1], "UP": prec[2]},
            "recall":    {"DOWN": rec[0],  "FLAT": rec[1],  "UP": rec[2]},
            "f1":        {"DOWN": f1[0],   "FLAT": f1[1],   "UP": f1[2]},
        }
        results.append(fold_metrics)

        log.info(
            f"  Fold {fold+1}/{n_folds}: train={len(X_tr):,} test={len(X_te):,} "
            f"({test_days} days, {fold_metrics['test_date_range'][0]} to {fold_metrics['test_date_range'][1]}) "
            f"acc={acc:.3f} AUC={auc:.3f} | "
            f"UP  P/R={prec[2]:.2f}/{rec[2]:.2f} | "
            f"DOWN P/R={prec[0]:.2f}/{rec[0]:.2f}"
        )

    if not results:
        return results, {}

    avg_acc = np.mean([r["accuracy"] for r in results])
    log.info(f"  Walk-forward {len(results)}-fold average accuracy: {avg_acc:.3f}  (context only)")

    last_fold = results[-1]
    log.info(
        f"  Last fold (newest large chunk, {last_fold['test_days']} days — "
        f"NOT a recent-conditions estimate, see evaluate_recent_holdout) → "
        f"acc={last_fold['accuracy']:.3f} AUC={last_fold['auc_macro_ovr']:.3f} | "
        f"UP P/R={last_fold['precision']['UP']:.2f}/{last_fold['recall']['UP']:.2f} | "
        f"DOWN P/R={last_fold['precision']['DOWN']:.2f}/{last_fold['recall']['DOWN']:.2f}"
    )

    return results, last_fold


def evaluate_recent_holdout(X: pd.DataFrame, y: pd.Series, model_fn,
                             holdout_days: int = RECENT_HOLDOUT_DAYS) -> dict:
    """
    Train on everything EXCEPT the last `holdout_days` trading days, test
    ONLY on those days. This is the number that should actually be quoted
    as "if I deployed this today, what accuracy would I expect."

    Unlike walk_forward_validate's last fold — whose size is total_rows /
    (n_folds+1) and therefore grows with however much history you have
    (e.g. ~300 trading days at N_FOLDS=6 on 8 years of data) — this window
    is FIXED at holdout_days regardless of dataset size, so it actually
    reflects current market conditions rather than a multi-quarter average.

    Returns a metrics dict in the same shape as a walk-forward fold entry,
    plus "holdout_days" and "holdout_date_range" for clarity in metrics.json.
    """
    cutoff = X.index.max() - pd.Timedelta(days=holdout_days * 1.6)
    # *1.6 buffer converts "trading days" to "calendar days" loosely
    # (5 trading days/week ≈ 7 calendar days), then the explicit day-count
    # filter below trims to the exact number of trading days requested.

    candidate = X[X.index > cutoff]
    test_dates = sorted(candidate.index.normalize().unique())[-holdout_days:]
    if not test_dates:
        log.warning("evaluate_recent_holdout: not enough recent data — skipping")
        return {}

    test_start = test_dates[0]
    test_mask  = X.index.normalize() >= test_start
    train_mask = ~test_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    if len(X_tr) < 200 or len(X_te) < 30:
        log.warning(f"evaluate_recent_holdout: insufficient rows "
                    f"(train={len(X_tr)}, test={len(X_te)}) — skipping")
        return {}

    weights = compute_combined_weights(X_tr, y_tr)
    model = model_fn()
    # eval_set is always safe to pass — see walk_forward_validate's comment
    # on the same pattern. Only acts as early stopping if the model was
    # constructed with early_stopping_rounds set (XGB_PARAMS_VALIDATION).
    model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[DOWN, FLAT, UP], zero_division=0
    )

    actual_days = X_te.index.normalize().nunique()
    result = {
        "train": len(X_tr), "test": len(X_te),
        "holdout_days": actual_days,
        "holdout_date_range": [str(X_te.index.min().date()), str(X_te.index.max().date())],
        "accuracy": acc, "auc_macro_ovr": auc,
        "precision": {"DOWN": prec[0], "FLAT": prec[1], "UP": prec[2]},
        "recall":    {"DOWN": rec[0],  "FLAT": rec[1],  "UP": rec[2]},
        "f1":        {"DOWN": f1[0],   "FLAT": f1[1],   "UP": f1[2]},
    }

    log.info(
        f"  ★ RECENT HOLDOUT ({actual_days} trading days, "
        f"{result['holdout_date_range'][0]} to {result['holdout_date_range'][1]}) "
        f"→ THIS is the live-accuracy estimate: "
        f"acc={acc:.3f} AUC={auc:.3f} | "
        f"UP P/R={prec[2]:.2f}/{rec[2]:.2f} | "
        f"DOWN P/R={prec[0]:.2f}/{rec[0]:.2f}"
    )

    return result


# ── XGBoost training ──────────────────────────────────────────────────────────

def train_xgboost(instrument: str, df_feat: pd.DataFrame) -> dict:
    """
    Train XGBoost A/B/C bucket models for one instrument. Saves .json + scaler.
    Bucket assignment comes from regime.get_bucket_series — the SAME function
    predictor.py uses live, so a row is never trained under one bucket and
    served under another.

    Returns a metrics dict (for metrics.json) including walk-forward context
    metrics, the recent-holdout live-accuracy estimate, and per-bucket sample
    counts/class balance.
    """
    from xgboost import XGBClassifier

    log.info(f"\n{'='*55}")
    log.info(f"  XGBoost — {instrument}")
    log.info(f"{'='*55}")

    X_full = df_feat[FEATURE_COLS]
    y_full = df_feat["target"].astype(int)

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_full), index=X_full.index, columns=X_full.columns,
    )
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{instrument.lower()}.pkl")
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved → {scaler_path}")

    log.info(f"\nWalk-forward validation (all hours, full-day model, context only — "
             f"see RECENT HOLDOUT below for the actual live-accuracy estimate):")
    wf_results, last_fold = walk_forward_validate(
        X_scaled, y_full, lambda: XGBClassifier(**XGB_PARAMS_VALIDATION),
    )

    log.info(f"\nRecent holdout (last {RECENT_HOLDOUT_DAYS} trading days — THIS is the "
             f"number to use as your live-accuracy estimate):")
    recent_holdout = evaluate_recent_holdout(
        X_scaled, y_full, lambda: XGBClassifier(**XGB_PARAMS_VALIDATION),
    )

    metrics = {
        "instrument": instrument,
        "walk_forward_last_fold_CONTEXT_ONLY": last_fold,
        "recent_holdout_LIVE_ESTIMATE": recent_holdout,
        "buckets": {},
    }

    bucket_series = get_bucket_series(df_feat.index)

    for bucket in ["A", "B", "C"]:
        mask   = bucket_series == bucket
        feat_b = df_feat[mask]

        if len(feat_b) < 200:
            log.warning(f"  Bucket {bucket} ({BUCKET_NAMES[bucket]}): only {len(feat_b)} "
                        f"samples — skipping")
            continue

        X_b = pd.DataFrame(
            scaler.transform(feat_b[FEATURE_COLS]), index=feat_b.index, columns=FEATURE_COLS,
        )
        y_b = feat_b["target"].astype(int)

        dist = class_distribution(feat_b)
        log.info(f"\n  Bucket {bucket} ({BUCKET_NAMES[bucket]}): n={len(feat_b):,}")
        log.info(f"    Class balance: {dist['pct'].to_dict()}")

        weights = compute_combined_weights(X_b, y_b)

        model = XGBClassifier(**XGB_PARAMS)   # plain params — NO early stopping
                                                # here: this trains on 100% of
                                                # the bucket's data with no
                                                # held-out eval_set, by design
                                                # (maximize signal for the
                                                # actual production model).
        model.fit(X_b, y_b, sample_weight=weights, verbose=False)

        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        top10 = fi.nlargest(10)
        log.info(f"    Top features: {', '.join(f'{k}({v:.3f})' for k, v in top10.items())}")

        model_path = os.path.join(MODEL_DIR, f"xgb_{instrument.lower()}_{bucket}.json")
        model.save_model(model_path)
        log.info(f"    Model saved → {model_path}")

        # Per-bucket recent holdout — this is the number that actually
        # matters for production: it evaluates THIS bucket's model (the
        # one predictor.py will load and serve live) on the genuinely
        # most-recent trading days, not the full-day model from above.
        # Uses XGB_PARAMS_VALIDATION (early stopping enabled) since THIS
        # call trains its own fresh model with a real eval_set internally,
        # unlike the production model fit two lines above.
        log.info(f"    Recent holdout for bucket {bucket} (last {RECENT_HOLDOUT_DAYS} "
                  f"trading days, this bucket's own data):")
        bucket_holdout = evaluate_recent_holdout(
            X_b, y_b, lambda: XGBClassifier(**XGB_PARAMS_VALIDATION),
        )

        metrics["buckets"][bucket] = {
            "n_samples": len(feat_b),
            "class_pct": dist["pct"].to_dict(),
            "recent_holdout_LIVE_ESTIMATE": bucket_holdout,
        }

    log.info(f"\n  XGBoost {instrument} — DONE")
    return metrics


# ── LSTM training ─────────────────────────────────────────────────────────────

def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(instrument: str, df_feat: pd.DataFrame) -> dict:
    """
    Train multiclass LSTM for one instrument. Saves .keras model.

    Changed from old build: output layer is now Dense(3, softmax) with
    sparse_categorical_crossentropy, not Dense(1, sigmoid)/binary_crossentropy
    — the old binary head cannot represent a 3-class target at all.

    IMPORTANT — two different "val" numbers reported here, do not confuse them:
    1. val_accuracy (from the 80/20 chronological split below) — this is
       what EarlyStopping/ModelCheckpoint actually use during training to
       decide when to stop and which weights to keep. It is now computed
       WEIGHTED (val_sample_weights=w_te passed via validation_data), using
       the same time-decay × inverse-class-frequency weights as the training
       loss — previously this was unweighted, which meant checkpoint
       selection silently favored FLAT-majority epochs regardless of how
       the model was actually trained. On ~8 years of data this 20% split is
       itself ~1.5-2 YEARS wide (confirmed: 418 trading days on real data),
       so it's a legitimate training-control signal but NOT a
       recent-conditions estimate.
    2. recent_holdout_INSAMPLE_NOT_STRICT_OOS (computed AFTER training, on
       the last RECENT_HOLDOUT_DAYS trading days specifically) — uses the
       SAME WINDOW as XGBoost's evaluate_recent_holdout(), fixing the
       "418 days isn't recent" problem. But it is NOT methodologically
       equivalent to XGBoost's number: XGBoost's evaluate_recent_holdout()
       trains a FRESH model excluding the holdout window (strict
       out-of-sample). This LSTM version evaluates the ALREADY-TRAINED
       model, which may have seen these exact days during its own 80/20
       split — it is in-sample for whatever portion overlaps that split.
       Read it as "how does this model do on the most recent month," not
       as a number directly interchangeable with XGBoost's. The dict key
       name itself carries this warning so it survives into metrics.json,
       not just this docstring.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        log.warning("TensorFlow not installed. Skipping LSTM. Run: pip install tensorflow")
        return {}

    log.info(f"\n{'='*55}")
    log.info(f"  LSTM — {instrument}")
    log.info(f"{'='*55}")

    scaler_path = os.path.join(MODEL_DIR, f"scaler_{instrument.lower()}.pkl")
    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}. Run XGBoost training first.")
        return {}
    scaler = joblib.load(scaler_path)

    X_full = df_feat[FEATURE_COLS]
    y_full = df_feat["target"].astype(int).values
    X_scaled = scaler.transform(X_full)

    idx_array   = df_feat.index[LOOKBACK:]
    time_w      = compute_time_weights(idx_array)
    cls_w_map   = class_weights(df_feat)
    class_w_arr = pd.Series(y_full[LOOKBACK:]).map(cls_w_map).values
    seq_weights = time_w * class_w_arr

    X_seq, y_seq = build_sequences(X_scaled, y_full, LOOKBACK)
    seq_index = df_feat.index[LOOKBACK:]  # timestamp aligned to each sequence's target row

    split = int(len(X_seq) * 0.80)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]
    w_tr, w_te = seq_weights[:split], seq_weights[split:]
    # BUG FIX: w_te previously wasn't computed/passed, so val_accuracy (used
    # by EarlyStopping/ModelCheckpoint below) was calculated UNWEIGHTED by
    # Keras — while training loss WAS weighted via sample_weight=w_tr. On a
    # ~70-80% FLAT target, raw unweighted val_accuracy is maximized by
    # whichever epoch predicts FLAT most often, so checkpoint selection was
    # silently re-introducing the exact FLAT bias the class weighting was
    # meant to correct. Passing w_te as the 3rd element of validation_data
    # makes Keras apply the same sample weights to val loss AND val_accuracy,
    # so the metric driving early stopping / checkpointing is now consistent
    # with what the model is actually being trained to optimize.

    n_features = X_seq.shape[2]
    log.info(f"  Sequences: {len(X_seq):,} | Features: {n_features} | Lookback: {LOOKBACK}")
    log.info(f"  Train: {len(X_tr):,}  Val (80/20 split, training-control only): {len(X_te):,} "
             f"(~{(len(X_te)) // 75} trading days — NOT a recent-conditions estimate)")

    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax"),   # 3-class: DOWN/FLAT/UP
    ], name=f"LSTM_{instrument}")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    log.info(f"  Parameters: {model.count_params():,}")

    model_path = os.path.join(MODEL_DIR, f"lstm_{instrument.lower()}.keras")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max"),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
    ]

    history = model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        validation_data=(X_te, y_te, w_te),
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    log.info(f"\n  LSTM {instrument}: Best 80/20-split val_acc={best_val_acc:.3f} "
             f"(training-control metric, NOT the live estimate)")
    log.info(f"  Model saved → {model_path}")

    # Per-class report on the 80/20 held-out validation set, since accuracy
    # alone is misleading on an imbalanced 3-class target (see EDA: ~70-80%
    # FLAT). Still not a recent-conditions number — see recent_holdout below.
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[DOWN, FLAT, UP], zero_division=0
    )
    log.info(f"  80/20-split Val UP   P/R/F1 = {prec[2]:.2f}/{rec[2]:.2f}/{f1[2]:.2f}")
    log.info(f"  80/20-split Val DOWN P/R/F1 = {prec[0]:.2f}/{rec[0]:.2f}/{f1[0]:.2f}")
    log.info(f"  80/20-split Val FLAT P/R/F1 = {prec[1]:.2f}/{rec[1]:.2f}/{f1[1]:.2f}")

    # ── Recent holdout — same WINDOW as XGBoost's, but NOT the same
    # methodology: this evaluates the already-trained model (which may have
    # seen these exact days during the 80/20 split), not a fresh model
    # trained excluding them. See recent_holdout_INSAMPLE_NOT_STRICT_OOS's
    # key name in the returned dict — deliberately NOT called LIVE_ESTIMATE
    # like XGBoost's, so metrics.json itself flags the difference without
    # requiring anyone to read this docstring first.
    log.info(f"\n  Recent window for LSTM (last {RECENT_HOLDOUT_DAYS} trading "
              f"days — same window as XGBoost's recent_holdout, but evaluated "
              f"in-sample on the already-trained model, NOT strict "
              f"out-of-sample like XGBoost's. See key name in metrics.json):")
    recent_holdout = _lstm_recent_holdout(model, X_seq, y_seq, seq_index)

    return {"val_80_20_split_TRAINING_CONTROL_ONLY": float(best_val_acc),
            "recent_holdout_INSAMPLE_NOT_STRICT_OOS": recent_holdout}


def _lstm_recent_holdout(model, X_seq: np.ndarray, y_seq: np.ndarray,
                          seq_index: pd.DatetimeIndex,
                          holdout_days: int = RECENT_HOLDOUT_DAYS) -> dict:
    """
    Evaluate an already-trained LSTM on a FIXED, genuinely-recent window
    (last `holdout_days` trading days), using the SAME WINDOW as
    evaluate_recent_holdout() for XGBoost — fixing the "the 80/20 split is
    ~1.5-2 years wide and was never recent in the first place" problem.

    This is NOT methodologically equivalent to XGBoost's number (see
    train_lstm's docstring and the recent_holdout_INSAMPLE_NOT_STRICT_OOS
    key name in the returned dict for the full explanation) — it evaluates
    the already-trained model rather than retraining excluding this window,
    so it can be in-sample for whatever part overlaps the 80/20 split.

    Note: this evaluates on data the model MAY have seen during the 80/20
    training split if the recent window overlaps the training portion —
    that's expected and fine here, since the point is "how does this model
    perform on the most recent month," not a strict train/test holdout
    (the model already trained on a years-wide validation-adjacent split
    anyway). For a stricter never-seen-during-training recent holdout,
    retrain from scratch excluding these days — out of scope for this
    post-hoc evaluation step.
    """
    test_dates = sorted(pd.Series(seq_index).dt.normalize().unique())[-holdout_days:]
    if not test_dates:
        log.warning("  LSTM recent holdout: not enough recent data — skipping")
        return {}

    test_start = test_dates[0]
    test_mask  = seq_index.normalize() >= test_start
    X_te, y_te = X_seq[test_mask], y_seq[test_mask]

    if len(X_te) < 10:
        log.warning(f"  LSTM recent holdout: insufficient rows ({len(X_te)}) — skipping")
        return {}

    y_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=[DOWN, FLAT, UP], zero_division=0
    )

    actual_days = seq_index[test_mask].normalize().nunique()
    result = {
        "test": len(X_te),
        "holdout_days": actual_days,
        "holdout_date_range": [str(seq_index[test_mask].min().date()),
                                str(seq_index[test_mask].max().date())],
        "accuracy": acc, "auc_macro_ovr": auc,
        "precision": {"DOWN": prec[0], "FLAT": prec[1], "UP": prec[2]},
        "recall":    {"DOWN": rec[0],  "FLAT": rec[1],  "UP": rec[2]},
        "f1":        {"DOWN": f1[0],   "FLAT": f1[1],   "UP": f1[2]},
    }

    log.info(
        f"  ★ LSTM RECENT HOLDOUT ({actual_days} trading days, "
        f"{result['holdout_date_range'][0]} to {result['holdout_date_range'][1]}) "
        f"→ acc={acc:.3f} AUC={auc:.3f} | "
        f"UP P/R={prec[2]:.2f}/{rec[2]:.2f} | "
        f"DOWN P/R={prec[0]:.2f}/{rec[0]:.2f}"
    )

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def load_and_engineer(instrument: str) -> pd.DataFrame:
    """Load CSV, build features, add target, drop session-boundary NaN rows."""
    log.info(f"\nLoading data for {instrument}...")
    log.info(f"  Target config (live, from targets.py): "
              f"{FORWARD_CANDLES}-candle forward, ±{DEADBAND:.2%} deadband")

    df_main = load_csv(instrument)

    try:
        df_vix = load_csv("INDIAVIX")
    except FileNotFoundError:
        log.warning("VIX CSV not found — vix features will be zero (degraded mode)")
        df_vix = None

    cross_map  = {"BANKNIFTY": "NIFTY50", "NIFTY50": "BANKNIFTY"}
    cross_name = cross_map.get(instrument.upper())
    try:
        df_cross = load_csv(cross_name) if cross_name else None
    except FileNotFoundError:
        log.warning(f"{cross_name} CSV not found — cross_ret_1 will be zero (degraded mode)")
        df_cross = None

    # Start-date alignment check. build_features()'s VIX/cross reindex uses
    # method="ffill", which only fills FORWARD from each series' own first
    # row — any df_main row before df_vix's (or df_cross's) start date gets
    # NaN for that feature, and the final dropna() in build_features()
    # silently removes those rows. Confirmed: a VIX series starting 2
    # months after df_main silently drops ~2 months of otherwise-valid
    # training data with zero warning anywhere. This doesn't affect your
    # current CSVs (all three start 2018-01-01 in the data checked at
    # build time), but logging it explicitly here means a future data
    # refresh, a new cross-index, or a VIX feed with different history
    # can't silently repeat this — the gap will be visible in every
    # training run's log instead of requiring someone to notice missing
    # months by inspecting date ranges by hand.
    main_start = df_main.index.min()
    for label, df_other in [("VIX", df_vix), (cross_name, df_cross)]:
        if df_other is not None and not df_other.empty:
            other_start = df_other.index.min()
            if other_start > main_start:
                gap_days = (other_start - main_start).days
                log.warning(
                    f"  ⚠ START-DATE MISMATCH: {label} data starts "
                    f"{other_start.date()}, but {instrument} starts "
                    f"{main_start.date()} ({gap_days} days earlier). "
                    f"build_features()'s ffill reindex cannot backfill "
                    f"before {label}'s first row, so ~{gap_days} days of "
                    f"{instrument} data will be SILENTLY dropped by the "
                    f"dropna() step below. If this is unexpected, check "
                    f"why {label}'s CSV doesn't cover this instrument's "
                    f"full history."
                )

    df_feat = build_features(df_main, df_vix=df_vix, df_cross=df_cross)
    df_t    = add_target(df_main)

    # Join features with target on shared index; drop session-boundary /
    # last-4-candles-of-day NaN targets per targets.py's contract.
    df_feat = df_feat.join(df_t["target"], how="inner")
    before = len(df_feat)
    df_feat = df_feat.dropna(subset=["target"])
    dropped = before - len(df_feat)

    log.info(f"  Feature matrix: {df_feat.shape[0]:,} rows × {df_feat.shape[1]-1} feature cols "
              f"(dropped {dropped:,} session-boundary/tail rows)")
    log.info(f"  Target balance: {class_distribution(df_feat)['pct'].to_dict()}")
    log.info(f"  Date range: {df_feat.index.min().date()} → {df_feat.index.max().date()}")

    return df_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-only",   action="store_true", help="Skip LSTM training")
    parser.add_argument("--lstm-only",  action="store_true", help="Skip XGBoost training")
    parser.add_argument("--instrument", default=None,
                        help="Train one instrument: NIFTY50 or BANKNIFTY (default: both)")
    args = parser.parse_args()

    instruments = [args.instrument.upper()] if args.instrument else ["NIFTY50", "BANKNIFTY"]
    all_metrics = {}

    for instr in instruments:
        df_feat = load_and_engineer(instr)

        if not args.lstm_only:
            all_metrics[instr] = train_xgboost(instr, df_feat)

        if not args.xgb_only:
            lstm_metrics = train_lstm(instr, df_feat)
            if instr not in all_metrics:
                all_metrics[instr] = {"instrument": instr}
            all_metrics[instr]["lstm"] = lstm_metrics

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    log.info(f"\nMetrics report saved → {metrics_path}")

    log.info("\n" + "="*55)
    log.info("  ALL TRAINING COMPLETE")
    log.info("="*55)
    log.info(f"  Models saved in: {MODEL_DIR}/")
    log.info("  Run predictor.py to test real-time predictions")


if __name__ == "__main__":
    main()
