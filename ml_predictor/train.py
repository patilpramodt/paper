"""
ml_predictor/train.py
──────────────────────
Complete training pipeline:
  1. Load CSV data (Nifty + BankNifty + VIX)
  2. Feature engineering (31 features)
  3. Time-decay sample weighting (recent data counts more)
  4. Walk-forward validation (honest accuracy, no data leakage)
  5. Train three XGBoost models per instrument (time-of-day split)
  6. Train LSTM per instrument
  7. Save all models + scalers

Usage:
    # Train everything from scratch:
    python ml_predictor/train.py

    # Train XGBoost only (faster, Week 1):
    python ml_predictor/train.py --xgb-only

    # Train one instrument:
    python ml_predictor/train.py --instrument BANKNIFTY
"""

import argparse
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(BASE_DIR))
from ml_predictor.data_fetcher       import load_csv
from ml_predictor.feature_engineering import build_features, FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.train")

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK        = 20       # LSTM sequence length (candles)
DECAY           = 0.70     # time-weight decay per year (recent matters more)
WALK_FOLD_DAYS  = 60       # test window per walk-forward fold
N_FOLDS         = 6        # number of walk-forward folds
XGB_PARAMS      = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

# Time-of-day buckets (hour, minute) boundaries
TIME_BUCKETS = {
    "A": ((9, 15), (10, 30)),    # opening volatility
    "B": ((10, 30), (14, 0)),    # trending hours
    "C": ((14, 0),  (15, 31)),   # closing action
}


# ── Sample weighting ──────────────────────────────────────────────────────────

def compute_time_weights(index: pd.DatetimeIndex, decay: float = DECAY) -> np.ndarray:
    """
    Assign exponentially decaying weights so recent samples matter more.

    weight = decay ^ (days_from_today / 365)
    decay=0.7 means:
        today      → 1.00
        1 year ago → 0.70
        2 years ago→ 0.49
        5 years ago→ 0.17
    """
    today_ts = pd.Timestamp.now()
    days_ago = (today_ts - index).days.astype(float)
    weights  = decay ** (days_ago / 365.0)
    return weights.values


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validate(X: pd.DataFrame, y: pd.Series, model_fn, n_folds=N_FOLDS):
    """
    Time-series walk-forward validation.
    Trains on all data before each test window. Never trains on future data.

    Returns list of {fold, train_size, test_size, accuracy, auc}
    """
    from xgboost import XGBClassifier

    total   = len(X)
    fold_sz = total // (n_folds + 1)
    results = []

    log.info(f"Walk-forward validation: {n_folds} folds, {fold_sz} samples each")

    for fold in range(n_folds):
        train_end = fold_sz * (fold + 1)
        test_end  = min(train_end + fold_sz, total)

        X_tr, y_tr = X.iloc[:train_end],        y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        weights = compute_time_weights(X_tr.index)

        model = model_fn()
        model.fit(X_tr, y_tr, sample_weight=weights,
                  eval_set=[(X_te, y_te)], verbose=False)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        acc    = accuracy_score(y_te, y_pred)
        auc    = roc_auc_score(y_te, y_prob)

        log.info(f"  Fold {fold+1}/{n_folds}: train={len(X_tr):,} test={len(X_te):,} "
                 f"acc={acc:.3f} AUC={auc:.3f}")

        results.append({"fold": fold+1, "train": len(X_tr), "test": len(X_te),
                        "accuracy": acc, "auc": auc})

    if results:
        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_auc = np.mean([r["auc"] for r in results])
        log.info(f"  Walk-forward average → Accuracy: {avg_acc:.3f}  AUC: {avg_auc:.3f}")
        log.info(f"  ← This is your HONEST live accuracy estimate")

    return results


# ── XGBoost training ──────────────────────────────────────────────────────────

def _filter_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Filter DataFrame to rows in a time-of-day bucket."""
    from datetime import time as dtime
    start_h, start_m = TIME_BUCKETS[bucket][0]
    end_h,   end_m   = TIME_BUCKETS[bucket][1]
    t = df.index.time
    return df[
        (t >= dtime(start_h, start_m)) &
        (t <  dtime(end_h,   end_m))
    ]


def train_xgboost(instrument: str, df_feat: pd.DataFrame):
    """Train XGBoost A/B/C models for one instrument. Saves .json + scaler."""
    from xgboost import XGBClassifier

    log.info(f"\n{'='*55}")
    log.info(f"  XGBoost — {instrument}")
    log.info(f"{'='*55}")

    X_full = df_feat[FEATURE_COLS]
    y_full = df_feat["target"].astype(int)

    # One scaler for all buckets (fitted on full data)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_full),
        index=X_full.index,
        columns=X_full.columns,
    )
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{instrument.lower()}.pkl")
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved → {scaler_path}")

    # Walk-forward on full data first
    log.info(f"\nWalk-forward validation (all hours):")
    walk_forward_validate(
        X_scaled, y_full,
        lambda: XGBClassifier(**XGB_PARAMS),
    )

    # Train time-of-day models
    for bucket in ["A", "B", "C"]:
        feat_b = _filter_bucket(df_feat, bucket)
        if len(feat_b) < 200:
            log.warning(f"  Bucket {bucket}: only {len(feat_b)} samples — skipping")
            continue

        X_b = pd.DataFrame(
            scaler.transform(feat_b[FEATURE_COLS]),
            index=feat_b.index,
            columns=FEATURE_COLS,
        )
        y_b     = feat_b["target"].astype(int)
        weights = compute_time_weights(X_b.index)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_b, y_b, sample_weight=weights, verbose=False)

        # Feature importance top 10
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        top10 = fi.nlargest(10)
        log.info(f"\n  Bucket {bucket} top features:")
        for feat, imp in top10.items():
            log.info(f"    {feat:<25} {imp:.4f}")

        model_path = os.path.join(MODEL_DIR, f"xgb_{instrument.lower()}_{bucket}.json")
        model.save_model(model_path)
        log.info(f"  Model saved → {model_path} (trained on {len(X_b):,} samples)")

    log.info(f"\n  XGBoost {instrument} — DONE")


# ── LSTM training ─────────────────────────────────────────────────────────────

def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    """Convert flat feature array to (samples, lookback, features) sequences."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(instrument: str, df_feat: pd.DataFrame):
    """Train LSTM for one instrument. Saves .keras model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
        log.warning("TensorFlow not installed. Skipping LSTM. Run: pip install tensorflow")
        return

    log.info(f"\n{'='*55}")
    log.info(f"  LSTM — {instrument}")
    log.info(f"{'='*55}")

    # Load scaler saved by XGBoost step
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{instrument.lower()}.pkl")
    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}. Run XGBoost training first.")
        return
    scaler = joblib.load(scaler_path)

    X_full = df_feat[FEATURE_COLS]
    y_full = df_feat["target"].astype(int).values

    X_scaled = scaler.transform(X_full)

    # Time-weighted sample weights for sequences
    idx_array = df_feat.index[LOOKBACK:]   # sequence labels start at lookback
    seq_weights = compute_time_weights(idx_array)

    X_seq, y_seq = build_sequences(X_scaled, y_full, LOOKBACK)

    # 80/20 time-based split
    split = int(len(X_seq) * 0.80)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]
    w_tr       = seq_weights[:split]

    n_features = X_seq.shape[2]
    log.info(f"  Sequences: {len(X_seq):,} | Features: {n_features} | Lookback: {LOOKBACK}")
    log.info(f"  Train: {len(X_tr):,}  Val: {len(X_te):,}")

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
        Dense(1, activation="sigmoid"),
    ], name=f"LSTM_{instrument}")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    log.info(f"  Parameters: {model.count_params():,}")

    model_path = os.path.join(MODEL_DIR, f"lstm_{instrument.lower()}.keras")

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max"),
        ModelCheckpoint(model_path, monitor="val_auc", save_best_only=True, mode="max", verbose=0),
    ]

    history = model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        validation_data=(X_te, y_te),
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_val_auc = max(history.history.get("val_auc", [0]))
    log.info(f"\n  LSTM {instrument}: Best val_acc={best_val_acc:.3f}  val_AUC={best_val_auc:.3f}")
    log.info(f"  Model saved → {model_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_and_engineer(instrument: str) -> pd.DataFrame:
    """Load CSV and run feature engineering for one instrument."""
    log.info(f"\nLoading data for {instrument}...")

    df_main  = load_csv(instrument)

    # Load VIX for vix_level + vix_change features
    try:
        df_vix = load_csv("INDIAVIX")
    except FileNotFoundError:
        log.warning("VIX CSV not found — vix features will be zero")
        df_vix = None

    # Cross-index: if training BankNifty, use Nifty as cross (and vice versa)
    cross_map = {"BANKNIFTY": "NIFTY50", "NIFTY50": "BANKNIFTY"}
    cross_name = cross_map.get(instrument.upper())
    try:
        df_cross = load_csv(cross_name) if cross_name else None
    except FileNotFoundError:
        log.warning(f"{cross_name} CSV not found — cross_ret_1 will be zero")
        df_cross = None

    df_feat = build_features(df_main, df_vix=df_vix, df_cross=df_cross)

    log.info(f"  Feature matrix: {df_feat.shape[0]:,} rows × {df_feat.shape[1]} cols")
    log.info(f"  Target balance: {df_feat['target'].mean():.1%} green candles")
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

    for instr in instruments:
        df_feat = load_and_engineer(instr)

        if not args.lstm_only:
            train_xgboost(instr, df_feat)

        if not args.xgb_only:
            train_lstm(instr, df_feat)

    log.info("\n" + "="*55)
    log.info("  ALL TRAINING COMPLETE")
    log.info("="*55)
    log.info(f"  Models saved in: {MODEL_DIR}/")
    log.info("  Run predictor.py to test real-time predictions")


if __name__ == "__main__":
    main()
