"""
ml_predictor/retrain_weekly.py
───────────────────────────────
Cron-driven retraining script.
Runs Sunday night (weekly fine-tune) or 1st Saturday (full retrain).

Crontab entries to add:
    # Daily: append today's candles (15:40 IST)
    40 15 * * 1-5 cd ~/paper && python ml_predictor/data_fetcher.py --append >> logs/ml_data.log 2>&1

    # Weekly: fine-tune on recent 30 days (Sunday 22:00 IST)
    0 22 * * 0 cd ~/paper && python ml_predictor/retrain_weekly.py >> logs/ml_retrain.log 2>&1

    # Monthly: full retrain from scratch (1st Saturday 21:00 IST)
    0 21 1-7 * 6 cd ~/paper && python ml_predictor/retrain_weekly.py --full >> logs/ml_retrain.log 2>&1

Usage:
    python ml_predictor/retrain_weekly.py           # weekly fine-tune (last 30 days)
    python ml_predictor/retrain_weekly.py --full    # full retrain (all history)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml.retrain")

FINE_TUNE_DAYS = 30    # fine-tune uses last N days of data


def run_data_append():
    """Append today's candles before retraining."""
    log.info("Step 1: Appending latest candles...")
    from ml_predictor.data_fetcher import _load_kite, fetch_append
    try:
        kite = _load_kite()
        fetch_append(kite)
        log.info("Data append complete.")
    except Exception as e:
        log.warning(f"Data append failed (continuing with existing data): {e}")


def run_full_retrain():
    """Full retrain on all historical data."""
    log.info("Step 2: Full retrain — all history")
    from ml_predictor.train import main as train_main
    import sys as _sys
    _sys.argv = ["train.py", "--xgb-only"]    # Start with XGB only for speed
    train_main()


def run_fine_tune():
    """
    Fine-tune XGBoost models using only the last FINE_TUNE_DAYS of data.
    Loads existing models and continues training on recent data.
    Helps model adapt to current market regime without forgetting long-term patterns.
    """
    log.info(f"Step 2: Fine-tune — last {FINE_TUNE_DAYS} days")

    try:
        from xgboost import XGBClassifier
        import joblib
        import pandas as pd
        from ml_predictor.data_fetcher        import load_csv
        from ml_predictor.feature_engineering import build_features, FEATURE_COLS
        from ml_predictor.train               import (
            compute_time_weights, TIME_BUCKETS, _filter_bucket, XGB_PARAMS
        )
    except ImportError as e:
        log.error(f"Import error: {e}. Run: pip install xgboost scikit-learn joblib")
        return

    model_dir  = os.path.join(BASE_DIR, "models")
    cutoff     = datetime.now() - timedelta(days=FINE_TUNE_DAYS)

    for instrument in ["NIFTY50", "BANKNIFTY"]:
        log.info(f"\n  Fine-tuning {instrument}...")

        try:
            df_main = load_csv(instrument)
        except FileNotFoundError:
            log.warning(f"  {instrument} CSV not found — skipping")
            continue

        # Filter to recent data only
        df_recent = df_main[df_main.index >= cutoff]
        if len(df_recent) < 100:
            log.warning(f"  {instrument}: only {len(df_recent)} recent rows — skipping fine-tune")
            continue

        try:
            df_vix = load_csv("INDIAVIX")
            df_vix = df_vix[df_vix.index >= cutoff]
        except Exception:
            df_vix = None

        df_feat = build_features(df_recent, df_vix=df_vix)
        if df_feat.empty:
            log.warning(f"  {instrument}: empty feature set after engineering")
            continue

        scaler_path = os.path.join(model_dir, f"scaler_{instrument.lower()}.pkl")
        if not os.path.exists(scaler_path):
            log.warning(f"  Scaler not found for {instrument} — run full retrain first")
            continue
        scaler = joblib.load(scaler_path)

        for bucket in ["A", "B", "C"]:
            model_path = os.path.join(model_dir, f"xgb_{instrument.lower()}_{bucket}.json")
            if not os.path.exists(model_path):
                log.warning(f"  {instrument} bucket {bucket}: model not found — skipping")
                continue

            feat_b = _filter_bucket(df_feat, bucket)
            if len(feat_b) < 30:
                continue

            X_b  = pd.DataFrame(
                scaler.transform(feat_b[FEATURE_COLS]),
                index=feat_b.index, columns=FEATURE_COLS
            )
            y_b     = feat_b["target"].astype(int)
            weights = compute_time_weights(X_b.index)

            # Load existing model and continue training
            model = XGBClassifier(**{**XGB_PARAMS, "n_estimators": 100})
            model.load_model(model_path)
            booster = model.get_booster()   # XGBClassifier.fit() xgb_model must be a Booster, not a path
            model.fit(X_b, y_b, sample_weight=weights,
                      xgb_model=booster,
                      verbose=False)

            model.save_model(model_path)
            log.info(f"  {instrument} bucket {bucket}: fine-tuned on {len(X_b)} samples → saved")

    log.info("Fine-tune complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Full retrain from scratch (default: fine-tune only)")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info(f"  ML Retrain — {'FULL' if args.full else 'FINE-TUNE'}")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    run_data_append()

    if args.full:
        run_full_retrain()
    else:
        run_fine_tune()

    log.info("\nRetrain complete.")


if __name__ == "__main__":
    main()

