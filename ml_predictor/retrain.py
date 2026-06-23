"""
ml_predictor/retrain.py
─────────────────────────
Cron-driven retraining script.

REBUILD NOTES — fixes vs. old retrain_weekly.py:

1. Tree bloat fixed. Old code did:
       model = XGBClassifier(n_estimators=100)
       model.load_model(model_path)
       model.fit(X, y, xgb_model=model_path, ...) # ← adds 100 MORE trees,
                                                   #    every single week,
                                                   #    forever (5,700 trees
                                                   #    after 1 year).
   New code checks the existing model's current tree count before fitting
   and skips/caps the fine-tune if it would exceed MAX_TOTAL_TREES — forcing
   a full retrain instead once the cap is hit, rather than growing unbounded.

2. Warm-start now passes an explicit Booster object, not a path string.
   xgb_model= accepts a Booster, a path string, or an XGBModel per
   xgboost's own type signature (Union[Booster, str, XGBModel]) — passing
   the path directly is technically valid. This version instead does
   model.load_model(model_path) explicitly first and passes the resulting
   .get_booster() object to fit(xgb_model=...). One extra load call,
   negligible cost, and it's the more universally-compatible form across
   xgboost versions/setups — a real-world production patch on this exact
   codebase's old retrain_weekly.py independently arrived at the same fix
   after hitting a warm-start issue with the path-string form.

3. Bucket filtering now goes through regime.get_bucket_series() — the SAME
   function train.py and predictor.py use — instead of a private
   _filter_bucket() with its own cutoff table that could drift out of sync
   (this drift is exactly what caused the old bucket-mismatch bug).

4. Combined time-decay × class-weight sample weighting (train.compute_combined_weights)
   instead of time-decay alone, consistent with the new 3-class imbalanced target.

Crontab entries:
    # Daily: append today's candles (15:40 IST)
    40 15 * * 1-5 cd ~/paper && python3 ml_predictor/data_fetcher.py --append >> logs/ml_data.log 2>&1

    # Weekly: fine-tune on recent 30 days (Sunday 22:00 IST)
    0 22 * * 0 cd ~/paper && python3 ml_predictor/retrain.py >> logs/ml_retrain.log 2>&1

    # Monthly: full retrain from scratch (1st Saturday 21:00 IST)
    0 21 1-7 * 6 cd ~/paper && python3 ml_predictor/retrain.py --full >> logs/ml_retrain.log 2>&1

Usage:
    python3 ml_predictor/retrain.py           # weekly fine-tune (last 30 days)
    python3 ml_predictor/retrain.py --full    # full retrain (all history)
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

FINE_TUNE_DAYS   = 30     # fine-tune uses last N days of data
FINE_TUNE_TREES  = 100    # trees added per fine-tune run, IF under the cap
# Sizing: full retrain resets to XGB_PARAMS["n_estimators"] = 500 trees.
# A monthly full retrain means at most ~4 weekly fine-tunes (4x100=400 trees)
# accumulate before the next reset. Cap is set with headroom above that
# (500 + 4*100 = 900) so the cap is rarely hit in practice as long as the
# monthly --full cron actually runs — it is NOT a substitute for that cron,
# just a safety ceiling in case a monthly run is missed for a cycle or two.
MAX_TOTAL_TREES  = 1200   # hard ceiling — once hit, fine-tune is skipped and
                           # a full retrain is required to reset tree count


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
    """Full retrain on all historical data. Resets tree count to XGB_PARAMS['n_estimators']."""
    log.info("Step 2: Full retrain — all history (resets tree count)")
    from ml_predictor.train import main as train_main
    import sys as _sys
    _sys.argv = ["train.py"]   # train both XGBoost and LSTM in a full retrain
    train_main()


def _current_tree_count(model_path: str) -> int:
    """Read how many trees an existing saved XGBoost model currently has."""
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(model_path)
    booster = m.get_booster()
    # num_boosted_rounds() returns total trees for multi:softprob too
    # (it already accounts for num_class internally in recent xgboost versions;
    # if unavailable, fall back to trees_to_dataframe length / num_class).
    try:
        return booster.num_boosted_rounds()
    except AttributeError:
        df = booster.trees_to_dataframe()
        return df["Tree"].nunique()


def run_fine_tune():
    """
    Fine-tune XGBoost models using only the last FINE_TUNE_DAYS of data.
    Loads existing models and continues training on recent data — UNLESS
    the model has already reached MAX_TOTAL_TREES, in which case fine-tune
    is skipped for that bucket and a full retrain is recommended instead.
    """
    log.info(f"Step 2: Fine-tune — last {FINE_TUNE_DAYS} days")

    try:
        from xgboost import XGBClassifier
        import joblib
        import pandas as pd
        from ml_predictor.data_fetcher import load_csv
        from ml_predictor.features     import build_features, FEATURE_COLS
        from ml_predictor.targets      import add_target, class_distribution
        from ml_predictor.regime       import get_bucket_series
        from ml_predictor.train        import compute_combined_weights, XGB_PARAMS
    except ImportError as e:
        log.error(f"Import error: {e}. Run: pip install xgboost scikit-learn joblib")
        return

    model_dir = os.path.join(BASE_DIR, "models")
    cutoff    = datetime.now() - timedelta(days=FINE_TUNE_DAYS)

    for instrument in ["NIFTY50", "BANKNIFTY"]:
        log.info(f"\n  Fine-tuning {instrument}...")

        try:
            df_main = load_csv(instrument)
        except FileNotFoundError:
            log.warning(f"  {instrument} CSV not found — skipping")
            continue

        df_recent = df_main[df_main.index >= cutoff]
        if len(df_recent) < 100:
            log.warning(f"  {instrument}: only {len(df_recent)} recent rows — skipping fine-tune")
            continue

        try:
            df_vix = load_csv("INDIAVIX")
            df_vix = df_vix[df_vix.index >= cutoff]
        except Exception:
            df_vix = None

        cross_map  = {"BANKNIFTY": "NIFTY50", "NIFTY50": "BANKNIFTY"}
        cross_name = cross_map.get(instrument)
        try:
            df_cross = load_csv(cross_name)
            df_cross = df_cross[df_cross.index >= cutoff]
        except Exception:
            df_cross = None

        df_feat = build_features(df_recent, df_vix=df_vix, df_cross=df_cross)
        df_t    = add_target(df_recent)
        df_feat = df_feat.join(df_t["target"], how="inner").dropna(subset=["target"])

        if df_feat.empty:
            log.warning(f"  {instrument}: empty feature set after engineering — skipping")
            continue

        scaler_path = os.path.join(model_dir, f"scaler_{instrument.lower()}.pkl")
        if not os.path.exists(scaler_path):
            log.warning(f"  Scaler not found for {instrument} — run full retrain first")
            continue
        scaler = joblib.load(scaler_path)

        bucket_series = get_bucket_series(df_feat.index)

        for bucket in ["A", "B", "C"]:
            model_path = os.path.join(model_dir, f"xgb_{instrument.lower()}_{bucket}.json")
            if not os.path.exists(model_path):
                log.warning(f"  {instrument} bucket {bucket}: model not found — skipping")
                continue

            # ── Tree cap check — this is the fix for BUG 8 ──────────────────
            current_trees = _current_tree_count(model_path)
            if current_trees + FINE_TUNE_TREES > MAX_TOTAL_TREES:
                log.warning(
                    f"  {instrument} bucket {bucket}: at {current_trees} trees, "
                    f"fine-tuning would exceed cap ({MAX_TOTAL_TREES}). "
                    f"Skipping fine-tune — run --full to reset this model."
                )
                continue

            feat_b = df_feat[bucket_series == bucket]
            if len(feat_b) < 30:
                continue

            X_b = pd.DataFrame(
                scaler.transform(feat_b[FEATURE_COLS]),
                index=feat_b.index, columns=FEATURE_COLS,
            )
            y_b     = feat_b["target"].astype(int)
            weights = compute_combined_weights(X_b, y_b)

            # Warm-start fit: xgb_model= accepts a Booster, a path string, or
            # an XGBModel per xgboost's own sklearn.py signature (confirmed
            # against the current source: xgb_model: Optional[Union[Booster,
            # str, XGBModel]]). Passing the path string directly is valid.
            # That said, this loads the existing booster explicitly first and
            # passes the Booster object itself, not the path — a stricter,
            # more universally-compatible form across xgboost versions, and
            # it sidesteps a real-world report of warm-start issues when
            # passing a path string on at least one production setup.
            # Belt-and-suspenders: costs one extra load_model() call, which
            # is negligible next to the fit() that follows.
            warm_start_model = XGBClassifier()
            warm_start_model.load_model(model_path)
            existing_booster = warm_start_model.get_booster()

            model = XGBClassifier(**{**XGB_PARAMS, "n_estimators": FINE_TUNE_TREES})
            model.fit(X_b, y_b, sample_weight=weights, xgb_model=existing_booster, verbose=False)
            model.save_model(model_path)

            new_trees = _current_tree_count(model_path)
            log.info(
                f"  {instrument} bucket {bucket}: fine-tuned on {len(X_b):,} samples "
                f"({current_trees} → {new_trees} trees) → saved"
            )

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


# Alias for direct programmatic use (e.g. `from ml_predictor.retrain import
# fine_tune`) rather than always shelling out to `python3 retrain.py`.
# run_fine_tune() is the canonical name used internally and in the crontab
# examples above; this just prevents an ImportError if future code (e.g. a
# strategy wanting to trigger an on-demand fine-tune) reaches for the more
# obvious short name instead.
fine_tune = run_fine_tune
full_retrain = run_full_retrain
