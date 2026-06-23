"""
ml_predictor/live_tracker.py
─────────────────────────────
Standalone script — runs independently alongside your algo system.
NO changes to any strategy file needed.

REBUILD NOTES — what changed vs. the old live_tracker.py:

1. Target is now 4-candle-forward (±0.15%), not next-candle green/red.
   This changes the resolution loop fundamentally: a prediction made when
   candle T closes is for the return measured at T+4 (20 minutes later),
   not T+1. The tracker now holds each prediction PENDING for 4 candles
   instead of 1, and resolves it against the actual forward return —
   computed with the same session-safe logic as targets.py (if the 4-candle
   window would cross a session boundary, the prediction is marked N/A
   rather than resolved against next-day data).

2. VIX fetch now returns real multi-row history via fetch_recent_candles
   (same function already used for the main instrument), not the old
   fetch_vix() which returned a single scalar — that was the same
   scalar-broadcast bug found in predictor.py, just in a different file.

3. Bucket selection goes through regime.get_bucket() — the same function
   train.py and predictor.py use.

4. Output is now DOWN / FLAT / UP (3-class), not GREEN / RED.

What it does every 5 minutes:
  1. Fetches last 80 completed 5-min candles from Kite (+ VIX + cross-index)
  2. Runs ML prediction → DOWN / FLAT / UP
  3. Holds the prediction pending for 4 candles (20 min)
  4. Resolves vs. actual 4-candle-forward return (session-safe)
  5. Logs result to CSV: ml_predictor/data/predictions_<instrument>_<date>.csv

Run:
    # Terminal 1 (your existing system):
    python3 t.py

    # Terminal 2 (this script — totally independent):
    python3 ml_predictor/live_tracker.py --instrument BANKNIFTY

Output CSV columns:
    candle_time | resolve_time | predicted | actual_class | actual_return_pct |
    confidence | correct | anchor_open | anchor_close | resolve_close |
    bucket | vix_regime | vix | reason

    predicted          UP / DOWN / FLAT — model's call at candle_time, for
                        the 20-minute (4-candle) window ending at resolve_time
    actual_class       UP / DOWN / FLAT / N/A — what really happened over
                        that same window (N/A if it crossed a session boundary)
    actual_return_pct  the real % move over that window (e.g. 0.183 = +0.183%)
    correct            YES / NO / N/A — predicted vs actual_class

Stop:
    Ctrl+C — saves CSV before exit.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone, time as dtime, date

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)

from ml_predictor.targets import FORWARD_CANDLES, DEADBAND
# Imported (not redefined) so this file can never drift from targets.py's
# actual target shape. Previously these were hardcoded local constants here
# with a comment promising they'd match targets.py — nothing enforced that.
# If FORWARD_CANDLES were ever changed in targets.py (e.g. 4 -> 6 candles)
# and training re-ran correctly against the new value, this file would have
# silently kept resolving predictions against a stale 4-candle window,
# making the entire accuracy CSV meaningless with no error or warning.

from ml_predictor.predictor import XGB_WEIGHT, LSTM_WEIGHT
# Same reasoning: predictor.py defines these as named constants, but this
# file previously hardcoded "0.60 * xgb_probs + 0.40 * lstm_probs" inline
# with no import. Tuning the ensemble weights after backtesting would
# silently have no effect on this file's accuracy CSV — it would keep
# measuring a DIFFERENT ensemble than the one predictor.py actually serves
# to your strategy, with no warning that the two had drifted apart.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.live_tracker")

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(tz=IST).replace(tzinfo=None)


INSTRUMENT_TOKENS = {
    "NIFTY50":   256265,
    "BANKNIFTY": 260105,
    "INDIAVIX":  264969,
}

MARKET_START   = dtime(9, 15)
MARKET_END     = dtime(15, 31)   # matches core/market_hub.py's MARKET_CLOSE exactly
                                   # — this is when the MAIN LOOP exits, not the
                                   # last tradeable candle (see LAST_CANDLE_CLOSE).
LAST_CANDLE_CLOSE = dtime(15, 25) # the LAST candle that will, in practice,
                                   # ever exist. The between_time filter
                                   # below is set to "15:30" (the nominal
                                   # NSE session end, matching predictor.py
                                   # and data_fetcher.py for consistency),
                                   # but Kite has never actually returned a
                                   # candle at 15:30 for this session in the
                                   # historical data checked (2083/2087
                                   # trading days end at 15:25). This
                                   # constant reflects that empirical reality
                                   # — widening the fetch filter doesn't
                                   # manufacture a candle that doesn't exist.
                                   # A resolve_at landing in the (15:25, 15:31]
                                   # gap can never be resolved against real
                                   # data and must be treated as a boundary
                                   # crossing, same as if it landed on the
                                   # next calendar day.
CANDLE_MIN     = 5
FETCH_BARS     = 80      # EMA50 warmup + buffer
FETCH_DAYS     = 3       # covers Monday morning gaps


# ── Kite loader ───────────────────────────────────────────────────────────────

def _load_kite():
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        log.error("kiteconnect not installed.")
        sys.exit(1)

    token_file = os.path.join(ROOT_DIR, "token.json")
    if not os.path.exists(token_file):
        log.error(f"token.json not found at {token_file}. Start t.py first.")
        sys.exit(1)

    with open(token_file) as f:
        data = json.load(f)

    if data.get("date") != str(date.today()):
        log.warning("token.json is from a previous day — trying auto_login...")
        try:
            from core.auto_login import auto_login
            auto_login()
            with open(token_file) as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Auto-login failed: {e}. Run t.py first.")
            sys.exit(1)

    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=data["api_key"])
    kite.set_access_token(data["access_token"])
    log.info("Kite session loaded from token.json")
    return kite


# ── Candle fetcher ────────────────────────────────────────────────────────────

def fetch_recent_candles(kite, token: int, n_bars: int = FETCH_BARS) -> pd.DataFrame:
    """
    Fetch last N completed 5-min candles for the given token.
    Used for the main instrument AND for VIX/cross-index — there is no
    separate scalar-only VIX fetcher in this rebuild; everything goes
    through this one real-history function.
    """
    now     = _now_ist()
    from_dt = now - timedelta(days=FETCH_DAYS)
    to_dt   = now - timedelta(minutes=CANDLE_MIN)

    try:
        raw = kite.historical_data(
            instrument_token=token,
            from_date=from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_date=to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval="5minute",
            continuous=False,
            oi=False,
        )
    except Exception as e:
        log.warning(f"Kite fetch error: {e}")
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.between_time("09:15", "15:30")
    return df.tail(n_bars).copy()


# ── Prediction ────────────────────────────────────────────────────────────────

def run_prediction(candle_time: datetime, df_candles: pd.DataFrame, df_vix: pd.DataFrame,
                    df_cross: pd.DataFrame, xgb_models: dict, lstm_model, scaler) -> dict:
    """
    Run ML prediction on the given candle DataFrame. Returns a 3-class
    (DOWN/FLAT/UP) prediction dict. df_vix and df_cross are REAL multi-row
    histories now (not scalar broadcasts) — same contract as predictor.py.

    candle_time: the ANCHOR candle's own close timestamp (current_candle_open
    in the main loop), used for bucket selection. This must NOT be the wall
    clock at call time — the main loop wakes up 5-10s after a candle closes,
    so at a bucket boundary (e.g. anchor=10:25, called at 10:30:05) the wall
    clock has already crossed into the next bucket while the candle being
    predicted on still belongs to the previous one. predictor.py already
    gets this right (it takes a ts param); this matches that contract.
    """
    from ml_predictor.features import build_features, FEATURE_COLS
    from ml_predictor.targets  import DOWN, FLAT, UP, TARGET_CLASSES
    from ml_predictor.regime   import get_bucket, get_vix_regime

    vix_level = float(df_vix["close"].iloc[-1]) if df_vix is not None and not df_vix.empty else 15.0
    vix_regime = get_vix_regime(vix_level)

    try:
        df_feat = build_features(df_candles.copy(), df_vix=df_vix, df_cross=df_cross)
    except Exception as e:
        log.warning(f"Feature engineering error: {e}")
        return {"predicted": "SKIP", "confidence": 0.0, "probs": {},
                "vix_regime": "ERROR", "reason": str(e)}

    if df_feat.empty:
        return {"predicted": "SKIP", "confidence": 0.0, "probs": {},
                "vix_regime": "UNKNOWN", "reason": "insufficient_features"}

    last_row = df_feat[FEATURE_COLS].iloc[[-1]]

    try:
        X = scaler.transform(last_row)
    except Exception as e:
        return {"predicted": "SKIP", "confidence": 0.0, "probs": {},
                "vix_regime": "ERROR", "reason": f"scaler_error: {e}"}

    bucket = get_bucket(candle_time)
    xgb_model = xgb_models.get(bucket) or next(iter(xgb_models.values()))
    xgb_probs = xgb_model.predict_proba(X)[0]  # [DOWN, FLAT, UP]

    lstm_probs = None
    if lstm_model is not None:
        try:
            X_full   = scaler.transform(df_feat[FEATURE_COLS])
            lookback = lstm_model.input_shape[1]
            if len(X_full) >= lookback:
                seq = X_full[-lookback:].reshape(1, lookback, -1)
                lstm_probs = lstm_model.predict(seq, verbose=0)[0]
        except Exception as e:
            log.debug(f"LSTM error: {e}")

    if lstm_probs is not None:
        probs  = XGB_WEIGHT * xgb_probs + LSTM_WEIGHT * lstm_probs
        source = f"ensemble xgb_up={xgb_probs[UP]:.3f} lstm_up={lstm_probs[UP]:.3f}"
    else:
        probs  = xgb_probs
        source = f"xgb_only up={xgb_probs[UP]:.3f}"

    prob_dict = {"DOWN": round(float(probs[DOWN]), 4),
                 "FLAT": round(float(probs[FLAT]), 4),
                 "UP":   round(float(probs[UP]), 4)}
    pred_idx   = int(np.argmax(probs))
    predicted  = TARGET_CLASSES[pred_idx]
    confidence = round(float(probs[pred_idx]), 4)

    return {
        "predicted":  predicted,
        "confidence": confidence,
        "probs":      prob_dict,
        "vix_regime": vix_regime,
        "vix":        vix_level,
        "bucket":     bucket,
        "reason":     source,
    }


# ── CSV logger ────────────────────────────────────────────────────────────────

class PredictionLogger:
    """
    Logs predictions and 4-candle-forward actuals to daily CSV.
    Each prediction is held pending for FORWARD_CANDLES (4) candle closes
    before being resolved — this is the core change from the old 1-candle
    resolution loop.
    """

    # Column order is deliberate: predicted -> actual -> confidence -> correct
    # is the primary line of sight a person reads. resolve_time makes it
    # obvious which 20-min window "actual" refers to without having to
    # compute candle_time + 20min by hand.
    COLUMNS = [
        "candle_time", "resolve_time",
        "predicted", "actual_class", "actual_return_pct",
        "confidence", "correct",
        "anchor_open", "anchor_close", "resolve_close",
        "bucket", "vix_regime", "vix", "reason",
    ]

    def __init__(self, instrument: str):
        today    = _now_ist().date()
        filename = f"predictions_{instrument.lower()}_{today}.csv"
        self.path = os.path.join(DATA_DIR, filename)

        if os.path.exists(self.path):
            self._df = pd.read_csv(self.path)
            log.info(f"Resuming existing log: {self.path} ({len(self._df)} rows)")
        else:
            self._df = pd.DataFrame(columns=self.COLUMNS)
            log.info(f"New prediction log: {self.path}")

        # Pending: {anchor_candle_time_str: {..., "resolve_at": datetime}}
        self._pending: dict = {}

    def record_prediction(self, anchor_time: datetime, anchor_candle: dict, pred: dict):
        """
        Record a prediction made at the close of `anchor_time`. It will be
        resolved FORWARD_CANDLES (4) candles later, against the actual
        forward return from anchor_close to that candle's close.
        """
        key = anchor_time.strftime("%Y-%m-%d %H:%M")
        if key in self._pending:
            return

        resolve_at = anchor_time + timedelta(minutes=CANDLE_MIN * FORWARD_CANDLES)

        self._pending[key] = {
            "candle_time":   key,
            "resolve_time":  resolve_at.strftime("%Y-%m-%d %H:%M"),
            "anchor_open":   round(anchor_candle["open"], 2),
            "anchor_close":  round(anchor_candle["close"], 2),
            "predicted":     pred["predicted"],
            "confidence":    pred["confidence"],
            "bucket":        pred.get("bucket", ""),
            "vix_regime":    pred.get("vix_regime", ""),
            "vix":           pred.get("vix", ""),
            "reason":        pred.get("reason", ""),
            "_resolve_at":   resolve_at,
        }
        log.info(
            f"📌 Prediction recorded | anchor={key} | resolve_at={resolve_at.strftime('%H:%M')} | "
            f"predicted={pred['predicted']} conf={pred['confidence']:.1%} | {pred.get('reason','')}"
        )

    def try_resolve(self, df_recent: pd.DataFrame):
        """
        Check all pending predictions: if the current time has passed
        their resolve_at candle close AND that candle exists in df_recent
        on the SAME trading day as the anchor (session-safe, mirrors
        targets.add_target's groupby(date) rule), resolve them.
        Predictions whose 4-candle window would cross a session boundary
        are marked N/A rather than resolved against next-day data.
        """
        if not self._pending or df_recent.empty:
            return

        now = _now_ist()
        resolved_keys = []

        for key, row in list(self._pending.items()):
            resolve_at = row["_resolve_at"]
            if now < resolve_at + timedelta(seconds=5):
                continue  # not due yet

            anchor_date = pd.Timestamp(key).date()
            # Session boundary check: a 4-candle (20 min) window starting
            # near 15:10-15:25 can produce a resolve_at that lands AFTER the
            # last candle that will ever actually exist (15:25), even though
            # it's still BEFORE the main loop's exit time (MARKET_END=15:31).
            # Checking against MARKET_END here was the bug — it let resolve_at
            # values in the (15:25, 15:31] gap through as "not a boundary
            # crossing," but no candle ever arrives there, so the prediction
            # would stay in _pending forever and silently vanish when the
            # process exits at MARKET_END with no N/A ever recorded.
            # The correct check is against LAST_CANDLE_CLOSE, the real data
            # ceiling, not the process's own exit time.
            crosses_boundary = (
                resolve_at.date() != anchor_date
                or resolve_at.time() > LAST_CANDLE_CLOSE
            )
            if crosses_boundary:
                # Session boundary crossed — same rule as targets.py's
                # groupby(date).shift(-4): do not resolve against next day.
                row["resolve_close"]      = None
                row["actual_return_pct"]  = None
                row["actual_class"]       = "N/A"
                row["correct"]            = "N/A"
                self._finalize(row)
                resolved_keys.append(key)
                log.info(f"⏭️  {key}: session boundary crossed — marked N/A (not resolved cross-day)")
                continue

            match = df_recent[df_recent.index.floor("5min") == pd.Timestamp(resolve_at)]
            if match.empty:
                continue  # data not available yet, try again next loop

            resolve_close = float(match.iloc[0]["close"])
            fwd_ret = (resolve_close - row["anchor_close"]) / row["anchor_close"]

            if fwd_ret > DEADBAND:
                actual_class = "UP"
            elif fwd_ret < -DEADBAND:
                actual_class = "DOWN"
            else:
                actual_class = "FLAT"

            predicted = row["predicted"]
            correct = "N/A" if predicted == "SKIP" else ("YES" if predicted == actual_class else "NO")

            row["resolve_close"]      = round(resolve_close, 2)
            row["actual_return_pct"]  = round(fwd_ret * 100, 3)   # e.g. 0.183 means +0.183%
            row["actual_class"]       = actual_class
            row["correct"]            = correct

            self._finalize(row)
            resolved_keys.append(key)

            icon = "✅" if correct == "YES" else ("❌" if correct == "NO" else "⏭️")
            log.info(
                f"{icon} Result | anchor={key} | predicted={predicted} actual={actual_class} "
                f"correct={correct} | fwd_ret={fwd_ret:+.3%}"
            )

        for key in resolved_keys:
            self._pending.pop(key)

        if resolved_keys:
            self._print_summary()

    def _finalize(self, row: dict):
        row = {k: v for k, v in row.items() if k != "_resolve_at"}
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self._save()

    def _save(self):
        self._df.to_csv(self.path, index=False)

    def _print_summary(self):
        df = self._df.copy()
        actionable = df[df["correct"].isin(["YES", "NO"])]
        if len(actionable) == 0:
            return
        correct_n = (actionable["correct"] == "YES").sum()
        total_n   = len(actionable)
        accuracy  = correct_n / total_n
        skipped   = (df["correct"] == "N/A").sum()

        log.info(
            f"📊 Today's accuracy: {correct_n}/{total_n} = {accuracy:.1%} "
            f"| skipped/NA={skipped} | total={len(df)}"
        )

    def save_on_exit(self):
        self._save()
        log.info(f"CSV saved on exit → {self.path}")


# ── Time helpers ──────────────────────────────────────────────────────────────

def next_candle_close(ts: datetime) -> datetime:
    minutes     = ts.minute
    next_minute = ((minutes // 5) + 1) * 5
    return ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)


def seconds_until(target: datetime) -> float:
    return max(0.0, (target - _now_ist()).total_seconds())


def last_completed_candle_time(ts: datetime) -> datetime:
    minutes     = ts.minute
    last_minute = (minutes // 5) * 5
    return ts.replace(minute=last_minute, second=0, microsecond=0) - timedelta(minutes=5)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_models(instrument: str):
    import joblib

    model_dir  = os.path.join(BASE_DIR, "models")
    inst_lower = instrument.lower()

    scaler_path = os.path.join(model_dir, f"scaler_{inst_lower}.pkl")
    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}")
        log.error("Run: python3 ml_predictor/train.py")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    log.info(f"Scaler loaded ← {scaler_path}")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        log.error("xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    xgb_models = {}
    for bucket in ["A", "B", "C"]:
        path = os.path.join(model_dir, f"xgb_{inst_lower}_{bucket}.json")
        if os.path.exists(path):
            m = XGBClassifier()
            m.load_model(path)
            xgb_models[bucket] = m
            log.info(f"XGBoost {bucket} loaded ← {path}")
        else:
            log.warning(f"XGBoost {bucket} not found: {path}")

    if not xgb_models:
        log.error("No XGBoost models found. Run train.py first.")
        sys.exit(1)

    lstm_model = None
    lstm_path  = os.path.join(model_dir, f"lstm_{inst_lower}.keras")
    if os.path.exists(lstm_path):
        try:
            from tensorflow.keras.models import load_model
            lstm_model = load_model(lstm_path)
            log.info(f"LSTM loaded ← {lstm_path}")
        except Exception as e:
            log.warning(f"LSTM load failed: {e} — using XGBoost only")
    else:
        log.info("No LSTM model — using XGBoost only")

    return xgb_models, lstm_model, scaler


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="BANKNIFTY",
                        choices=["NIFTY50", "BANKNIFTY"],
                        help="Instrument to track")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run once immediately (for testing)")
    args = parser.parse_args()

    instrument = args.instrument.upper()
    token      = INSTRUMENT_TOKENS[instrument]
    cross_map  = {"BANKNIFTY": "NIFTY50", "NIFTY50": "BANKNIFTY"}
    cross_token = INSTRUMENT_TOKENS[cross_map[instrument]]

    log.info("=" * 60)
    log.info(f"  ML LIVE TRACKER — {instrument}")
    log.info(f"  Predicts {FORWARD_CANDLES}-candle forward move (±{DEADBAND:.2%}) | Logs to CSV")
    log.info("=" * 60)

    xgb_models, lstm_model, scaler = load_models(instrument)
    kite = _load_kite()
    logger = PredictionLogger(instrument)

    def _shutdown(sig, frame):
        log.info("\nShutting down...")
        logger.save_on_exit()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    last_predicted_candle = None

    log.info("\nWaiting for market hours (09:15–15:25 IST)...")

    while True:
        now = _now_ist()

        if not args.dry_run:
            if now.time() < MARKET_START:
                wait = (datetime.combine(now.date(), MARKET_START) - now).total_seconds()
                log.info(f"Pre-market. Market opens in {wait/60:.0f} min. Sleeping...")
                time.sleep(min(wait, 300))
                continue

            if now.time() > MARKET_END:
                log.info("Market closed. Final save and exit.")
                logger.save_on_exit()
                sys.exit(0)

        current_candle_open = last_completed_candle_time(now)

        # ── STEP 1: Try to resolve any pending predictions ──────────────────
        df_recent = fetch_recent_candles(kite, token, n_bars=20)
        if not df_recent.empty:
            logger.try_resolve(df_recent)

        # ── STEP 2: Make new prediction ──────────────────────────────────────
        if current_candle_open != last_predicted_candle:
            log.info(f"\n{'─'*55}")
            log.info(f"New candle closed: {current_candle_open.strftime('%H:%M')} — making prediction...")

            df_candles = fetch_recent_candles(kite, token, n_bars=FETCH_BARS)

            if df_candles.empty or len(df_candles) < 30:
                log.warning(f"Insufficient candles ({len(df_candles)}) — skipping")
            else:
                df_vix   = fetch_recent_candles(kite, INSTRUMENT_TOKENS["INDIAVIX"], n_bars=FETCH_BARS)
                df_cross = fetch_recent_candles(kite, cross_token, n_bars=FETCH_BARS)

                pred = run_prediction(current_candle_open, df_candles, df_vix, df_cross,
                                       xgb_models, lstm_model, scaler)

                last_candle = df_candles.iloc[-1].to_dict()
                logger.record_prediction(current_candle_open, last_candle, pred)
                last_predicted_candle = current_candle_open

            if args.dry_run:
                log.info("Dry run complete.")
                logger.save_on_exit()
                sys.exit(0)

        next_close = next_candle_close(now)
        sleep_secs = seconds_until(next_close) + 5

        if sleep_secs > 1:
            log.info(f"Next candle closes at {next_close.strftime('%H:%M:%S')} "
                     f"— sleeping {sleep_secs:.0f}s...")
            slept = 0
            while slept < sleep_secs:
                chunk = min(30, sleep_secs - slept)
                time.sleep(chunk)
                slept += chunk


if __name__ == "__main__":
    main()
