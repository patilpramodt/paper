"""
ml_predictor/live_tracker.py
─────────────────────────────
Standalone script — runs independently alongside your algo system.
NO changes to any strategy file needed.

What it does every 5 minutes:
  1. Fetches last 60 completed 5-min candles from Kite (ATM index candle)
  2. Runs ML prediction → GREEN or RED
  3. Waits for next candle to close
  4. Compares prediction vs actual
  5. Logs result to CSV: ml_predictor/data/predictions_YYYY-MM-DD.csv

Run:
    # Terminal 1 (your existing system):
    python t.py

    # Terminal 2 (this script — totally independent):
    python ml_predictor/live_tracker.py --instrument BANKNIFTY

    # Or for Nifty:
    python ml_predictor/live_tracker.py --instrument NIFTY50

Output CSV columns:
    candle_time | open | high | low | close | actual_color |
    predicted | confidence | correct | regime | vix | reason

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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml.live_tracker")

# ── IST ───────────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=IST).replace(tzinfo=None)

# ── Instrument config ─────────────────────────────────────────────────────────
INSTRUMENT_TOKENS = {
    "NIFTY50":   256265,
    "BANKNIFTY": 260105,
    "INDIAVIX":  264969,
}

MARKET_START = dtime(9, 15)
MARKET_END   = dtime(15, 30)
CANDLE_MIN   = 5          # 5-minute candles
FETCH_BARS   = 70         # fetch last 70 bars (covers EMA50 warmup + buffer)
FETCH_DAYS   = 3          # look back 3 days to cover Monday morning gaps


# ── Kite loader ───────────────────────────────────────────────────────────────

def _load_kite():
    """Load KiteConnect from token.json — reuses your existing session."""
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
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (IST)
    """
    now      = _now_ist()
    from_dt  = now - timedelta(days=FETCH_DAYS)
    to_dt    = now - timedelta(minutes=CANDLE_MIN)  # exclude currently forming candle

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
    # Strip tz — Kite returns IST-aware timestamps but all internal logic
    # uses tz-naive datetimes (via _now_ist which already strips tz).
    # Keeping tz-aware here causes "Cannot compare dtypes" errors downstream.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    # Market hours only
    df = df.between_time("09:15", "15:25")

    # Return last N bars
    return df.tail(n_bars).copy()


def fetch_vix(kite) -> float | None:
    """Fetch current India VIX value."""
    try:
        raw = kite.historical_data(
            instrument_token=INSTRUMENT_TOKENS["INDIAVIX"],
            from_date=(_now_ist() - timedelta(days=1)).strftime("%Y-%m-%d"),
            to_date=_now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            interval="5minute",
        )
        if raw:
            return float(raw[-1]["close"])
    except Exception:
        pass
    return None


# ── Prediction ────────────────────────────────────────────────────────────────

def run_prediction(df_candles: pd.DataFrame, instrument: str, vix: float,
                   xgb_models: dict, lstm_model, scaler) -> dict:
    """
    Run ML prediction on the given candle DataFrame.
    Returns prediction dict.
    """
    from ml_predictor.feature_engineering import build_features, FEATURE_COLS
    from ml_predictor.regime_detector     import RegimeDetector

    rd = RegimeDetector()

    if not rd.is_tradeable(vix or 15.0):
        return {
            "predicted": "SKIP", "confidence": 0.0, "raw_prob": 0.5,
            "regime": "CRISIS", "reason": f"VIX={vix} CRISIS"
        }

    # Build VIX DataFrame aligned to candle index
    df_vix = None
    if vix is not None:
        df_vix = pd.DataFrame({"close": vix}, index=df_candles.index)

    try:
        df_feat = build_features(df_candles.copy(), df_vix=df_vix)
    except Exception as e:
        log.warning(f"Feature engineering error: {e}")
        return {"predicted": "SKIP", "confidence": 0.0, "raw_prob": 0.5,
                "regime": "ERROR", "reason": str(e)}

    if df_feat.empty or len(df_feat) < 2:
        return {"predicted": "SKIP", "confidence": 0.0, "raw_prob": 0.5,
                "regime": "UNKNOWN", "reason": "insufficient_features"}

    # Use second-to-last row — the LAST COMPLETED candle
    # (last row's target = prediction for the candle we're about to observe)
    last_row = df_feat[FEATURE_COLS].iloc[[-1]]

    try:
        X = scaler.transform(last_row)
    except Exception as e:
        return {"predicted": "SKIP", "confidence": 0.0, "raw_prob": 0.5,
                "regime": "ERROR", "reason": f"scaler_error: {e}"}

    # Determine time bucket
    now_time = _now_ist().time()
    if now_time < dtime(10, 30):
        bucket = "A"
    elif now_time < dtime(14, 0):
        bucket = "B"
    else:
        bucket = "C"

    xgb_model = xgb_models.get(bucket) or next(iter(xgb_models.values()))
    xgb_prob  = float(xgb_model.predict_proba(X)[0, 1])

    # LSTM
    lstm_prob = None
    if lstm_model is not None:
        try:
            X_full   = scaler.transform(df_feat[FEATURE_COLS])
            lookback = lstm_model.input_shape[1]
            if len(X_full) >= lookback:
                seq      = X_full[-lookback:].reshape(1, lookback, -1)
                lstm_prob = float(lstm_model.predict(seq, verbose=0)[0, 0])
        except Exception as e:
            log.debug(f"LSTM error: {e}")

    # Ensemble
    if lstm_prob is not None:
        raw_prob = 0.60 * xgb_prob + 0.40 * lstm_prob
        source   = f"ensemble xgb={xgb_prob:.3f} lstm={lstm_prob:.3f}"
    else:
        raw_prob = xgb_prob
        source   = f"xgb_only={xgb_prob:.3f}"

    # Regime threshold
    rd_info   = rd.describe(vix or 15.0, _now_ist())
    threshold = rd_info["threshold"]
    inv_thr   = 1.0 - threshold

    if raw_prob >= threshold:
        predicted  = "GREEN"
        confidence = raw_prob
    elif raw_prob <= inv_thr:
        predicted  = "RED"
        confidence = 1.0 - raw_prob
    else:
        predicted  = "SKIP"
        confidence = abs(raw_prob - 0.5) * 2

    return {
        "predicted":  predicted,
        "confidence": round(confidence, 4),
        "raw_prob":   round(raw_prob, 4),
        "regime":     rd_info["regime"],
        "threshold":  threshold,
        "reason":     source,
        "bucket":     bucket,
    }


# ── CSV logger ────────────────────────────────────────────────────────────────

class PredictionLogger:
    """Logs predictions and actuals to daily CSV."""

    COLUMNS = [
        "candle_time", "open", "high", "low", "close",
        "actual_color", "predicted", "confidence",
        "correct", "regime", "vix", "raw_prob", "bucket", "reason"
    ]

    def __init__(self, instrument: str):
        today    = _now_ist().date()
        filename = f"predictions_{instrument.lower()}_{today}.csv"
        self.path = os.path.join(DATA_DIR, filename)

        # Load existing if present (resume after restart)
        if os.path.exists(self.path):
            self._df = pd.read_csv(self.path)
            log.info(f"Resuming existing log: {self.path} ({len(self._df)} rows)")
        else:
            self._df = pd.DataFrame(columns=self.COLUMNS)
            log.info(f"New prediction log: {self.path}")

        # Pending: {candle_time_str: prediction_dict}
        self._pending: dict = {}

    def record_prediction(self, candle_time: datetime, candle: dict, pred: dict):
        """Record a prediction. Actual will be filled when next candle closes."""
        key = candle_time.strftime("%Y-%m-%d %H:%M")

        if key in self._pending:
            return  # already recorded

        self._pending[key] = {
            "candle_time": key,
            "open":        round(candle["open"],  2),
            "high":        round(candle["high"],  2),
            "low":         round(candle["low"],   2),
            "close":       round(candle["close"], 2),
            "predicted":   pred["predicted"],
            "confidence":  pred["confidence"],
            "raw_prob":    pred["raw_prob"],
            "regime":      pred["regime"],
            "vix":         pred.get("vix", ""),
            "bucket":      pred.get("bucket", ""),
            "reason":      pred.get("reason", ""),
        }
        log.info(
            f"📌 Prediction recorded | candle={key} | "
            f"predicted={pred['predicted']} conf={pred['confidence']:.1%} | "
            f"regime={pred['regime']} | {pred.get('reason','')}"
        )

    def resolve_actual(self, candle_time: datetime, actual_candle: dict):
        """
        Called after the predicted candle closes.
        Fills actual_color and correct, writes to CSV.
        """
        key = candle_time.strftime("%Y-%m-%d %H:%M")

        if key not in self._pending:
            return

        row = self._pending.pop(key)

        actual_color = "GREEN" if actual_candle["close"] >= actual_candle["open"] else "RED"
        predicted    = row["predicted"]

        if predicted == "SKIP":
            correct = "N/A"
        else:
            correct = "YES" if predicted == actual_color else "NO"

        row["actual_color"] = actual_color
        row["correct"]      = correct

        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self._save()

        icon = "✅" if correct == "YES" else ("❌" if correct == "NO" else "⏭️")
        log.info(
            f"{icon} Result | candle={key} | "
            f"predicted={predicted} actual={actual_color} correct={correct} | "
            f"open={actual_candle['open']:.1f} close={actual_candle['close']:.1f}"
        )

        # Rolling accuracy summary
        self._print_summary()

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
            f"| skipped={skipped} | total={len(df)}"
        )

    def save_on_exit(self):
        self._save()
        log.info(f"CSV saved on exit → {self.path}")


# ── Next candle time ──────────────────────────────────────────────────────────

def next_candle_close(ts: datetime) -> datetime:
    """Return the close time of the next 5-min candle after ts."""
    minutes     = ts.minute
    next_minute = ((minutes // 5) + 1) * 5
    next_close  = ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
    return next_close


def seconds_until(target: datetime) -> float:
    return max(0.0, (target - _now_ist()).total_seconds())


def last_completed_candle_time(ts: datetime) -> datetime:
    """Returns the open time of the last COMPLETED 5-min candle."""
    minutes     = ts.minute
    last_minute = (minutes // 5) * 5
    return ts.replace(minute=last_minute, second=0, microsecond=0) - timedelta(minutes=5)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_models(instrument: str):
    """Load XGBoost A/B/C + LSTM + scaler for given instrument."""
    import joblib

    model_dir  = os.path.join(BASE_DIR, "models")
    inst_lower = instrument.lower()

    # Scaler
    scaler_path = os.path.join(model_dir, f"scaler_{inst_lower}.pkl")
    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}")
        log.error("Run: python ml_predictor/train.py --xgb-only")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    log.info(f"Scaler loaded ← {scaler_path}")

    # XGBoost models
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

    # LSTM (optional)
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

    log.info("=" * 60)
    log.info(f"  ML LIVE TRACKER — {instrument}")
    log.info(f"  Predicts next 5-min candle color | Logs to CSV")
    log.info("=" * 60)

    # Load models
    xgb_models, lstm_model, scaler = load_models(instrument)

    # Load Kite
    kite = _load_kite()

    # Prediction logger
    logger = PredictionLogger(instrument)

    # Graceful shutdown
    def _shutdown(sig, frame):
        log.info("\nShutting down...")
        logger.save_on_exit()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── State ─────────────────────────────────────────────────────────────────
    last_predicted_candle = None    # candle_time we last made a prediction for
    last_resolved_candle  = None    # candle_time we last resolved actual for

    log.info("\nWaiting for market hours (09:15–15:25 IST)...")

    while True:
        now = _now_ist()

        # ── Outside market hours — sleep ───────────────────────────────────────
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

        # ── Determine current candle boundary ─────────────────────────────────
        # The candle that just CLOSED (we predict its NEXT candle)
        current_candle_open = last_completed_candle_time(now)

        # ── STEP 1: Resolve previous prediction (actual outcome) ───────────────
        # If we predicted for the candle before current_candle_open,
        # that candle has now completed — we can record actual.
        if last_predicted_candle and last_predicted_candle != last_resolved_candle:
            if current_candle_open > last_predicted_candle:
                # The predicted candle (last_predicted_candle) has now closed
                # Fetch it to get actual OHLC
                log.info(f"Resolving actual for candle {last_predicted_candle.strftime('%H:%M')}...")
                df_recent = fetch_recent_candles(kite, token, n_bars=10)
                if not df_recent.empty:
                    # Find the candle at last_predicted_candle time
                    match = df_recent[
                        df_recent.index.floor("5min") == pd.Timestamp(last_predicted_candle)
                    ]
                    if not match.empty:
                        actual = match.iloc[0].to_dict()
                        logger.resolve_actual(last_predicted_candle, actual)
                        last_resolved_candle = last_predicted_candle
                    else:
                        # Fallback: use the candle just before current
                        if len(df_recent) >= 2:
                            actual = df_recent.iloc[-2].to_dict()
                            logger.resolve_actual(last_predicted_candle, actual)
                            last_resolved_candle = last_predicted_candle

        # ── STEP 2: Make new prediction ────────────────────────────────────────
        if current_candle_open != last_predicted_candle:
            log.info(f"\n{'─'*55}")
            log.info(f"New candle closed: {current_candle_open.strftime('%H:%M')} — making prediction...")

            # Fetch recent candles for features
            df_candles = fetch_recent_candles(kite, token, n_bars=FETCH_BARS)

            if df_candles.empty or len(df_candles) < 30:
                log.warning(f"Insufficient candles ({len(df_candles)}) — skipping")
            else:
                # Fetch VIX
                vix = fetch_vix(kite)

                # Run prediction
                pred = run_prediction(df_candles, instrument, vix, xgb_models, lstm_model, scaler)
                pred["vix"] = vix

                # The candle we just used is current_candle_open
                # Our prediction is for the NEXT candle (current_candle_open + 5min)
                predicted_for = current_candle_open + timedelta(minutes=5)

                # Record: we save against the candle we're predicting FOR
                last_candle = df_candles.iloc[-1].to_dict()
                logger.record_prediction(predicted_for, last_candle, pred)
                last_predicted_candle = predicted_for

            if args.dry_run:
                log.info("Dry run complete.")
                logger.save_on_exit()
                sys.exit(0)

        # ── STEP 3: Sleep until next candle close + 5s buffer ─────────────────
        next_close = next_candle_close(now)
        sleep_secs = seconds_until(next_close) + 5   # +5s ensures candle is fully closed

        if sleep_secs > 1:
            log.info(f"Next candle closes at {next_close.strftime('%H:%M:%S')} "
                     f"— sleeping {sleep_secs:.0f}s...")
            # Sleep in chunks so we can catch KeyboardInterrupt
            slept = 0
            while slept < sleep_secs:
                chunk = min(30, sleep_secs - slept)
                time.sleep(chunk)
                slept += chunk


if __name__ == "__main__":
    main()

