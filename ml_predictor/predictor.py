"""
ml_predictor/predictor.py
──────────────────────────
Real-time 4-candle-forward (±0.15% deadband) direction predictor.
Ensemble of XGBoost (3 time-of-day bucket models) + LSTM.
Integrates with your existing MarketHub and PreMarketData.

REBUILD NOTES — what changed vs. the old predictor.py, and why:

1. VIX / cross-index are no longer scalar broadcasts.
   Old code did `pd.DataFrame({"close": vix}, index=df_raw.index)` — a
   single number repeated down every row, which makes vix_change and
   cross_ret_1 collapse to 0 at live time even though they were trained
   on real, varying historical series. This predictor now fetches actual
   recent VIX and cross-index candle HISTORY via Kite (same pattern as
   live_tracker.py's fetch_recent_candles), cached for a few minutes so
   we aren't hammering the API every 5-min tick.

2. hub._candles[token] doesn't exist in the real MarketHub.
   MarketHub owns exactly ONE index_candles CandleBuilder for whichever
   index that hub instance is tracking (see core/market_hub.py). It is
   NOT a dict keyed by token. The old _build_raw_df's assumption was
   wrong for this codebase. This predictor pulls its own OHLC history for
   the main instrument, VIX, and cross-index directly from Kite via
   hub.kite (the same authenticated session MarketHub already holds —
   no second login needed), rather than reading a structure that isn't there.

3. OI features are gone entirely (dropped in features.py). No more
   add_live_oi / ATM-token lookup / OI-delta tracking in this file —
   that entire code path is deleted, not patched.

4. Bucket selection goes through regime.get_bucket(), the SAME function
   train.py uses to assign rows to buckets during training. No more
   separate RegimeDetector cutoff table that can drift out of sync.

5. Output is now 3-class (DOWN / FLAT / UP) matching the new target,
   not GREEN/RED binary.

Usage in strategy (unchanged shape, see strategy_integration.py):
    from ml_predictor.predictor import NiftyPredictor

    # In pre_market():
    self._ml = NiftyPredictor(instrument="BANKNIFTY")
    self._ml.warm_up(hub, pm)

    # In _evaluate_entry():
    signal = self._ml.predict(hub, pm)
    if not signal["tradeable"]:
        return
"""

import logging
import os
import sys
import time as _time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from ml_predictor.features import build_features, FEATURE_COLS
from ml_predictor.targets   import DOWN, FLAT, UP, TARGET_CLASSES
from ml_predictor.regime    import get_bucket, get_vix_regime

log = logging.getLogger("ml.predictor")

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist():
    return datetime.now(tz=IST).replace(tzinfo=None)


# Instrument tokens (must match data_fetcher.py's INSTRUMENTS table)
INDEX_TOKENS = {
    "NIFTY50":   256265,
    "BANKNIFTY": 260105,
    "INDIAVIX":  264969,
}

# LSTM / XGB ensemble weights
XGB_WEIGHT  = 0.60
LSTM_WEIGHT = 0.40

# Minimum candles needed before prediction (EMA50 warmup + buffer)
MIN_CANDLES = 60

# How long fetched VIX/cross/main history is cached before re-fetching,
# to avoid hitting Kite's historical_data endpoint on every 5-min tick.
HISTORY_CACHE_SECONDS = 90
HISTORY_FETCH_DAYS    = 5     # enough days of 5-min candles for warmup + indicators
HISTORY_BARS_NEEDED   = 80    # keep a bit more than MIN_CANDLES after warmup loss

# Kite's Historical Data API is capped at 2 req/sec (confirmed via Zerodha's
# developer forum — narrower than the general 10 req/sec "all other endpoints"
# rule). predict() can fire up to 3 calls back-to-back (main + VIX + cross)
# whenever the cache expires; this enforces a floor between ANY two calls.
MIN_SECONDS_BETWEEN_CALLS = 0.6   # ~1.67 req/sec, safe margin under the 2/sec cap

# Hard ceiling on how old a cached fetch can be before it's refused as a
# fallback during a Kite outage. Without this, a sustained outage would let
# predict() keep serving arbitrarily old data forever — confirmed: 30+ min
# of continuous failures still produced predictions with no warning beyond
# a per-call log line, and no upper bound on how stale it could get.
MAX_STALE_SECONDS = 600   # 10 minutes — beyond this, skip the prediction
                           # rather than act on data too old for a 20-min-
                           # forward target.


class NiftyPredictor:
    """
    Real-time 4-candle-forward direction predictor for Nifty / BankNifty.

    Loads saved XGBoost A/B/C bucket models + LSTM + scaler.
    Fetches its own recent OHLC history (main instrument, VIX, cross-index)
    directly from Kite via hub.kite, builds features, and returns a
    DOWN / FLAT / UP signal with confidence.
    """

    def __init__(self, instrument: str = "BANKNIFTY"):
        self.instrument = instrument.upper()
        self._cross_name = "NIFTY50" if self.instrument == "BANKNIFTY" else "BANKNIFTY"

        self._xgb_models = {}      # {"A": model, "B": model, "C": model}
        self._lstm_model = None
        self._scaler     = None
        self._loaded      = False

        # Simple time-based cache for fetched history so we don't call
        # Kite's historical_data endpoint on every single prediction.
        self._cache = {}  # name -> (fetched_at: float, df: pd.DataFrame)

        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        model_dir = os.path.join(BASE_DIR, "models")

        scaler_path = os.path.join(model_dir, f"scaler_{self.instrument.lower()}.pkl")
        if not os.path.exists(scaler_path):
            log.warning(f"[ML] Scaler not found: {scaler_path}. Run train.py first.")
            return
        self._scaler = joblib.load(scaler_path)
        log.info(f"[ML] Scaler loaded ← {scaler_path}")

        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.error("[ML] xgboost not installed. Run: pip install xgboost")
            return

        for bucket in ["A", "B", "C"]:
            path = os.path.join(model_dir, f"xgb_{self.instrument.lower()}_{bucket}.json")
            if os.path.exists(path):
                m = XGBClassifier()
                m.load_model(path)
                self._xgb_models[bucket] = m
                log.info(f"[ML] XGBoost {bucket} loaded ← {path}")
            else:
                log.warning(f"[ML] XGBoost {bucket} not found: {path}")

        lstm_path = os.path.join(model_dir, f"lstm_{self.instrument.lower()}.keras")
        if os.path.exists(lstm_path):
            try:
                from tensorflow.keras.models import load_model
                self._lstm_model = load_model(lstm_path)
                log.info(f"[ML] LSTM loaded ← {lstm_path}")
            except Exception as e:
                log.warning(f"[ML] LSTM load failed: {e}. Falling back to XGBoost only.")
        else:
            log.info("[ML] No LSTM model found — using XGBoost only")

        if self._xgb_models and self._scaler:
            self._loaded = True
            log.info(f"[ML] NiftyPredictor ready for {self.instrument}")

    # ── Public API ────────────────────────────────────────────────────────────

    def warm_up(self, hub, pm):
        """Call once in pre_market() to validate everything is loaded."""
        vix = pm.vix if pm else None
        regime = get_vix_regime(vix if vix is not None else 15.0)
        log.info(
            f"[ML] {self.instrument} predictor | loaded={self._loaded} | "
            f"models={list(self._xgb_models.keys())} | "
            f"lstm={'yes' if self._lstm_model else 'no'} | "
            f"vix_regime={regime} vix={vix}"
        )

    def predict(self, hub, pm, ts: datetime = None) -> dict:
        """
        Main prediction call. Called every 5-min candle close from strategy.

        Returns dict:
            {
                "prediction":  "UP" | "DOWN" | "FLAT" | "SKIP",
                "confidence":  float 0.0-1.0,
                "probs":       {"UP": float, "DOWN": float, "FLAT": float},
                "tradeable":   bool,
                "bucket":      "A" | "B" | "C",
                "vix_regime":  str,
                "reason":      str,
            }
        """
        if ts is None:
            ts = _now_ist()

        skip = lambda reason: {
            "prediction": "SKIP", "confidence": 0.0,
            "probs": {"UP": 0.0, "DOWN": 0.0, "FLAT": 1.0},
            "tradeable": False, "bucket": None, "vix_regime": "UNKNOWN",
            "reason": reason,
        }

        if not self._loaded:
            return skip("models_not_loaded")

        vix = pm.vix if pm else None
        vix_regime = get_vix_regime(vix if vix is not None else 15.0)

        kite = getattr(hub, "kite", None)
        if kite is None:
            return skip("no_kite_session_on_hub")

        # ── Fetch real OHLC history for main instrument + VIX + cross-index ───
        df_main = self._get_history(kite, self.instrument)
        if df_main is None or len(df_main) < MIN_CANDLES:
            n = len(df_main) if df_main is not None else 0
            return skip(f"insufficient_candles({n}<{MIN_CANDLES})")

        df_vix   = self._get_history(kite, "INDIAVIX")
        df_cross = self._get_history(kite, self._cross_name)

        try:
            df_feat = build_features(df_main, df_vix=df_vix, df_cross=df_cross)
        except Exception as e:
            log.warning(f"[ML] Feature engineering error: {e}")
            return skip(f"feature_error: {e}")

        if df_feat.empty:
            return skip("empty_features_after_dropna")

        last_row = df_feat[FEATURE_COLS].iloc[[-1]]
        X = self._scaler.transform(last_row[FEATURE_COLS])

        # ── Bucket selection — SAME function train.py used to label rows ──────
        bucket = get_bucket(ts)
        xgb_model = self._xgb_models.get(bucket) or next(iter(self._xgb_models.values()))

        xgb_probs = xgb_model.predict_proba(X)[0]   # [P(DOWN), P(FLAT), P(UP)]

        # ── LSTM prediction (optional) ──────────────────────────────────────────
        lstm_probs = None
        if self._lstm_model is not None:
            try:
                X_full   = self._scaler.transform(df_feat[FEATURE_COLS])
                lookback = self._lstm_model.input_shape[1]
                if len(X_full) >= lookback:
                    seq = X_full[-lookback:].reshape(1, lookback, -1)
                    lstm_probs = self._lstm_model.predict(seq, verbose=0)[0]  # [P(D),P(F),P(U)]
            except Exception as e:
                log.debug(f"[ML] LSTM inference error: {e}")

        if lstm_probs is not None:
            probs  = XGB_WEIGHT * xgb_probs + LSTM_WEIGHT * lstm_probs
            source = f"ensemble(xgb_up={xgb_probs[UP]:.3f} lstm_up={lstm_probs[UP]:.3f})"
        else:
            probs  = xgb_probs
            source = f"xgb_only(up={xgb_probs[UP]:.3f})"

        prob_dict = {"DOWN": float(probs[DOWN]), "FLAT": float(probs[FLAT]), "UP": float(probs[UP])}
        pred_idx  = int(np.argmax(probs))
        prediction = TARGET_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # FLAT prediction is, by definition, not tradeable — it means "no
        # edge over the deadband", not a direction to act on.
        tradeable = prediction in ("UP", "DOWN")

        log.info(
            f"[ML] {self.instrument} {ts.strftime('%H:%M')} bucket={bucket} | "
            f"{prediction} {confidence:.1%} | "
            f"probs={ {k: round(v,3) for k,v in prob_dict.items()} } | "
            f"vix_regime={vix_regime} | {source}"
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probs":      prob_dict,
            "tradeable":  tradeable,
            "bucket":     bucket,
            "vix_regime": vix_regime,
            "reason":     source,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_history(self, kite, name: str) -> pd.DataFrame:
        """
        Fetch (or return cached) recent 5-min OHLC history for `name`
        ("NIFTY50" / "BANKNIFTY" / "INDIAVIX") via the hub's authenticated
        Kite session. This replaces the old scalar-broadcast approach —
        callers get a real multi-row DataFrame with real variation, exactly
        like what build_features() saw during training.

        Cached for HISTORY_CACHE_SECONDS so a 5-min prediction loop doesn't
        re-fetch all three series from Kite on every single call.

        Rate limiting: Kite's Historical Data API is capped at 2 req/sec
        (separate, narrower limit than the general 10 req/sec "all other
        endpoints" rule — confirmed on Zerodha's developer forum). predict()
        calls this 3x in a row (main + VIX + cross) whenever the cache
        expires, which would burst all 3 requests within a fraction of a
        second. self._last_call_ts enforces a minimum gap between ANY two
        real network calls this predictor instance makes, not just per-series,
        so a 3-call burst can't exceed the 2 req/sec ceiling.

        Stale-cache ceiling: on fetch failure, this falls back to the last
        successfully cached data rather than failing outright (a single
        Kite hiccup shouldn't block a prediction). But the fallback's cache
        TIMESTAMP is never refreshed on failure, so previously this could
        be served forever if Kite stayed down — confirmed: after 30+ min of
        continuous failures, predict() kept silently using 30-minute-old
        data with no escalation. MAX_STALE_SECONDS enforces a hard ceiling:
        past that age, the fallback is refused (returns None) instead of
        served, forcing the caller to skip the prediction cycle rather than
        act on data that's too old to be meaningful for a 20-minute-forward
        target.
        """
        now = _time.time()
        cached = self._cache.get(name)
        if cached is not None and (now - cached[0]) < HISTORY_CACHE_SECONDS:
            return cached[1]

        token = INDEX_TOKENS.get(name)
        if token is None:
            return None

        last_call = getattr(self, "_last_call_ts", 0.0)
        elapsed = _time.time() - last_call
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            _time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        self._last_call_ts = _time.time()

        to_dt   = _now_ist()
        from_dt = to_dt - timedelta(days=HISTORY_FETCH_DAYS)

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
            log.warning(f"[ML] Kite fetch failed for {name}: {e}")
            return self._stale_or_none(name, cached, now)

        if not raw:
            return self._stale_or_none(name, cached, now)

        df = pd.DataFrame(raw)
        df.rename(columns={"date": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.between_time("09:15", "15:30")
        df = df.tail(HISTORY_BARS_NEEDED).copy()

        self._cache[name] = (now, df)
        return df

    def _stale_or_none(self, name: str, cached, now: float):
        """
        Fallback decision when a real fetch fails or returns empty.
        Serves the cached value ONLY if it's still within MAX_STALE_SECONDS
        of its last successful fetch. Beyond that, returns None — forcing
        predict() to skip this cycle rather than silently act on data that's
        too old to be meaningful for a 20-minute-forward target.

        Note: cached[0] is the timestamp of the LAST SUCCESSFUL fetch, not
        updated on failed retries — so staleness here correctly reflects
        real wall-clock age since real data was last obtained, not just
        time since the last (possibly failed) attempt.
        """
        if cached is None:
            return None
        age = now - cached[0]
        if age > MAX_STALE_SECONDS:
            log.warning(
                f"[ML] {name}: cached data is {age:.0f}s old "
                f"(> MAX_STALE_SECONDS={MAX_STALE_SECONDS}s) — refusing stale "
                f"fallback, returning None instead of risking a prediction "
                f"on outdated market data."
            )
            return None
        log.info(f"[ML] {name}: serving cached data, {age:.0f}s old (within "
                 f"{MAX_STALE_SECONDS}s staleness ceiling)")
        return cached[1]
