"""
ml_predictor/predictor.py
──────────────────────────
Real-time 5-minute candle color predictor.
Ensemble of XGBoost (3 time-of-day models) + LSTM.
Integrates with your existing MarketHub and PreMarketData.

Usage in strategy:
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
from datetime import datetime, timedelta, timezone, time as dtime

import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))

from ml_predictor.feature_engineering import build_features, add_live_oi, FEATURE_COLS
from ml_predictor.regime_detector      import RegimeDetector

log = logging.getLogger("ml.predictor")

IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist():
    return datetime.now(tz=IST).replace(tzinfo=None)

# Instrument tokens
INDEX_TOKENS = {
    "NIFTY50":   256265,
    "BANKNIFTY": 260105,
    "INDIAVIX":  264969,
}

# LSTM XGB ensemble weights
XGB_WEIGHT  = 0.60
LSTM_WEIGHT = 0.40

# Minimum candles needed in live buffer before prediction
MIN_CANDLES = 55   # covers EMA50 warmup (50) + a few extra


class NiftyPredictor:
    """
    Real-time candle color predictor for Nifty / BankNifty.

    Loads saved XGBoost A/B/C models + LSTM + scaler.
    Builds features from live candle buffer every 5 minutes.
    Returns GREEN / RED / SKIP signal with confidence.
    """

    def __init__(self, instrument: str = "BANKNIFTY"):
        self.instrument   = instrument.upper()
        self._regime      = RegimeDetector()
        self._xgb_models  = {}      # {"A": model, "B": model, "C": model}
        self._lstm_model  = None
        self._scaler      = None
        self._loaded      = False

        # OI tracking for delta features
        self._prev_ce_oi  = 0
        self._prev_pe_oi  = 0

        # Cross-index buffer (opposite index for cross_ret_1)
        self._cross_token = INDEX_TOKENS.get(
            "NIFTY50" if self.instrument == "BANKNIFTY" else "BANKNIFTY"
        )

        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        model_dir = os.path.join(BASE_DIR, "models")

        # Scaler
        scaler_path = os.path.join(model_dir, f"scaler_{self.instrument.lower()}.pkl")
        if not os.path.exists(scaler_path):
            log.warning(f"[ML] Scaler not found: {scaler_path}. Run train.py first.")
            return
        self._scaler = joblib.load(scaler_path)
        log.info(f"[ML] Scaler loaded ← {scaler_path}")

        # XGBoost A / B / C
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

        # LSTM (optional — works without it)
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
        """
        Call once in pre_market() to validate everything is loaded.
        Logs a summary so you can see ML status in morning logs.
        """
        vix = pm.vix if pm else None
        rd  = self._regime.describe(vix or 15.0)
        log.info(
            f"[ML] {self.instrument} predictor | "
            f"loaded={self._loaded} | "
            f"models={list(self._xgb_models.keys())} | "
            f"lstm={'yes' if self._lstm_model else 'no'} | "
            f"regime={rd['regime']} vix={vix}"
        )

    def predict(self, hub, pm, ts: datetime = None) -> dict:
        """
        Main prediction call. Called every 5-min candle close from strategy.

        Args:
            hub:  MarketHub instance (provides last_price, last_oi)
            pm:   PreMarketData (provides vix, expiry, atm tokens)
            ts:   Timestamp (defaults to now IST)

        Returns dict:
            {
                "prediction":  "GREEN" | "RED" | "SKIP",
                "confidence":  float 0.0–1.0,
                "raw_prob":    float (raw P(green)),
                "tradeable":   bool,
                "regime":      str,
                "threshold":   float,
                "reason":      str,
            }
        """
        if ts is None:
            ts = _now_ist()

        # Default "skip" response
        skip = lambda reason: {
            "prediction": "SKIP", "confidence": 0.0, "raw_prob": 0.5,
            "tradeable": False, "regime": "UNKNOWN", "threshold": 1.0,
            "reason": reason,
        }

        if not self._loaded:
            return skip("models_not_loaded")

        # ── Regime gate ───────────────────────────────────────────────────────
        vix = pm.vix if pm else None
        rd  = self._regime.describe(vix or 15.0, ts)

        if not rd["tradeable"]:
            return skip(f"regime_CRISIS vix={vix}")

        # ── Build feature DataFrame from hub's candle history ─────────────────
        df_raw = self._build_raw_df(hub, ts)
        if df_raw is None or len(df_raw) < MIN_CANDLES:
            n = len(df_raw) if df_raw is not None else 0
            return skip(f"insufficient_candles({n}<{MIN_CANDLES})")

        # VIX series (single value repeated — approximation for live)
        df_vix = None
        if vix is not None:
            df_vix = pd.DataFrame(
                {"close": vix},
                index=df_raw.index,
            )

        # Cross-index (last price as proxy for cross_ret_1)
        df_cross = self._build_cross_df(hub, df_raw.index)

        try:
            df_feat = build_features(df_raw, df_vix=df_vix, df_cross=df_cross)
        except Exception as e:
            log.warning(f"[ML] Feature engineering error: {e}")
            return skip(f"feature_error: {e}")

        if df_feat.empty:
            return skip("empty_features_after_dropna")

        # Take the LAST row = prediction for NEXT candle
        last_row = df_feat[FEATURE_COLS].iloc[[-1]]

        # ── Inject live OI features ───────────────────────────────────────────
        atm_ce_tok, atm_pe_tok = self._get_atm_tokens(hub, pm, ts)
        if atm_ce_tok and atm_pe_tok:
            row_dict = last_row.iloc[0].to_dict()
            row_dict = add_live_oi(
                row_dict, hub, atm_ce_tok, atm_pe_tok,
                self._prev_ce_oi, self._prev_pe_oi,
            )
            # Update prev OI for next candle delta
            self._prev_ce_oi = hub.last_oi(atm_ce_tok)
            self._prev_pe_oi = hub.last_oi(atm_pe_tok)
            last_row = pd.DataFrame([row_dict], index=last_row.index)

        # ── Scale ─────────────────────────────────────────────────────────────
        X = self._scaler.transform(last_row[FEATURE_COLS])

        # ── XGBoost prediction ────────────────────────────────────────────────
        bucket = rd["time_bucket"][0].upper()   # "open"→"A", "normal"→"B", "close"→"C"
        bucket_map = {"o": "A", "n": "B", "c": "C"}
        bucket_key = bucket_map.get(rd["time_bucket"][0], "B")
        xgb_model  = self._xgb_models.get(bucket_key) or next(iter(self._xgb_models.values()))

        xgb_prob = float(xgb_model.predict_proba(X)[0, 1])

        # ── LSTM prediction ───────────────────────────────────────────────────
        lstm_prob = None
        if self._lstm_model is not None:
            try:
                X_full   = self._scaler.transform(df_feat[FEATURE_COLS])
                lookback = self._lstm_model.input_shape[1]
                if len(X_full) >= lookback:
                    seq      = X_full[-lookback:].reshape(1, lookback, -1)
                    lstm_prob = float(self._lstm_model.predict(seq, verbose=0)[0, 0])
            except Exception as e:
                log.debug(f"[ML] LSTM inference error: {e}")

        # ── Ensemble ──────────────────────────────────────────────────────────
        if lstm_prob is not None:
            raw_prob = XGB_WEIGHT * xgb_prob + LSTM_WEIGHT * lstm_prob
            source   = f"ensemble(xgb={xgb_prob:.3f} lstm={lstm_prob:.3f})"
        else:
            raw_prob = xgb_prob
            source   = f"xgb_only({xgb_prob:.3f})"

        # ── Decision ─────────────────────────────────────────────────────────
        threshold = rd["threshold"]
        inv_thr   = 1.0 - threshold     # threshold for RED (bearish)

        if raw_prob >= threshold:
            prediction = "GREEN"
            confidence = raw_prob
            tradeable  = True
        elif raw_prob <= inv_thr:
            prediction = "RED"
            confidence = 1.0 - raw_prob
            tradeable  = True
        else:
            prediction = "SKIP"
            confidence = abs(raw_prob - 0.5) * 2
            tradeable  = False

        log.info(
            f"[ML] {self.instrument} {ts.strftime('%H:%M')} | "
            f"{prediction} {confidence:.1%} | "
            f"prob={raw_prob:.3f} thr={threshold} | "
            f"regime={rd['regime']} | {source}"
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_prob":   raw_prob,
            "tradeable":  tradeable,
            "regime":     rd["regime"],
            "threshold":  threshold,
            "reason":     source,
            "size_mult":  rd["size_mult"],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_raw_df(self, hub, ts: datetime) -> pd.DataFrame | None:
        """
        Build OHLCV DataFrame from hub's internal candle buffer.
        Works with any strategy that has hub._candles_5m or similar.
        Falls back to requesting historical if buffer is thin.
        """
        # Try to get candles from hub
        candles = None

        # MarketHub exposes _candles dict keyed by token
        token = INDEX_TOKENS.get(self.instrument)
        if hasattr(hub, "_candles") and token in hub._candles:
            candles = list(hub._candles[token])
        elif hasattr(hub, "get_candles"):
            candles = hub.get_candles(token)

        if not candles:
            return None

        df = pd.DataFrame(candles)
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

        df.sort_index(inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df

    def _build_cross_df(self, hub, index: pd.DatetimeIndex) -> pd.DataFrame | None:
        """Build minimal cross-index DataFrame for cross_ret_1 feature."""
        if self._cross_token is None:
            return None
        try:
            price = hub.last_price(self._cross_token)
            if price:
                return pd.DataFrame({"close": price}, index=index)
        except Exception:
            pass
        return None

    def _get_atm_tokens(self, hub, pm, ts: datetime):
        """
        Get current ATM CE and PE tokens.
        Uses the same logic as your strategies (get_atm_strike).
        Returns (ce_token, pe_token) or (None, None).
        """
        try:
            sys.path.insert(0, os.path.dirname(BASE_DIR))
            from strategies.bb_stoch_strategy import get_atm_strike

            spot = hub.last_price(INDEX_TOKENS[self.instrument])
            if not spot or not pm or not pm.expiry_date:
                return None, None

            atm      = get_atm_strike(spot)
            instr_obj = getattr(hub, "instruments", None) or getattr(hub, "_instruments", None)
            if instr_obj is None:
                return None, None

            ce_tok, _ = instr_obj.get_option_token(atm, "CE", pm.expiry_date)
            pe_tok, _ = instr_obj.get_option_token(atm, "PE", pm.expiry_date)
            return ce_tok, pe_tok
        except Exception as e:
            log.debug(f"[ML] ATM token lookup failed: {e}")
            return None, None
