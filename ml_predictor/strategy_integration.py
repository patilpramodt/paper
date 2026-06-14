"""
ml_predictor/strategy_integration.py
──────────────────────────────────────
Drop-in mixin that adds ML confluence gate to any BaseStrategy subclass.

HOW TO ADD TO YOUR STRATEGIES — 3 steps each:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Import at top of strategy file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from ml_predictor.strategy_integration import MLGateMixin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: Add mixin to class definition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Before:
    class BBStochStrategy(BaseStrategy):

    # After:
    class BBStochStrategy(MLGateMixin, BaseStrategy):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: Add to pre_market() and _evaluate_entry()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def pre_market(self, premarket_data, instruments) -> bool:
        # ... your existing code ...
        self.ml_init("BANKNIFTY", premarket_data)   # ← ADD THIS LINE
        return True

    def _evaluate_entry(self, ts):
        # ADD THESE 5 LINES at the very top of _evaluate_entry:
        ml_block = self.ml_gate(self._hub, self._pm, ts, required_direction="BUY")
        if ml_block:
            return
        # ... rest of your existing entry logic unchanged ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For strategies with both BUY and SELL signals:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # After you know the direction:
    if action == "BUY":
        if self.ml_gate(self._hub, self._pm, ts, required_direction="BUY"):
            return   # ML says market is bearish — skip CE buy
    elif action == "SELL":
        if self.ml_gate(self._hub, self._pm, ts, required_direction="SELL"):
            return   # ML says market is bullish — skip PE buy
"""

import logging
import os
import sys

log = logging.getLogger("ml.gate")

# Enable/disable ML gate globally without touching strategy files
ML_GATE_ENABLED = True


class MLGateMixin:
    """
    Mixin that adds ML candle-color prediction gate to any strategy.
    Add to class definition: class MyStrategy(MLGateMixin, BaseStrategy)
    """

    def ml_init(self, instrument: str, pm=None):
        """
        Call from pre_market() to initialize the predictor.
        instrument: "BANKNIFTY" or "NIFTY50"
        pm:         PreMarketData (for warm-up logging)
        """
        if not ML_GATE_ENABLED:
            self._ml_predictor = None
            return

        try:
            BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if BASE not in sys.path:
                sys.path.insert(0, BASE)

            from ml_predictor.predictor import NiftyPredictor
            self._ml_predictor = NiftyPredictor(instrument=instrument)

            if pm is not None:
                hub = getattr(self, "_hub", None)
                self._ml_predictor.warm_up(hub, pm)

            log.info(f"[ML] Gate initialized for {instrument}")

        except Exception as e:
            log.warning(f"[ML] Gate init failed (trading continues without ML): {e}")
            self._ml_predictor = None

    def ml_gate(
        self,
        hub,
        pm,
        ts,
        required_direction: str = "BUY",
        min_confidence: float = None,
    ) -> bool:
        """
        Check ML gate. Returns True if trade should be BLOCKED.

        Args:
            hub:                MarketHub instance
            pm:                 PreMarketData instance
            ts:                 Current timestamp
            required_direction: "BUY" (expecting GREEN) or "SELL" (expecting RED)
            min_confidence:     Override confidence threshold (None = use regime default)

        Returns:
            True  → BLOCK the trade (ML disagrees or uncertain)
            False → ALLOW the trade (ML agrees or gate disabled)
        """
        if not ML_GATE_ENABLED:
            return False

        predictor = getattr(self, "_ml_predictor", None)
        if predictor is None:
            return False    # gate disabled/failed → don't block

        try:
            signal = predictor.predict(hub, pm, ts)
        except Exception as e:
            log.warning(f"[ML] predict() error — allowing trade: {e}")
            return False

        prediction  = signal["prediction"]
        confidence  = signal["confidence"]
        regime      = signal["regime"]
        threshold   = signal.get("threshold", 0.55)
        reason      = signal.get("reason", "")

        # Map required_direction to expected ML prediction
        expected = "GREEN" if required_direction == "BUY" else "RED"

        # ── Decision logic ────────────────────────────────────────────────────

        if prediction == "SKIP":
            # Uncertain zone — block the trade
            log.info(
                f"[ML] BLOCK (uncertain) | "
                f"required={expected} got=SKIP conf={confidence:.1%} | {reason}"
            )
            return True

        if prediction != expected:
            # ML predicts opposite direction — block
            log.info(
                f"[ML] BLOCK (direction) | "
                f"required={expected} got={prediction} conf={confidence:.1%} | {reason}"
            )
            return True

        # ML agrees — check confidence meets threshold
        effective_thresh = min_confidence if min_confidence is not None else threshold
        if confidence < effective_thresh:
            log.info(
                f"[ML] BLOCK (low conf) | "
                f"{prediction} {confidence:.1%} < thr={effective_thresh:.1%} | {reason}"
            )
            return True

        # ── All checks passed — allow trade ───────────────────────────────────
        log.info(
            f"[ML] ALLOW | "
            f"{prediction} {confidence:.1%} ≥ {effective_thresh:.1%} | "
            f"regime={regime} | {reason}"
        )
        return False

    def ml_signal(self, hub, pm, ts) -> dict:
        """
        Get full ML signal dict without blocking logic.
        Useful for logging or making custom decisions.
        """
        predictor = getattr(self, "_ml_predictor", None)
        if predictor is None:
            return {"prediction": "SKIP", "confidence": 0.0, "tradeable": False,
                    "reason": "ml_disabled"}
        try:
            return predictor.predict(hub, pm, ts)
        except Exception as e:
            return {"prediction": "SKIP", "confidence": 0.0, "tradeable": False,
                    "reason": str(e)}
