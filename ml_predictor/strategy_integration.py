"""
ml_predictor/strategy_integration.py
──────────────────────────────────────
Drop-in mixin that adds ML confluence gate to any BaseStrategy subclass.

REBUILD NOTES — what changed vs. old strategy_integration.py:
  - predictor.py now returns UP / DOWN / FLAT (not GREEN / RED / SKIP).
    FLAT means "no edge beyond the ±0.15% deadband" — it is the new
    "uncertain zone" and is treated the same way SKIP used to be (blocks
    the trade rather than allowing it).
  - signal["regime"] is gone; signal["vix_regime"] and signal["bucket"]
    are the new fields (see predictor.py). threshold-based confidence
    gating is replaced by signal["confidence"] vs min_confidence directly,
    since the new predictor doesn't compute a regime-driven threshold
    internally (that logic lives in regime.py if/when you want to re-add
    a per-bucket confidence floor — see NOTE below).

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
            return   # ML says market is bearish/flat — skip CE buy
    elif action == "SELL":
        if self.ml_gate(self._hub, self._pm, ts, required_direction="SELL"):
            return   # ML says market is bullish/flat — skip PE buy

NOTE on confidence thresholds:
    The old regime_detector.py supplied a per-VIX-regime confidence
    threshold (e.g. 0.70 in QUIET, 0.55 in NORMAL bucket B). That logic
    was tuned for the OLD binary target and is intentionally NOT carried
    forward as-is — see the bug audit (BUG 11, BUG 12). Until new
    thresholds are backtested against the 3-class target, ml_gate() here
    uses a single DEFAULT_MIN_CONFIDENCE for all buckets/regimes. Pass
    min_confidence= explicitly per call if you want per-bucket tuning
    sooner, using signal["bucket"] / signal["vix_regime"] to decide.
"""

import logging
import os
import sys
from datetime import timedelta

log = logging.getLogger("ml.gate")

ML_GATE_ENABLED = True
DEFAULT_MIN_CONFIDENCE = 0.55   # provisional — re-tune after backtesting the new target


def _floor_to_candle(ts):
    """
    Convert a wall-clock timestamp to the bar-start of the candle that JUST
    CLOSED at or before that moment (e.g. 10:30:03 -> 10:25:00, the candle
    covering 10:25-10:30 that closed when this tick arrived).

    Why this exists: in this codebase's live tick path, core/market_hub.py
    calls strat.on_candle(closed, now) where `now` is the wall-clock time
    the CLOSING TICK arrived, not the candle's own bar-start time (which
    does exist, as closed["ts"], but isn't what's threaded through to
    on_candle's ts parameter on that path). A candle covering 10:25-10:30
    closes when a tick arrives at e.g. 10:30:03 — naively flooring that to
    10:30:00 lands on the START of the NEXT candle (already bucket B per
    regime.py's A=09:15-10:30 cutoff), not the bar-start of the candle that
    actually just closed (10:25:00, still bucket A). Subtracting one 5-min
    interval before flooring corrects this: it maps any wall-clock moment
    back to the start of the candle whose CLOSE triggered this call.

    Idempotent in the sense that matters: calling this on the wall-clock
    arrival time of a candle's closing tick always yields that candle's
    own bar-start, regardless of exactly how many seconds into the new
    minute the tick happened to land (10:30:01 and 10:30:58 both correctly
    map to 10:25:00).
    """
    shifted = ts - timedelta(minutes=5)
    floored_minute = (shifted.minute // 5) * 5
    return shifted.replace(minute=floored_minute, second=0, microsecond=0)


class MLGateMixin:
    """
    Mixin that adds ML 4-candle-forward direction gate to any strategy.
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
            ts:                 Current timestamp (see _floor_to_candle note below)
            required_direction: "BUY" (expecting UP) or "SELL" (expecting DOWN)
            min_confidence:     Override confidence threshold
                                 (None = use DEFAULT_MIN_CONFIDENCE)

        Returns:
            True  → BLOCK the trade (ML disagrees, is flat, or uncertain)
            False → ALLOW the trade (ML agrees or gate disabled)

        Bucket-boundary note: ts here may be the strategy's on_candle()
        wall-clock parameter, which in this codebase's live tick path is
        the time the closing tick ARRIVED (core/market_hub.py's `now`), not
        the candle's own bar-start time (which IS available on the candle
        dict itself as closed["ts"], but isn't always what gets threaded
        through to here). A candle covering 10:25-10:30 can close on a tick
        that arrives at 10:30:03 — using that raw timestamp for bucket
        selection would put a 10:25 (bucket A) candle's prediction through
        bucket B's model for the few seconds before the next real candle
        accumulates. _floor_to_candle() corrects for this defensively by
        always flooring to the nearest 5-min boundary, regardless of
        whether the caller already did so — idempotent either way, so it's
        safe even if a future caller passes the correct bar-start time.
        """
        if not ML_GATE_ENABLED:
            return False

        predictor = getattr(self, "_ml_predictor", None)
        if predictor is None:
            return False    # gate disabled/failed → don't block

        candle_ts = _floor_to_candle(ts)

        try:
            signal = predictor.predict(hub, pm, candle_ts)
        except Exception as e:
            log.warning(f"[ML] predict() error — allowing trade: {e}")
            return False

        prediction  = signal["prediction"]       # "UP" | "DOWN" | "FLAT" | "SKIP"
        confidence  = signal["confidence"]
        vix_regime  = signal.get("vix_regime", "UNKNOWN")
        bucket      = signal.get("bucket", "?")
        reason      = signal.get("reason", "")

        expected = "UP" if required_direction == "BUY" else "DOWN"

        # ── Decision logic ────────────────────────────────────────────────────

        if prediction in ("SKIP", "FLAT"):
            # FLAT means "no edge beyond ±0.15% deadband" — same treatment
            # as the old SKIP/uncertain zone: block the trade.
            log.info(
                f"[ML] BLOCK ({prediction.lower()}) | "
                f"required={expected} got={prediction} conf={confidence:.1%} | "
                f"bucket={bucket} | {reason}"
            )
            return True

        if prediction != expected:
            log.info(
                f"[ML] BLOCK (direction) | "
                f"required={expected} got={prediction} conf={confidence:.1%} | "
                f"bucket={bucket} | {reason}"
            )
            return True

        effective_thresh = min_confidence if min_confidence is not None else DEFAULT_MIN_CONFIDENCE
        if confidence < effective_thresh:
            log.info(
                f"[ML] BLOCK (low conf) | "
                f"{prediction} {confidence:.1%} < thr={effective_thresh:.1%} | "
                f"bucket={bucket} | {reason}"
            )
            return True

        log.info(
            f"[ML] ALLOW | "
            f"{prediction} {confidence:.1%} ≥ {effective_thresh:.1%} | "
            f"vix_regime={vix_regime} bucket={bucket} | {reason}"
        )
        return False

    def ml_signal(self, hub, pm, ts) -> dict:
        """
        Get full ML signal dict without blocking logic.
        Useful for logging or making custom decisions.

        ts is floored to the nearest 5-min candle boundary before being
        passed to predict() — see ml_gate()'s docstring for why.
        """
        predictor = getattr(self, "_ml_predictor", None)
        if predictor is None:
            return {"prediction": "SKIP", "confidence": 0.0, "tradeable": False,
                    "reason": "ml_disabled"}
        try:
            return predictor.predict(hub, pm, _floor_to_candle(ts))
        except Exception as e:
            return {"prediction": "SKIP", "confidence": 0.0, "tradeable": False,
                    "reason": str(e)}
