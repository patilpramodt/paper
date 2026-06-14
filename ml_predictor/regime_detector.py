"""
ml_predictor/regime_detector.py
─────────────────────────────────
VIX-based market regime detection.
Adjusts ML confidence thresholds and trade size based on fear level.

Regimes:
    QUIET   VIX < 13  → Model unreliable (low vol = random drift)
    NORMAL  13–20     → Full trust
    FEARFUL 20–25     → Reduce confidence threshold, smaller size
    CRISIS  > 25      → Block all ML signals

Usage:
    from ml_predictor.regime_detector import RegimeDetector
    rd = RegimeDetector()
    regime = rd.get(vix=16.5)
    if rd.is_tradeable(vix):
        threshold = rd.threshold(vix, base_time="normal")
"""


class RegimeDetector:

    QUIET   = "QUIET"
    NORMAL  = "NORMAL"
    FEARFUL = "FEARFUL"
    CRISIS  = "CRISIS"

    # VIX bands
    _QUIET_MAX  = 13.0
    _NORMAL_MAX = 20.0
    _FEAR_MAX   = 25.0

    # Confidence thresholds per regime × time-of-day
    # Format: {regime: {time_bucket: threshold}}
    _THRESHOLDS = {
        QUIET:   {"open": 0.70, "normal": 0.70, "close": 0.70},  # raise bar in low-vol
        NORMAL:  {"open": 0.65, "normal": 0.55, "close": 0.60},
        FEARFUL: {"open": 0.70, "normal": 0.62, "close": 0.68},
        CRISIS:  {"open": 1.00, "normal": 1.00, "close": 1.00},  # effectively blocked
    }

    # Position size multipliers per regime
    _SIZE = {
        QUIET:   0.50,
        NORMAL:  1.00,
        FEARFUL: 0.50,
        CRISIS:  0.00,
    }

    def get(self, vix: float) -> str:
        """Return regime string for given VIX value."""
        if vix is None:
            return self.NORMAL   # default if VIX unavailable
        if vix < self._QUIET_MAX:
            return self.QUIET
        if vix < self._NORMAL_MAX:
            return self.NORMAL
        if vix < self._FEAR_MAX:
            return self.FEARFUL
        return self.CRISIS

    def is_tradeable(self, vix: float) -> bool:
        """Returns False during CRISIS regime — block all ML signals."""
        return self.get(vix) != self.CRISIS

    def threshold(self, vix: float, time_bucket: str = "normal") -> float:
        """
        Return confidence threshold for given VIX and time bucket.

        time_bucket:
            "open"   → 09:15–09:45 (chaotic)
            "normal" → 09:45–14:00 (trending)
            "close"  → 14:00–15:30 (EOD)
        """
        regime = self.get(vix)
        return self._THRESHOLDS[regime].get(time_bucket, 0.55)

    def size_multiplier(self, vix: float) -> float:
        """Return position size multiplier (0.0 = no trade, 1.0 = full size)."""
        return self._SIZE[self.get(vix)]

    def time_bucket(self, ts) -> str:
        """Classify a datetime into time bucket."""
        from datetime import time as dtime
        t = ts.time() if hasattr(ts, "time") else ts
        if t < dtime(9, 45):
            return "open"
        if t < dtime(14, 0):
            return "normal"
        return "close"

    def describe(self, vix: float, ts=None) -> dict:
        """Full regime status dict — useful for logging."""
        regime  = self.get(vix)
        bucket  = self.time_bucket(ts) if ts else "normal"
        thresh  = self.threshold(vix, bucket)
        size    = self.size_multiplier(vix)
        return {
            "regime":     regime,
            "tradeable":  regime != self.CRISIS,
            "time_bucket": bucket,
            "threshold":  thresh,
            "size_mult":  size,
            "vix":        vix,
        }
