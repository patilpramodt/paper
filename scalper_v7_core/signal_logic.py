# ==============================================================================
# signal_logic.py  V6 Signal Engine (All Filters Active)
#
# Entry filter stack (in order of application):
#   1. ATR Gate            block dead/low-vol markets
#   2. Regime Check        5-min must be TRENDING (ATR-relative EMA gap)
#   3. Sideways Filter     RSI 4654 + flat MACD = no trade
#   4. EMA Alignment       1-min EMA gap with ATR-relative minimum
#   5. RSI Z-Score         adaptive threshold (replaces fixed RSI levels)
#   6. RSI Slope           momentum must be RISING (for CE) or FALLING (for PE)
#   7. MACD Expansion      histogram expanding for 2 consecutive bars
#   8. ATR-Scaled MACD     histogram magnitude vs ATR
#   9. Volume Confirm      bullish/bearish volume required
#  10. Structure Break     price must break above 5-candle high (for CE)
#  11. Exhaustion Guard    block entries at RSI extremes with fading slope
# ==============================================================================

import datetime as dt

# IST FIX: GitHub Actions runners are UTC — bare datetime.now() returns UTC
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

def _now_ist() -> dt.datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return dt.datetime.now(tz=_IST).replace(tzinfo=None)
from typing import Dict, Any

from scalper_v7_core.config import (
    # ATR gate
    ATR_MIN_PCT, LUNCH_ATR_MULTIPLIER,
    # ATR vol-ratio (V7 new)
    ATR_VOL_RATIO_MIN, ATR_VOL_RATIO_LUNCH_MIN,
    # Regime
    REGIME_EMA_ATR_RATIO, REGIME_EMA_ATR_RATIO_1M,
    EMA_GAP_5M_FIXED, LUNCH_EMA_ATR_RATIO_5M,
    # RSI Z-score
    RSI_ZSCORE_BULL, RSI_ZSCORE_BEAR, RSI_ZSCORE_EXTREME,
    RSI_BULL_RAW, RSI_BEAR_RAW, RSI_OVERBOUGHT, RSI_OVERSOLD,
    LUNCH_RSI_ZSCORE_BULL, LUNCH_RSI_ZSCORE_BEAR,
    # RSI slope + acceleration (V7)
    RSI_SLOPE_MIN, RSI_ACC_MIN,
    # MACD
    MACD_EXPANSION_BARS, MACD_ATR_RATIO, MACD_HIST_FLOOR,
    MACD_HIST_CAP, MACD_SLOPE_MIN,
    LUNCH_MACD_ATR_RATIO,
    # VWAP (V7 new)
    VWAP_FILTER_ENABLED, VWAP_BUFFER_PTS,
    # Session
    LUNCH_START, LUNCH_END,
)
from scalper_v7_core.logger_setup import log


#  Volume labels (must match indicators.py output) 
BULL_VOL = ("bullish",)
BEAR_VOL = ("bearish",)


# ==============================================================================
# Helpers
# ==============================================================================

def _in_lunch() -> bool:
    now = _now_ist().time()  # FIX: was dt.datetime.now() — UTC on GitHub Actions
    return dt.time(*LUNCH_START) <= now < dt.time(*LUNCH_END)


def _get_5m_regime(ind5m: dict, lunch: bool) -> str:
    """
    Classify 5-min trend as BULL / BEAR / SIDEWAYS.
    Uses ATR-relative EMA gap (adapts to volatility).
    Falls back to fixed gap if ATR unavailable.
    """
    ema_gap  = float(ind5m.get("ema_gap",  0.0))
    atr5     = float(ind5m.get("atr14",    0.0))
    ratio    = LUNCH_EMA_ATR_RATIO_5M if lunch else REGIME_EMA_ATR_RATIO

    if atr5 > 0:
        threshold = atr5 * ratio
    else:
        threshold = EMA_GAP_5M_FIXED  # fallback

    if   ema_gap >=  threshold: return "BULL"
    elif ema_gap <= -threshold: return "BEAR"
    else:                       return "SIDEWAYS"


def _atr_gate_ok(ind1m: dict, lunch: bool) -> bool:
    """
    Return True only if ATR % of spot exceeds the minimum.
    In lunch, require 30% more volatility.
    """
    atr_pct  = float(ind1m.get("atr_pct", 0.0))
    required = ATR_MIN_PCT * (LUNCH_ATR_MULTIPLIER if lunch else 1.0)
    ok       = atr_pct >= required
    if not ok:
        log.info(f"[SCALPER] ATR gate BLOCKED: atr_pct={atr_pct:.4f} < required={required:.4f} (lunch={lunch})")
    return ok


def _atr_vol_ratio_ok(ind1m: dict, lunch: bool) -> bool:
    """
    NEW V7: ATR must be above its own rolling mean by a ratio.
    Checks if volatility is EXPANDING, not just present.
    Complements _atr_gate_ok (which checks absolute level).
    """
    vol_ratio = float(ind1m.get("atr_vol_ratio", 1.0))
    required  = ATR_VOL_RATIO_LUNCH_MIN if lunch else ATR_VOL_RATIO_MIN
    ok        = vol_ratio >= required
    if not ok:
        log.info(f"[SCALPER] ATR vol_ratio BLOCKED: {vol_ratio:.3f} < {required:.3f} (lunch={lunch})")
    return ok


def _vwap_ok(ind1m: dict, direction: str) -> bool:
    """
    NEW V7: Price position relative to VWAP.
    VWAP is the only non-price-derivative signal  addresses collinearity.
    For CE: price > VWAP - buffer (price above average buy cost = bullish)
    For PE: price < VWAP + buffer (price below average sell cost = bearish)
    """
    if not VWAP_FILTER_ENABLED:
        return True
    vwap       = float(ind1m.get("vwap",       0.0))
    last_close = float(ind1m.get("last_close", 0.0))
    if vwap <= 0 or last_close <= 0:
        return True   # data unavailable  don't block
    if direction == "bull":
        ok = last_close >= (vwap - VWAP_BUFFER_PTS)
        if not ok:
            log.debug(f"VWAP BLOCKED bull: close={last_close:.2f} < vwap={vwap:.2f}-{VWAP_BUFFER_PTS}")
        return ok
    else:
        ok = last_close <= (vwap + VWAP_BUFFER_PTS)
        if not ok:
            log.debug(f"VWAP BLOCKED bear: close={last_close:.2f} > vwap={vwap:.2f}+{VWAP_BUFFER_PTS}")
        return ok


def _macd_threshold(ind1m: dict, lunch: bool) -> float:
    """
    ATR-scaled MACD threshold.
    Replaces the fixed 0.40 from V4/V5.
    """
    atr   = float(ind1m.get("atr14", 0.0))
    ratio = LUNCH_MACD_ATR_RATIO if lunch else MACD_ATR_RATIO
    return float(max(MACD_HIST_FLOOR, min(atr * ratio, MACD_HIST_CAP)))


def _macd_expanding(hist_series: list, direction: str) -> bool:
    """
    Return True if histogram is expanding in the correct direction
    for MACD_EXPANSION_BARS consecutive bars.
    direction: "bull" (histogram rising) or "bear" (histogram falling)
    """
    if len(hist_series) < MACD_EXPANSION_BARS + 1:
        return False
    bars = hist_series[-(MACD_EXPANSION_BARS + 1):]
    if direction == "bull":
        return all(bars[i] < bars[i + 1] for i in range(len(bars) - 1))
    else:
        return all(bars[i] > bars[i + 1] for i in range(len(bars) - 1))


def _structure_break(ind1m: dict, direction: str) -> bool:
    """
    Price must break above the N-candle high (for CE)
    or below the N-candle low (for PE).
    This is the leading signal that reduces lag-stack problem.
    """
    last_close = float(ind1m.get("last_close",    0.0))
    struct_hi  = float(ind1m.get("structure_high",0.0))
    struct_lo  = float(ind1m.get("structure_low", 0.0))

    if direction == "bull":
        ok = last_close > struct_hi
        if not ok:
            log.debug(f"Structure break BLOCKED (bull): close={last_close:.2f} <= struct_hi={struct_hi:.2f}")
        return ok
    else:
        ok = last_close < struct_lo
        if not ok:
            log.debug(f"Structure break BLOCKED (bear): close={last_close:.2f} >= struct_lo={struct_lo:.2f}")
        return ok


# ==============================================================================
# Main Signal Builder
# ==============================================================================

def _build_signal(ind1m: dict, ind5m: dict, lunch: bool) -> tuple[str, str]:
    """
    Apply all filters. Returns (action, blocked_by_reason).
    action: "BUY_CE" | "BUY_PE" | "HOLD"
    """

    #  Filter 1: ATR Gate 
    if not _atr_gate_ok(ind1m, lunch):
        return "HOLD", "atr_gate"

    #  Filter 1b: ATR Vol-Ratio (V7 NEW) 
    if not _atr_vol_ratio_ok(ind1m, lunch):
        return "HOLD", "atr_vol_ratio"

    #  Filter 2: 5-Min Regime 
    trend_bias = _get_5m_regime(ind5m, lunch)
    if trend_bias == "SIDEWAYS":
        return "HOLD", "5m_sideways"

    if not ind5m.get("is_trending", False):
        return "HOLD", "5m_not_trending"

    #  Filter 3: Sideways / Ranging filter 
    rsi   = float(ind1m.get("rsi14",     50.0))
    hist  = float(ind1m.get("macd_hist",  0.0))
    if 46 <= rsi <= 54 and abs(hist) < 0.15:
        return "HOLD", "ranging"

    #  Filter 4: 1-Min Regime (EMA gap vs ATR) 
    ema_gap  = float(ind1m.get("ema_gap",  0.0))
    atr1     = float(ind1m.get("atr14",    0.0))
    ema_ok_bull = ema_gap >= max(3.0, atr1 * REGIME_EMA_ATR_RATIO_1M)
    ema_ok_bear = ema_gap <= -max(3.0, atr1 * REGIME_EMA_ATR_RATIO_1M)

    #  Filter 5: RSI Z-Score 
    zscore_ready = ind1m.get("zscore_ready", False)
    rsi_z        = float(ind1m.get("rsi_z", 0.0))

    if zscore_ready:
        bull_rsi_ok        = rsi_z >= (LUNCH_RSI_ZSCORE_BULL if lunch else RSI_ZSCORE_BULL)
        bear_rsi_ok        = rsi_z <= (LUNCH_RSI_ZSCORE_BEAR if lunch else RSI_ZSCORE_BEAR)
        rsi_exhausted_bull = rsi_z >= RSI_ZSCORE_EXTREME
        rsi_exhausted_bear = rsi_z <= -RSI_ZSCORE_EXTREME
    else:
        bull_rsi_ok        = rsi > RSI_BULL_RAW
        bear_rsi_ok        = rsi < RSI_BEAR_RAW
        rsi_exhausted_bull = rsi >= RSI_OVERBOUGHT
        rsi_exhausted_bear = rsi <= RSI_OVERSOLD

    #  Filter 6: RSI Slope 
    rsi_slope         = float(ind1m.get("rsi_slope", 0.0))
    rsi_slope_ok_bull = rsi_slope >= RSI_SLOPE_MIN
    rsi_slope_ok_bear = rsi_slope <= -RSI_SLOPE_MIN

    #  Filter 6b: RSI Acceleration (V7 NEW) 
    # 2nd derivative: slope must be increasing (impulse start, not plateau).
    rsi_acc           = float(ind1m.get("rsi_acc", 0.0))
    rsi_acc_ok_bull   = rsi_acc >= RSI_ACC_MIN
    rsi_acc_ok_bear   = rsi_acc <= -RSI_ACC_MIN

    #  Filter 7+8: MACD Expansion + ATR-Scaled threshold 
    macd_thresh  = _macd_threshold(ind1m, lunch)
    hist_series  = ind1m.get("macd_hist_series", [hist, hist, hist])
    macd_slope   = float(ind1m.get("macd_slope", 0.0))

    macd_bull_ok = (
        hist > macd_thresh and
        macd_slope >= MACD_SLOPE_MIN and
        _macd_expanding(hist_series, "bull")
    )
    macd_bear_ok = (
        hist < -macd_thresh and
        macd_slope <= -MACD_SLOPE_MIN and
        _macd_expanding(hist_series, "bear")
    )

    #  Filter 9: Volume Confirmation 
    vol          = str(ind1m.get("volume_trend", "flat")).lower()
    vol_bull_ok  = vol in BULL_VOL
    vol_bear_ok  = vol in BEAR_VOL

    #  Combine: Bullish 
    if (
        trend_bias == "BULL"   and
        ema_ok_bull            and
        bull_rsi_ok            and
        rsi_slope_ok_bull      and
        rsi_acc_ok_bull        and
        macd_bull_ok           and
        vol_bull_ok            and
        not rsi_exhausted_bull
    ):
        if not _structure_break(ind1m, "bull"):
            return "HOLD", "no_structure_break_bull"
        #  Filter 11: VWAP (V7 NEW) 
        if not _vwap_ok(ind1m, "bull"):
            return "HOLD", "vwap_bull"
        return "BUY_CE", ""

    #  Combine: Bearish 
    if (
        trend_bias == "BEAR"   and
        ema_ok_bear            and
        bear_rsi_ok            and
        rsi_slope_ok_bear      and
        rsi_acc_ok_bear        and
        macd_bear_ok           and
        vol_bear_ok            and
        not rsi_exhausted_bear
    ):
        if not _structure_break(ind1m, "bear"):
            return "HOLD", "no_structure_break_bear"
        if not _vwap_ok(ind1m, "bear"):
            return "HOLD", "vwap_bear"
        return "BUY_PE", ""

    #  Granular block reason 
    if trend_bias == "BULL":
        if not ema_ok_bull:          return "HOLD", "ema_gap_bull"
        if not bull_rsi_ok:          return "HOLD", "rsi_z_bull"
        if not rsi_slope_ok_bull:    return "HOLD", "rsi_slope_bull"
        if not rsi_acc_ok_bull:      return "HOLD", "rsi_acc_bull"
        if not macd_bull_ok:         return "HOLD", "macd_bull"
        if not vol_bull_ok:          return "HOLD", "volume_bull"
        if rsi_exhausted_bull:       return "HOLD", "rsi_exhausted_bull"
    elif trend_bias == "BEAR":
        if not ema_ok_bear:          return "HOLD", "ema_gap_bear"
        if not bear_rsi_ok:          return "HOLD", "rsi_z_bear"
        if not rsi_slope_ok_bear:    return "HOLD", "rsi_slope_bear"
        if not rsi_acc_ok_bear:      return "HOLD", "rsi_acc_bear"
        if not macd_bear_ok:         return "HOLD", "macd_bear"
        if not vol_bear_ok:          return "HOLD", "volume_bear"
        if rsi_exhausted_bear:       return "HOLD", "rsi_exhausted_bear"

    return "HOLD", "no_setup"


# ==============================================================================
# Public Interface
# ==============================================================================

def get_signal(snapshot: dict) -> Dict[str, Any]:
    """
    Generate trade signal from market snapshot.

    Returns:
        action       : "BUY_CE" | "BUY_PE" | "HOLD"
        trend_bias   : "BULL" | "BEAR" | "SIDEWAYS"
        blocked_by   : reason string if HOLD
        in_lunch     : bool
        ind1m, ind5m : indicator dicts for logging
    """
    ind1m  = snapshot.get("indicators_1m") or {}
    ind5m  = snapshot.get("indicators_5m") or {}
    lunch  = _in_lunch()

    trend_bias        = _get_5m_regime(ind5m, lunch)
    action, blocked   = _build_signal(ind1m, ind5m, lunch)

    return {
        "action":     action,
        "trend_bias": trend_bias,
        "blocked_by": blocked,
        "in_lunch":   lunch,
        "ind1m":      ind1m,
        "ind5m":      ind5m,
    }
