"""
core/fast_indicators.py

Fast/leading confirmation indicators for the Candle Breakout v1 strategies
(NIFTY_CANDLE_BREAKOUT + BANKNIFTY_CANDLE_BREAKOUT).

LOG-ONLY — nothing computed here gates entries. Both strategies call
compute_fast_indicators() once per signal event (TRIGGER, CONFIRM_MATCH,
CONFIRM_MISMATCH, NO_BREAKOUT, BREAKOUT_ENTER, FAST_TICK_ENTER) and write
the result as extra columns on the signals CSV and the trades CSV. The
existing state machine (C1 body > threshold -> 5s confirm -> tick
breakout) is completely untouched.

DATA SOURCES
────────────
  candles : rolling window of the strategy's own closed 10-second candles,
            via self._c10.last_n_closed(N) (core/candle.py — already
            tracked by both strategies for the pattern state machine, no
            new plumbing needed).
  vwap    : self._hub.session_vwap.value   (core/vwap.py — true session
            VWAP, already live/instant on every tick).
  pcr     : self._pm.pcr                   (core/pcr_kite.py's WsPCR via
            the live PreMarketData reference — updated every pcr_interval
            seconds by the background refresh thread in premarket.py).

WHY "LEADING" (FAST) PERIODS, NOT THE STANDARD ONES  (per user request, 2026-07-22)
────────────────────────────────────────────────────────────────────────────────────
Standard EMA20/50, RSI14, MACD(12,26,9), ATR14, Supertrend(10,3) were built
for 5-minute-ish bars. On this strategy's 10-second candles those periods
span 3-8+ minutes of history before they react — far slower than the
pattern itself, which resolves in 10-15 seconds. Every period below is
deliberately short so the indicator reacts within a handful of 10s candles
(tens of seconds, not minutes) instead of lagging behind the move:

    EMA          5 / 10         (vs standard 20/50)
    RSI          7, 3-bar slope (vs standard 14)
    MACD         5 / 13 / 4     (vs standard 12/26/9)
    ATR          7              (vs standard 14)
    Supertrend   period=7, multiplier=1.5  (vs common 10/3 — flips faster)

These are still the conventional EMA/RSI/MACD/ATR/Supertrend formulas —
just tuned to this strategy's much shorter timeframe so they lead rather
than lag it.
"""

import pandas as pd

# ── Fast/leading periods (10s-candle tuned — see module docstring) ──────────
EMA_FAST_PERIOD    = 5
EMA_SLOW_PERIOD    = 10
EMA_SLOPE_LOOKBACK = 3

RSI_PERIOD         = 7
RSI_SLOPE_LOOKBACK = 3

MACD_FAST          = 5
MACD_SLOW          = 13
MACD_SIGNAL        = 4

ATR_PERIOD         = 7

SUPERTREND_PERIOD  = 7
SUPERTREND_MULT    = 1.5

# Rolling window of closed 10s candles pulled via self._c10.last_n_closed(N).
# 50 candles = ~8.3 min of history — enough warmup for all periods above
# while still being "recent tape", not the whole session.
CANDLE_WINDOW = 50

# Need at least this many closed 10s candles before trusting any of the
# derived (non-VWAP/PCR) indicators below.
MIN_CANDLES_FOR_INDICATORS = 12

# CSV column order — both strategies' _log_signal_csv / _log_csv use this
# list so the signals CSV and the trades CSV carry identical indicator
# columns.
INDICATOR_FIELDS = [
    "vwap", "spot_vs_vwap", "pcr",
    "ema_fast", "ema_slow", "ema_gap", "ema_slope",
    "rsi", "rsi_slope",
    "macd_hist", "macd_slope",
    "atr", "atr_pct",
    "supertrend", "supertrend_dir",
    "indicators_ready",
]

_EMPTY = {k: "" for k in INDICATOR_FIELDS}
_EMPTY["indicators_ready"] = False


def compute_fast_indicators(candles: list, spot: float = None,
                             vwap: float = None, pcr: float = None) -> dict:
    """
    candles : list of closed 10s candle dicts (ts/open/high/low/close),
              oldest first — pass self._c10.last_n_closed(CANDLE_WINDOW).
    spot    : current index price, for the spot_vs_vwap flag. Optional.
    vwap    : self._hub.session_vwap.value. Already live/fast — passed
              through as-is (no alternate period needed).
    pcr     : self._pm.pcr. Already live/fast — passed through as-is.

    Returns a flat dict (see INDICATOR_FIELDS) with blank "" for anything
    not yet computable, so callers can always do meta.get(col, "") safely
    without special-casing. Never raises.
    """
    out = dict(_EMPTY)
    out["vwap"] = round(vwap, 2) if vwap else ""
    out["pcr"]  = round(pcr, 3) if pcr else ""
    if spot and vwap:
        if spot > vwap:
            out["spot_vs_vwap"] = "ABOVE"
        elif spot < vwap:
            out["spot_vs_vwap"] = "BELOW"
        else:
            out["spot_vs_vwap"] = "FLAT"

    if not candles or len(candles) < MIN_CANDLES_FOR_INDICATORS:
        return out

    try:
        df    = pd.DataFrame(candles)
        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        # ── EMA fast/slow, gap, slope ────────────────────────────────────
        ema_fast_s = close.ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
        ema_slow_s = close.ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
        out["ema_fast"] = round(float(ema_fast_s.iloc[-1]), 2)
        out["ema_slow"] = round(float(ema_slow_s.iloc[-1]), 2)
        out["ema_gap"]  = round(float(ema_fast_s.iloc[-1] - ema_slow_s.iloc[-1]), 2)
        if len(ema_fast_s) >= EMA_SLOPE_LOOKBACK + 1:
            out["ema_slope"] = round(
                float(ema_fast_s.iloc[-1] - ema_fast_s.iloc[-(EMA_SLOPE_LOOKBACK + 1)]), 2
            )

        # ── RSI + slope (Wilder's smoothing, short period) ──────────────
        delta = close.diff()
        gain  = delta.clip(lower=0.0)
        loss  = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        out["rsi"] = round(float(rsi_series.iloc[-1]), 2)
        if len(rsi_series) >= RSI_SLOPE_LOOKBACK + 1:
            out["rsi_slope"] = round(
                float(rsi_series.iloc[-1] - rsi_series.iloc[-(RSI_SLOPE_LOOKBACK + 1)]), 2
            )

        # ── MACD histogram + slope (fast/short spans) ───────────────────
        ema_macd_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_macd_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line     = ema_macd_fast - ema_macd_slow
        signal_line   = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist          = macd_line - signal_line
        out["macd_hist"] = round(float(hist.iloc[-1]), 4)
        if len(hist) >= 2:
            out["macd_slope"] = round(float(hist.iloc[-1] - hist.iloc[-2]), 4)

        # ── ATR + ATR% (Wilder's, short period) ─────────────────────────
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
        atr_val    = float(atr_series.iloc[-1])
        out["atr"] = round(atr_val, 2)
        last_close = float(close.iloc[-1])
        if last_close > 0:
            out["atr_pct"] = round(atr_val / last_close * 100, 4)

        # ── Supertrend (short period/multiplier -> flips fast) ──────────
        st_val, st_dir = _supertrend(high, low, close, SUPERTREND_PERIOD, SUPERTREND_MULT)
        out["supertrend"]     = round(st_val, 2) if st_val is not None else ""
        out["supertrend_dir"] = st_dir or ""

        out["indicators_ready"] = True
    except Exception:
        # Log-only feature: never let indicator computation break the
        # actual trading state machine. Return whatever was computed
        # so far (vwap/pcr survive even if the pandas block throws).
        pass

    return out


def _supertrend(high: "pd.Series", low: "pd.Series", close: "pd.Series",
                 period: int, multiplier: float):
    """
    Standard Supertrend, computed with a short period/multiplier so it
    flips quickly relative to this strategy's 10s candles (leading, not
    lagging). Returns (last_value, "UP"|"DOWN"), or (None, None) if there
    isn't enough history yet.
    """
    if len(close) < period + 1:
        return None, None

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2         = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = [True] * len(close)  # True = uptrend

    for i in range(1, len(close)):
        upper.iloc[i] = (
            upper_basic.iloc[i]
            if (upper_basic.iloc[i] < upper.iloc[i - 1] or close.iloc[i - 1] > upper.iloc[i - 1])
            else upper.iloc[i - 1]
        )
        lower.iloc[i] = (
            lower_basic.iloc[i]
            if (lower_basic.iloc[i] > lower.iloc[i - 1] or close.iloc[i - 1] < lower.iloc[i - 1])
            else lower.iloc[i - 1]
        )

        if close.iloc[i] > upper.iloc[i - 1]:
            direction[i] = True
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction[i] = False
        else:
            direction[i] = direction[i - 1]

    st_line = lower.iloc[-1] if direction[-1] else upper.iloc[-1]
    return float(st_line), ("UP" if direction[-1] else "DOWN")

