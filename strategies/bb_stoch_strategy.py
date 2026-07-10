"""
strategies/bb_stoch_strategy.py
-------------------------------------------------------------
BBStochStrategy -- BankNifty options scalper using:
  * Bollinger Bands  (BB)       -- identifies breakout / volatility expansion
  * Volume Filter               -- confirms genuine institutional participation
  * Stochastic momentum filter  -- confirms momentum direction at band touch
                                   Stochastic(5, 3, 3) computed on the live
                                   5-min candle buffer; no seeding required.

SIGNAL LOGIC (all filters must agree before entry):
--------------------------------------------------------------
  CE entry (buy call):
    1. BB Breakout : last close breaks ABOVE upper BB band
       OR BB Bounce : last close crosses back above lower BB band from below
       OR BB Middle : last close crosses ABOVE the BB middle band
    2. Volume     : current bar volume >= VOL_MULT * rolling avg vol
    3. VWAP       : close above session VWAP
    4. Stochastic : K < 80 (not overbought) AND K crosses above D
                    (momentum turning up at/near band touch)
    5. Session    : only between SESSION_START and ENTRY_CUTOFF

  PE entry (buy put):
    Mirror of the above -- BB break below lower band, bounce below upper
    band, or middle band cross DOWN; K > 20 (not oversold) AND K crosses
    below D (momentum turning down).

TRADE MANAGEMENT (on every option WebSocket tick):
--------------------------------------------------------------
  * SL    = entry_price - ATR_SL_MULT * atr   (anchored to fill price)
  * TP    = entry_price + ATR_TP_MULT * atr   (anchored to fill price)
  * Trail = moves SL to breakeven when profit >= TRAIL_ARM pts,
            then trails every TRAIL_STEP pts after that

FIXES APPLIED:
  Bug 1  -- time.sleep() inside WebSocket callback blocked all tick
             processing. Removed entirely.
  Bug 2  -- _persist_buf cleared before fill; now cleared only on
             genuine fill inside _execute_fill().
  Bug 3  -- _close_trade() race condition between WS thread and main
             thread. Fixed with atomic lock at top.
  Bug 4  -- ATM options not pre-subscribed. Fixed in pre_market().
  Bug 5  -- Only opening ATM pre-subscribed. Now ATM+-200 covered.
  Bug 6  -- Large gap days: spot-based ATM+-400 subscription on
             first candle (~9:20 AM).
  Bug 7  -- LTP unavailable on entry: stored as _pending_entry,
             filled on first tick via on_option_tick().
  Bug 8  -- SL/TP anchored to LTP not fill price. Fixed in BBTrade.
  Bug 9  -- Trail breakeven SL at entry, not entry+slippage. Fixed.
  Bug 10 -- _force_close fired on every tick after 15:15. Fixed with
             _squareoff_done flag.
  Bug 11 -- BBTrade __slots__ missing "order_id". Fixed.
  Bug 12 -- Stale cached price used for entry when token was previously
             unsubscribed by another strategy (e.g. HEDGED_SELL closes and
             unsubscribes 53300PE; price freezes at 348.90; BB_STOCH
             re-subscribes 104 min later, get_price() returns 348.90 while
             actual market is 524.50; first live tick hits TP instantly).
             Fix: check get_price_ts(); treat price as unavailable when
             age > 30 s, forcing _pending_entry path for first live tick.
  Bug 13 -- _place_buy() returns Tuple[str, float] (order_id, fill_price)
             but _execute_fill stored the whole tuple as trade.order_id and
             ignored the actual exchange fill price, anchoring SL/TP to the
             ref ltp instead.  Fix: unpack the tuple; use fill_price for
             BBTrade construction in live mode.
  Bug 14 -- No historical seeding: _buf_5m started empty every session and
             needed min_bars=10 (~50 live minutes from 9:15) before the
             first possible signal, missing the opening volatility window
             every single day.  Fix: pre_market() now seeds _buf_5m with
             the prior session's last N BankNifty 5-min candles via
             hub.kite.historical_data(). See _seed_historical_buffer().
  Bug 15 -- ATR unit mismatch: _compute_sl_tp() used raw 5-min BankNifty
             INDEX-point ATR (typically 60-150+) directly as the option's
             SL/TP distance in premium points. atr*atr_sl_mult almost always
             exceeded the old sl_max=50 ceiling, so SL was effectively a
             flat 50pt premium stop every day regardless of real option
             volatility -- the "ATR-adaptive" sizing was never adaptive.
             Fix: scale index-point ATR to premium-point ATR via
             premium_atr_scale (0.45) before sizing, and recalibrate
             sl_min/sl_max/tp_min/tp_max to the new range. Mirrors the
             same fix already validated in bb_stoch_nifty_strategy.py.
  Bug 16 -- Routing collision: a class-level `INDEX_TOKEN = 260105`
             attribute (BankNifty's OWN token) made MarketHub treat this
             strategy as tracking a *secondary* index and skip it from
             every BankNifty on_tick()/on_candle() broadcast, with no
             error logged. Fix: removed the class attribute; the BankNifty
             token is now only a module-level constant
             (BANKNIFTY_INDEX_TOKEN) used solely for the historical_data()
             seed call. See comment above BANKNIFTY_INDEX_TOKEN below.
  Bug 17 -- Spread filter silently zeroed out every real trade. The gate
             in _execute_fill() used `est_spread = ltp * 0.03` (flat 3% of
             premium) against `tp_pts`, which is capped in ABSOLUTE premium
             points (tp_min=15..tp_max=90) regardless of premium size.
             BankNifty ATM premiums run ~700-1200 (a function of the
             index's absolute level and 100-pt strike spacing, NOT of
             liquidity), so 3% of premium is ~21-36 pts -- routinely 40-55%
             of a TP capped at 53-90 pts. Every genuine setup on
             2026-07-09 (10:45, 12:55, 14:25) failed this gate; BB_STOCH
             finished the session with 3 valid signals and 0 trades, which
             matches the pattern of EOD Trades=0 seen historically. The
             same constant almost never binds on bb_stoch_nifty_strategy.py
             because Nifty ATM premiums (~130-150) make 3% only ~4 pts --
             which is why the two strategies looked functionally different
             in production despite sharing identical filter code.
             Fix: replaced the flat 3%-of-premium guess with
             _estimate_spread() -- a modest percentage WITH an absolute
             point cap, since real bid-ask spreads on liquid near-ATM index
             options don't keep growing linearly with premium; they're
             bounded by tick size and market-maker competition, not premium
             magnitude. See CFG["spread_pct"] / CFG["spread_cap_pts"].
  Bug 18 -- Flat CFG["slippage"]=1.5 constant used everywhere (entry markup,
             breakeven-trail target, exit markdown) regardless of the
             option's actual premium size -- the same "one flat number
             across very different premium scales" mistake as Bug 17, just
             applied to fill/exit modeling instead of the entry gate.
             A 1.5pt assumed half-spread is plausible for a ~130-150
             Nifty premium but understates the real cost of crossing the
             market on a ~900+ BankNifty premium, and overstates it on thin
             far-OTM contracts trading near the sl_min floor.
             Fix: BBTrade now carries a per-trade `slip` computed once at
             entry from _estimate_spread(entry_ltp)/2 (floored at
             CFG["slippage_min"]), and _update_trail()/_close_trade() reuse
             that same stored value -- consistent with each other (so the
             Bug 9 breakeven guarantee still holds exactly) and scaled to
             what was actually traded, not a single constant borrowed from
             Nifty's economics and silently reused for BankNifty.
"""

import csv
import logging
import os
import threading
import time as time_module
from collections import deque
from datetime import datetime, date, time as dtime, timezone, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

# BankNifty NSE index token — fixed Zerodha instrument token (260105).
# Used exclusively in _seed_historical_buffer() to fetch prior-session
# 5-min OHLC candles via kite.historical_data().
#
# BUG FIX (routing collision): this MUST be a plain module-level constant,
# NOT a class-level `INDEX_TOKEN` attribute on the strategy. MarketHub's
# _handle_index_tick() treats any strategy with a non-None `INDEX_TOKEN`
# class attribute as tracking a *different* (secondary) index — e.g.
# BBStochNiftyStrategy sets INDEX_TOKEN=256265 so it gets routed Nifty
# ticks/candles via _handle_extra_index_tick() instead of the main
# BankNifty feed. Because this strategy previously set
# `INDEX_TOKEN = 260105` (BankNifty's OWN token) as a class attribute,
# MarketHub's broadcast loop:
#     strat_index = getattr(strat, "INDEX_TOKEN", None)
#     if strat_index is not None:
#         continue   # "tracks a different index" — WRONG for this strategy
# silently skipped this strategy on every single BankNifty on_tick() and
# on_candle() call, all day, with no error or warning logged. The token
# also isn't in MarketHub._extra_index_tokens (that set is only populated
# for genuinely secondary indices), so it was never routed anywhere —
# fully orphaned despite successful pre_market() init and option
# subscriptions. Result: zero HOLD logs, zero entries, EOD Trades=0,
# every session, with no diagnostic trace.
#
# scalper_v7_strategy.py already uses the correct pattern for the same
# need (BANKNIFTY_INDEX_TOKEN module constant, no class attribute) —
# mirrored here. This strategy now has no class-level INDEX_TOKEN
# attribute, so `getattr(strat, "INDEX_TOKEN", None)` correctly returns
# None for it, and it receives the main BankNifty tick/candle broadcast
# exactly like every other primary-index strategy (SPIKE, ORB_v2,
# ScalperV7, etc).
BANKNIFTY_INDEX_TOKEN = 260105

# IST FIX: GitHub Actions runners are UTC.
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


log = logging.getLogger("strategy.bb_stoch")

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False


# ============================================================
# CONFIG
# ============================================================
CFG = {
    # Bollinger Bands
    "bb_period"          : 20,
    "bb_std"             : 2.0,
    "bb_squeeze_pct"     : 0.002,   # band-width/close < this → squeeze, skip all entries
    "bb_mid_squeeze_pct" : 0.003,   # stricter threshold for middle-band cross only
    "bb_min_bw_breakout" : 0.36,    # breakout entries only when BW/close >= this

    # Volume filter
    "vol_avg_period"     : 10,
    "vol_mult"           : 0.85,

    # VWAP
    "vwap_buffer"        : 10.0,    # allow entry within N pts of VWAP

    # Session windows (HH, MM)
    "session_start"      : (9, 45),
    "session_cutoff"     : (15, 0),
    "auto_squareoff"     : (15, 15),

    # Trade management
    "atr_period"         : 14,
    "atr_sl_mult"        : 0.8,
    "atr_tp_mult"        : 2.0,
    # BUG FIX (ATR unit mismatch): `atr` computed by _compute_atr(df, ...) is
    # measured on the 5-min BankNifty INDEX candle buffer, i.e. raw index
    # points (typically 60-150+ on BankNifty) -- NOT option premium points.
    # _compute_sl_tp() previously used this index-point ATR directly as the
    # option's SL/TP distance in premium points. Since atr*atr_sl_mult
    # (≈48-120) almost always exceeded the old sl_max=50 ceiling, SL was
    # effectively pinned to a flat 50pt premium stop every day regardless of
    # real option volatility -- the "ATR-adaptive" sizing was never actually
    # adaptive. Same issue for TP, usually pinned near the old tp_max=120.
    # Fix: scale index-point ATR down to an approximate premium-point ATR
    # before sizing, mirroring the same fix already applied and validated
    # in bb_stoch_nifty_strategy.py. sl_min/sl_max/tp_min/tp_max recalibrated
    # to match the new (correctly-scaled) premium ATR range.
    "premium_atr_scale"  : 0.45,
    "sl_min"             : 8.0,
    "sl_max"             : 35.0,
    "tp_min"             : 15.0,
    "tp_max"             : 90.0,
    "trail_arm"          : 25.0,    # move SL to BE when profit >= this
    "trail_step"         : 12.0,    # then trail every N pts

    # Bug 17/18 fix: spread/slippage now modeled from the option's own
    # premium instead of a single flat constant borrowed from Nifty's
    # (much cheaper) premium economics and silently reused for BankNifty.
    # Real bid-ask spreads on liquid near-ATM index options are bounded by
    # tick size / market-maker competition, not premium magnitude -- so
    # this is a modest percentage WITH an absolute cap, not a pure %.
    # See _estimate_spread().
    "spread_pct"         : 0.012,   # assumed spread as % of premium
    "spread_cap_pts"     : 8.0,     # absolute cap on assumed spread (pts)
    "slippage_min"       : 1.0,     # floor for exit-slippage modeling
    "exit_cooldown"      : 2.0,

    # Risk
    "max_trades_day"     : 999999,
    "post_sl_cooldown"   : 120,
    "max_daily_loss"     : 6000,
    "quantity"           : 30,

    # Signal persistence
    "persistence"        : 1,

    # Minimum 5-min bars before first signal.
    # Stochastic(5,3,3) needs k_period+d_smooth-1 = 7 bars minimum.
    # BB needs 7 bars, ATR needs 15 (falls back to sl_min if not ready),
    # Volume needs 11 (bypasses if not ready).
    # 10 bars = 50 min from 9:15 → first signal ~10:05 AM.
    "min_bars"           : 10,

    # Stochastic momentum filter -- replaces SuperTrend
    # Stochastic(k_period=5, k_smooth=3, d_smooth=3) on 5-min candles.
    # CE entry: K < stoch_ob (not overbought) AND K crosses above D
    # PE entry: K > stoch_os (not oversold)  AND K crosses below D
    "stoch_k_period"     : 5,
    "stoch_k_smooth"     : 3,
    "stoch_d_smooth"     : 3,
    "stoch_ob"           : 80,   # overbought ceiling for CE entries
    "stoch_os"           : 20,   # oversold floor for PE entries

    # Spot-based strike offsets for Bug 6 subscription
    "spot_atm_offsets"   : [0, 200, -200, 400, -400],

    # CSV paths
    "entry_csv"          : "logs/bb_stoch_entry.csv",
    "exit_csv"           : "logs/bb_stoch_exit.csv",
    "signal_csv"         : "logs/bb_stoch_signals.csv",
}


# ============================================================
# SPREAD / SLIPPAGE MODEL  (Bug 17/18 fix)
# ============================================================

def _estimate_spread(ltp: float) -> float:
    """
    Realistic bid-ask spread estimate for a near-ATM weekly/monthly index
    option, expressed in premium points.

    BUG FIX: the old model was a flat `ltp * 0.03` with no cap, compared
    against a TP that's capped in ABSOLUTE premium points regardless of
    premium size. BankNifty ATM premiums run ~700-1200 simply because of
    the index's absolute level and 100-pt strike spacing -- not because
    the option is illiquid. 3% of a 900 premium is ~27 pts, which is 2-3x
    wider than the real quoted spread on a liquid ATM BankNifty option
    (typically 1-4 pts in normal conditions). That mismatch made every
    genuine BankNifty setup fail the spread/TP gate. On Nifty (~130-150
    premium) the same 3% only ever produces ~4 pts, so it never binds --
    which is why this bug was invisible on BB_STOCH_NIFTY but fatal to
    BB_STOCH.

    Fix: model spread as a modest percentage of premium WITH an absolute
    cap. Real spreads on liquid near-ATM index options don't keep growing
    linearly with premium -- they're bounded by tick size and
    market-maker competition, not premium magnitude.
    """
    if ltp <= 0:
        return 0.0
    pct_component = ltp * CFG["spread_pct"]
    return round(min(pct_component, CFG["spread_cap_pts"]), 2)


# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    prev  = close.shift(1)
    tr    = pd.concat([
                high - low,
                (high - prev).abs(),
                (low  - prev).abs()
            ], axis=1).max(axis=1)
    atr   = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(round(atr.iloc[-1], 2))


def compute_bb(df: pd.DataFrame, period: int, nstd: float) -> dict:
    """
    Bollinger Bands on 'close'.
    Returns current + previous bar values for crossover detection.
    """
    if len(df) < 2:
        return {
            "mid": 0, "upper": 0, "lower": 0,
            "bw_pct": 0, "squeeze": True,
            "prev_mid": 0, "prev_upper": 0,
            "prev_lower": 0, "prev_close": 0,
        }
    close = df["close"].astype(float)
    min_p = max(2, period // 3)
    mid   = close.rolling(period, min_periods=min_p).mean()
    std   = close.rolling(period, min_periods=min_p).std().fillna(0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    m     = float(mid.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    bw    = (u - l) / m if m > 0 else 0
    return {
        "mid"        : round(m, 2),
        "upper"      : round(u, 2),
        "lower"      : round(l, 2),
        "bw_pct"     : round(bw, 5),
        "squeeze"    : bw < CFG["bb_squeeze_pct"],
        "prev_mid"   : round(float(mid.iloc[-2]),   2),
        "prev_upper" : round(float(upper.iloc[-2]), 2),
        "prev_lower" : round(float(lower.iloc[-2]), 2),
        "prev_close" : round(float(close.iloc[-2]), 2),
    }


def compute_vol_ratio(df: pd.DataFrame, avg_period: int) -> float:
    """
    Ratio of last bar's volume vs rolling N-bar average.
    Returns -1.0 sentinel when data unavailable -- caller bypasses filter.
    """
    if "volume" not in df.columns:
        return -1.0
    vol = df["volume"].astype(float)
    if vol.sum() == 0 or vol.isna().all():
        return -1.0
    if len(df) < avg_period + 1:
        return -1.0
    avg     = float(vol.iloc[-(avg_period + 1):-1].mean())
    current = float(vol.iloc[-1])
    if avg <= 0:
        return -1.0
    return round(current / avg, 3)


def compute_stochastic(df: pd.DataFrame,
                       k_period: int = 5,
                       k_smooth: int = 3,
                       d_smooth: int = 3) -> dict:
    """
    Stochastic oscillator (Slow Stochastic) on OHLC data.

    Algorithm:
      1. Raw %K[i] = 100 * (Close[i] - LowestLow[k_period])
                         / (HighestHigh[k_period] - LowestLow[k_period])
      2. Smoothed K = SMA(raw %K, k_smooth)
      3. Signal D   = SMA(K, d_smooth)

    Returns dict:
      k     : current smoothed %K value (0-100)
      d     : current %D signal line value (0-100)
      ready : False when insufficient data
    """
    min_bars = k_period + k_smooth + d_smooth - 2
    insufficient = {"k": 50.0, "d": 50.0, "ready": False}
    if len(df) < min_bars:
        return insufficient

    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom        = highest_high - lowest_low
    # Avoid division by zero on flat bars
    raw_k = pd.Series(
        np.where(denom > 0, 100.0 * (close - lowest_low) / denom, 50.0),
        index=close.index,
    )
    k_series = raw_k.rolling(k_smooth).mean()
    d_series = k_series.rolling(d_smooth).mean()

    k = float(round(k_series.iloc[-1], 2))
    d = float(round(d_series.iloc[-1], 2))

    return {
        "k"     : k,
        "d"     : d,
        "ready" : True,
    }



# ============================================================
# SIGNAL BUILDER
# ============================================================

def evaluate_signal(df: pd.DataFrame,
                    vwap: Optional[float]) -> dict:
    """
    Run all filters. Returns:
        action     : "BUY_CE" | "BUY_PE" | "HOLD"
        blocked_by : reason string if HOLD
        bb, vol_ratio, atr, stoch_k, stoch_d, close : snapshots for logging
    """
    empty = {"action": "HOLD", "blocked_by": "no_data",
             "bb": {}, "vol_ratio": 0, "atr": 0}

    if df is None or len(df) < CFG["min_bars"]:
        return {**empty, "blocked_by": "insufficient_bars"}

    # ---- Indicators ----
    bb        = compute_bb(df, CFG["bb_period"], CFG["bb_std"])
    vol_ratio = compute_vol_ratio(df, CFG["vol_avg_period"])
    atr       = _compute_atr(df, CFG["atr_period"])
    close     = float(df["close"].iloc[-1])

    # Stochastic momentum filter
    stoch      = compute_stochastic(df, CFG["stoch_k_period"],
                                    CFG["stoch_k_smooth"], CFG["stoch_d_smooth"])
    # Crossover: K crosses above D (bullish momentum) or below D (bearish)
    k_cross_up = stoch["k"] > stoch["d"]
    k_cross_dn = stoch["k"] < stoch["d"]

    base = {
        "bb": bb, "vol_ratio": vol_ratio, "atr": atr,
        "close": close, "stoch_k": stoch["k"], "stoch_d": stoch["d"],
    }

    # Gate 0: Stochastic not yet ready (insufficient bars)
    if not stoch["ready"]:
        return {"action": "HOLD", "blocked_by": "stoch_warmup", **base}

    # Gate 1: BB squeeze
    if bb["squeeze"]:
        return {"action": "HOLD", "blocked_by": "bb_squeeze", **base}

    # Gate 2: Volume
    if vol_ratio == -1.0:
        vol_ok = True   # data unavailable -- bypass
        log.debug("[BB_STOCH] Volume data unavailable -- vol filter bypassed")
    else:
        vol_ok = round(vol_ratio, 4) >= CFG["vol_mult"]

    if not vol_ok:
        return {"action": "HOLD", "blocked_by": "volume_low", **base}

    # Gate 3: VWAP bias
    if vwap and vwap > 0:
        above_vwap = close >= (vwap - CFG["vwap_buffer"])
        below_vwap = close <= (vwap + CFG["vwap_buffer"])
    else:
        above_vwap = True
        below_vwap = True

    # ================================================================
    # CE CONDITIONS
    #   Breakout : close crosses ABOVE upper band (this bar vs prev bar)
    #   Bounce   : close crosses back ABOVE lower band
    #   Middle   : close crosses ABOVE middle band
    # ================================================================
    bb_breakout_up  = close > bb["upper"] and bb["prev_close"] <= bb["prev_upper"]
    bb_bounce_up    = close > bb["lower"] and bb["prev_close"] <= bb["prev_lower"]
    bb_mid_cross_up = close > bb["mid"]   and bb["prev_close"] <= bb["prev_mid"]

    ce_bb_trigger = bb_breakout_up or bb_bounce_up or bb_mid_cross_up

    # Stochastic CE gate: K not overbought AND K just crossed above D
    stoch_ce_ok = stoch["k"] < CFG["stoch_ob"] and k_cross_up

    if ce_bb_trigger and above_vwap:
        if not stoch_ce_ok:
            reason = "stoch_ob" if stoch["k"] >= CFG["stoch_ob"] else "stoch_no_cross_ce"
            return {"action": "HOLD", "blocked_by": reason, **base}
        if bb_breakout_up:
            if bb["bw_pct"] < CFG["bb_min_bw_breakout"]:
                return {"action": "HOLD", "blocked_by": "bw_squeeze", **base}
            mode = "breakout"
        elif bb_bounce_up:
            mode = "bounce"
        else:
            if bb["bw_pct"] < CFG["bb_mid_squeeze_pct"]:
                return {"action": "HOLD", "blocked_by": "bb_mid_squeeze", **base}
            mode = "middle"
        return {"action": "BUY_CE", "blocked_by": "", **base, "mode": mode}

    # ================================================================
    # PE CONDITIONS
    #   Breakout : close crosses BELOW lower band
    #   Bounce   : close crosses back BELOW upper band
    #   Middle   : close crosses BELOW middle band
    # ================================================================
    bb_breakout_dn  = close < bb["lower"] and bb["prev_close"] >= bb["prev_lower"]
    bb_bounce_dn    = close < bb["upper"] and bb["prev_close"] >= bb["prev_upper"]
    bb_mid_cross_dn = close < bb["mid"]   and bb["prev_close"] >= bb["prev_mid"]

    pe_bb_trigger = bb_breakout_dn or bb_bounce_dn or bb_mid_cross_dn

    # Stochastic PE gate: K not oversold AND K just crossed below D
    stoch_pe_ok = stoch["k"] > CFG["stoch_os"] and k_cross_dn

    if pe_bb_trigger and below_vwap:
        if not stoch_pe_ok:
            reason = "stoch_os" if stoch["k"] <= CFG["stoch_os"] else "stoch_no_cross_pe"
            return {"action": "HOLD", "blocked_by": reason, **base}
        if bb_breakout_dn:
            if bb["bw_pct"] < CFG["bb_min_bw_breakout"]:
                return {"action": "HOLD", "blocked_by": "bw_squeeze", **base}
            mode = "breakout"
        elif bb_bounce_dn:
            mode = "bounce"
        else:
            if bb["bw_pct"] < CFG["bb_mid_squeeze_pct"]:
                return {"action": "HOLD", "blocked_by": "bb_mid_squeeze", **base}
            mode = "middle"
        return {"action": "BUY_PE", "blocked_by": "", **base, "mode": mode}

    # Granular block reasons for log analysis
    if ce_bb_trigger:
        if not above_vwap:
            return {"action": "HOLD", "blocked_by": "below_vwap", **base}
        if not stoch_ce_ok:
            reason = "stoch_ob" if stoch["k"] >= CFG["stoch_ob"] else "stoch_no_cross_ce"
            return {"action": "HOLD", "blocked_by": reason, **base}
    if pe_bb_trigger:
        if not below_vwap:
            return {"action": "HOLD", "blocked_by": "above_vwap", **base}
        if not stoch_pe_ok:
            reason = "stoch_os" if stoch["k"] <= CFG["stoch_os"] else "stoch_no_cross_pe"
            return {"action": "HOLD", "blocked_by": reason, **base}

    return {"action": "HOLD", "blocked_by": "no_setup", **base}



# ============================================================
# PAPER TRADE STATE
# ============================================================

class BBTrade:
    __slots__ = (
        "symbol", "token", "option_type", "qty",
        "entry", "sl", "target", "sl_pts", "tp_pts",
        "trail_stage", "spot", "atr", "timestamp",
        "exit_pending", "last_exit_ts", "order_id", "slip",
    )

    def __init__(self, symbol, token, opt_type, ltp, qty,
                 spot, sl_pts, tp_pts, atr, slip):
        # Bug 18 fix: `slip` is now computed by the caller (_execute_fill)
        # from _estimate_spread(ltp)/2, floored at CFG["slippage_min"] --
        # not a single flat constant reused across BankNifty and Nifty's
        # very different premium scales. Stored on the trade so
        # _update_trail() and _close_trade() apply the exact same value
        # this trade was priced with (keeps the Bug 9 breakeven guarantee
        # exact, since entry markup and exit markdown must match).
        self.slip        = slip
        self.symbol      = symbol
        self.token       = token
        self.option_type = opt_type
        self.qty         = qty
        self.entry       = round(ltp + slip, 2)
        # Bug 8 fix: anchor SL/TP to actual fill price
        self.sl          = round(self.entry - sl_pts, 2)
        self.target      = round(self.entry + tp_pts, 2)
        self.sl_pts      = sl_pts
        self.tp_pts      = tp_pts
        self.trail_stage = 0
        self.spot        = spot
        self.atr         = atr
        self.timestamp   = _now_ist().isoformat()
        self.exit_pending  = False
        self.last_exit_ts  = 0.0
        self.order_id      = ""   # Bug 11 fix


# ============================================================
# CSV HELPERS
# ============================================================

def _csv_append(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# STRATEGY
# ============================================================

class BBStochStrategy(BaseStrategy):
    """
    BankNifty options strategy: Bollinger Bands + Volume + Stochastic momentum filter.

    Stochastic(5, 3, 3) is computed live from the 5-min candle buffer.
    _buf_5m is seeded with the prior session's last N BankNifty candles in
    pre_market() so indicators are warm from the very first live candle.
    """

    LIVE_MODE = LIVE_MODE

    # No class-level INDEX_TOKEN attribute here, deliberately.
    # This strategy tracks the MAIN BankNifty index, the same one MarketHub
    # treats as primary. Strategies that track a secondary index (e.g.
    # BBStochNiftyStrategy → Nifty 50) set INDEX_TOKEN as a class attribute
    # so MarketHub routes them via _handle_extra_index_tick() instead of the
    # main broadcast. Setting INDEX_TOKEN here (even to BankNifty's own
    # token) would make MarketHub treat this strategy as secondary and skip
    # it from every BankNifty on_tick()/on_candle() call — see the
    # BANKNIFTY_INDEX_TOKEN module constant above for the historical bug
    # this caused. The module-level constant covers the one place this
    # strategy actually needs the raw token: the historical_data() seed call.

    @property
    def name(self) -> str:
        return "BB_STOCH"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # 5-min candle buffer -- size covers BB and Stochastic lookbacks + headroom
        buf_size = max(CFG["bb_period"], CFG["stoch_k_period"]) + 20
        self._buf_5m: deque = deque(maxlen=buf_size)

        # Trade state
        self._trade: Optional[BBTrade] = None
        self._active_token: Optional[int] = None
        self._lock = threading.Lock()

        # Risk state
        self._daily_pnl   : float = 0.0
        self._trades_today: int   = 0
        self._last_sl_time: float = 0.0
        self._is_halted   : bool  = False
        self._results     : list  = []
        self._blocked_log : dict  = {}

        # Signal persistence
        self._persist_buf: deque = deque(maxlen=CFG["persistence"])

        # Pre-market data
        self._instruments  = None
        self._expiry_date  = None
        self._dte          = 0
        self._session_date = None

        # Bug 10 fix
        self._squareoff_done: bool = False

        # Bug 4/5 fix
        self._pre_ce_token: Optional[int] = None
        self._pre_pe_token: Optional[int] = None

        # Bug 6 fix
        self._open_atm_subscribed: bool = False

        # Bug 7 fix
        self._pending_entry: Optional[dict] = None

        # Change B: entry delay (next-candle breakout confirmation)
        self._entry_delay_pending: Optional[dict] = None

        # Bug A fix: track every token subscribed as part of the ambient range
        # (pre_market + spot_atm) so we never call hub.subscribe() twice for the
        # same token from this strategy.  pre_market subscribes [ATM, ATM±200];
        # _subscribe_spot_atm later subscribes [ATM, ATM±200, ATM±400].
        # The overlapping tokens (ATM, ATM±200) would get refcount=2 without this
        # guard, causing the hub to never reach refcount=0 on those tokens from
        # BB_STOCH's side and silently blocking cross-strategy cleanup.
        self._subscribed_range: set[int] = set()

        log.info("[BB_STOCH] Strategy initialized")

    # ----------------------------------------------------------
    # PRE-MARKET
    # ----------------------------------------------------------

    def pre_market(self, premarket_data, instruments) -> bool:
        self._instruments  = instruments
        self._expiry_date  = premarket_data.expiry_date
        self._dte          = premarket_data.dte_days
        self._session_date = _now_ist().date()
        self._reset_day()

        # Bug 14 fix: seed _buf_5m with the prior session's BankNifty candles
        # so BB and Stochastic are already warm at 9:15 AM.
        self._seed_historical_buffer()

        log.info(
            f"[BB_STOCH] Pre-market | VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr} Expiry={self._expiry_date} DTE={self._dte}"
        )

        # ── PRE-SUBSCRIBE STRIKES ─────────────────────────────────────────────
        # Bug 4+5 fix: warm up ATM and ATM+-200 tokens before signal fires.
        ref_price = premarket_data.prev_close or premarket_data.prev_last5m_close
        if ref_price and self._expiry_date:
            atm = get_atm_strike(ref_price)
            for offset in [0, 200, -200]:
                for opt_type in ("CE", "PE"):
                    strike = atm + offset
                    if offset == 0 and self._dte == 0:
                        strike = atm - 100 if opt_type == "CE" else atm + 100
                    tok, sym = instruments.get_option_token(strike, opt_type, self._expiry_date)
                    if tok:
                        # Bug A fix: guard with _subscribed_range to prevent
                        # double-subscribe when _subscribe_spot_atm() later
                        # covers the same [ATM, ATM±200] offsets.
                        if tok not in self._subscribed_range:
                            self.subscribe_option(tok)
                            self._subscribed_range.add(tok)
                        if offset == 0:
                            if opt_type == "CE":
                                self._pre_ce_token = tok
                            else:
                                self._pre_pe_token = tok
                        log.info(
                            f"[BB_STOCH] Pre-subscribed {sym} ({tok})"
                            f"{' [ATM]' if offset == 0 else f' [ATM{offset:+d}]'}"
                        )
        else:
            log.warning("[BB_STOCH] No ref price for pre-subscription")

        return True

    def _reset_day(self):
        self._daily_pnl           = 0.0
        self._trades_today        = 0
        self._last_sl_time        = 0.0
        self._is_halted           = False
        self._results.clear()
        self._blocked_log.clear()
        self._persist_buf.clear()
        self._squareoff_done      = False
        self._open_atm_subscribed = False
        self._pending_entry       = None
        self._entry_delay_pending = None
        # Bug A fix: clear range-subscription tracking on each new session
        self._subscribed_range.clear()

    # ----------------------------------------------------------
    # Bug 14: Historical buffer seeding
    # ----------------------------------------------------------

    def _seed_historical_buffer(self):
        """
        Fetch the prior trading session's last N BankNifty 5-min candles via
        hub.kite.historical_data() and load them into _buf_5m, so that BB(20)
        and Stochastic(5,3,3) already have enough bars to be valid the moment
        the market opens -- instead of needing min_bars=10 (~50 live minutes
        from 9:15) to accumulate from empty every day.

        This is separate from hub.backfill(), which only replays TODAY's
        already-elapsed candles for late starts. This seeds yesterday's
        close-of-session data so today's first candle isn't starting cold.

        There is a genuine overnight gap between the last seeded bar and
        the first live 9:15 bar -- BB and Stochastic self-correct within
        a few live bars, which is an accepted tradeoff.

        Fails safe: if kite is unavailable, the fetch errors, or no data
        comes back, this silently falls back to live-only warmup (no
        regression -- the strategy just trades ~50 min later that day,
        exactly as it did before this fix).
        """
        kite = getattr(self._hub, "kite", None)
        if kite is None:
            log.warning(
                "[BB_STOCH] No kite handle on hub -- cannot seed "
                "historical candles, falling back to live-only warmup "
                f"(~{CFG['min_bars'] * 5}min)"
            )
            return

        needed = self._buf_5m.maxlen or (CFG["bb_period"] + CFG["stoch_k_period"] + 10)
        today  = _now_ist().date()

        try:
            raw = kite.historical_data(
                instrument_token = BANKNIFTY_INDEX_TOKEN,
                from_date        = datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                to_date          = datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                interval         = "5minute",
            )
        except Exception as e:
            log.warning(
                f"[BB_STOCH] Historical seed fetch failed: {e} -- "
                f"falling back to live-only warmup (~{CFG['min_bars'] * 5}min)"
            )
            return

        if not raw:
            log.warning(
                "[BB_STOCH] Historical seed returned no candles -- "
                f"falling back to live-only warmup (~{CFG['min_bars'] * 5}min)"
            )
            return

        tail = raw[-needed:]
        with self._lock:
            self._buf_5m.clear()
            for bar in tail:
                try:
                    self._buf_5m.append({
                        "ts"     : bar["date"],
                        "open"   : float(bar["open"]),
                        "high"   : float(bar["high"]),
                        "low"    : float(bar["low"]),
                        "close"  : float(bar["close"]),
                        "volume" : float(bar.get("volume", 0)),
                    })
                except (KeyError, TypeError, ValueError) as e:
                    log.warning(f"[BB_STOCH] Skipping malformed seed bar {bar}: {e}")

        if self._buf_5m:
            log.info(
                f"[BB_STOCH] Seeded {len(self._buf_5m)} prior-session 5-min "
                f"candles ({tail[0].get('date', '?')} -> {tail[-1].get('date', '?')}) "
                f"-- BB/Stochastic warm from first live tick, "
                f"no {CFG['min_bars'] * 5}min wait"
            )
        else:
            log.warning(
                "[BB_STOCH] Seed buffer empty after load -- "
                "falling back to live-only warmup"
            )

    # ----------------------------------------------------------
    # Bug 6: Dynamic spot-based subscription on first candle
    # ----------------------------------------------------------

    def _subscribe_spot_atm(self, spot: float):
        """
        Subscribe ATM+-400 based on actual live spot price.
        Called on first 5-min candle (~9:20 AM) to cover large gap days
        where pre_market()'s prev_close-based subscriptions are off-market.
        """
        if not spot or spot <= 0:
            return
        if not self._expiry_date or not self._instruments:
            return

        atm        = get_atm_strike(spot)
        subscribed = 0
        for offset in CFG["spot_atm_offsets"]:
            for opt_type in ("CE", "PE"):
                tok, sym = self._instruments.get_option_token(
                    atm + offset, opt_type, self._expiry_date
                )
                if tok:
                    # Bug A fix: skip if already subscribed by pre_market()
                    # (offsets [0, ±200] overlap between pre_market and spot_atm).
                    if tok not in self._subscribed_range:
                        self.subscribe_option(tok)
                        self._subscribed_range.add(tok)
                        subscribed += 1
                        log.info(
                            f"[BB_STOCH] Spot-sub {sym} ({tok}) "
                            f"[spot-ATM{offset:+d}]"
                        )
                    else:
                        log.debug(
                            f"[BB_STOCH] Spot-sub skip (already held) {sym} ({tok}) "
                            f"[spot-ATM{offset:+d}]"
                        )
        log.info(
            f"[BB_STOCH] Spot subscription done: "
            f"{subscribed} new tokens around ATM={atm} (spot={spot:.0f}); "
            f"{len(self._subscribed_range)} total range tokens held"
        )

    # ----------------------------------------------------------
    # TICK CALLBACKS
    # ----------------------------------------------------------

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Index price tick.
        Bug 10 fix: _force_close fires exactly once via _squareoff_done.
        Bug D  fix: subscribe spot ATM range on the very first BankNifty tick
                    at or after 9:15, not waiting for the first 5-min candle
                    to close at ~9:20.  SPIKE may unsubscribe a shared BankNifty
                    ATM option at 9:15 (after entering its CE/PE leg); subscribing
                    the full ATM range immediately keeps those tokens live.
        Change B: confirms or expires the entry delay trigger level.
        """
        # Bug 10 fix
        if ts.time() >= dtime(*CFG["auto_squareoff"]) and not self._squareoff_done:
            self._squareoff_done = True
            self._force_close("AUTO-SQUAREOFF")
            return

        # Bug D fix: subscribe spot range on first tick at/after market open.
        # _open_atm_subscribed is also checked in on_candle() — whichever fires
        # first wins; the flag ensures only one call goes through.
        if not self._open_atm_subscribed and ts.time() >= dtime(9, 15):
            if price and price > 0:
                self._subscribe_spot_atm(price)
                self._open_atm_subscribed = True

        # Change B: entry delay confirmation
        confirm = self._entry_delay_pending
        if confirm and not self._trade and not self._is_halted:
            now_time = ts.time()
            action   = confirm["action"]
            trigger  = confirm["trigger_level"]

            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(
                    f"[BB_STOCH] Entry delay expired (past cutoff) "
                    f"| {action} trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
            elif action == "BUY_CE" and price > trigger:
                log.info(
                    f"[BB_STOCH] Entry delay CONFIRMED | BUY_CE "
                    f"| index={price:.2f} > trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])
            elif action == "BUY_PE" and price < trigger:
                log.info(
                    f"[BB_STOCH] Entry delay CONFIRMED | BUY_PE "
                    f"| index={price:.2f} < trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])

    def on_candle(self, candle: dict, ts: datetime):
        """
        Called by MarketHub when a 5-min index candle closes.
        Bug 6 fix: on first candle subscribe ATM+-400 from actual spot.
        Change B: expire unconfirmed entry delay (fake breakout).
        """
        with self._lock:
            self._buf_5m.append(candle)

        # Bug 6 / Bug D fix: spot-based subscription on first candle of session.
        # In normal conditions on_tick() has already done this at 9:15.
        # This path is a safety net for backfill / test scenarios where on_tick
        # is not called before the first candle arrives.
        if not self._open_atm_subscribed:
            spot = candle.get("close", 0)
            if spot and spot > 0:
                self._subscribe_spot_atm(spot)
                self._open_atm_subscribed = True

        # Change B: expire any entry delay that survived a full candle
        if self._entry_delay_pending:
            edp = self._entry_delay_pending
            log.info(
                f"[BB_STOCH] Entry delay expired (no index confirmation "
                f"in previous candle) | {edp['action']} trigger={edp['trigger_level']:.2f}"
            )
            self._entry_delay_pending = None

        if self._trade:
            return

        self._evaluate_entry(ts)

    def on_option_tick(self, token: int, price: float, ts: datetime,
                       tick_ts: datetime = None):
        """
        Live option price tick.
        Bug 7 fix: fills pending entry on first tick of subscribed token.
        """
        # Bug 7: pending entry fill
        pending = self._pending_entry
        if pending is not None and token == pending["token"] and not self._trade:
            now_time = ts.time()
            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(f"[BB_STOCH] Pending entry expired (past cutoff) for "
                         f"{pending['symbol']} -- discarding")
                self._pending_entry = None
            elif self._is_halted or self._trades_today >= CFG["max_trades_day"]:
                log.info(f"[BB_STOCH] Pending entry cancelled (risk gate) for "
                         f"{pending['symbol']}")
                self._pending_entry = None
            elif price and price >= 5:
                self._fill_pending(pending, price, ts)
            return

        if token != self._active_token or not self._trade:
            return
        self._manage_trade(price, ts)

    # ----------------------------------------------------------
    # SIGNAL EVALUATION
    # ----------------------------------------------------------

    def _evaluate_entry(self, ts: datetime):
        now_time = ts.time()

        if now_time < dtime(*CFG["session_start"]):
            return
        if now_time >= dtime(*CFG["session_cutoff"]):
            return
        if self._is_halted:
            return
        if self._trades_today >= CFG["max_trades_day"]:
            return
        elapsed = time_module.time() - self._last_sl_time
        if self._last_sl_time > 0 and elapsed < CFG["post_sl_cooldown"]:
            log.info(f"[BB_STOCH] Post-SL cooldown: {CFG['post_sl_cooldown'] - elapsed:.0f}s remaining")
            return

        with self._lock:
            candles = list(self._buf_5m)
        df = self._to_df(candles)
        if df.empty:
            return

        n_bars = len(df)
        if n_bars < CFG["min_bars"]:
            log.debug(
                f"[BB_STOCH] Warming up: {n_bars}/{CFG['min_bars']} bars "
                f"(ready in ~{(CFG['min_bars'] - n_bars) * 5} mins)"
            )
            return

        vwap = self._hub.session_vwap.value

        sig    = evaluate_signal(df, vwap)
        action = sig["action"]

        self._log_signal(ts, sig)

        # Persistence check
        self._persist_buf.append(action)
        confirmed = (
            len(self._persist_buf) >= CFG["persistence"]
            and all(a == action and a != "HOLD" for a in self._persist_buf)
        )

        bb            = sig.get("bb", {})
        vwap_str      = f"{vwap:.2f}" if vwap is not None else "N/A"
        vol_ratio_val = sig.get("vol_ratio", -1)
        vol_str       = f"{vol_ratio_val:.2f}x" if vol_ratio_val >= 0 else "N/A(no-data)"
        stoch_k_val   = sig.get("stoch_k", 50.0)
        stoch_d_val   = sig.get("stoch_d", 50.0)
        stoch_str     = f"K={stoch_k_val:.1f} D={stoch_d_val:.1f}"

        log.info(
            f"[BB_STOCH] {action:8s} | "
            f"Close={sig.get('close', 0):.2f} | "
            f"BB=[{bb.get('lower', 0):.1f}~{bb.get('upper', 0):.1f}] "
            f"BW={bb.get('bw_pct', 0)*100:.2f}% | "
            f"Vol={vol_str} | VWAP={vwap_str} | "
            f"Stoch={stoch_str} | "
            f"Block={sig.get('blocked_by') or 'none'} | Persist={confirmed} | "
            f"PnL={self._daily_pnl:+.1f}Rs T={self._trades_today}/{CFG['max_trades_day']}"
        )

        if not confirmed or action == "HOLD":
            if action == "HOLD":
                bl = sig.get("blocked_by", "")
                self._blocked_log[bl] = self._blocked_log.get(bl, 0) + 1
            return

        # Change B: set trigger level, wait for next-candle index confirmation.
        # on_tick() fires entry the moment index price crosses signal-candle's
        # high (CE) or low (PE). Full next candle without confirmation = expired.
        last_candle   = candles[-1] if candles else {}
        trigger_level = (
            float(last_candle.get("high", 0)) if action == "BUY_CE"
            else float(last_candle.get("low", 0))
        )

        if trigger_level <= 0:
            log.warning(
                f"[BB_STOCH] Entry delay: trigger level unavailable for {action} "
                f"-- entering immediately"
            )
            self._enter_trade(action, sig, ts)
            return

        log.info(
            f"[BB_STOCH] Entry delay SET | {action} | "
            f"trigger={'>' if action == 'BUY_CE' else '<'}{trigger_level:.2f} | "
            f"waiting for next-candle index break via on_tick()"
        )
        self._entry_delay_pending = {
            "action"        : action,
            "trigger_level" : trigger_level,
            "sig"           : sig,
            "ts"            : ts,
        }

    # ----------------------------------------------------------
    # ENTRY
    # ----------------------------------------------------------

    def _enter_trade(self, action: str, sig: dict, ts: datetime):
        """
        Open a new options position.
        Bug 1 fix: no sleep.
        Bug 2 fix: persist_buf cleared only on genuine fill.
        Bug 7 fix: on LTP miss store _pending_entry for fill on first tick.
        """
        opt    = "CE" if action == "BUY_CE" else "PE"
        spot   = sig.get("close", 0)
        atr    = sig.get("atr", 0)
        expiry = self._expiry_date

        if expiry is None:
            log.warning("[BB_STOCH] No expiry date -- skipping entry")
            return

        atm = get_atm_strike(spot)
        if self._dte == 0:
            atm = atm - 100 if opt == "CE" else atm + 100

        token, tsym = self._instruments.get_option_token(atm, opt, expiry)
        if not token:
            log.warning(f"[BB_STOCH] Option not found: ATM={atm} {opt} expiry={expiry}")
            return

        self.subscribe_option(token)
        ltp      = self.get_price(token)
        price_ts = self.get_price_ts(token)

        # Bug 12 fix: treat cached price as unavailable when it is stale.
        # Another strategy may have unsubscribed this token earlier, freezing
        # the hub's cached price while the market moved.  If we use that stale
        # price for entry, the first real tick (Zerodha's re-subscription
        # snapshot) arrives in the same _on_ticks batch and instantly triggers
        # SL or TP — producing a phantom fill.
        # 30 s threshold: anything older than one WebSocket heartbeat interval
        # is unreliable.
        price_age   = (_now_ist() - price_ts).total_seconds() if price_ts else None
        price_stale = (price_ts is None) or (price_age is not None and price_age > 30)

        if not ltp or ltp < 5 or price_stale:
            if price_stale and ltp and ltp >= 5:
                log.warning(
                    f"[BB_STOCH] LTP stale for {tsym} (token={token}, "
                    f"age={price_age:.0f}s) -- storing pending entry, "
                    f"will fill on first live tick"
                )
            else:
                log.warning(
                    f"[BB_STOCH] LTP unavailable for {tsym} (token={token}) -- "
                    f"storing pending entry, will fill on first tick"
                )
            self._pending_entry = {
                "action" : action,
                "opt"    : opt,
                "sig"    : sig,
                "token"  : token,
                "symbol" : tsym,
                "ts"     : ts,
                "atm"    : atm,
            }
            return

        self._execute_fill(opt, sig, token, tsym, ltp, ts)

    def _execute_fill(self, opt: str, sig: dict, token: int,
                      tsym: str, ltp: float, ts: datetime,
                      pending_delay_ms: Optional[float] = None):
        """Shared fill logic for _enter_trade() and _fill_pending()."""
        atr  = sig.get("atr", 0)
        spot = sig.get("close", 0)

        sl_pts, tp_pts = self._compute_sl_tp(ltp, atr)

        # Bug 17 fix: was `ltp * 0.03` uncapped -- scaled linearly with
        # premium while tp_pts is capped in absolute points, so every
        # high-premium BankNifty setup failed this gate. See
        # _estimate_spread() for the full explanation.
        est_spread = _estimate_spread(ltp)
        if tp_pts > 0 and (est_spread / tp_pts) > 0.40:
            log.warning(
                f"[BB_STOCH] Spread {est_spread:.1f} too large vs TP {tp_pts:.1f} -- skipping"
            )
            self.unsubscribe_option(token)
            return

        if not self._acquire_slot():
            log.warning("[BB_STOCH] Trade slot blocked -- another live strategy has a position")
            self.unsubscribe_option(token)
            return

        buy_result = self._place_buy(tsym, token, CFG["quantity"], ltp)
        if LIVE_MODE and buy_result is None:
            self._release_slot()
            log.error(f"[BB_STOCH] BUY order FAILED for {tsym} -- entry aborted")
            self.unsubscribe_option(token)
            return

        # Bug 13 fix: _place_buy returns (order_id_str, fill_price).
        # Unpack and use the actual exchange fill price in live mode so that
        # SL/TP are anchored to the real fill, not the ref ltp.
        # Paper mode: fill_price == ltp (paper fill is always at ref ltp).
        order_id_str: Optional[str]
        if buy_result is not None:
            order_id_str, fill_price = buy_result
        else:
            order_id_str, fill_price = None, ltp   # paper-only safety fallback

        # In live mode use actual exchange fill; paper fill already equals ltp.
        entry_ltp = fill_price if LIVE_MODE else ltp

        # Recompute sl/tp from actual fill (live fill may differ from ref ltp)
        if LIVE_MODE and abs(entry_ltp - ltp) > 0.5:
            sl_pts, tp_pts = self._compute_sl_tp(entry_ltp, atr)

        # Bug 18 fix: per-trade half-spread, sized to what was actually
        # traded (not a flat constant shared across BankNifty/Nifty).
        slip = max(CFG["slippage_min"], _estimate_spread(entry_ltp) / 2)

        trade = BBTrade(
            symbol   = tsym,
            token    = token,
            opt_type = opt,
            ltp      = entry_ltp,
            qty      = CFG["quantity"],
            spot     = spot,
            sl_pts   = sl_pts,
            tp_pts   = tp_pts,
            atr      = atr,
            slip     = slip,
        )
        trade.order_id = order_id_str

        with self._lock:
            self._trade        = trade
            self._active_token = token

        self._trades_today += 1
        self._persist_buf.clear()   # Bug 2 fix

        rr        = round(tp_pts / sl_pts, 2) if sl_pts else 0
        mode_tag  = "LIVE" if LIVE_MODE else "PAPER"
        delay_str = (f" [pending_delay={pending_delay_ms:.0f}ms]"
                     if pending_delay_ms is not None else "")

        log.info(
            f"[BB_STOCH] [{mode_tag}] ENTRY {opt} | {tsym} | "
            f"Fill={trade.entry:.2f} LTP={ltp:.2f}{delay_str} | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) TP={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} ATR={atr:.1f} | buy_order={order_id_str} | "
            f"Mode={sig.get('mode', '?')} BB-BW={sig['bb'].get('bw_pct', 0)*100:.2f}%"
        )

        bb = sig.get("bb", {})
        _csv_append(CFG["entry_csv"], {
            "timestamp"        : trade.timestamp,
            "symbol"           : tsym,
            "opt_type"         : opt,
            "qty"              : CFG["quantity"],
            "ltp"              : ltp,
            "fill"             : trade.entry,
            "sl"               : trade.sl,
            "target"           : trade.target,
            "sl_pts"           : sl_pts,
            "tp_pts"           : tp_pts,
            "rr"               : rr,
            "spot"             : spot,
            "atr"              : atr,
            "exec_mode"        : mode_tag,
            "order_id"         : order_id_str,
            "mode"             : sig.get("mode", "pending" if pending_delay_ms else ""),
            "bb_upper"         : bb.get("upper", ""),
            "bb_mid"           : bb.get("mid", ""),
            "bb_lower"         : bb.get("lower", ""),
            "bb_bw_pct"        : bb.get("bw_pct", ""),
            "stoch_k"          : sig.get("stoch_k", ""),
            "stoch_d"          : sig.get("stoch_d", ""),
            "vol_ratio"        : sig.get("vol_ratio", ""),
            "vwap"             : (self._hub.session_vwap.value
                                  if self._hub.session_vwap.value is not None else "N/A"),
            "pending_delay_ms" : (round(pending_delay_ms, 0)
                                  if pending_delay_ms is not None else ""),
        })

    # ----------------------------------------------------------
    # Bug 7: Pending entry fill on first WebSocket tick
    # ----------------------------------------------------------

    def _fill_pending(self, pending: dict, ltp: float, ts: datetime):
        """Fill a pending entry on its first live tick. No blocking."""
        self._pending_entry = None

        if ltp < 5:
            log.warning(
                f"[BB_STOCH] Pending fill: LTP {ltp:.2f} too low for "
                f"{pending['symbol']} -- discarding"
            )
            return

        now_time = ts.time()
        if now_time >= dtime(*CFG["session_cutoff"]):
            log.info(f"[BB_STOCH] Pending fill aborted -- past session cutoff ({pending['symbol']})")
            return
        if self._is_halted:
            log.info(f"[BB_STOCH] Pending fill aborted -- strategy halted ({pending['symbol']})")
            return
        if self._trades_today >= CFG["max_trades_day"]:
            log.info(f"[BB_STOCH] Pending fill aborted -- max trades reached ({pending['symbol']})")
            return
        if self._trade:
            log.info(f"[BB_STOCH] Pending fill aborted -- trade already open ({pending['symbol']})")
            return

        try:
            delay_ms = (ts - pending["ts"]).total_seconds() * 1000
        except Exception:
            delay_ms = None

        log.info(
            f"[BB_STOCH] PENDING FILL | {pending['symbol']} "
            f"token={pending['token']} LTP={ltp:.2f}"
            + (f" delay={delay_ms:.0f}ms" if delay_ms is not None else "")
        )
        self._execute_fill(
            opt              = pending["opt"],
            sig              = pending["sig"],
            token            = pending["token"],
            tsym             = pending["symbol"],
            ltp              = ltp,
            ts               = ts,
            pending_delay_ms = delay_ms,
        )

    # ----------------------------------------------------------
    # TRADE MANAGEMENT
    # ----------------------------------------------------------

    def _manage_trade(self, ltp: float, ts: datetime):
        trade = self._trade
        if not trade:
            return
        now_ts = time_module.time()
        if trade.exit_pending and (now_ts - trade.last_exit_ts) < CFG["exit_cooldown"]:
            return
        pnl_pts = ltp - trade.entry
        self._update_trail(trade, ltp, pnl_pts)
        if ltp >= trade.target:
            self._close_trade(ltp, "TARGET")
        elif ltp <= trade.sl:
            self._close_trade(ltp, "SL")

    def _update_trail(self, trade: BBTrade, ltp: float, pnl_pts: float):
        """
        Bug 9 fix: trail SL to entry + slippage so exit at trail SL
        yields net P&L = 0 (true breakeven), not -slippage.
        """
        if pnl_pts >= CFG["trail_arm"] and trade.trail_stage == 0:
            # Bug 18 fix: use this trade's own stored slip (matches what
            # was charged on entry / will be charged on exit) instead of
            # the flat CFG["slippage"] constant, so breakeven is exact.
            new_sl = round(trade.entry + trade.slip, 2)
            if new_sl > trade.sl:
                trade.sl          = new_sl
                trade.trail_stage = 1
                log.info(f"[BB_STOCH] Trail -> BE | SL={trade.sl:.2f}")
        elif pnl_pts > CFG["trail_arm"] and trade.trail_stage >= 1:
            new_stage = int((pnl_pts - CFG["trail_arm"]) / CFG["trail_step"]) + 1
            if new_stage > trade.trail_stage:
                inc               = (new_stage - trade.trail_stage) * CFG["trail_step"]
                trade.sl          = round(trade.sl + inc, 2)
                trade.trail_stage = new_stage
                log.info(f"[BB_STOCH] Trail stage {new_stage} | SL={trade.sl:.2f}")

    def _close_trade(self, ltp: float, reason: str):
        """Bug 3 fix: atomically claim self._trade at the top."""
        with self._lock:
            trade = self._trade
            if not trade:
                return
            self._trade = None

        trade.exit_pending = True
        trade.last_exit_ts = time_module.time()

        # Bug 18 fix: use this trade's own stored slip, not a flat
        # constant borrowed from a different instrument's premium scale.
        slip    = trade.slip
        exit_px = round(ltp - slip, 2)
        pnl_pts = round(exit_px - trade.entry, 2)
        pnl_rs  = round(pnl_pts * trade.qty, 2)
        rr_act  = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl += pnl_rs
        if reason == "SL":
            self._last_sl_time = time_module.time()

        # Use retry-aware sell: checks position status before each retry so we
        # never fire a phantom sell against an already-closed position, and never
        # silently drop a real open position after a transient exchange rejection.
        sell_order_id = self._place_sell_with_retry(
            trade.symbol, trade.token, trade.qty, ltp,
            max_retries=3,
        )
        # Release slot AFTER retry loop so no other strategy can enter while we
        # are still trying to close this position.
        self._release_slot()

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        tag      = "[TARGET]" if reason == "TARGET" else ("[SL]" if reason == "SL" else "[EXIT]")
        log.info(
            f"[BB_STOCH] [{mode_tag}] {tag} {trade.option_type} | {trade.symbol} | {reason} | "
            f"RawTick={ltp:.2f} Slip={slip:.2f} Exit={exit_px:.2f} | "
            f"PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}Rs) | "
            f"RR={rr_act:+.2f} | DayPnL={self._daily_pnl:+.2f}Rs | "
            f"sell_order={sell_order_id}"
        )

        result = {
            "timestamp"   : _now_ist().isoformat(),
            "symbol"      : trade.symbol,
            "opt_type"    : trade.option_type,
            "qty"         : trade.qty,
            "entry"       : trade.entry,
            "exit"        : exit_px,
            "raw_tick"    : ltp,
            "slip_pts"    : slip,
            "pnl_pts"     : pnl_pts,
            "pnl_rs"      : pnl_rs,
            "rr_actual"   : rr_act,
            "sl_pts"      : trade.sl_pts,
            "tp_pts"      : trade.tp_pts,
            "reason"      : reason,
            "trail_stage" : trade.trail_stage,
            "day_pnl_rs"  : round(self._daily_pnl, 2),
            "exec_mode"   : mode_tag,
            "sell_order"  : sell_order_id,
        }
        _csv_append(CFG["exit_csv"], result)
        self._results.append(result)

        if self._daily_pnl <= -abs(CFG["max_daily_loss"]):
            self._is_halted = True
            log.warning(
                f"[BB_STOCH] HALTED -- max daily loss breached "
                f"({self._daily_pnl:.0f}Rs)"
            )

        with self._lock:
            self._unsubscribe_active()

    def _force_close(self, reason: str):
        if self._pending_entry:
            log.info(
                f"[BB_STOCH] Discarding pending entry for "
                f"{self._pending_entry.get('symbol', '?')} on {reason}"
            )
            self._pending_entry = None
        if self._entry_delay_pending:
            log.info(
                f"[BB_STOCH] Discarding entry delay "
                f"({self._entry_delay_pending.get('action', '?')}) on {reason}"
            )
            self._entry_delay_pending = None
        with self._lock:
            trade = self._trade
        if not trade:
            return
        ltp = self.get_price(trade.token) or trade.entry
        self._close_trade(ltp, reason)

    def _unsubscribe_active(self):
        """Must be called with self._lock held or after trade cleared."""
        if self._active_token:
            self.unsubscribe_option(self._active_token)
            log.info(f"[BB_STOCH] Unsubscribed token {self._active_token}")
            self._active_token = None

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _compute_sl_tp(self, ltp: float, atr: float) -> Tuple[float, float]:
        # BUG FIX: `atr` is raw 5-min BankNifty INDEX-point ATR, not option
        # premium-point ATR. Scale it down before using it as a premium SL/TP
        # distance -- see premium_atr_scale comment in CFG above.
        premium_atr = atr * CFG["premium_atr_scale"] if atr > 0 else 0.0
        sl_pts = float(premium_atr * CFG["atr_sl_mult"]) if premium_atr > 0 else CFG["sl_min"]
        tp_pts = float(premium_atr * CFG["atr_tp_mult"]) if premium_atr > 0 else CFG["tp_min"]
        sl_pts = max(CFG["sl_min"], min(sl_pts, CFG["sl_max"]))
        tp_pts = max(CFG["tp_min"], min(tp_pts, CFG["tp_max"]))
        if self._dte == 0:
            tp_pts = max(CFG["tp_min"], tp_pts * 0.75)
        rr = round(tp_pts / sl_pts, 2) if sl_pts else 0
        log.info(
            f"[BB_STOCH] SL/TP | ATR={atr:.1f} PremiumATR={premium_atr:.1f} "
            f"SL_pts={sl_pts:.1f} TP_pts={tp_pts:.1f} RR=1:{rr}"
        )
        return sl_pts, tp_pts

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).reset_index(drop=True)

    def _log_signal(self, ts: datetime, sig: dict):
        bb = sig.get("bb", {})
        _csv_append(CFG["signal_csv"], {
            "timestamp"    : ts.isoformat(),
            "action"       : sig.get("action", "HOLD"),
            "mode"         : sig.get("mode", ""),
            "blocked_by"   : sig.get("blocked_by", ""),
            "close"        : sig.get("close", ""),
            "bb_upper"     : bb.get("upper", ""),
            "bb_mid"       : bb.get("mid", ""),
            "bb_lower"     : bb.get("lower", ""),
            "bb_bw_pct"    : bb.get("bw_pct", ""),
            "bb_squeeze"   : bb.get("squeeze", ""),
            "stoch_k"      : sig.get("stoch_k", ""),
            "stoch_d"      : sig.get("stoch_d", ""),
            "vol_ratio"    : sig.get("vol_ratio", ""),
            "atr"          : sig.get("atr", ""),
            "vwap"         : (self._hub.session_vwap.value
                              if self._hub.session_vwap.value is not None else "N/A"),
        })

    # ----------------------------------------------------------
    # EOD
    # ----------------------------------------------------------

    def eod_summary(self):
        self._force_close("EOD-SQUAREOFF")

        # Bug A / Bug E fix: release all range-subscribed tokens (pre_market +
        # spot_atm).  _unsubscribe_active() inside _force_close already handled
        # the active trade token; the ambient range tokens are cleaned up here.
        # This correctly handles the case where a range token == active token:
        # _unsubscribe_active() decremented its refcount by 1 (from the trade
        # subscribe in _enter_trade); this loop decrements it by 1 more (from
        # the pre_market/spot_atm subscribe).  Net = 0 from BB_STOCH's side.
        for tok in set(self._subscribed_range):
            self.unsubscribe_option(tok)
        n = len(self._subscribed_range)
        self._subscribed_range.clear()
        if n:
            log.info(f"[BB_STOCH] EOD range cleanup: released {n} ambient tokens")

        results  = self._results
        total    = len(results)
        wins     = [r for r in results if r["pnl_pts"] > 0]
        losses   = [r for r in results if r["pnl_pts"] <= 0]
        win_pct  = len(wins) / total if total else 0
        avg_win  = sum(r["pnl_pts"] for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0
        exp      = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        log.info(
            f"[BB_STOCH] EOD | "
            f"Trades={total} W={len(wins)} L={len(losses)} "
            f"WinRate={win_pct*100:.1f}% Expect={exp:+.3f}pts "
            f"DayPnL={self._daily_pnl:+.2f}Rs | "
            f"5m-bars={len(self._buf_5m)} | "
            f"Stoch({CFG['stoch_k_period']},{CFG['stoch_k_smooth']},{CFG['stoch_d_smooth']})-momentum-filter=ACTIVE"
        )
        if self._blocked_log:
            top = sorted(self._blocked_log.items(), key=lambda x: -x[1])[:5]
            log.info(f"[BB_STOCH] Top blocks today: {top}")
