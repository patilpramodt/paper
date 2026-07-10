"""
strategies/bb_stoch_nifty_strategy.py
-------------------------------------------------------------
BBStochNiftyStrategy -- Nifty 50 options scalper using:
  * Bollinger Bands  (BB)       -- identifies breakout / volatility expansion
  * Volume Filter               -- confirms genuine institutional participation
  * Stochastic momentum filter  -- confirms momentum direction at band touch
                                   Stochastic(5, 3, 3) on 5-min candle buffer.

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

NIFTY-SPECIFIC DETAILS:
--------------------------------------------------------------
  Index          : NSE:NIFTY 50
  INDEX_TOKEN    : 256265  (fixed Zerodha instrument token)
  Expiry         : Weekly  (Nifty 50 retained weekly expiry after SEBI Oct-2024)
  Strike step    : 50 pts  (BankNifty uses 100)
  Lot size       : 65      (verify before going live -- SEBI revises periodically)
  DTE==0 adjust  : ±50 pts (one Nifty strike ITM on expiry day)
  Pre-sub range  : ATM ± 100 pts (2 Nifty strikes each side)
  Spot-sub range : ATM ± 200 pts (4 Nifty strikes each side)

INTERNAL CANDLE + VWAP ARCHITECTURE:
--------------------------------------------------------------
  MarketHub only builds 5-min candles and VWAP from the BankNifty index
  (token 260105). Nifty strategies CANNOT use hub.index_candles or
  hub.session_vwap — those reflect BankNifty prices.

  This strategy therefore maintains:
    self._nifty_candles : CandleBuilder(minutes=5)
        Built in on_tick() from every Nifty index tick delivered by MarketHub
        via the INDEX_TOKEN routing. When a 5-min bar closes, _process_candle()
        is called just as MarketHub would call on_candle() for BankNifty strats.

    self._nifty_vwap : SessionVWAP()
        Updated on every on_tick() with Nifty price (proxy_weight=1, same as
        MarketHub uses for BankNifty since the index has no traded volume).
        Reset in _reset_day() which is called from pre_market().

  Backfill (today's catch-up, for late starts):
    t.py calls hub.backfill(hub.kite, index_token=256265) AFTER the BankNifty
    backfill. MarketHub's backfill filtering (INDEX_TOKEN routing) ensures only
    Nifty strategies receive those candles via on_candle(). Both paths
    (backfill on_candle + live on_tick) funnel into the same _process_candle()
    so indicator warmup is identical regardless of how candles arrive.

  Historical seeding (Bug 14 fix, prior-session warmup):
    Separately from the same-day backfill above, pre_market() now calls
    _seed_historical_buffer(), which fetches the PRIOR trading session's
    last N 5-min candles directly via hub.kite.historical_data() (same kite
    handle hub.backfill() uses) and loads them straight into _buf_5m.
    This means Supertrend(10,3) and BB(20) already have enough bars to be
    valid the moment the market opens, instead of needing min_bars=12
    (~60 live minutes) to accumulate from empty every single day. There is
    a real overnight gap between the last seeded bar and the first live
    9:15 bar -- Supertrend/BB self-correct within a few live bars, which is
    an accepted tradeoff (same one already used for BankNifty BB_STOCH's
    EMA seeding). If the historical fetch fails for any reason (holiday,
    API error, no kite handle), this falls back silently to the old
    live-only warmup path -- no regression risk.

TRADE MANAGEMENT (on every option WebSocket tick):
--------------------------------------------------------------
  * SL    = entry_price - ATR_SL_MULT * (index_atr * PREMIUM_ATR_SCALE)
            (anchored to fill price; scaled to premium points, Bug 12 fix)
  * TP    = entry_price + ATR_TP_MULT * (index_atr * PREMIUM_ATR_SCALE)
  * Trail = once profit >= TRAIL_ARM pts, SL locks TRAIL_LOCK_PCT of the
            profit reached so far (not flat breakeven), re-locking every
            TRAIL_STEP pts thereafter (Bug 13 fix)

ALL BUG FIXES FROM BB_STOCH (BankNifty) ARE INHERITED:
  Bug 1  -- time.sleep() removed from WebSocket callbacks.
  Bug 2  -- _persist_buf cleared only on genuine fill.
  Bug 3  -- _close_trade() race condition fixed with atomic lock.
  Bug 4  -- ATM options pre-subscribed in pre_market().
  Bug 5  -- ATM+-100 pts (2 Nifty strikes) pre-subscribed.
  Bug 6  -- Large gap days: spot-based ATM+-200 subscription on first candle.
  Bug 7  -- LTP unavailable on entry: stored as _pending_entry, filled on tick.
  Bug 8  -- SL/TP anchored to fill price, not pre-order LTP.
  Bug 9  -- Trail breakeven SL at entry+slippage (true net-zero breakeven).
  Bug 10 -- _force_close fires exactly once via _squareoff_done flag.
  Bug 11 -- BBTrade __slots__ includes "order_id".

NEW FIXES (this revision):
  Bug 12 -- ATR unit mismatch: SL/TP were previously sized directly off the
            raw NIFTY *index*-point ATR (typically 15-25 pts) and applied
            as if it were option-*premium* points. That mismatch meant
            sl_pts (atr*0.8) almost never exceeded the old sl_min=20 floor,
            so the "ATR-adaptive" SL was actually a flat 20 on nearly every
            trade. Fix: index ATR is scaled by PREMIUM_ATR_SCALE (a rough
            ATM-delta proxy) into an estimated premium-point ATR before
            sizing, and sl_min/sl_max/tp_min/tp_max were recalibrated to
            that smaller scale so the multiplier can actually flex with
            volatility instead of pinning at a floor every time.
            See _compute_sl_tp().
  Bug 13 -- Trail give-back: the old trail jumped straight from "no
            protection" to a flat breakeven the instant profit crossed
            trail_arm, then only added a new stage every trail_step points
            *after* that -- so a trade that ran most of the way to TP and
            reversed gave back the ENTIRE move (observed live 2026-06-18:
            PE ran to +25pts vs a +30.4 TP, reversed, exited at -0.25pts).
            Fix: once armed, SL locks TRAIL_LOCK_PCT of the profit reached
            so far (not just breakeven), re-locking at each new stage.
            See _update_trail().
  Bug 14 -- No historical seeding: _buf_5m started empty every session and
            needed min_bars=10 (~50 live minutes from 9:15) before the
            first possible signal, missing the most volatile opening hour
            every single day. Fix: pre_market() now seeds _buf_5m with the
            prior session's last N 5-min candles via
            hub.kite.historical_data(). See _seed_historical_buffer().
  Bug 15 -- Spread filter used a flat `ltp * 0.03` guess (shared verbatim
            with bb_stoch_strategy.py) against a TP capped in absolute
            premium points. This constant almost never binds here because
            Nifty ATM premiums (~130-150) make 3% only ~4 pts, but the
            identical code in the BankNifty file (~900+ premiums) blocked
            every real trade there -- see bb_stoch_strategy.py Bug 17 for
            the full mechanism. Fixed here too for consistency, since
            future DTE/strike changes could push Nifty premiums into the
            range where this constant starts silently blocking trades the
            same way it already did on BankNifty.
            Fix: replaced with _estimate_spread() -- percentage of premium
            WITH an absolute point cap. See CFG["spread_pct"] /
            CFG["spread_cap_pts"].
  Bug 16 -- Flat CFG["slippage"]=1.5 constant used for both entry markup
            and exit markdown regardless of the option's actual premium.
            On 2026-07-09 this contributed a flat 1.5pt debit on TOP of
            whatever real tick-to-tick gap already occurred at the SL
            level (entry 137.05, SL=127.51, but the triggering tick was
            127.10 -- only a genuine 0.41pt gap -- yet the logged exit was
            125.60, i.e. 1.91pts worse than the SL level, 1.5 of which was
            this flat constant, not market movement). Sized correctly for
            Nifty's own premium level in isolation, but not scaled to what
            was actually traded, and not shared/consistent with the same
            constant's use in bb_stoch_strategy.py's BankNifty trail math.
            Fix: BBTrade now carries a per-trade `slip` computed once at
            entry from _estimate_spread(entry_ltp)/2 (floored at
            CFG["slippage_min"]), stored and reused at exit so the two
            numbers are always consistent with each other and with what
            was actually filled. Exit logs now also show the raw
            triggering tick separately from the modeled slip, so real
            market gap vs modeled cost can be told apart at a glance.
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
from core.candle import CandleBuilder
from core.instruments import get_atm_strike
from core.vwap import SessionVWAP

# IST FIX: GitHub Actions runners are UTC.
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


log = logging.getLogger("strategy.bb_stoch_nifty")

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  NIFTY STRIKE STEP
#  Nifty 50 options are quoted at 50-pt intervals. BankNifty uses 100 pts.
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_STRIKE_STEP = 50

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

    # Bug 12 fix: `atr` from evaluate_signal() is computed on the NIFTY
    # *index* 5-min candles (15-25 pt range), not the option premium.
    # Scale it into an estimated premium-point ATR (rough ATM-delta proxy)
    # before using it to size SL/TP in premium terms. 0.45 is a starting
    # estimate for near-ATM weekly options -- revisit once real option-tick
    # ATR data is available (see _seed/_compute_sl_tp TODO).
    "premium_atr_scale"  : 0.45,

    # Recalibrated to the premium-point scale above (old values of
    # 20/50/30/120 were sized for raw index points and pinned SL to the
    # floor on almost every trade -- see Bug 12).
    "sl_min"             : 8.0,
    "sl_max"             : 35.0,
    "tp_min"             : 15.0,
    "tp_max"             : 90.0,

    # Bug 13 fix: trail now locks a FRACTION of profit reached at each
    # stage instead of jumping straight to flat breakeven.
    "trail_arm"          : 15.0,    # arm trail once profit >= this
    "trail_lock_pct"     : 0.5,     # lock this fraction of profit reached
    "trail_step"         : 8.0,     # re-lock every N pts of profit after arming

    # Bug 15/16 fix: spread/slippage now modeled from the option's own
    # premium instead of a single flat constant. Real bid-ask spreads on
    # liquid near-ATM index options are bounded by tick size /
    # market-maker competition, not premium magnitude -- so this is a
    # modest percentage WITH an absolute cap, not a pure %.
    # See _estimate_spread().
    "spread_pct"         : 0.012,   # assumed spread as % of premium
    "spread_cap_pts"     : 8.0,     # absolute cap on assumed spread (pts)
    "slippage_min"       : 1.0,     # floor for exit-slippage modeling
    "exit_cooldown"      : 2.0,

    # Risk
    "max_trades_day"     : 999999,
    "post_sl_cooldown"   : 120,
    "max_daily_loss"     : 6000,

    # ── Nifty-specific ────────────────────────────────────────────────────────
    # Lot size: Nifty 50. SEBI revises periodically — verify before live.
    "quantity"           : 65,

    # Pre-subscription offsets (pts from ATM).
    # Nifty strike step = 50 pts, so 100 = 2 strikes, 200 = 4 strikes each side.
    "pre_sub_offsets"    : [0, 100, -100],

    # Spot-based subscription on first candle (~9:20 AM) — covers gap days.
    # [0, 100, -100, 200, -200] = ATM and ±2 and ±4 Nifty strikes.
    "spot_atm_offsets"   : [0, 100, -100, 200, -200],

    # Signal persistence
    "persistence"        : 1,

    # Minimum 5-min bars before first signal.
    # Stochastic(5,3,3) needs k_period+d_smooth-1 = 7 bars minimum.
    # 10 bars = 50 min from 9:15 → first signal ~10:05 AM.
    "min_bars"           : 10,

    # Stochastic momentum filter -- replaces SuperTrend
    "stoch_k_period"     : 5,
    "stoch_k_smooth"     : 3,
    "stoch_d_smooth"     : 3,
    "stoch_ob"           : 80,   # overbought ceiling for CE entries
    "stoch_os"           : 20,   # oversold floor for PE entries

    # CSV paths
    "entry_csv"          : "logs/bb_stoch_nifty_entry.csv",
    "exit_csv"           : "logs/bb_stoch_nifty_exit.csv",
    "signal_csv"         : "logs/bb_stoch_nifty_signals.csv",
}


# ============================================================
# SPREAD / SLIPPAGE MODEL  (Bug 15/16 fix)
# ============================================================

def _estimate_spread(ltp: float) -> float:
    """
    Realistic bid-ask spread estimate for a near-ATM weekly index option,
    expressed in premium points.

    BUG FIX: the old model was a flat `ltp * 0.03` with no cap. On Nifty
    (~130-150 ATM premium) this only ever produces ~4 pts, so it rarely
    binds -- but the identical constant in bb_stoch_strategy.py blocks
    every real BankNifty trade (~900+ premium), because that file's TP is
    capped in ABSOLUTE premium points regardless of premium size. Fixed
    here too for consistency and to protect against future DTE/strike
    changes pushing Nifty premiums into the range where this constant
    would start silently blocking trades.

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
# INDICATOR FUNCTIONS  (pure, index-agnostic)
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
    raw_k = pd.Series(
        np.where(denom > 0, 100.0 * (close - lowest_low) / denom, 50.0),
        index=close.index,
    )
    k_series = raw_k.rolling(k_smooth).mean()
    d_series = k_series.rolling(d_smooth).mean()

    k = float(round(k_series.iloc[-1], 2))
    d = float(round(d_series.iloc[-1], 2))

    return {"k": k, "d": d, "ready": True}



# ============================================================
# SIGNAL BUILDER
# ============================================================

def evaluate_signal(df: pd.DataFrame, vwap: Optional[float]) -> dict:
    """
    Run all filters. Returns action = "BUY_CE" | "BUY_PE" | "HOLD".
    Identical logic to BBStochStrategy — pure function, index-agnostic.
    """
    empty = {"action": "HOLD", "blocked_by": "no_data",
             "bb": {}, "vol_ratio": 0, "atr": 0}

    if df is None or len(df) < CFG["min_bars"]:
        return {**empty, "blocked_by": "insufficient_bars"}

    bb        = compute_bb(df, CFG["bb_period"], CFG["bb_std"])
    vol_ratio = compute_vol_ratio(df, CFG["vol_avg_period"])
    atr       = _compute_atr(df, CFG["atr_period"])
    close     = float(df["close"].iloc[-1])

    stoch      = compute_stochastic(df, CFG["stoch_k_period"],
                                    CFG["stoch_k_smooth"], CFG["stoch_d_smooth"])
    k_cross_up = stoch["k"] > stoch["d"]
    k_cross_dn = stoch["k"] < stoch["d"]

    base = {
        "bb": bb, "vol_ratio": vol_ratio, "atr": atr,
        "close": close, "stoch_k": stoch["k"], "stoch_d": stoch["d"],
    }

    if not stoch["ready"]:
        return {"action": "HOLD", "blocked_by": "stoch_warmup", **base}

    if bb["squeeze"]:
        return {"action": "HOLD", "blocked_by": "bb_squeeze", **base}

    if vol_ratio == -1.0:
        vol_ok = True
        log.debug("[BB_STOCH_NIFTY] Volume data unavailable -- vol filter bypassed")
    else:
        vol_ok = round(vol_ratio, 4) >= CFG["vol_mult"]

    if not vol_ok:
        return {"action": "HOLD", "blocked_by": "volume_low", **base}

    if vwap and vwap > 0:
        above_vwap = close >= (vwap - CFG["vwap_buffer"])
        below_vwap = close <= (vwap + CFG["vwap_buffer"])
    else:
        above_vwap = True
        below_vwap = True

    bb_breakout_up  = close > bb["upper"] and bb["prev_close"] <= bb["prev_upper"]
    bb_bounce_up    = close > bb["lower"] and bb["prev_close"] <= bb["prev_lower"]
    bb_mid_cross_up = close > bb["mid"]   and bb["prev_close"] <= bb["prev_mid"]
    ce_bb_trigger   = bb_breakout_up or bb_bounce_up or bb_mid_cross_up
    stoch_ce_ok     = stoch["k"] < CFG["stoch_ob"] and k_cross_up

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

    bb_breakout_dn  = close < bb["lower"] and bb["prev_close"] >= bb["prev_lower"]
    bb_bounce_dn    = close < bb["upper"] and bb["prev_close"] >= bb["prev_upper"]
    bb_mid_cross_dn = close < bb["mid"]   and bb["prev_close"] >= bb["prev_mid"]
    pe_bb_trigger   = bb_breakout_dn or bb_bounce_dn or bb_mid_cross_dn
    stoch_pe_ok     = stoch["k"] > CFG["stoch_os"] and k_cross_dn

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
        # Bug 16 fix: `slip` is computed by the caller (_execute_fill)
        # from _estimate_spread(ltp)/2, floored at CFG["slippage_min"] --
        # not a flat constant. Stored on the trade so _close_trade()
        # reuses the exact value this trade was priced with.
        self.slip        = slip
        self.symbol      = symbol
        self.token       = token
        self.option_type = opt_type
        self.qty         = qty
        self.entry       = round(ltp + slip, 2)
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
        self.order_id      = ""


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

class BBStochNiftyStrategy(BaseStrategy):
    """
    Nifty 50 options strategy: Bollinger Bands + Volume + Supertrend trend filter.
    Identical signal logic to BBStochStrategy (BankNifty).

    Key architectural difference:
      MarketHub's 5-min candles and VWAP are built from BankNifty ticks.
      This strategy maintains its own internal CandleBuilder and SessionVWAP
      updated from Nifty index ticks delivered via INDEX_TOKEN routing.

    Backfill candles arrive via on_candle() (from hub.backfill index_token=256265).
    Live candles are built from on_tick() via self._nifty_candles CandleBuilder.
    Both paths call _process_candle() so indicator warmup is consistent.
    """

    # ── MarketHub routes Nifty ticks exclusively here via INDEX_TOKEN ─────────
    INDEX_TOKEN = 256265   # NSE:NIFTY 50 — fixed Zerodha instrument token

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "BB_STOCH_NIFTY"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        # 5-min candle buffer — size covers BB and Supertrend lookbacks + headroom
        buf_size = max(CFG["bb_period"], CFG["stoch_k_period"]) + 20
        self._buf_5m: deque = deque(maxlen=buf_size)

        # Internal Nifty 5-min candle builder (MarketHub builds BankNifty candles)
        # Updated on every Nifty index tick in on_tick().
        self._nifty_candles = CandleBuilder(minutes=5)

        # Internal Nifty VWAP (MarketHub's session_vwap tracks BankNifty)
        # Reset in _reset_day() which is called from pre_market().
        self._nifty_vwap = SessionVWAP()

        # Trade state
        self._trade: Optional[BBTrade] = None
        self._active_token: Optional[int] = None
        self._lock = threading.Lock()

        # Risk state
        self._daily_pnl    : float = 0.0
        self._trades_today : int   = 0
        self._last_sl_time : float = 0.0
        self._is_halted    : bool  = False
        self._results      : list  = []
        self._blocked_log  : dict  = {}

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

        # Entry delay (next-candle breakout confirmation)
        self._entry_delay_pending: Optional[dict] = None

        # Bug A fix: track every token subscribed as part of the ambient range
        # (pre_market + spot_atm) — identical guard to BB_STOCH (BankNifty).
        # pre_sub_offsets [0, ±100] ⊂ spot_atm_offsets [0, ±100, ±200].
        # Without this guard the overlapping Nifty ATM tokens get refcount=2
        # from BB_STOCH_NIFTY alone and are never fully released by this strategy.
        self._subscribed_range: set[int] = set()

        log.info("[BB_STOCH_NIFTY] Strategy initialized")

    # ----------------------------------------------------------
    # PRE-MARKET
    # ----------------------------------------------------------

    def pre_market(self, premarket_data, instruments) -> bool:
        """
        Called from t.py with Nifty-specific PreMarketData and InstrumentStore.
          premarket_data : PreMarketData fetched with index_token=256265
          instruments    : InstrumentStore loaded with option_root="NIFTY"
        """
        self._instruments  = instruments
        self._expiry_date  = premarket_data.expiry_date
        self._dte          = premarket_data.dte_days
        self._session_date = _now_ist().date()
        self._reset_day()

        # Bug 14 fix: seed _buf_5m with the prior session's candles so
        # Supertrend/BB are warm from market open instead of needing
        # min_bars=12 (~60 live minutes) to fill up from empty.
        self._seed_historical_buffer()

        log.info(
            f"[BB_STOCH_NIFTY] Pre-market | VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr} Expiry={self._expiry_date} DTE={self._dte}"
        )

        # ── PRE-SUBSCRIBE STRIKES ─────────────────────────────────────────────
        # Bug 4+5 fix: warm up ATM and ATM±100 pts (2 Nifty strikes) before signal fires.
        ref_price = premarket_data.prev_close or premarket_data.prev_last5m_close
        if ref_price and self._expiry_date:
            atm = get_atm_strike(ref_price, step=NIFTY_STRIKE_STEP)
            for offset in CFG["pre_sub_offsets"]:
                for opt_type in ("CE", "PE"):
                    strike = atm + offset
                    # On expiry day (DTE==0): shift ATM one Nifty strike toward
                    # ITM for better delta and liquidity.
                    if offset == 0 and self._dte == 0:
                        strike = (atm - NIFTY_STRIKE_STEP if opt_type == "CE"
                                  else atm + NIFTY_STRIKE_STEP)
                    tok, sym = instruments.get_option_token(strike, opt_type, self._expiry_date)
                    if tok:
                        # Bug A fix: guard with _subscribed_range to prevent
                        # double-subscribe when _subscribe_spot_atm() later
                        # covers the same offsets [0, ±100].
                        if tok not in self._subscribed_range:
                            self.subscribe_option(tok)
                            self._subscribed_range.add(tok)
                        if offset == 0:
                            if opt_type == "CE":
                                self._pre_ce_token = tok
                            else:
                                self._pre_pe_token = tok
                        log.info(
                            f"[BB_STOCH_NIFTY] Pre-subscribed {sym} ({tok})"
                            f"{' [ATM]' if offset == 0 else f' [ATM{offset:+d}]'}"
                        )
        else:
            log.warning("[BB_STOCH_NIFTY] No ref price for pre-subscription")

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
        self._nifty_vwap.reset()   # Reset internal Nifty VWAP for fresh session
        # Bug A fix: clear range-subscription tracking on each new session
        self._subscribed_range.clear()

    # ----------------------------------------------------------
    # Bug 14: Historical candle seeding (prior-session warmup)
    # ----------------------------------------------------------

    def _seed_historical_buffer(self):
        """
        Fetch the prior trading session's last N 5-min Nifty candles via
        hub.kite.historical_data() and load them into _buf_5m, so that
        Supertrend(10,3) and BB(20) already have enough bars to be valid
        the moment the market opens -- instead of needing min_bars=12
        (~60 live minutes from 9:15) to accumulate from empty every day.

        This is separate from hub.backfill(), which only replays TODAY's
        already-elapsed candles for late starts. This seeds yesterday's
        close-of-session data so today's first candle isn't starting cold.

        There is a genuine overnight gap between the last seeded bar and
        the first live 9:15 bar -- Supertrend/BB self-correct within a few
        live bars, which is an accepted tradeoff (mirrors the BankNifty
        BB_STOCH EMA-seeding approach already used elsewhere).

        Fails safe: if kite is unavailable, the fetch errors, or no data
        comes back, this silently falls back to the old live-only warmup
        path (no regression -- the strategy just trades a bit later that
        day, exactly as it did before this fix).
        """
        kite = getattr(self._hub, "kite", None)
        if kite is None:
            log.warning(
                "[BB_STOCH_NIFTY] No kite handle on hub -- cannot seed "
                "historical candles, falling back to live-only warmup "
                f"(~{CFG['min_bars'] * 5}min)"
            )
            return

        needed = self._buf_5m.maxlen or (CFG["bb_period"] + CFG["stoch_k_period"] + 10)
        today  = _now_ist().date()

        try:
            raw = kite.historical_data(
                instrument_token = self.INDEX_TOKEN,
                from_date        = datetime.combine(today - timedelta(days=7), dtime(9, 15)),
                to_date          = datetime.combine(today - timedelta(days=1), dtime(15, 30)),
                interval         = "5minute",
            )
        except Exception as e:
            log.warning(
                f"[BB_STOCH_NIFTY] Historical seed fetch failed: {e} -- "
                f"falling back to live-only warmup (~{CFG['min_bars'] * 5}min)"
            )
            return

        if not raw:
            log.warning(
                "[BB_STOCH_NIFTY] Historical seed returned no candles -- "
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
                    log.warning(f"[BB_STOCH_NIFTY] Skipping malformed seed bar {bar}: {e}")

        if self._buf_5m:
            log.info(
                f"[BB_STOCH_NIFTY] Seeded {len(self._buf_5m)} prior-session 5-min "
                f"candles ({tail[0].get('date', '?')} -> {tail[-1].get('date', '?')}) "
                f"-- Supertrend/BB warm from first live tick, "
                f"no {CFG['min_bars'] * 5}min wait"
            )
        else:
            log.warning(
                "[BB_STOCH_NIFTY] Seed buffer empty after load -- "
                "falling back to live-only warmup"
            )


    # ----------------------------------------------------------
    # Bug 6: Dynamic spot-based subscription on first live candle
    # ----------------------------------------------------------

    def _subscribe_spot_atm(self, spot: float):
        """
        Subscribe ATM±200 pts (4 Nifty strikes) based on actual live spot.
        Called on first 5-min candle (~9:20 AM) to cover large gap days
        where pre_market()'s prev_close-based subscriptions are off-market.
        """
        if not spot or spot <= 0:
            return
        if not self._expiry_date or not self._instruments:
            return

        atm        = get_atm_strike(spot, step=NIFTY_STRIKE_STEP)
        subscribed = 0
        for offset in CFG["spot_atm_offsets"]:
            for opt_type in ("CE", "PE"):
                tok, sym = self._instruments.get_option_token(
                    atm + offset, opt_type, self._expiry_date
                )
                if tok:
                    # Bug A fix: skip tokens already subscribed in pre_market()
                    # (offsets [0, ±100] overlap between pre_sub_offsets and
                    # spot_atm_offsets — those exact tokens must not be
                    # subscribed twice by this strategy).
                    if tok not in self._subscribed_range:
                        self.subscribe_option(tok)
                        self._subscribed_range.add(tok)
                        subscribed += 1
                        log.info(
                            f"[BB_STOCH_NIFTY] Spot-sub {sym} ({tok}) "
                            f"[spot-ATM{offset:+d}]"
                        )
                    else:
                        log.debug(
                            f"[BB_STOCH_NIFTY] Spot-sub skip (already held) "
                            f"{sym} ({tok}) [spot-ATM{offset:+d}]"
                        )
        log.info(
            f"[BB_STOCH_NIFTY] Spot subscription done: "
            f"{subscribed} new tokens around ATM={atm} (spot={spot:.0f}); "
            f"{len(self._subscribed_range)} total range tokens held"
        )

    # ----------------------------------------------------------
    # TICK CALLBACKS
    # ----------------------------------------------------------

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """
        Nifty 50 index tick — delivered exclusively by MarketHub via INDEX_TOKEN=256265.
        BankNifty ticks are NEVER delivered here.

        Three jobs:
          1. Update internal Nifty VWAP (proxy_weight=1, same as MarketHub uses
             for BankNifty — the index has no traded volume).
          2. Feed the internal 5-min CandleBuilder; call _process_candle() when
             a bar closes. This is the live equivalent of MarketHub calling
             on_candle() for BankNifty strategies.
          3. Check squareoff time + entry delay confirmation.
        """
        # Bug 10 fix: auto-squareoff fires exactly once
        if ts.time() >= dtime(*CFG["auto_squareoff"]) and not self._squareoff_done:
            self._squareoff_done = True
            self._force_close("AUTO-SQUAREOFF")
            return

        # Bug D fix: subscribe the full Nifty ATM range on the very first tick
        # at or after market open (9:15), not waiting for the first 5-min candle
        # close at ~9:20.
        #
        # Why it matters:
        #   SPIKE_NIFTY also subscribes Nifty ATM CE+PE in pre_market.  At 9:15
        #   SPIKE_NIFTY enters its trade and unsubscribes the unused leg
        #   (e.g. unsubscribes ATM PE when going CE).  If BB_STOCH_NIFTY's
        #   pre_market happened to use a slightly different ATM (prev-close vs
        #   live) that token's hub refcount from BB_STOCH_NIFTY may be 0 for
        #   that exact token.  SPIKE_NIFTY's unsubscribe then brings the hub
        #   refcount to 0 → tick removed from WebSocket for 5 minutes (until
        #   the first _process_candle fires at ~9:20).
        #
        #   Calling _subscribe_spot_atm here on the first live tick guarantees
        #   the full ATM ± range is live from 9:15 onward.
        #
        # _open_atm_subscribed is also checked inside _process_candle() so that
        # path remains a safety-net for backfill/test scenarios.
        if not self._open_atm_subscribed and ts.time() >= dtime(9, 15):
            if price and price > 0:
                self._subscribe_spot_atm(price)
                self._open_atm_subscribed = True

        # Job 1: update Nifty VWAP
        self._nifty_vwap.update(price, price, price, volume=0, proxy_weight=1)

        # Entry delay confirmation (Change B pattern)
        confirm = self._entry_delay_pending
        if confirm and not self._trade and not self._is_halted:
            action  = confirm["action"]
            trigger = confirm["trigger_level"]

            if ts.time() >= dtime(*CFG["session_cutoff"]):
                log.info(
                    f"[BB_STOCH_NIFTY] Entry delay expired (past cutoff) "
                    f"| {action} trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
            elif action == "BUY_CE" and price > trigger:
                log.info(
                    f"[BB_STOCH_NIFTY] Entry delay CONFIRMED | BUY_CE "
                    f"| nifty={price:.2f} > trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])
            elif action == "BUY_PE" and price < trigger:
                log.info(
                    f"[BB_STOCH_NIFTY] Entry delay CONFIRMED | BUY_PE "
                    f"| nifty={price:.2f} < trigger={trigger:.2f}"
                )
                self._entry_delay_pending = None
                self._enter_trade(action, confirm["sig"], confirm["ts"])

        # Job 2: build 5-min Nifty candle from tick
        # proxy_vol=1: Nifty index has no traded volume; use tick count as proxy
        # (same convention MarketHub uses for BankNifty index ticks).
        closed = self._nifty_candles.feed_tick(price, 1, ts)
        if closed is not None:
            self._process_candle(closed, ts)

    def on_candle(self, candle: dict, ts: datetime):
        """
        Receives Nifty 5-min candles from hub.backfill(index_token=256265).
        MarketHub's backfill filtering (INDEX_TOKEN routing) ensures only
        Nifty candles arrive here — BankNifty candles are never delivered.

        During live trading, on_tick() builds candles internally and calls
        _process_candle() directly — on_candle() is only used for backfill.
        """
        self._process_candle(candle, ts)

    def _process_candle(self, candle: dict, ts: datetime):
        """
        Shared handler for both backfill candles (on_candle) and live candles
        (built by on_tick via internal CandleBuilder).

        Appends to _buf_5m, handles one-time spot subscription (Bug 6),
        expires stale entry delays, then evaluates entry signal.
        """
        with self._lock:
            self._buf_5m.append(candle)

        # Bug 6 / Bug D fix: spot-based subscription safety-net.
        # In live trading on_tick() has already subscribed the spot range at
        # 9:15 on the first tick.  This path covers backfill/test runs where
        # on_tick() is not called and the first signal arrives via on_candle().
        if not self._open_atm_subscribed:
            spot = candle.get("close", 0)
            if spot and spot > 0:
                self._subscribe_spot_atm(spot)
                self._open_atm_subscribed = True

        # Expire entry delay that survived a full candle without index confirmation
        if self._entry_delay_pending:
            edp = self._entry_delay_pending
            log.info(
                f"[BB_STOCH_NIFTY] Entry delay expired (no Nifty confirmation "
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
        # Bug 7: pending entry fill on first live tick
        pending = self._pending_entry
        if pending is not None and token == pending["token"] and not self._trade:
            now_time = ts.time()
            if now_time >= dtime(*CFG["session_cutoff"]):
                log.info(f"[BB_STOCH_NIFTY] Pending entry expired (past cutoff) for "
                         f"{pending['symbol']} -- discarding")
                self._pending_entry = None
            elif self._is_halted or self._trades_today >= CFG["max_trades_day"]:
                log.info(f"[BB_STOCH_NIFTY] Pending entry cancelled (risk gate) for "
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
            log.info(
                f"[BB_STOCH_NIFTY] Post-SL cooldown: "
                f"{CFG['post_sl_cooldown'] - elapsed:.0f}s remaining"
            )
            return

        with self._lock:
            candles = list(self._buf_5m)
        df = self._to_df(candles)
        if df.empty:
            return

        n_bars = len(df)
        if n_bars < CFG["min_bars"]:
            log.debug(
                f"[BB_STOCH_NIFTY] Warming up: {n_bars}/{CFG['min_bars']} bars "
                f"(ready in ~{(CFG['min_bars'] - n_bars) * 5} mins)"
            )
            return

        # Use internal Nifty VWAP — hub.session_vwap tracks BankNifty
        vwap = self._nifty_vwap.value

        sig    = evaluate_signal(df, vwap)
        action = sig["action"]

        self._log_signal(ts, sig)

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
            f"[BB_STOCH_NIFTY] {action:8s} | "
            f"Close={sig.get('close', 0):.2f} | "
            f"BB=[{bb.get('lower', 0):.1f}~{bb.get('upper', 0):.1f}] "
            f"BW={bb.get('bw_pct', 0)*100:.2f}% | "
            f"Vol={vol_str} | VWAP(Nifty)={vwap_str} | "
            f"Stoch={stoch_str} | "
            f"Block={sig.get('blocked_by') or 'none'} | Persist={confirmed} | "
            f"PnL={self._daily_pnl:+.1f}Rs T={self._trades_today}/{CFG['max_trades_day']}"
        )

        if not confirmed or action == "HOLD":
            if action == "HOLD":
                bl = sig.get("blocked_by", "")
                self._blocked_log[bl] = self._blocked_log.get(bl, 0) + 1
            return

        # Entry delay: set trigger level, wait for next-tick Nifty confirmation.
        # on_tick() fires entry the moment Nifty crosses signal-candle H (CE) or L (PE).
        last_candle   = candles[-1] if candles else {}
        trigger_level = (
            float(last_candle.get("high", 0)) if action == "BUY_CE"
            else float(last_candle.get("low", 0))
        )

        if trigger_level <= 0:
            log.warning(
                f"[BB_STOCH_NIFTY] Entry delay: trigger level unavailable for {action} "
                f"-- entering immediately"
            )
            self._enter_trade(action, sig, ts)
            return

        log.info(
            f"[BB_STOCH_NIFTY] Entry delay SET | {action} | "
            f"trigger={'>' if action == 'BUY_CE' else '<'}{trigger_level:.2f} | "
            f"waiting for Nifty index break via on_tick()"
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
        Open a new Nifty options position.
        Uses NIFTY_STRIKE_STEP=50 for ATM calculation.
        DTE==0 adjustment: one Nifty strike (50 pts) toward ITM.
        Bug 7: stores _pending_entry if LTP unavailable.
        """
        opt    = "CE" if action == "BUY_CE" else "PE"
        spot   = sig.get("close", 0)
        atr    = sig.get("atr", 0)
        expiry = self._expiry_date

        if expiry is None:
            log.warning("[BB_STOCH_NIFTY] No expiry date -- skipping entry")
            return

        atm = get_atm_strike(spot, step=NIFTY_STRIKE_STEP)
        # On expiry day: shift one Nifty strike (50 pts) toward ITM
        if self._dte == 0:
            atm = atm - NIFTY_STRIKE_STEP if opt == "CE" else atm + NIFTY_STRIKE_STEP

        token, tsym = self._instruments.get_option_token(atm, opt, expiry)
        if not token:
            log.warning(f"[BB_STOCH_NIFTY] Option not found: ATM={atm} {opt} expiry={expiry}")
            return

        self.subscribe_option(token)
        ltp = self.get_price(token)

        if not ltp or ltp < 5:
            log.warning(
                f"[BB_STOCH_NIFTY] LTP unavailable for {tsym} (token={token}) -- "
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

        # Bug 15 fix: was `ltp * 0.03` uncapped -- see _estimate_spread()
        # for why that's wrong once premium size varies across strikes/
        # instruments (mirrors bb_stoch_strategy.py Bug 17).
        est_spread = _estimate_spread(ltp)
        if tp_pts > 0 and (est_spread / tp_pts) > 0.40:
            log.warning(
                f"[BB_STOCH_NIFTY] Spread {est_spread:.1f} too large vs TP {tp_pts:.1f} -- skipping"
            )
            self.unsubscribe_option(token)
            return

        if not self._acquire_slot():
            log.warning("[BB_STOCH_NIFTY] Trade slot blocked -- another live strategy has a position")
            self.unsubscribe_option(token)
            return

        buy_order_id = self._place_buy(tsym, token, CFG["quantity"], ltp)
        if LIVE_MODE and buy_order_id is None:
            self._release_slot()
            log.error(f"[BB_STOCH_NIFTY] BUY order FAILED for {tsym} -- entry aborted")
            self.unsubscribe_option(token)
            return

        # Bug 16 fix: per-trade half-spread, sized to what was actually
        # traded (not a flat constant shared across BankNifty/Nifty).
        slip = max(CFG["slippage_min"], _estimate_spread(ltp) / 2)

        trade = BBTrade(
            symbol   = tsym,
            token    = token,
            opt_type = opt,
            ltp      = ltp,
            qty      = CFG["quantity"],
            spot     = spot,
            sl_pts   = sl_pts,
            tp_pts   = tp_pts,
            atr      = atr,
            slip     = slip,
        )
        trade.order_id = buy_order_id

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
            f"[BB_STOCH_NIFTY] [{mode_tag}] ENTRY {opt} | {tsym} | "
            f"Fill={trade.entry:.2f} LTP={ltp:.2f}{delay_str} | "
            f"SL={trade.sl:.2f}(-{sl_pts:.1f}) TP={trade.target:.2f}(+{tp_pts:.1f}) | "
            f"RR=1:{rr} ATR={atr:.1f} | buy_order={buy_order_id} | "
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
            "order_id"         : buy_order_id,
            "mode"             : sig.get("mode", "pending" if pending_delay_ms else ""),
            "bb_upper"         : bb.get("upper", ""),
            "bb_mid"           : bb.get("mid", ""),
            "bb_lower"         : bb.get("lower", ""),
            "bb_bw_pct"        : bb.get("bw_pct", ""),
            "stoch_k"          : sig.get("stoch_k", ""),
            "stoch_d"          : sig.get("stoch_d", ""),
            "vol_ratio"        : sig.get("vol_ratio", ""),
            "vwap"             : (self._nifty_vwap.value
                                  if self._nifty_vwap.value is not None else "N/A"),
            "pending_delay_ms" : (round(pending_delay_ms, 0)
                                  if pending_delay_ms is not None else ""),
        })

    # ----------------------------------------------------------
    # Bug 7: Pending entry fill on first WebSocket tick
    # ----------------------------------------------------------

    def _fill_pending(self, pending: dict, ltp: float, ts: datetime):
        """Fill a pending entry on its first live option tick."""
        self._pending_entry = None

        if ltp < 5:
            log.warning(
                f"[BB_STOCH_NIFTY] Pending fill: LTP {ltp:.2f} too low for "
                f"{pending['symbol']} -- discarding"
            )
            return

        now_time = ts.time()
        if now_time >= dtime(*CFG["session_cutoff"]):
            log.info(f"[BB_STOCH_NIFTY] Pending fill aborted -- past session cutoff ({pending['symbol']})")
            return
        if self._is_halted:
            log.info(f"[BB_STOCH_NIFTY] Pending fill aborted -- strategy halted ({pending['symbol']})")
            return
        if self._trades_today >= CFG["max_trades_day"]:
            log.info(f"[BB_STOCH_NIFTY] Pending fill aborted -- max trades reached ({pending['symbol']})")
            return
        if self._trade:
            log.info(f"[BB_STOCH_NIFTY] Pending fill aborted -- trade already open ({pending['symbol']})")
            return

        try:
            delay_ms = (ts - pending["ts"]).total_seconds() * 1000
        except Exception:
            delay_ms = None

        log.info(
            f"[BB_STOCH_NIFTY] PENDING FILL | {pending['symbol']} "
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
        Bug 13 fix (trail give-back): the old version jumped straight from
        "no protection" to a flat breakeven SL the instant profit crossed
        trail_arm, then only added a new stage every trail_step points
        *after* that. That left a dead zone where a trade running most of
        the way to TP and reversing gave back the ENTIRE move (observed
        live 2026-06-18: PE ran to +25pts vs a +30.4pt TP, reversed, and
        was stopped out for -0.25pts net -- a near-winner turned scratch).

        Fix: once armed (profit >= trail_arm), SL locks TRAIL_LOCK_PCT of
        the profit reached *at that point* (not just breakeven+slippage),
        and re-locks at the same fraction every trail_step points beyond
        the arm threshold. SL only ever moves up, never down.
        """
        if pnl_pts < CFG["trail_arm"]:
            return

        stage = int((pnl_pts - CFG["trail_arm"]) / CFG["trail_step"]) + 1
        if stage <= trade.trail_stage:
            return

        locked_pts = round(pnl_pts * CFG["trail_lock_pct"], 2)
        new_sl     = round(trade.entry + locked_pts, 2)

        if new_sl > trade.sl:
            trade.sl          = new_sl
            trade.trail_stage = stage
            log.info(
                f"[BB_STOCH_NIFTY] Trail stage {stage} | "
                f"profit={pnl_pts:+.1f}pts locked={locked_pts:+.1f}pts | "
                f"SL={trade.sl:.2f}"
            )

    def _close_trade(self, ltp: float, reason: str):
        """Bug 3 fix: atomically claim self._trade at the top."""
        with self._lock:
            trade = self._trade
            if not trade:
                return
            self._trade = None

        trade.exit_pending = True
        trade.last_exit_ts = time_module.time()

        # Bug 16 fix: use this trade's own stored slip, not a flat
        # constant borrowed from a different instrument's premium scale.
        slip    = trade.slip
        exit_px = round(ltp - slip, 2)
        pnl_pts = round(exit_px - trade.entry, 2)
        pnl_rs  = round(pnl_pts * trade.qty, 2)
        rr_act  = round(pnl_pts / trade.sl_pts, 2) if trade.sl_pts else 0

        self._daily_pnl += pnl_rs
        if reason == "SL":
            self._last_sl_time = time_module.time()

        sell_order_id = self._place_sell_with_retry(
            trade.symbol, trade.token, trade.qty, ltp,
            max_retries=3,
        )
        self._release_slot()

        mode_tag = "LIVE" if LIVE_MODE else "PAPER"
        tag      = "[TARGET]" if reason == "TARGET" else ("[SL]" if reason == "SL" else "[EXIT]")
        log.info(
            f"[BB_STOCH_NIFTY] [{mode_tag}] {tag} {trade.option_type} | {trade.symbol} | {reason} | "
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
                f"[BB_STOCH_NIFTY] HALTED -- max daily loss breached "
                f"({self._daily_pnl:.0f}Rs)"
            )

        with self._lock:
            self._unsubscribe_active()

    def _force_close(self, reason: str):
        if self._pending_entry:
            log.info(
                f"[BB_STOCH_NIFTY] Discarding pending entry for "
                f"{self._pending_entry.get('symbol', '?')} on {reason}"
            )
            self._pending_entry = None
        if self._entry_delay_pending:
            log.info(
                f"[BB_STOCH_NIFTY] Discarding entry delay "
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
            log.info(f"[BB_STOCH_NIFTY] Unsubscribed token {self._active_token}")
            self._active_token = None

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _compute_sl_tp(self, ltp: float, atr: float) -> Tuple[float, float]:
        """
        Bug 12 fix (ATR unit mismatch): `atr` here is the NIFTY *index*
        ATR (5-min OHLC of the spot index, typically 15-25 pts) -- NOT the
        option premium's own ATR. Previously this index-point value was
        subtracted/added directly onto the option premium (rupees), a unit
        mismatch that meant sl_pts = atr*0.8 almost never exceeded the old
        sl_min=20 floor: the "ATR-adaptive" SL was actually a flat 20 on
        nearly every trade, regardless of real volatility.

        Fix: scale the index ATR by PREMIUM_ATR_SCALE -- a rough proxy for
        an ATM weekly option's delta -- into an estimated premium-point
        ATR before sizing. sl_min/sl_max/tp_min/tp_max were recalibrated
        to that smaller scale (see CFG) so the multiplier actually flexes
        with volatility instead of pinning at a floor every trade.

        TODO (future improvement): replace this scale-factor proxy with
        ATR computed directly from the option's own premium tick/candle
        series once that buffer exists -- this is an interim fix, not the
        final correct measurement.
        """
        premium_atr = atr * CFG["premium_atr_scale"] if atr > 0 else 0.0

        sl_pts = float(premium_atr * CFG["atr_sl_mult"]) if premium_atr > 0 else CFG["sl_min"]
        tp_pts = float(premium_atr * CFG["atr_tp_mult"]) if premium_atr > 0 else CFG["tp_min"]
        sl_pts = max(CFG["sl_min"], min(sl_pts, CFG["sl_max"]))
        tp_pts = max(CFG["tp_min"], min(tp_pts, CFG["tp_max"]))
        if self._dte == 0:
            tp_pts = max(CFG["tp_min"], tp_pts * 0.75)
        rr = round(tp_pts / sl_pts, 2) if sl_pts else 0
        log.info(
            f"[BB_STOCH_NIFTY] SL/TP | IndexATR={atr:.1f} "
            f"PremiumATR~{premium_atr:.1f} "
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
            "vwap"         : (self._nifty_vwap.value
                              if self._nifty_vwap.value is not None else "N/A"),
        })

    # ----------------------------------------------------------
    # EOD
    # ----------------------------------------------------------

    def eod_summary(self):
        self._force_close("EOD-SQUAREOFF")

        # Bug A / Bug E fix: release all ambient range tokens (pre_market +
        # spot_atm).  Mirror of the same pattern in BB_STOCH (BankNifty).
        # _unsubscribe_active() inside _force_close already handled the active
        # trade token.  If that token was also in _subscribed_range (it was
        # pre-subscribed as part of the ATM range), this loop correctly
        # decrements its refcount a second time to zero out BB_STOCH_NIFTY's
        # full share.
        for tok in set(self._subscribed_range):
            self.unsubscribe_option(tok)
        n = len(self._subscribed_range)
        self._subscribed_range.clear()
        if n:
            log.info(f"[BB_STOCH_NIFTY] EOD range cleanup: released {n} ambient tokens")

        results  = self._results
        total    = len(results)
        wins     = [r for r in results if r["pnl_pts"] > 0]
        losses   = [r for r in results if r["pnl_pts"] <= 0]
        win_pct  = len(wins) / total if total else 0
        avg_win  = sum(r["pnl_pts"] for r in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(r["pnl_pts"] for r in losses) / len(losses) if losses else 0
        exp      = round(win_pct * avg_win + (1 - win_pct) * avg_loss, 3) if total else 0

        log.info(
            f"[BB_STOCH_NIFTY] EOD | "
            f"Trades={total} W={len(wins)} L={len(losses)} "
            f"WinRate={win_pct*100:.1f}% Expect={exp:+.3f}pts "
            f"DayPnL={self._daily_pnl:+.2f}Rs | "
            f"5m-bars={len(self._buf_5m)} | "
            f"Stoch({CFG['stoch_k_period']},{CFG['stoch_k_smooth']},{CFG['stoch_d_smooth']})-momentum-filter=ACTIVE"
        )
        if self._blocked_log:
            top = sorted(self._blocked_log.items(), key=lambda x: -x[1])[:5]
            log.info(f"[BB_STOCH_NIFTY] Top blocks today: {top}")

