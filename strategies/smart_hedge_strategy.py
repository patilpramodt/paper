"""
strategies/smart_hedge_strategy.py

SMART HEDGE — Unified Directional Spread Strategy for BANKNIFTY
═══════════════════════════════════════════════════════════════════════════════

ONE TRADE PER DAY. Market direction is scored at entry time using 4 signals.
Based on the score, ONE of three option spread structures is selected:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  SCORE ≥ +2  →  BULLISH  → Bull Put Spread   (sell OTM PE + buy hedge)  │
  │  SCORE ≤ -2  →  BEARISH  → Bear Call Spread  (sell OTM CE + buy hedge)  │
  │  -1 to +1   →  NEUTRAL  → Iron Condor        (sell both sides)          │
  └──────────────────────────────────────────────────────────────────────────┘

DIRECTION SIGNALS (each scored individually and logged):
──────────────────────────────────────────────────────
  Signal 1 – PCR (Put-Call Ratio via WebSocket OI):
    PCR > 1.15  → +2  (strong bullish — heavy put writing, market held)
    PCR > 1.05  → +1  (mild bullish)
    PCR 0.95–1.05 → 0 (neutral)
    PCR < 0.95  → -1  (mild bearish)
    PCR < 0.85  → -2  (strong bearish — heavy call writing, market selling off)

  Signal 2 – Spot vs EMA200 (from premarket data):
    Spot > EMA200  → +1  (bullish bias)
    Spot < EMA200  → -1  (bearish bias)

  Signal 3 – Gap from previous close:
    Gap > +150 pts   → +1  (gap-up, bullish momentum)
    Gap < -150 pts   → -1  (gap-down, bearish momentum)
    Within ±150 pts  →  0  (flat open)

  Signal 4 – First completed 5-min candle direction:
    Green candle (close > open) → +1  (early buying)
    Red candle   (close < open) → -1  (early selling)

TRADE STRUCTURES:
─────────────────
  BULLISH — Bull Put Spread (2 legs):
    SELL PE at (ATM - short_offset)   ← collect premium
    BUY  PE at (ATM - long_offset)    ← hedge, caps max loss
    Profits when BANKNIFTY stays ABOVE short_pe_strike
    Net credit = sell_pe_ltp - buy_pe_ltp

  BEARISH — Bear Call Spread (2 legs):
    SELL CE at (ATM + short_offset)   ← collect premium
    BUY  CE at (ATM + long_offset)    ← hedge, caps max loss
    Profits when BANKNIFTY stays BELOW short_ce_strike
    Net credit = sell_ce_ltp - buy_ce_ltp

  NEUTRAL — Iron Condor (4 legs):
    SELL CE at (ATM + short_offset)   ← collect premium
    BUY  CE at (ATM + long_offset)    ← hedge
    SELL PE at (ATM - short_offset)   ← collect premium
    BUY  PE at (ATM - long_offset)    ← hedge
    Profits when BANKNIFTY stays inside [short_pe – short_ce] range

ENTRY FILTERS (all three trade types):
──────────────────────────────────────
  Time window  : 9:35 AM – 10:15 AM (after first candle is available)
  Gap guard    : skip if |gap| > 400 pts AND score is neutral (too risky for condor)
  VIX filter   : VIX > 20 → trade half-size (IV is rich, still trade)
  Net credit   : must be > 0 (sanity check)
  One trade    : once entered, no re-entry same day

EXIT RULES (same for all three):
──────────────────────────────────
  TARGET      : exit when current net value ≤ net_credit × (1 - 0.50)
                → keep 50% of credit collected
  SL          : exit when current net value ≥ net_credit × 2.0
                → position has doubled in cost against us
  BREACH      : emergency exit when spot crosses the short strike(s)
                Condor: breach of either short_CE or short_PE
                Bull Put: breach below short_PE
                Bear Call: breach above short_CE
  TIME STOP   : force close at 3:10 PM
  EOD         : hard close if still open at eod_summary()

LOGGING PHILOSOPHY:
───────────────────
  Every filter check is logged with the value and decision.
  Every signal is logged with score contribution.
  Entry log shows why this trade type was chosen (score breakdown).
  Exit log shows what triggered exit and final P&L.
  CSV captures everything for post-market analysis.
  A separate signal_audit.csv logs every entry attempt (including skips)
  so you can review what blocked entries and improve the filter logic.

PAPER MODE ONLY:
────────────────
  LIVE_MODE = False → all orders are simulated, no real API calls.
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional, Tuple

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

log = logging.getLogger("strategy.smart_hedge")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE — always paper for now
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Lot size
    "quantity"          : 30,       # BANKNIFTY lot (multiples of 30)

    # Strike offsets from ATM
    # short_offset → the leg we SELL (closer to ATM, collects more premium)
    # long_offset  → the leg we BUY  (farther OTM, limits max loss)
    # Spread width = long_offset - short_offset = 300 pts per side
    "short_offset"      : 200,      # pts OTM from ATM for sold leg
    "long_offset"       : 500,      # pts OTM from ATM for hedge leg

    # Entry window
    "entry_start"       : dtime(9, 35),   # after first candle, so signal 4 is ready
    "entry_cutoff"      : dtime(10, 15),

    # Time stop
    "close_time"        : dtime(15, 10),

    # Exit thresholds
    "profit_target_pct" : 0.50,     # exit when value ≤ (1 - 0.50) × credit
    "sl_multiplier"     : 2.0,      # exit when value ≥ 2.0 × credit

    # Gap guard: if |gap| > this AND score is neutral → skip (too directional for condor)
    "max_gap_pts"       : 400,

    # PCR scoring thresholds
    "pcr_bull_strong"   : 1.15,     # PCR above → score +2
    "pcr_bull_mild"     : 1.05,     # PCR above (but below strong) → score +1
    "pcr_bear_mild"     : 0.95,     # PCR below → score -1
    "pcr_bear_strong"   : 0.85,     # PCR below → score -2

    # Gap scoring threshold (pts)
    "gap_score_pts"     : 150,

    # Direction threshold for directional trade
    "bullish_threshold" : 2,        # score ≥ this → bull put spread
    "bearish_threshold" : -2,       # score ≤ this → bear call spread

    # VIX half-size threshold
    "vix_max"           : 20.0,

    # Output files
    "trade_csv"         : "logs/smart_hedge_trades.csv",
    "signal_csv"        : "logs/smart_hedge_signals.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE TYPE ENUM
# ─────────────────────────────────────────────────────────────────────────────
IRON_CONDOR  = "IRON_CONDOR"
BULL_PUT     = "BULL_PUT_SPREAD"
BEAR_CALL    = "BEAR_CALL_SPREAD"


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE STATE
# ─────────────────────────────────────────────────────────────────────────────

class SmartHedgeTrade:
    """
    Holds all open position state regardless of trade type.
    Directional trades (BULL_PUT / BEAR_CALL) use only 2 legs.
    Iron Condor uses all 4 legs.
    Unused legs are None.

    P&L:
      net_credit    = sum of sold leg premiums − sum of bought leg premiums
      current_value = live cost to close the position
      pnl           = (net_credit − current_value) × qty
    """
    __slots__ = (
        "trade_type", "entry_time", "score", "score_breakdown",
        # Short legs (sold — we receive premium)
        "short_ce_sym",  "short_ce_token",  "short_ce_entry",  "short_ce_strike",
        "short_pe_sym",  "short_pe_token",  "short_pe_entry",  "short_pe_strike",
        # Long legs (bought — hedge)
        "long_ce_sym",   "long_ce_token",   "long_ce_entry",   "long_ce_strike",
        "long_pe_sym",   "long_pe_token",   "long_pe_entry",   "long_pe_strike",
        # Credit / SL levels
        "net_credit", "profit_target", "sl_level",
        # Live prices — updated on every option tick
        "lp_short_ce", "lp_short_pe",
        "lp_long_ce",  "lp_long_pe",
        # Position size
        "qty",
    )

    def __init__(
        self,
        trade_type: str,
        entry_time: datetime,
        score: int,
        score_breakdown: str,
        qty: int,
        net_credit: float,
        profit_target: float,
        sl_level: float,
        # optional legs (None for unused legs)
        short_ce_sym=None, short_ce_token=None, short_ce_entry=0.0, short_ce_strike=0,
        short_pe_sym=None, short_pe_token=None, short_pe_entry=0.0, short_pe_strike=0,
        long_ce_sym=None,  long_ce_token=None,  long_ce_entry=0.0,  long_ce_strike=0,
        long_pe_sym=None,  long_pe_token=None,  long_pe_entry=0.0,  long_pe_strike=0,
    ):
        self.trade_type      = trade_type
        self.entry_time      = entry_time
        self.score           = score
        self.score_breakdown = score_breakdown
        self.qty             = qty
        self.net_credit      = net_credit
        self.profit_target   = profit_target
        self.sl_level        = sl_level

        self.short_ce_sym    = short_ce_sym
        self.short_ce_token  = short_ce_token
        self.short_ce_entry  = short_ce_entry
        self.short_ce_strike = short_ce_strike

        self.short_pe_sym    = short_pe_sym
        self.short_pe_token  = short_pe_token
        self.short_pe_entry  = short_pe_entry
        self.short_pe_strike = short_pe_strike

        self.long_ce_sym     = long_ce_sym
        self.long_ce_token   = long_ce_token
        self.long_ce_entry   = long_ce_entry
        self.long_ce_strike  = long_ce_strike

        self.long_pe_sym     = long_pe_sym
        self.long_pe_token   = long_pe_token
        self.long_pe_entry   = long_pe_entry
        self.long_pe_strike  = long_pe_strike

        # live prices mirror entry prices at start
        self.lp_short_ce = short_ce_entry
        self.lp_short_pe = short_pe_entry
        self.lp_long_ce  = long_ce_entry
        self.lp_long_pe  = long_pe_entry

    def current_net_value(self) -> float:
        """
        Real-time cost to close the position.
        Sold legs: we pay to close them (positive cost).
        Bought legs: we receive when closing (negative cost).

        For BULL_PUT:  lp_short_pe - lp_long_pe       (CE legs are 0)
        For BEAR_CALL: lp_short_ce - lp_long_ce       (PE legs are 0)
        For CONDOR:    (lp_short_ce + lp_short_pe) - (lp_long_ce + lp_long_pe)
        """
        return (self.lp_short_ce + self.lp_short_pe) - (self.lp_long_ce + self.lp_long_pe)

    def pnl(self) -> float:
        """Net P&L in rupees (positive = profit)."""
        return (self.net_credit - self.current_net_value()) * self.qty

    def active_tokens(self) -> list:
        """Return list of all active (non-None) option tokens."""
        tokens = []
        for tok in (self.short_ce_token, self.short_pe_token,
                    self.long_ce_token,  self.long_pe_token):
            if tok is not None:
                tokens.append(tok)
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SmartHedgeStrategy(BaseStrategy):
    """
    Unified directional hedge strategy for BANKNIFTY.
    Scores market direction at entry time, picks one of three spread structures,
    and manages the trade with detailed logging for analysis.
    """

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "SMART_HEDGE"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._lock   = threading.Lock()
        self._trade: Optional[SmartHedgeTrade] = None

        self._done_today      = False
        self._squareoff_done  = False
        self._close_in_flight = False

        # Premarket data
        self._pm            = None
        self._instruments   = None
        self._expiry_date   = None
        self._prev_close    = None
        self._vix           = None
        self._ema200        = None  # from premarket historical data

        # First candle tracking (for signal 4)
        self._first_candle       = None   # dict with open/close
        self._first_candle_done  = False

        self._subscribed_tokens: set = set()

        log.info("[SMART_HEDGE] Strategy initialised — PAPER MODE | "
                 "Modes: IRON_CONDOR / BULL_PUT_SPREAD / BEAR_CALL_SPREAD")

    # ─────────────────────────────────────────────────────────────────────────
    #  PRE-MARKET
    # ─────────────────────────────────────────────────────────────────────────

    def pre_market(self, premarket_data, instruments) -> bool:
        self._pm          = premarket_data
        self._instruments = instruments
        self._expiry_date = premarket_data.expiry_date
        self._prev_close  = premarket_data.prev_close
        self._vix         = premarket_data.vix

        # Try to get EMA200 from premarket (set by ORB or premarket module)
        self._ema200 = getattr(premarket_data, "ema200_daily", None)

        if self._expiry_date is None:
            log.warning("[SMART_HEDGE] expiry_date not available — skipping today")
            return False

        log.info(
            f"[SMART_HEDGE] pre_market OK | expiry={self._expiry_date} | "
            f"prev_close={self._prev_close} | vix={self._vix} | ema200={self._ema200}"
        )

        # Subscribe likely strikes early so WebSocket warms prices before entry
        if self._prev_close:
            self._subscribe_all_legs(self._prev_close, tag="pre_market")

        self._ensure_csvs()
        return True

    def _subscribe_all_legs(self, spot: float, tag: str = ""):
        """Subscribe all possible legs (all 4 strikes) for warming WebSocket prices."""
        atm  = get_atm_strike(spot)
        legs = [
            (atm + CFG["short_offset"], "CE", "short_CE"),
            (atm - CFG["short_offset"], "PE", "short_PE"),
            (atm + CFG["long_offset"],  "CE", "long_CE"),
            (atm - CFG["long_offset"],  "PE", "long_PE"),
        ]
        for strike, opt_type, label in legs:
            tok, sym = self._instruments.get_option_token(strike, opt_type, self._expiry_date)
            if tok and tok not in self._subscribed_tokens:
                self.subscribe_option(tok)
                self._subscribed_tokens.add(tok)
                log.info(f"[SMART_HEDGE] [{tag}] subscribed {label} {sym} (token={tok})")

    # ─────────────────────────────────────────────────────────────────────────
    #  INDEX TICK — time stop + breach detection
    # ─────────────────────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        # ── Time Stop ──────────────────────────────────────────────────────
        if ts.time() >= CFG["close_time"] and not self._squareoff_done:
            with self._lock:
                if self._trade is not None and not self._squareoff_done:
                    log.info(
                        f"[SMART_HEDGE] ⏰ TIME STOP | {ts.strftime('%H:%M:%S')} | "
                        f"spot={price:.0f} | pnl=₹{self._trade.pnl():.0f}"
                    )
                    self._initiate_close("TIME_STOP", ts)
            return

        # ── Strike Breach ──────────────────────────────────────────────────
        with self._lock:
            trade = self._trade
            if trade is None or self._squareoff_done:
                return

            if trade.trade_type in (IRON_CONDOR, BEAR_CALL):
                if trade.short_ce_token and price >= trade.short_ce_strike:
                    log.warning(
                        f"[SMART_HEDGE] 🚨 BREACH CE | spot={price:.0f} ≥ "
                        f"short_CE={trade.short_ce_strike} | type={trade.trade_type}"
                    )
                    self._initiate_close("BREACH_CE", ts)
                    return

            if trade.trade_type in (IRON_CONDOR, BULL_PUT):
                if trade.short_pe_token and price <= trade.short_pe_strike:
                    log.warning(
                        f"[SMART_HEDGE] 🚨 BREACH PE | spot={price:.0f} ≤ "
                        f"short_PE={trade.short_pe_strike} | type={trade.trade_type}"
                    )
                    self._initiate_close("BREACH_PE", ts)
                    return

        # ── Gap refresh subscription ────────────────────────────────────────
        if ts.time() >= dtime(9, 20) and len(self._subscribed_tokens) < 4:
            self._subscribe_all_legs(price, tag="gap_refresh")

    def _initiate_close(self, reason: str, ts: datetime):
        """Must be called while self._lock is held."""
        self._squareoff_done  = True
        self._close_in_flight = True
        threading.Thread(
            target=self._close_trade,
            args=(reason, ts),
            daemon=True,
        ).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  CANDLE CLOSE
    # ─────────────────────────────────────────────────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        t = ts.time()

        # Track first candle (9:15–9:20 candle closes at 9:20) for signal 4
        if not self._first_candle_done and t >= dtime(9, 20) and t < dtime(9, 30):
            with self._lock:
                if not self._first_candle_done:
                    self._first_candle      = candle
                    self._first_candle_done = True
            candle_dir = "GREEN" if candle["close"] > candle["open"] else "RED"
            log.info(
                f"[SMART_HEDGE] First candle recorded | "
                f"open={candle['open']:.0f} close={candle['close']:.0f} → {candle_dir}"
            )

        # Entry window check
        if not (CFG["entry_start"] <= t < CFG["entry_cutoff"]):
            return

        with self._lock:
            if self._trade is not None or self._done_today:
                return

        # Attempt entry
        self._check_entry(candle["close"], ts)

    # ─────────────────────────────────────────────────────────────────────────
    #  ENTRY LOGIC — Score → Trade Type → Place Orders
    # ─────────────────────────────────────────────────────────────────────────

    def _check_entry(self, spot: float, ts: datetime):
        """
        Score market direction, pick trade type, validate entry conditions,
        and place orders. All decisions are logged for post-market review.
        """
        log.info(
            f"[SMART_HEDGE] ─── Entry Check | {ts.strftime('%H:%M:%S')} | "
            f"spot={spot:.0f} ───"
        )

        # ── Score market direction ─────────────────────────────────────────
        score        = 0
        score_parts  = []

        # Signal 1 — PCR
        pcr = getattr(self._pm, "pcr", None)
        if pcr is None:
            log.info("[SMART_HEDGE]   Signal-1 PCR = None (WebSocket OI not warm yet) → 0")
            score_parts.append("PCR=None(0)")
        elif pcr >= CFG["pcr_bull_strong"]:
            score += 2
            score_parts.append(f"PCR={pcr:.2f}(+2)")
            log.info(f"[SMART_HEDGE]   Signal-1 PCR={pcr:.2f} ≥ {CFG['pcr_bull_strong']} → STRONG BULLISH +2")
        elif pcr >= CFG["pcr_bull_mild"]:
            score += 1
            score_parts.append(f"PCR={pcr:.2f}(+1)")
            log.info(f"[SMART_HEDGE]   Signal-1 PCR={pcr:.2f} ≥ {CFG['pcr_bull_mild']} → mild bullish +1")
        elif pcr <= CFG["pcr_bear_strong"]:
            score -= 2
            score_parts.append(f"PCR={pcr:.2f}(-2)")
            log.info(f"[SMART_HEDGE]   Signal-1 PCR={pcr:.2f} ≤ {CFG['pcr_bear_strong']} → STRONG BEARISH -2")
        elif pcr <= CFG["pcr_bear_mild"]:
            score -= 1
            score_parts.append(f"PCR={pcr:.2f}(-1)")
            log.info(f"[SMART_HEDGE]   Signal-1 PCR={pcr:.2f} ≤ {CFG['pcr_bear_mild']} → mild bearish -1")
        else:
            score_parts.append(f"PCR={pcr:.2f}(0)")
            log.info(f"[SMART_HEDGE]   Signal-1 PCR={pcr:.2f} → neutral 0")

        # Signal 2 — EMA200
        ema200 = self._ema200
        if ema200 is None:
            log.info("[SMART_HEDGE]   Signal-2 EMA200 = None (not available) → 0")
            score_parts.append("EMA=None(0)")
        elif spot > ema200:
            score += 1
            score_parts.append(f"EMA={ema200:.0f}(+1)")
            log.info(f"[SMART_HEDGE]   Signal-2 spot={spot:.0f} > EMA200={ema200:.0f} → bullish +1")
        else:
            score -= 1
            score_parts.append(f"EMA={ema200:.0f}(-1)")
            log.info(f"[SMART_HEDGE]   Signal-2 spot={spot:.0f} < EMA200={ema200:.0f} → bearish -1")

        # Signal 3 — Gap
        gap = 0.0
        if self._prev_close:
            gap = spot - self._prev_close
            if gap > CFG["gap_score_pts"]:
                score += 1
                score_parts.append(f"GAP=+{gap:.0f}(+1)")
                log.info(f"[SMART_HEDGE]   Signal-3 gap={gap:+.0f}pts → gap-up bullish +1")
            elif gap < -CFG["gap_score_pts"]:
                score -= 1
                score_parts.append(f"GAP={gap:.0f}(-1)")
                log.info(f"[SMART_HEDGE]   Signal-3 gap={gap:+.0f}pts → gap-down bearish -1")
            else:
                score_parts.append(f"GAP={gap:+.0f}(0)")
                log.info(f"[SMART_HEDGE]   Signal-3 gap={gap:+.0f}pts → flat open 0")
        else:
            score_parts.append("GAP=None(0)")
            log.info("[SMART_HEDGE]   Signal-3 gap = None (prev_close unavailable) → 0")

        # Signal 4 — First candle direction
        if self._first_candle is None:
            log.info("[SMART_HEDGE]   Signal-4 first candle = not ready → 0")
            score_parts.append("CANDLE=None(0)")
        elif self._first_candle["close"] > self._first_candle["open"]:
            score += 1
            score_parts.append("CANDLE=GREEN(+1)")
            log.info(
                f"[SMART_HEDGE]   Signal-4 first candle GREEN "
                f"({self._first_candle['open']:.0f}→{self._first_candle['close']:.0f}) → bullish +1"
            )
        else:
            score -= 1
            score_parts.append("CANDLE=RED(-1)")
            log.info(
                f"[SMART_HEDGE]   Signal-4 first candle RED "
                f"({self._first_candle['open']:.0f}→{self._first_candle['close']:.0f}) → bearish -1"
            )

        score_breakdown = " | ".join(score_parts)
        log.info(f"[SMART_HEDGE]   ── Total score = {score:+d} | Breakdown: {score_breakdown}")

        # ── Determine trade type ───────────────────────────────────────────
        if score >= CFG["bullish_threshold"]:
            trade_type = BULL_PUT
            log.info(
                f"[SMART_HEDGE]   📈 BULLISH (score={score:+d} ≥ +{CFG['bullish_threshold']}) "
                f"→ BULL PUT SPREAD selected"
            )
        elif score <= CFG["bearish_threshold"]:
            trade_type = BEAR_CALL
            log.info(
                f"[SMART_HEDGE]   📉 BEARISH (score={score:+d} ≤ {CFG['bearish_threshold']}) "
                f"→ BEAR CALL SPREAD selected"
            )
        else:
            trade_type = IRON_CONDOR
            log.info(
                f"[SMART_HEDGE]   ↔️  NEUTRAL (score={score:+d}) → IRON CONDOR selected"
            )

        # ── Gap guard for Iron Condor only ─────────────────────────────────
        if trade_type == IRON_CONDOR and self._prev_close:
            abs_gap = abs(gap)
            if abs_gap > CFG["max_gap_pts"]:
                log.info(
                    f"[SMART_HEDGE]   SKIP | Iron Condor on gap day is risky | "
                    f"|gap|={abs_gap:.0f} > {CFG['max_gap_pts']}pts | "
                    f"Consider going directional if this recurs often"
                )
                self._log_signal_audit(
                    ts, spot, score, score_breakdown, trade_type,
                    reason_skipped=f"GAP_TOO_LARGE_FOR_CONDOR({abs_gap:.0f}pts)"
                )
                with self._lock:
                    self._done_today = True
                return

        # ── VIX check (half-size, not skip) ───────────────────────────────
        vix = getattr(self._pm, "vix", None) or self._vix
        qty = CFG["quantity"]
        if vix and vix > CFG["vix_max"]:
            qty = max(round(CFG["quantity"] / 2 / 30) * 30, 30)
            log.info(
                f"[SMART_HEDGE]   VIX={vix:.1f} > {CFG['vix_max']} → half-size qty={qty} "
                f"(IV is rich, still trade)"
            )
        else:
            log.info(
                f"[SMART_HEDGE]   VIX={f'{vix:.1f}' if vix else 'N/A'} ≤ {CFG['vix_max']} → "
                f"full size qty={qty}"
            )

        # ── Get strikes ────────────────────────────────────────────────────
        atm         = get_atm_strike(spot)
        short_c_str = atm + CFG["short_offset"]
        short_p_str = atm - CFG["short_offset"]
        long_c_str  = atm + CFG["long_offset"]
        long_p_str  = atm - CFG["long_offset"]

        log.info(
            f"[SMART_HEDGE]   ATM={atm} | short offsets: CE={short_c_str} PE={short_p_str} | "
            f"hedge offsets: CE={long_c_str} PE={long_p_str}"
        )

        # ── Token lookup (only for legs we need) ──────────────────────────
        sc_tok = sc_sym = sc_ltp = None
        sp_tok = sp_sym = sp_ltp = None
        lc_tok = lc_sym = lc_ltp = None
        lp_tok = lp_sym = lp_ltp = None

        need_ce = (trade_type in (IRON_CONDOR, BEAR_CALL))
        need_pe = (trade_type in (IRON_CONDOR, BULL_PUT))

        if need_ce:
            sc_tok, sc_sym = self._instruments.get_option_token(short_c_str, "CE", self._expiry_date)
            lc_tok, lc_sym = self._instruments.get_option_token(long_c_str,  "CE", self._expiry_date)
            if None in (sc_tok, lc_tok):
                log.error(
                    f"[SMART_HEDGE]   SKIP | CE token lookup failed | "
                    f"short_CE tok={sc_tok} long_CE tok={lc_tok}"
                )
                self._log_signal_audit(
                    ts, spot, score, score_breakdown, trade_type,
                    reason_skipped="CE_TOKEN_LOOKUP_FAILED"
                )
                with self._lock:
                    self._done_today = True
                return

        if need_pe:
            sp_tok, sp_sym = self._instruments.get_option_token(short_p_str, "PE", self._expiry_date)
            lp_tok, lp_sym = self._instruments.get_option_token(long_p_str,  "PE", self._expiry_date)
            if None in (sp_tok, lp_tok):
                log.error(
                    f"[SMART_HEDGE]   SKIP | PE token lookup failed | "
                    f"short_PE tok={sp_tok} long_PE tok={lp_tok}"
                )
                self._log_signal_audit(
                    ts, spot, score, score_breakdown, trade_type,
                    reason_skipped="PE_TOKEN_LOOKUP_FAILED"
                )
                with self._lock:
                    self._done_today = True
                return

        # ── Late-subscribe any un-warmed tokens ───────────────────────────
        for tok, sym in filter(lambda x: x[0] is not None, [
            (sc_tok, sc_sym), (lc_tok, lc_sym),
            (sp_tok, sp_sym), (lp_tok, lp_sym),
        ]):
            if tok not in self._subscribed_tokens:
                self.subscribe_option(tok)
                self._subscribed_tokens.add(tok)
                log.warning(f"[SMART_HEDGE]   Late-subscribed {sym} ({tok}) — price may not be warm")

        # ── Get live prices ────────────────────────────────────────────────
        if need_ce:
            sc_ltp = self.get_price(sc_tok)
            lc_ltp = self.get_price(lc_tok)

        if need_pe:
            sp_ltp = self.get_price(sp_tok)
            lp_ltp = self.get_price(lp_tok)

        prices_needed = []
        if need_ce: prices_needed += [("short_CE", sc_ltp), ("long_CE", lc_ltp)]
        if need_pe: prices_needed += [("short_PE", sp_ltp), ("long_PE", lp_ltp)]

        missing_prices = [label for label, p in prices_needed if p is None]
        if missing_prices:
            log.warning(
                f"[SMART_HEDGE]   LTP not ready for: {missing_prices} | "
                f"will retry next candle (prices warm up after a few ticks)"
            )
            # Do NOT set _done_today — retry next candle in window
            return

        # ── Compute net credit ─────────────────────────────────────────────
        sold_premium   = (sc_ltp or 0.0) + (sp_ltp or 0.0)
        bought_premium = (lc_ltp or 0.0) + (lp_ltp or 0.0)
        net_credit     = sold_premium - bought_premium

        log.info(
            f"[SMART_HEDGE]   Premiums | "
            + (f"sell_CE={sc_ltp:.1f} buy_CE={lc_ltp:.1f} " if need_ce else "")
            + (f"sell_PE={sp_ltp:.1f} buy_PE={lp_ltp:.1f} " if need_pe else "")
            + f"| net_credit={net_credit:.2f}"
        )

        if net_credit <= 0:
            log.warning(
                f"[SMART_HEDGE]   SKIP | net_credit={net_credit:.2f} ≤ 0 | "
                f"bid-ask too wide or legs mis-priced | skipping today"
            )
            self._log_signal_audit(
                ts, spot, score, score_breakdown, trade_type,
                reason_skipped=f"ZERO_CREDIT({net_credit:.2f})"
            )
            with self._lock:
                self._done_today = True
            return

        spread_width  = CFG["long_offset"] - CFG["short_offset"]
        profit_target = net_credit * (1.0 - CFG["profit_target_pct"])
        sl_level      = net_credit * CFG["sl_multiplier"]
        max_profit    = net_credit * qty
        max_loss      = (spread_width - net_credit) * qty

        log.info(
            f"[SMART_HEDGE]   ENTRY VALID ✓ | type={trade_type} | "
            f"net_credit={net_credit:.2f} | "
            f"profit_target≤{profit_target:.2f} | SL≥{sl_level:.2f} | "
            f"max_profit=₹{max_profit:.0f} | max_loss=₹{max_loss:.0f}"
        )

        # ── Acquire slot ───────────────────────────────────────────────────
        if not self._acquire_slot():
            log.info("[SMART_HEDGE]   SLOT BLOCKED — another live strategy has active position")
            return

        # ── Place orders ───────────────────────────────────────────────────
        if trade_type == BULL_PUT:
            oid_sp = self._place_sell(sp_sym, sp_tok, qty, sp_ltp)
            oid_lp = self._place_buy( lp_sym, lp_tok, qty, lp_ltp)
            if None in (oid_sp, oid_lp):
                log.error("[SMART_HEDGE]   BULL_PUT order FAILED — rolling back placed legs")
                if oid_sp: self._place_buy( sp_sym, sp_tok, qty, self.get_price(sp_tok) or sp_ltp)
                if oid_lp: self._place_sell(lp_sym, lp_tok, qty, self.get_price(lp_tok) or lp_ltp)
                self._release_slot()
                with self._lock:
                    self._done_today = True
                return

        elif trade_type == BEAR_CALL:
            oid_sc = self._place_sell(sc_sym, sc_tok, qty, sc_ltp)
            oid_lc = self._place_buy( lc_sym, lc_tok, qty, lc_ltp)
            if None in (oid_sc, oid_lc):
                log.error("[SMART_HEDGE]   BEAR_CALL order FAILED — rolling back placed legs")
                if oid_sc: self._place_buy( sc_sym, sc_tok, qty, self.get_price(sc_tok) or sc_ltp)
                if oid_lc: self._place_sell(lc_sym, lc_tok, qty, self.get_price(lc_tok) or lc_ltp)
                self._release_slot()
                with self._lock:
                    self._done_today = True
                return

        else:  # IRON_CONDOR
            oid_sc = self._place_sell(sc_sym, sc_tok, qty, sc_ltp)
            oid_sp = self._place_sell(sp_sym, sp_tok, qty, sp_ltp)
            oid_lc = self._place_buy( lc_sym, lc_tok, qty, lc_ltp)
            oid_lp = self._place_buy( lp_sym, lp_tok, qty, lp_ltp)
            if None in (oid_sc, oid_sp, oid_lc, oid_lp):
                log.error("[SMART_HEDGE]   IRON_CONDOR order FAILED — rolling back placed legs")
                if oid_sc: self._place_buy( sc_sym, sc_tok, qty, self.get_price(sc_tok) or sc_ltp)
                if oid_sp: self._place_buy( sp_sym, sp_tok, qty, self.get_price(sp_tok) or sp_ltp)
                if oid_lc: self._place_sell(lc_sym, lc_tok, qty, self.get_price(lc_tok) or lc_ltp)
                if oid_lp: self._place_sell(lp_sym, lp_tok, qty, self.get_price(lp_tok) or lp_ltp)
                self._release_slot()
                with self._lock:
                    self._done_today = True
                return

        # ── Build trade object ─────────────────────────────────────────────
        trade = SmartHedgeTrade(
            trade_type      = trade_type,
            entry_time      = ts,
            score           = score,
            score_breakdown = score_breakdown,
            qty             = qty,
            net_credit      = net_credit,
            profit_target   = profit_target,
            sl_level        = sl_level,
            short_ce_sym    = sc_sym,   short_ce_token  = sc_tok,
            short_ce_entry  = sc_ltp or 0.0, short_ce_strike = short_c_str if need_ce else 0,
            short_pe_sym    = sp_sym,   short_pe_token  = sp_tok,
            short_pe_entry  = sp_ltp or 0.0, short_pe_strike = short_p_str if need_pe else 0,
            long_ce_sym     = lc_sym,   long_ce_token   = lc_tok,
            long_ce_entry   = lc_ltp or 0.0, long_ce_strike  = long_c_str if need_ce else 0,
            long_pe_sym     = lp_sym,   long_pe_token   = lp_tok,
            long_pe_entry   = lp_ltp or 0.0, long_pe_strike  = long_p_str if need_pe else 0,
        )

        with self._lock:
            self._trade      = trade
            self._done_today = True

        # Log the successful signal too (with ENTERED status)
        self._log_signal_audit(
            ts, spot, score, score_breakdown, trade_type,
            reason_skipped="ENTERED"
        )

        if trade_type == BULL_PUT:
            leg_str = (
                f"sell_PE {short_p_str}@{sp_ltp:.1f} | "
                f"buy_PE  {long_p_str}@{lp_ltp:.1f}"
            )
        elif trade_type == BEAR_CALL:
            leg_str = (
                f"sell_CE {short_c_str}@{sc_ltp:.1f} | "
                f"buy_CE  {long_c_str}@{lc_ltp:.1f}"
            )
        else:
            leg_str = (
                f"sell_CE {short_c_str}@{sc_ltp:.1f} | "
                f"sell_PE {short_p_str}@{sp_ltp:.1f} | "
                f"buy_CE  {long_c_str}@{lc_ltp:.1f} | "
                f"buy_PE  {long_p_str}@{lp_ltp:.1f}"
            )

        log.info(
            f"[SMART_HEDGE] ✅ POSITION OPEN | {trade_type} | score={score:+d} | "
            f"{leg_str} | net_credit={net_credit:.2f} | qty={qty} | "
            f"target≤{profit_target:.2f} | SL≥{sl_level:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  OPTION TICK — update live prices, check exit
    # ─────────────────────────────────────────────────────────────────────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        with self._lock:
            trade = self._trade
            if trade is None:
                return

            updated = False
            if   token == trade.short_ce_token and trade.short_ce_token is not None:
                trade.lp_short_ce = price; updated = True
            elif token == trade.short_pe_token and trade.short_pe_token is not None:
                trade.lp_short_pe = price; updated = True
            elif token == trade.long_ce_token  and trade.long_ce_token  is not None:
                trade.lp_long_ce  = price; updated = True
            elif token == trade.long_pe_token  and trade.long_pe_token  is not None:
                trade.lp_long_pe  = price; updated = True

            if updated:
                self._check_exit_locked(trade, ts)

    def _check_exit_locked(self, trade: SmartHedgeTrade, ts: datetime):
        """Must be called while self._lock is held."""
        if self._squareoff_done or self._close_in_flight:
            return

        cv  = trade.current_net_value()
        pnl = trade.pnl()

        if cv <= trade.profit_target:
            log.info(
                f"[SMART_HEDGE] 🎯 TARGET HIT | type={trade.trade_type} | "
                f"current_value={cv:.2f} ≤ target={trade.profit_target:.2f} | "
                f"pnl=₹{pnl:.0f}"
            )
            self._initiate_close("TARGET", ts)
            return

        if cv >= trade.sl_level:
            log.warning(
                f"[SMART_HEDGE] 🛑 SL HIT | type={trade.trade_type} | "
                f"current_value={cv:.2f} ≥ sl={trade.sl_level:.2f} | "
                f"loss=₹{pnl:.0f}"
            )
            self._initiate_close("SL", ts)

    # ─────────────────────────────────────────────────────────────────────────
    #  CLOSE TRADE (daemon thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _close_trade(self, reason: str, ts: datetime):
        """
        Exit all active legs.
        Order: buy back short legs first (kill gamma risk), then sell hedge legs.
        """
        with self._lock:
            trade = self._trade
            if trade is None:
                self._close_in_flight = False
                return
            self._trade = None

        exit_ts = _now_ist()

        # Snapshot exit prices
        sc_exit = (self.get_price(trade.short_ce_token) or trade.lp_short_ce) if trade.short_ce_token else 0.0
        sp_exit = (self.get_price(trade.short_pe_token) or trade.lp_short_pe) if trade.short_pe_token else 0.0
        lc_exit = (self.get_price(trade.long_ce_token)  or trade.lp_long_ce)  if trade.long_ce_token  else 0.0
        lp_exit = (self.get_price(trade.long_pe_token)  or trade.lp_long_pe)  if trade.long_pe_token  else 0.0

        log.info(
            f"[SMART_HEDGE] CLOSING | reason={reason} | type={trade.trade_type} | "
            + (f"sc={sc_exit:.2f} " if trade.short_ce_token else "")
            + (f"sp={sp_exit:.2f} " if trade.short_pe_token else "")
            + (f"lc={lc_exit:.2f} " if trade.long_ce_token  else "")
            + (f"lp={lp_exit:.2f} " if trade.long_pe_token  else "")
        )

        # Buy back short legs first (priority — these have unlimited/large risk)
        if trade.short_ce_token:
            self._place_buy(trade.short_ce_sym, trade.short_ce_token, trade.qty, sc_exit)
        if trade.short_pe_token:
            self._place_buy(trade.short_pe_sym, trade.short_pe_token, trade.qty, sp_exit)

        # Sell hedge legs
        if trade.long_ce_token:
            self._place_sell(trade.long_ce_sym, trade.long_ce_token, trade.qty, lc_exit)
        if trade.long_pe_token:
            self._place_sell(trade.long_pe_sym, trade.long_pe_token, trade.qty, lp_exit)

        self._release_slot()

        # Unsubscribe
        for tok in trade.active_tokens():
            self.unsubscribe_option(tok)
            self._subscribed_tokens.discard(tok)

        # Final P&L
        exit_net  = (sc_exit + sp_exit) - (lc_exit + lp_exit)
        total_pnl = (trade.net_credit - exit_net) * trade.qty
        outcome   = "WIN" if total_pnl > 0 else ("LOSS" if total_pnl < 0 else "BE")
        duration  = (exit_ts - trade.entry_time).seconds // 60

        log.info(
            f"[SMART_HEDGE] {'✅' if outcome == 'WIN' else '❌'} CLOSED | "
            f"{outcome} | reason={reason} | type={trade.trade_type} | "
            f"entry_credit={trade.net_credit:.2f} | exit_value={exit_net:.2f} | "
            f"pnl=₹{total_pnl:.0f} | qty={trade.qty} | duration={duration}min | "
            f"score={trade.score:+d} ({trade.score_breakdown})"
        )

        self._log_trade_csv(
            trade, reason, sc_exit, sp_exit, lc_exit, lp_exit,
            exit_net, total_pnl, outcome, exit_ts
        )

        with self._lock:
            self._close_in_flight = False

    # ─────────────────────────────────────────────────────────────────────────
    #  EOD
    # ─────────────────────────────────────────────────────────────────────────

    def eod_summary(self):
        with self._lock:
            trade = self._trade

        if trade is not None:
            log.warning("[SMART_HEDGE] EOD: position still open — forcing close")
            with self._lock:
                if not self._squareoff_done:
                    self._initiate_close("EOD_FORCE", _now_ist())
        else:
            log.info(
                f"[SMART_HEDGE] EOD | done_today={self._done_today} | "
                f"no open position | logs → {CFG['trade_csv']}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_csvs(self):
        os.makedirs("logs", exist_ok=True)

        # Trade CSV — one row per completed trade
        if not os.path.isfile(CFG["trade_csv"]):
            with open(CFG["trade_csv"], "w", newline="") as f:
                csv.writer(f).writerow([
                    "date", "entry_time", "exit_time",
                    "trade_type", "score", "score_breakdown",
                    "atm_approx", "short_ce_strike", "short_pe_strike",
                    "long_ce_strike", "long_pe_strike",
                    "spread_width_pts",
                    "short_ce_entry", "short_pe_entry",
                    "long_ce_entry",  "long_pe_entry",
                    "net_credit", "profit_target", "sl_level",
                    "short_ce_exit", "short_pe_exit",
                    "long_ce_exit",  "long_pe_exit",
                    "exit_net_value", "pnl_per_unit", "qty", "total_pnl",
                    "outcome", "exit_reason", "duration_min",
                    "vix", "pcr",
                ])
            log.info(f"[SMART_HEDGE] Created trade CSV: {CFG['trade_csv']}")

        # Signal audit CSV — one row per entry attempt (including skipped)
        if not os.path.isfile(CFG["signal_csv"]):
            with open(CFG["signal_csv"], "w", newline="") as f:
                csv.writer(f).writerow([
                    "date", "time", "spot",
                    "score", "score_breakdown", "trade_type_selected",
                    "result",   # ENTERED / reason why skipped
                    "pcr", "ema200", "vix", "gap",
                ])
            log.info(f"[SMART_HEDGE] Created signal audit CSV: {CFG['signal_csv']}")

    def _log_trade_csv(
        self, trade: SmartHedgeTrade, reason: str,
        sc_exit, sp_exit, lc_exit, lp_exit,
        exit_net, total_pnl, outcome, exit_ts,
    ):
        spread_width = CFG["long_offset"] - CFG["short_offset"]
        pnl_per_unit = trade.net_credit - exit_net
        duration     = (exit_ts - trade.entry_time).seconds // 60
        atm_approx   = (
            trade.short_ce_strike - CFG["short_offset"]
            if trade.short_ce_strike else
            trade.short_pe_strike + CFG["short_offset"]
        )
        pcr  = getattr(self._pm, "pcr",  "N/A")
        vix  = getattr(self._pm, "vix",  "N/A") or self._vix

        with open(CFG["trade_csv"], "a", newline="") as f:
            csv.writer(f).writerow([
                exit_ts.strftime("%Y-%m-%d"),
                trade.entry_time.strftime("%H:%M:%S"),
                exit_ts.strftime("%H:%M:%S"),
                trade.trade_type,
                trade.score,
                trade.score_breakdown,
                atm_approx,
                trade.short_ce_strike or "",
                trade.short_pe_strike or "",
                trade.long_ce_strike  or "",
                trade.long_pe_strike  or "",
                spread_width,
                f"{trade.short_ce_entry:.2f}" if trade.short_ce_token else "",
                f"{trade.short_pe_entry:.2f}" if trade.short_pe_token else "",
                f"{trade.long_ce_entry:.2f}"  if trade.long_ce_token  else "",
                f"{trade.long_pe_entry:.2f}"  if trade.long_pe_token  else "",
                f"{trade.net_credit:.2f}",
                f"{trade.profit_target:.2f}",
                f"{trade.sl_level:.2f}",
                f"{sc_exit:.2f}" if trade.short_ce_token else "",
                f"{sp_exit:.2f}" if trade.short_pe_token else "",
                f"{lc_exit:.2f}" if trade.long_ce_token  else "",
                f"{lp_exit:.2f}" if trade.long_pe_token  else "",
                f"{exit_net:.2f}",
                f"{pnl_per_unit:.2f}",
                trade.qty,
                f"{total_pnl:.0f}",
                outcome,
                reason,
                duration,
                f"{vix:.1f}" if isinstance(vix, float) else vix,
                f"{pcr:.2f}" if isinstance(pcr, float) else pcr,
            ])

    def _log_signal_audit(
        self, ts: datetime, spot: float,
        score: int, score_breakdown: str,
        trade_type: str, reason_skipped: str,
    ):
        pcr    = getattr(self._pm, "pcr",  "N/A")
        vix    = getattr(self._pm, "vix",  "N/A") or self._vix
        ema200 = self._ema200 or "N/A"
        gap    = (spot - self._prev_close) if self._prev_close else "N/A"

        with open(CFG["signal_csv"], "a", newline="") as f:
            csv.writer(f).writerow([
                ts.strftime("%Y-%m-%d"),
                ts.strftime("%H:%M:%S"),
                f"{spot:.0f}",
                score,
                score_breakdown,
                trade_type,
                reason_skipped,
                f"{pcr:.2f}"    if isinstance(pcr,    float) else pcr,
                f"{ema200:.0f}" if isinstance(ema200, float) else ema200,
                f"{vix:.1f}"    if isinstance(vix,    float) else vix,
                f"{gap:.0f}"    if isinstance(gap,    float) else gap,
            ])


