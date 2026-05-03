"""
strategies/hedged_sell_strategy.py

HEDGED OPTION SELLING — Iron Condor on BANKNIFTY
═══════════════════════════════════════════════════════════════════

STRUCTURE (Iron Condor — 4 legs)
─────────────────────────────────
  SELL OTM Call  (ATM + SHORT_OFFSET)   ← collect premium
  BUY  OTM Call  (ATM + LONG_OFFSET)    ← hedge (limits max loss)
  SELL OTM Put   (ATM - SHORT_OFFSET)   ← collect premium
  BUY  OTM Put   (ATM - LONG_OFFSET)    ← hedge (limits max loss)

  Net Credit = (short_CE + short_PE) − (long_CE + long_PE)
  Max Profit = net_credit × qty          (if BankNifty stays between short strikes)
  Max Loss   = (spread_width − net_credit) × qty

ENTRY FILTERS
─────────────
  1. Time window     : 9:30 – 10:15 AM only
  2. Gap filter      : skip if BankNifty gapped > 400 pts (directional day)
  3. PCR filter      : skip if PCR < 0.70 or > 1.40 (extreme OI skew)
  4. VIX filter      : VIX > 20 → reduce to half size (still trade, just smaller)
  5. Net credit > 0  : spread must actually yield a credit (sanity check)
  6. One trade/day   : once entered, no re-entry on same day

EXIT / STOP LOSS
─────────────────
  TARGET  : Exit when current net value ≤ net_credit × 0.50
            (you keep 50% of the credit → profit)
  SL      : Exit when current net value ≥ net_credit × 2.0
            (position has doubled against you → take the loss)
  STRIKE  : Force exit when spot crosses either short strike
            (underlying has moved too far into your sold zone)
  TIME    : Force exit at 3:10 PM regardless of P&L
  EOD     : Hard close at eod_summary() if somehow still open

LIVE / PAPER MODE
──────────────────
  LIVE_MODE = False → all trades are paper-simulated, no real orders placed.
  Set LIVE_MODE = True only when ready to go live with real money.
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

log = logging.getLogger("strategy.hedged_sell")

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return datetime.now(tz=_IST).replace(tzinfo=None)


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE FLAG — False = paper simulation only (no real orders)
# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Lot size
    "quantity"          : 30,        # BANKNIFTY lot size (multiples of 30)

    # Strike selection
    # short_offset = pts OTM from ATM for the SOLD legs  (e.g. ATM ± 200)
    # long_offset  = pts OTM from ATM for the HEDGE legs (e.g. ATM ± 500)
    # Spread width = long_offset - short_offset = 300 pts per side
    "short_offset"      : 200,
    "long_offset"       : 500,

    # Entry window (candle close triggers entry check)
    "entry_start"       : dtime(9, 30),
    "entry_cutoff"      : dtime(10, 15),

    # Forced exit time
    "close_time"        : dtime(15, 10),

    # Profit target: exit when current position value = (1 - pct) × net_credit
    # 0.50 → keep 50% of credit (exit when position decays to half its value)
    "profit_target_pct" : 0.50,

    # SL multiplier: exit when current value = sl_mult × net_credit
    # 2.0 → SL fires when the position is worth 2× what we collected
    "sl_multiplier"     : 2.0,

    # Gap filter: skip if BankNifty gap from prev_close exceeds this
    "max_gap_pts"       : 400,

    # PCR filter: extreme PCR means extreme OI skew (one-sided market risk)
    "pcr_min"           : 0.70,
    "pcr_max"           : 1.40,

    # VIX filter: above this, reduce size to half (not skip — IV is rich)
    "vix_max"           : 20.0,

    # Log file
    "csv_file"          : "hedged_sell_trades.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE STATE CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

class HedgedSellTrade:
    """
    Holds all state for one open Iron Condor position.

    P&L tracking:
      net_credit    = (short_ce_entry + short_pe_entry) − (long_ce_entry + long_pe_entry)
      current_value = (lp_short_ce + lp_short_pe) − (lp_long_ce + lp_long_pe)
      pnl           = (net_credit − current_value) × qty
        → pnl > 0 when position decays (theta working for us)
        → pnl < 0 when market moves against us
    """
    __slots__ = (
        "entry_time",
        # Short legs (sold — receive premium)
        "short_ce_sym",   "short_ce_token",  "short_ce_entry",  "short_ce_strike",
        "short_pe_sym",   "short_pe_token",  "short_pe_entry",  "short_pe_strike",
        # Long legs (bought — pay premium, hedge)
        "long_ce_sym",    "long_ce_token",   "long_ce_entry",   "long_ce_strike",
        "long_pe_sym",    "long_pe_token",   "long_pe_entry",   "long_pe_strike",
        # Credit / SL levels
        "net_credit",
        "profit_target",
        "sl_level",
        # Live option prices (updated on every tick)
        "lp_short_ce", "lp_short_pe",
        "lp_long_ce",  "lp_long_pe",
        # Position size
        "qty",
        # Order IDs
        "oid_sell_ce", "oid_sell_pe",
        "oid_buy_ce",  "oid_buy_pe",
    )

    def __init__(
        self,
        entry_time,
        short_ce_sym, short_ce_token, short_ce_entry, short_ce_strike,
        short_pe_sym, short_pe_token, short_pe_entry, short_pe_strike,
        long_ce_sym,  long_ce_token,  long_ce_entry,  long_ce_strike,
        long_pe_sym,  long_pe_token,  long_pe_entry,  long_pe_strike,
        net_credit, profit_target, sl_level, qty,
    ):
        self.entry_time = entry_time
        self.short_ce_sym    = short_ce_sym;   self.short_ce_token  = short_ce_token
        self.short_ce_entry  = short_ce_entry; self.short_ce_strike = short_ce_strike
        self.short_pe_sym    = short_pe_sym;   self.short_pe_token  = short_pe_token
        self.short_pe_entry  = short_pe_entry; self.short_pe_strike = short_pe_strike
        self.long_ce_sym     = long_ce_sym;    self.long_ce_token   = long_ce_token
        self.long_ce_entry   = long_ce_entry;  self.long_ce_strike  = long_ce_strike
        self.long_pe_sym     = long_pe_sym;    self.long_pe_token   = long_pe_token
        self.long_pe_entry   = long_pe_entry;  self.long_pe_strike  = long_pe_strike
        self.net_credit    = net_credit
        self.profit_target = profit_target
        self.sl_level      = sl_level
        self.lp_short_ce   = short_ce_entry
        self.lp_short_pe   = short_pe_entry
        self.lp_long_ce    = long_ce_entry
        self.lp_long_pe    = long_pe_entry
        self.qty           = qty
        self.oid_sell_ce   = ""
        self.oid_sell_pe   = ""
        self.oid_buy_ce    = ""
        self.oid_buy_pe    = ""

    def current_net_value(self) -> float:
        """
        Real-time cost to close the position.
        Falls as theta decays (profit grows). Rises as market moves against us.
        """
        return (
            (self.lp_short_ce + self.lp_short_pe)
            - (self.lp_long_ce + self.lp_long_pe)
        )

    def pnl(self) -> float:
        """Net P&L in rupees for the full position (positive = profit)."""
        return (self.net_credit - self.current_net_value()) * self.qty


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY CLASS
# ─────────────────────────────────────────────────────────────────────────────

class HedgedSellStrategy(BaseStrategy):
    """
    Hedged Option Selling (Iron Condor) for BANKNIFTY.
    One trade per day. 4 legs. Premium-based P&L tracking via WebSocket ticks.
    """

    LIVE_MODE = LIVE_MODE

    @property
    def name(self) -> str:
        return "HEDGED_SELL"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._lock   = threading.Lock()
        self._trade: Optional[HedgedSellTrade] = None

        self._done_today      = False
        self._squareoff_done  = False
        self._close_in_flight = False

        self._pm            = None
        self._instruments   = None
        self._expiry_date   = None
        self._prev_close    = None
        self._vix           = None

        self._subscribed_tokens: set = set()

        log.info("[HEDGED_SELL] Strategy initialised (PAPER MODE)")

    # ─────────────────────────────────────────────────────────────────────────
    #  PRE-MARKET
    # ─────────────────────────────────────────────────────────────────────────

    def pre_market(self, premarket_data, instruments) -> bool:
        self._pm          = premarket_data
        self._instruments = instruments
        self._expiry_date = premarket_data.expiry_date
        self._prev_close  = premarket_data.prev_close
        self._vix         = premarket_data.vix

        if self._expiry_date is None:
            log.warning("[HEDGED_SELL] expiry_date unavailable — skipping today")
            return False

        log.info(
            f"[HEDGED_SELL] pre_market OK | expiry={self._expiry_date} | "
            f"prev_close={self._prev_close} | vix={self._vix}"
        )

        if self._prev_close:
            self._subscribe_condor_legs(self._prev_close, tag="pre_market")

        self._ensure_csv()
        return True

    def _subscribe_condor_legs(self, spot: float, tag: str = ""):
        """Subscribe all 4 Iron Condor tokens for the given spot price."""
        atm  = get_atm_strike(spot)
        legs = [
            (atm + CFG["short_offset"], "CE", "short_CE"),
            (atm - CFG["short_offset"], "PE", "short_PE"),
            (atm + CFG["long_offset"],  "CE", "long_CE"),
            (atm - CFG["long_offset"],  "PE", "long_PE"),
        ]
        for strike, opt_type, label in legs:
            token, sym = self._instruments.get_option_token(
                strike, opt_type, self._expiry_date
            )
            if token and token not in self._subscribed_tokens:
                self.subscribe_option(token)
                self._subscribed_tokens.add(token)
                log.info(f"[HEDGED_SELL] [{tag}] subscribed {label} {sym} (token={token})")

    # ─────────────────────────────────────────────────────────────────────────
    #  INDEX TICK
    # ─────────────────────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        # Time stop
        if ts.time() >= CFG["close_time"] and not self._squareoff_done:
            with self._lock:
                if self._trade is not None and not self._squareoff_done:
                    log.info(
                        f"[HEDGED_SELL] TIME STOP | {ts.strftime('%H:%M:%S')} | spot={price:.0f}"
                    )
                    self._squareoff_done  = True
                    self._close_in_flight = True
                    threading.Thread(
                        target=self._close_trade,
                        args=("TIME_STOP", ts),
                        daemon=True,
                    ).start()
            return

        # Short-strike breach
        with self._lock:
            trade = self._trade
            if trade is not None and not self._squareoff_done:
                if price >= trade.short_ce_strike:
                    log.warning(
                        f"[HEDGED_SELL] STRIKE BREACH (CE) | "
                        f"spot={price:.0f} >= short_CE={trade.short_ce_strike}"
                    )
                    self._squareoff_done  = True
                    self._close_in_flight = True
                    threading.Thread(
                        target=self._close_trade,
                        args=("STRIKE_BREACH_CE", ts),
                        daemon=True,
                    ).start()
                elif price <= trade.short_pe_strike:
                    log.warning(
                        f"[HEDGED_SELL] STRIKE BREACH (PE) | "
                        f"spot={price:.0f} <= short_PE={trade.short_pe_strike}"
                    )
                    self._squareoff_done  = True
                    self._close_in_flight = True
                    threading.Thread(
                        target=self._close_trade,
                        args=("STRIKE_BREACH_PE", ts),
                        daemon=True,
                    ).start()

        # Gap-day token refresh (re-subscribe if spot moved far from prev_close)
        if ts.time() >= dtime(9, 20) and len(self._subscribed_tokens) < 4:
            self._subscribe_condor_legs(price, tag="gap_refresh")

    # ─────────────────────────────────────────────────────────────────────────
    #  CANDLE CLOSE
    # ─────────────────────────────────────────────────────────────────────────

    def on_candle(self, candle: dict, ts: datetime):
        t = ts.time()
        if not (CFG["entry_start"] <= t < CFG["entry_cutoff"]):
            return

        with self._lock:
            if self._trade is not None or self._done_today:
                return

        self._check_entry(candle["close"], ts)

    def _check_entry(self, spot: float, ts: datetime):
        # Gap filter
        if self._prev_close:
            gap = abs(spot - self._prev_close)
            if gap > CFG["max_gap_pts"]:
                log.info(
                    f"[HEDGED_SELL] SKIP | gap={gap:.0f}pts > {CFG['max_gap_pts']}pts"
                )
                self._done_today = True
                return

        # PCR filter
        pcr = getattr(self._pm, "pcr", None)
        if pcr is not None:
            if not (CFG["pcr_min"] <= pcr <= CFG["pcr_max"]):
                log.info(
                    f"[HEDGED_SELL] SKIP | PCR={pcr:.2f} outside "
                    f"[{CFG['pcr_min']}, {CFG['pcr_max']}]"
                )
                self._done_today = True
                return

        # VIX filter (size reduction, not skip)
        vix = getattr(self._pm, "vix", None) or self._vix
        qty = CFG["quantity"]
        if vix and vix > CFG["vix_max"]:
            qty = max((CFG["quantity"] // 2 // 30) * 30, 30)
            log.info(f"[HEDGED_SELL] VIX={vix:.1f} > {CFG['vix_max']} — half size qty={qty}")

        # Strikes
        atm         = get_atm_strike(spot)
        short_c_str = atm + CFG["short_offset"]
        short_p_str = atm - CFG["short_offset"]
        long_c_str  = atm + CFG["long_offset"]
        long_p_str  = atm - CFG["long_offset"]

        # Tokens
        sc_tok, sc_sym = self._instruments.get_option_token(short_c_str, "CE", self._expiry_date)
        sp_tok, sp_sym = self._instruments.get_option_token(short_p_str, "PE", self._expiry_date)
        lc_tok, lc_sym = self._instruments.get_option_token(long_c_str,  "CE", self._expiry_date)
        lp_tok, lp_sym = self._instruments.get_option_token(long_p_str,  "PE", self._expiry_date)

        if None in (sc_tok, sp_tok, lc_tok, lp_tok):
            log.error(
                f"[HEDGED_SELL] Token lookup failed — "
                f"sc={sc_tok} sp={sp_tok} lc={lc_tok} lp={lp_tok} | skipping"
            )
            self._done_today = True
            return

        # Subscribe any un-warmed tokens
        for tok, sym in ((sc_tok, sc_sym), (sp_tok, sp_sym),
                          (lc_tok, lc_sym), (lp_tok, lp_sym)):
            if tok not in self._subscribed_tokens:
                self.subscribe_option(tok)
                self._subscribed_tokens.add(tok)
                log.warning(f"[HEDGED_SELL] Late-subscribed {sym} ({tok})")

        # Live prices
        sc_ltp = self.get_price(sc_tok)
        sp_ltp = self.get_price(sp_tok)
        lc_ltp = self.get_price(lc_tok)
        lp_ltp = self.get_price(lp_tok)

        if None in (sc_ltp, sp_ltp, lc_ltp, lp_ltp):
            log.warning(
                f"[HEDGED_SELL] LTP unavailable | "
                f"sc={sc_ltp} sp={sp_ltp} lc={lc_ltp} lp={lp_ltp} | retry next candle"
            )
            return   # do NOT set _done_today — retry next candle in window

        # Net credit sanity
        net_credit = (sc_ltp + sp_ltp) - (lc_ltp + lp_ltp)
        if net_credit <= 0:
            log.warning(
                f"[HEDGED_SELL] net_credit={net_credit:.2f} ≤ 0 | skipping"
            )
            self._done_today = True
            return

        profit_target_val = net_credit * (1.0 - CFG["profit_target_pct"])
        sl_level_val      = net_credit * CFG["sl_multiplier"]
        spread_width      = CFG["long_offset"] - CFG["short_offset"]

        log.info(
            f"[HEDGED_SELL] ENTRY | spot={spot:.0f} ATM={atm} | "
            f"short_CE={short_c_str}@{sc_ltp:.1f} short_PE={short_p_str}@{sp_ltp:.1f} | "
            f"long_CE={long_c_str}@{lc_ltp:.1f} long_PE={long_p_str}@{lp_ltp:.1f} | "
            f"net_credit={net_credit:.2f} target≤{profit_target_val:.2f} SL≥{sl_level_val:.2f} | "
            f"max_profit=₹{net_credit * qty:.0f} max_loss=₹{(spread_width - net_credit) * qty:.0f}"
        )

        if not self._acquire_slot():
            log.info("[HEDGED_SELL] Slot blocked — another LIVE strategy is active")
            return

        # Place all 4 orders (short legs first, then hedges)
        oid_sell_ce = self._place_sell(sc_sym, sc_tok, qty, sc_ltp)
        oid_sell_pe = self._place_sell(sp_sym, sp_tok, qty, sp_ltp)
        oid_buy_ce  = self._place_buy( lc_sym, lc_tok, qty, lc_ltp)
        oid_buy_pe  = self._place_buy( lp_sym, lp_tok, qty, lp_ltp)

        if None in (oid_sell_ce, oid_sell_pe, oid_buy_ce, oid_buy_pe):
            log.error(
                "[HEDGED_SELL] One or more entry orders FAILED — "
                "releasing slot. Check for partial fills!"
            )
            self._release_slot()
            self._done_today = True
            return

        trade = HedgedSellTrade(
            entry_time     = ts,
            short_ce_sym   = sc_sym,   short_ce_token  = sc_tok,
            short_ce_entry = sc_ltp,   short_ce_strike = short_c_str,
            short_pe_sym   = sp_sym,   short_pe_token  = sp_tok,
            short_pe_entry = sp_ltp,   short_pe_strike = short_p_str,
            long_ce_sym    = lc_sym,   long_ce_token   = lc_tok,
            long_ce_entry  = lc_ltp,   long_ce_strike  = long_c_str,
            long_pe_sym    = lp_sym,   long_pe_token   = lp_tok,
            long_pe_entry  = lp_ltp,   long_pe_strike  = long_p_str,
            net_credit     = net_credit,
            profit_target  = profit_target_val,
            sl_level       = sl_level_val,
            qty            = qty,
        )
        trade.oid_sell_ce = oid_sell_ce or ""
        trade.oid_sell_pe = oid_sell_pe or ""
        trade.oid_buy_ce  = oid_buy_ce  or ""
        trade.oid_buy_pe  = oid_buy_pe  or ""

        with self._lock:
            self._trade      = trade
            self._done_today = True

        log.info(
            f"[HEDGED_SELL] POSITION OPEN ✓ | net_credit={net_credit:.2f} | qty={qty} | "
            f"range [{short_p_str} – {short_c_str}] | "
            f"profit_target≤{profit_target_val:.2f} | SL≥{sl_level_val:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  OPTION TICK
    # ─────────────────────────────────────────────────────────────────────────

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        with self._lock:
            trade = self._trade
            if trade is None:
                return

            if   token == trade.short_ce_token: trade.lp_short_ce = price
            elif token == trade.short_pe_token: trade.lp_short_pe = price
            elif token == trade.long_ce_token:  trade.lp_long_ce  = price
            elif token == trade.long_pe_token:  trade.lp_long_pe  = price
            else:
                return

            self._check_exit_locked(trade, ts)

    def _check_exit_locked(self, trade: HedgedSellTrade, ts: datetime):
        """Must be called while self._lock is held."""
        if self._squareoff_done or self._close_in_flight:
            return

        cv = trade.current_net_value()

        if cv <= trade.profit_target:
            log.info(
                f"[HEDGED_SELL] TARGET HIT ✓ | "
                f"current_value={cv:.2f} ≤ target={trade.profit_target:.2f} | "
                f"pnl=₹{trade.pnl():.0f}"
            )
            self._squareoff_done  = True
            self._close_in_flight = True
            threading.Thread(
                target=self._close_trade, args=("TARGET", ts), daemon=True
            ).start()
            return

        if cv >= trade.sl_level:
            log.warning(
                f"[HEDGED_SELL] SL HIT | "
                f"current_value={cv:.2f} ≥ sl={trade.sl_level:.2f} | "
                f"loss=₹{trade.pnl():.0f}"
            )
            self._squareoff_done  = True
            self._close_in_flight = True
            threading.Thread(
                target=self._close_trade, args=("SL", ts), daemon=True
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  CLOSE TRADE (runs in daemon thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _close_trade(self, reason: str, ts: datetime):
        """
        Exit all 4 legs.
        Order: buy back short legs first (kill gamma risk), then sell hedges.
        """
        with self._lock:
            trade = self._trade
            if trade is None:
                self._close_in_flight = False
                return
            self._trade = None

        exit_ts = _now_ist()

        sc_exit = self.get_price(trade.short_ce_token) or trade.lp_short_ce
        sp_exit = self.get_price(trade.short_pe_token) or trade.lp_short_pe
        lc_exit = self.get_price(trade.long_ce_token)  or trade.lp_long_ce
        lp_exit = self.get_price(trade.long_pe_token)  or trade.lp_long_pe

        log.info(
            f"[HEDGED_SELL] CLOSING | reason={reason} | "
            f"sc={sc_exit:.2f} sp={sp_exit:.2f} lc={lc_exit:.2f} lp={lp_exit:.2f}"
        )

        # 1. Buy back short legs (buy-to-close)
        self._place_buy(trade.short_ce_sym, trade.short_ce_token, trade.qty, sc_exit)
        self._place_buy(trade.short_pe_sym, trade.short_pe_token, trade.qty, sp_exit)
        # 2. Sell long hedge legs (sell-to-close)
        self._place_sell(trade.long_ce_sym, trade.long_ce_token, trade.qty, lc_exit)
        self._place_sell(trade.long_pe_sym, trade.long_pe_token, trade.qty, lp_exit)

        self._release_slot()

        for tok in (trade.short_ce_token, trade.short_pe_token,
                    trade.long_ce_token,  trade.long_pe_token):
            self.unsubscribe_option(tok)
            self._subscribed_tokens.discard(tok)

        exit_net   = (sc_exit + sp_exit) - (lc_exit + lp_exit)
        total_pnl  = (trade.net_credit - exit_net) * trade.qty
        outcome    = "WIN" if total_pnl > 0 else ("LOSS" if total_pnl < 0 else "BE")

        log.info(
            f"[HEDGED_SELL] CLOSED | {outcome} | reason={reason} | "
            f"entry_credit={trade.net_credit:.2f} | exit_value={exit_net:.2f} | "
            f"total_pnl=₹{total_pnl:.0f} | qty={trade.qty} | "
            f"duration={(exit_ts - trade.entry_time).seconds // 60}min"
        )

        self._log_csv(trade, reason, sc_exit, sp_exit, lc_exit, lp_exit,
                      exit_net, total_pnl, exit_ts)

        with self._lock:
            self._close_in_flight = False

    # ─────────────────────────────────────────────────────────────────────────
    #  EOD SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def eod_summary(self):
        with self._lock:
            trade = self._trade

        if trade is not None:
            log.warning("[HEDGED_SELL] EOD: position still open — forcing close")
            if not self._squareoff_done:
                self._squareoff_done  = True
                self._close_in_flight = True
                threading.Thread(
                    target=self._close_trade,
                    args=("EOD_FORCE", _now_ist()),
                    daemon=True,
                ).start()
        else:
            log.info(
                f"[HEDGED_SELL] EOD summary | done_today={self._done_today} | "
                f"log → {CFG['csv_file']}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  CSV LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_csv(self):
        path = CFG["csv_file"]
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "date", "entry_time", "exit_time", "reason",
                    "short_ce_strike", "short_pe_strike",
                    "long_ce_strike",  "long_pe_strike",
                    "spread_width_pts",
                    "short_ce_entry", "short_pe_entry",
                    "long_ce_entry",  "long_pe_entry",
                    "net_credit", "profit_target", "sl_level",
                    "short_ce_exit", "short_pe_exit",
                    "long_ce_exit",  "long_pe_exit",
                    "exit_net_value", "pnl_per_unit", "qty", "total_pnl", "outcome",
                ])
            log.info(f"[HEDGED_SELL] CSV created: {path}")

    def _log_csv(
        self, trade: HedgedSellTrade, reason: str,
        sc_exit, sp_exit, lc_exit, lp_exit,
        exit_net, total_pnl, exit_ts,
    ):
        outcome      = "WIN" if total_pnl > 0 else ("LOSS" if total_pnl < 0 else "BE")
        spread_width = CFG["long_offset"] - CFG["short_offset"]
        pnl_per_unit = (trade.net_credit - exit_net)

        with open(CFG["csv_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                exit_ts.strftime("%Y-%m-%d"),
                trade.entry_time.strftime("%H:%M:%S"),
                exit_ts.strftime("%H:%M:%S"),
                reason,
                trade.short_ce_strike,
                trade.short_pe_strike,
                trade.long_ce_strike,
                trade.long_pe_strike,
                spread_width,
                f"{trade.short_ce_entry:.2f}",
                f"{trade.short_pe_entry:.2f}",
                f"{trade.long_ce_entry:.2f}",
                f"{trade.long_pe_entry:.2f}",
                f"{trade.net_credit:.2f}",
                f"{trade.profit_target:.2f}",
                f"{trade.sl_level:.2f}",
                f"{sc_exit:.2f}",
                f"{sp_exit:.2f}",
                f"{lc_exit:.2f}",
                f"{lp_exit:.2f}",
                f"{exit_net:.2f}",
                f"{pnl_per_unit:.2f}",
                trade.qty,
                f"{total_pnl:.0f}",
                outcome,
            ])
