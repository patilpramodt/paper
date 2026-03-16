# ==============================================================================
# risk_manager.py  V6 Risk Management Engine
#
# New in V6:
#   [OK] Directional consecutive-loss blocker (CE or PE independently blocked)
#   [OK] Expiry day rules (Thursday  cutoff at 12:00, ITM offset)
#   [OK] Afternoon block 14:3015:00 (no new entries)
#   [OK] Option LTP sanity validation
#   [OK] ATR-scaled SL/TP computation (called at entry time)
#   [OK] All V5 features retained
# ==============================================================================

import time
import datetime as dt
from collections import deque
from typing import Optional, Tuple

# IST FIX: GitHub Actions runners are UTC — bare datetime.now() returns UTC
_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

def _now_ist() -> dt.datetime:
    """Always returns current datetime in IST — works on GitHub Actions (UTC) and local."""
    return dt.datetime.now(tz=_IST).replace(tzinfo=None)

from scalper_v7_core.config import (
    MAX_DAILY_LOSS, MAX_TRADES_DAY, MAX_TRADES_PER_HOUR,
    POST_SL_COOLDOWN, 
    SESSION_START,
    AFTERNOON_BLOCK, AUTO_SQUAREOFF,
    EXPIRY_WEEKDAY, EXPIRY_ENTRY_CUTOFF,
    CONSEC_LOSS_BLOCK_COUNT, CONSEC_LOSS_BLOCK_MINS,
    OPTION_LTP_MIN, OPTION_LTP_MAX_PCT, OPTION_SPREAD_MAX_PCT,
    ESTIMATED_SPREAD_PCT,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    SL_MIN_POINTS, SL_MAX_POINTS, TP_MIN_POINTS, TP_MAX_POINTS,
    EXPIRY_TP_REDUCTION, EXPIRY_USE_ITM_OFFSET,
)
from scalper_v7_core.logger_setup import log


class RiskManager:
    """
    Full risk controller for V6.
    Stateful  must call reset_day() at session start.
    """

    def __init__(self):
        # Daily accumulators
        self.daily_pnl:        float = 0.0
        self.trades_today:     int   = 0
        self.sl_hits_today:    int   = 0
        self._last_sl_time:    float = 0.0
        self._is_halted:       bool  = False
        self._halt_reason:     str   = ""
        self._session_date:    Optional[dt.date] = None

        # Directional consecutive loss tracker
        self._consec_losses:   dict  = {"CE": 0, "PE": 0}
        self._dir_blocked_at:  dict  = {"CE": 0.0, "PE": 0.0}

        # Hourly trade limiter (V7 NEW)
        # Stores entry timestamps as epoch floats in a rolling deque
        self._hourly_entries: deque = deque()

    # 
    # Day Reset
    # 

    def reset_day(self):
        today = dt.date.today()
        if self._session_date == today:
            return
        self._session_date    = today
        self.daily_pnl        = 0.0
        self.trades_today     = 0
        self.sl_hits_today    = 0
        self._last_sl_time    = 0.0
        self._is_halted       = False
        self._halt_reason     = ""
        self._consec_losses   = {"CE": 0, "PE": 0}
        self._dir_blocked_at  = {"CE": 0.0, "PE": 0.0}
        self._hourly_entries  = deque()
        expiry = self._is_expiry_day()
        log.info(f"[DATE] RiskManager reset for {today} {'[EXPIRY DAY]' if expiry else ''}")

    # 
    # Expiry Day Detection
    # 

    def _is_expiry_day(self) -> bool:
        return dt.date.today().weekday() == EXPIRY_WEEKDAY

    def expiry_strike_offset(self) -> int:
        """
        On expiry day, return +EXPIRY_USE_ITM_OFFSET for CE (buy lower strike)
        or -EXPIRY_USE_ITM_OFFSET for PE (buy higher strike) to use ITM options.
        Returns 0 on normal days.
        """
        return EXPIRY_USE_ITM_OFFSET if self._is_expiry_day() else 0

    # 
    # SL / TP Computation (ATR-scaled)
    # 

    def compute_sl_tp(
        self,
        entry_price: float,
        atr14: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute ATR-scaled SL and TP.
        Returns (sl_price, tp_price, sl_pts, tp_pts)

        SL  = entry - clamp(ATR * SL_MULTIPLIER, SL_MIN, SL_MAX)
        TP  = entry + clamp(ATR * TP_MULTIPLIER, TP_MIN, TP_MAX)
        On expiry day: TP is reduced by EXPIRY_TP_REDUCTION factor.
        """
        sl_pts = float(atr14 * ATR_SL_MULTIPLIER) if atr14 > 0 else SL_MIN_POINTS
        tp_pts = float(atr14 * ATR_TP_MULTIPLIER) if atr14 > 0 else TP_MIN_POINTS

        sl_pts = max(SL_MIN_POINTS, min(sl_pts, SL_MAX_POINTS))
        tp_pts = max(TP_MIN_POINTS, min(tp_pts, TP_MAX_POINTS))

        # Expiry day: tighter TP (theta erosion)
        if self._is_expiry_day():
            tp_pts = max(TP_MIN_POINTS, tp_pts * EXPIRY_TP_REDUCTION)

        sl_price = round(entry_price - sl_pts, 2)
        tp_price = round(entry_price + tp_pts, 2)

        log.info(
            f"SL/TP computed | ATR={atr14:.1f} | "
            f"SL={sl_price:.2f}(-{sl_pts:.1f}) | TP={tp_price:.2f}(+{tp_pts:.1f})"
            + (" [EXPIRY]" if self._is_expiry_day() else "")
        )
        return sl_price, tp_price, sl_pts, tp_pts

    # 
    # Event Callbacks
    # 

    def on_trade_entry(self, option_type: str):
        self.trades_today += 1
        self._hourly_entries.append(time.time())
        log.info(f"Trade #{self.trades_today} entered ({option_type})")

    def on_trade_exit(self, pnl_pts: float, pnl_rs: float, reason: str, option_type: str):
        self.daily_pnl += pnl_rs

        if reason == "SL":
            self.sl_hits_today += 1
            self._last_sl_time  = time.time()
            self._consec_losses[option_type] = self._consec_losses.get(option_type, 0) + 1

            count = self._consec_losses[option_type]
            if count >= CONSEC_LOSS_BLOCK_COUNT:
                self._dir_blocked_at[option_type] = time.time()
                log.warning(
                    f"[BLOCK] {option_type} direction BLOCKED after {count} consecutive losses. "
                    f"Cooldown: {CONSEC_LOSS_BLOCK_MINS} min"
                )
        else:
            # Win resets the consecutive loss counter for that direction
            self._consec_losses[option_type] = 0

        log.info(
            f"Exit ({reason}) {option_type} | "
            f"PnL={pnl_pts:+.2f}pts ({pnl_rs:+.2f}) | "
            f"DayPnL={self.daily_pnl:+.2f}"
        )

        if self.daily_pnl <= -abs(MAX_DAILY_LOSS):
            self._halt("Max daily loss breached")

    # 
    # Option LTP Sanity Check
    # 

    def validate_option_ltp(
        self,
        ltp: float,
        spot: float,
        tp_pts: float,
    ) -> Tuple[bool, str]:
        """
        Validate LTP before entering a paper trade.
        Returns (ok: bool, reason: str)
        """
        if ltp < OPTION_LTP_MIN:
            return False, f"LTP={ltp:.2f} too low (illiquid, min={OPTION_LTP_MIN})"

        if spot > 0 and (ltp / spot * 100) > OPTION_LTP_MAX_PCT:
            return False, f"LTP={ltp:.2f} too high (>{OPTION_LTP_MAX_PCT}% of spot, wrong strike?)"

        est_spread = ltp * ESTIMATED_SPREAD_PCT
        if tp_pts > 0 and (est_spread / tp_pts * 100) > OPTION_SPREAD_MAX_PCT:
            return False, (
                f"Spread={est_spread:.2f} consumes "
                f"{est_spread/tp_pts*100:.0f}% of target={tp_pts:.1f}pts "
                f"(max {OPTION_SPREAD_MAX_PCT}%)"
            )

        return True, ""

    # 
    # Entry Gate
    # 

    def can_enter(self, option_type: str = "") -> Tuple[bool, str]:
        """
        Returns (True, "") if entry allowed, else (False, reason).
        option_type: "CE" or "PE"  checked against directional block.
        """
        if self._is_halted:
            return False, f"HALTED: {self._halt_reason}"

        if self.trades_today >= MAX_TRADES_DAY:
            return False, f"Max trades/day ({MAX_TRADES_DAY}) reached"

        now      = _now_ist()  # FIX: was dt.datetime.now() — UTC on GitHub Actions
        now_time = now.time()

        # Session window
        if now_time < dt.time(*SESSION_START):
            return False, f"Pre-session (opens {SESSION_START[0]}:{SESSION_START[1]:02d})"

        if now_time >= dt.time(*AUTO_SQUAREOFF):
            return False, "Post square-off time"

        # Afternoon block  no new entries
        if now_time >= dt.time(*AFTERNOON_BLOCK):
            return False, f"Afternoon block (no entries after {AFTERNOON_BLOCK[0]}:{AFTERNOON_BLOCK[1]:02d})"

        # Expiry day cutoff
        if self._is_expiry_day() and now_time >= dt.time(*EXPIRY_ENTRY_CUTOFF):
            return False, f"Expiry day cutoff ({EXPIRY_ENTRY_CUTOFF[0]}:{EXPIRY_ENTRY_CUTOFF[1]:02d})"

        # Hourly trade limit (V7 NEW)
        # Prune entries older than 60 minutes from rolling window
        cutoff = time.time() - 3600
        while self._hourly_entries and self._hourly_entries[0] < cutoff:
            self._hourly_entries.popleft()
        if len(self._hourly_entries) >= MAX_TRADES_PER_HOUR:
            return False, f"Hourly trade cap ({MAX_TRADES_PER_HOUR}/hr) reached"

        # Global post-SL cooldown
        if self._last_sl_time > 0:
            elapsed   = time.time() - self._last_sl_time
            remaining = POST_SL_COOLDOWN - elapsed
            if remaining > 0:
                m, s = int(remaining // 60), int(remaining % 60)
                return False, f"Post-SL cooldown: {m}m {s}s remaining"

        # Directional consecutive-loss block
        if option_type in ("CE", "PE"):
            blocked_at = self._dir_blocked_at.get(option_type, 0.0)
            if blocked_at > 0:
                elapsed   = time.time() - blocked_at
                remaining = CONSEC_LOSS_BLOCK_MINS * 60 - elapsed
                if remaining > 0:
                    m = int(remaining // 60)
                    return False, f"{option_type} direction blocked: {m}m remaining (consecutive losses)"
                else:
                    # Block expired  reset
                    self._dir_blocked_at[option_type]  = 0.0
                    self._consec_losses[option_type]   = 0
                    log.info(f"{option_type} direction block expired  re-enabled")

        return True, ""

    # 
    # Square-off Check
    # 

    def should_squareoff(self) -> bool:
        now = _now_ist().time()  # FIX: was dt.datetime.now() — UTC on GitHub Actions
        return now >= dt.time(*AUTO_SQUAREOFF)

    # 
    # Status
    # 

    def status_line(self) -> str:
        cooldown = max(0, POST_SL_COOLDOWN - (time.time() - self._last_sl_time))
        ce_block  = self._consec_losses.get("CE", 0)
        pe_block  = self._consec_losses.get("PE", 0)
        return (
            f"PnL={self.daily_pnl:+.1f} | "
            f"T={self.trades_today}/{MAX_TRADES_DAY} | "
            f"SL={self.sl_hits_today} | "
            f"Cooldown={cooldown:.0f}s | "
            f"CE_loss={ce_block} PE_loss={pe_block}"
        )

    # 
    # Internal
    # 

    def _halt(self, reason: str):
        self._is_halted   = True
        self._halt_reason = reason
        log.warning(f"[ALERT] Trading HALTED: {reason}")
