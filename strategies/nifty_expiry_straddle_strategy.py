"""
strategies/nifty_expiry_straddle_strategy.py
──────────────────────────────────────────────────────────────────────────────
NiftyExpiryStraddleStrategy  —  Nifty weekly expiry day short straddle

RUNS ONLY:  when DTE == 0  (Nifty weekly expiry day, typically Thursday)

PHASE 1  (9:20 AM – 11:30 AM):
  Entry (9:20 AM, once):
    • Sell ATM CE  +  Sell ATM PE at current LTP
    • Filter: combined premium >= MIN_COMBINED_PREMIUM (120 pts)
    • Filter: PCR between PCR_LOW (0.75) and PCR_HIGH (1.30)
              (bypassed if nifty_pm.pcr is still None at 9:20)

  Trade management (on every option tick):
    • Per-leg SL   : leg_ltp > LEG_SL_MULT × leg_credit (2×) → close that leg
    • Combined SL  : unrealised loss > COMBINED_SL_MULT × combined_credit (1×) → close all
    • Target       : combined gain >= TARGET_PCT × combined_credit (40%) → close all

  Hard exit:
    • 11:30 AM cutoff
    • 15:15 PM safety squareoff

INDEX / INSTRUMENTS:
    NSE:NIFTY 50  |  INDEX_TOKEN = 256265
    t.py routes Nifty ticks and passes nifty_pm + nifty_instruments
    to any strategy with INDEX_TOKEN == 256265.
    PCR: nifty_pm.pcr (Nifty 50 WsPCR, warmed from 9:16 AM)

ORDER FLOW:
    Entry → _place_sell()  (SELL to open short leg)
    Exit  → _place_buy()   (BUY to close short leg)
    PnL   = (sell_price − buy_price) × qty  per leg
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Optional, Tuple

from core.base_strategy import BaseStrategy
from core.instruments import get_atm_strike

_IST = timezone(timedelta(hours=5, minutes=30))
def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

log = logging.getLogger("strategy.nifty_expiry_straddle")

# ─────────────────────────────────────────────────────────────────────────────
LIVE_MODE = False

NIFTY_STRIKE_STEP = 50

CFG = {
    "min_combined_premium"  : 120,    # skip if CE+PE combined premium < this
    "pcr_low"               : 0.75,   # skip if PCR below (strong bull trend)
    "pcr_high"              : 1.30,   # skip if PCR above (strong bear trend)
    "target_pct"            : 0.40,   # close when gain >= 40% of combined credit
    "leg_sl_mult"           : 2.0,    # close leg when ltp > 2× leg credit
    "combined_sl_mult"      : 1.0,    # close all when loss >= 100% of combined credit
    "entry_time"            : (9, 20),
    "cutoff_time"           : (11, 30),
    "auto_squareoff"        : (15, 15),
    "quantity"              : 75,     # Nifty lot size (post Oct-2024 SEBI revision)
    "entry_csv"             : "logs/nifty_expiry_straddle_entry.csv",
    "exit_csv"              : "logs/nifty_expiry_straddle_exit.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
class _Leg:
    __slots__ = ["token", "symbol", "credit", "order_id", "closed", "qty"]
    def __init__(self, token, symbol, credit, order_id, qty):
        self.token    = token
        self.symbol   = symbol
        self.credit   = credit
        self.order_id = order_id
        self.closed   = False
        self.qty      = qty


def _csv_append(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    write_header = not os.path.isfile(filepath)
    try:
        with open(filepath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        log.error(f"[NIFTY_EXPIRY_STRADDLE] CSV write error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
class NiftyExpiryStraddleStrategy(BaseStrategy):
    """
    Nifty weekly expiry day short straddle.
    INDEX_TOKEN=256265 → t.py passes nifty_pm + nifty_instruments to pre_market()
    and routes Nifty 50 spot ticks to on_tick().
    """

    INDEX_TOKEN = 256265   # NSE:NIFTY 50 — fixed Zerodha instrument token
    LIVE_MODE   = LIVE_MODE

    @property
    def name(self) -> str:
        return "NIFTY_EXPIRY_STRADDLE"

    def __init__(self, hub):
        super().__init__(hub)
        self._lock = threading.Lock()

        self._leg_ce            : Optional[_Leg] = None
        self._leg_pe            : Optional[_Leg] = None
        self._position_open     : bool  = False
        self._combined_credit   : float = 0.0
        self._entry_attempted   : bool  = False
        self._squareoff_done    : bool  = False

        self._daily_pnl         : float = 0.0
        self._trade_count       : int   = 0

        self._active            : bool  = False   # True only when DTE == 0
        self._expiry_date       = None
        self._pm                = None
        self._instruments       = None
        self._spot              : Optional[float] = None
        self._subscribed        : set   = set()

    # ─────────────────────────────────────────────────────────────────────────
    def pre_market(self, premarket_data, instruments) -> bool:
        """
        Called with nifty_pm + nifty_instruments by t.py
        (because INDEX_TOKEN == 256265).
        Returns True only on Nifty expiry day (DTE == 0).
        """
        self._pm          = premarket_data
        self._instruments = instruments
        self._expiry_date = premarket_data.expiry_date
        dte               = premarket_data.dte_days

        if dte != 0:
            log.info(
                f"[NIFTY_EXPIRY_STRADDLE] DTE={dte} — not expiry day, "
                "strategy inactive today"
            )
            return False

        self._active = True
        log.info(
            f"[NIFTY_EXPIRY_STRADDLE] ✓ EXPIRY DAY | "
            f"expiry={self._expiry_date} VIX={premarket_data.vix} "
            f"PCR={premarket_data.pcr}"
        )

        # Pre-subscribe ATM ± 2 strikes so LTP is ready by 9:20
        ref_price = premarket_data.prev_close
        if ref_price:
            atm = get_atm_strike(ref_price, step=NIFTY_STRIKE_STEP)
            for opt_type in ("CE", "PE"):
                for offset in (0, -NIFTY_STRIKE_STEP, NIFTY_STRIKE_STEP,
                               -2 * NIFTY_STRIKE_STEP, 2 * NIFTY_STRIKE_STEP):
                    tok, sym = instruments.get_option_token(
                        atm + offset, opt_type, self._expiry_date
                    )
                    if tok and tok not in self._subscribed:
                        self.subscribe_option(tok)
                        self._subscribed.add(tok)
                        log.info(
                            f"[NIFTY_EXPIRY_STRADDLE] Pre-subscribed {sym} ({tok})"
                        )
        else:
            log.warning("[NIFTY_EXPIRY_STRADDLE] No prev_close — pre-subscription skipped")

        return True

    # ─────────────────────────────────────────────────────────────────────────
    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        """Nifty 50 spot tick — routed exclusively here via INDEX_TOKEN."""
        if not self._active:
            return
        self._spot = price

        if self._entry_attempted or self._squareoff_done:
            return

        t = ts.time()
        if t < dtime(*CFG["entry_time"]) or t >= dtime(*CFG["cutoff_time"]):
            return

        self._try_entry(price, ts)

    # ─────────────────────────────────────────────────────────────────────────
    def on_candle(self, candle: dict, ts: datetime):
        """Not called for INDEX_TOKEN strategies — MarketHub routes only on_tick."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    def on_option_tick(self, token: int, price: float,
                       ts: datetime, tick_ts: datetime = None):
        """All subscribed option ticks arrive here — drives trade management."""
        if not self._active:
            return
        if not self._position_open:
            return

        t = ts.time()

        # Safety squareoff at 15:15
        if t >= dtime(*CFG["auto_squareoff"]) and not self._squareoff_done:
            with self._lock:
                if not self._squareoff_done:
                    log.info("[NIFTY_EXPIRY_STRADDLE] Auto-squareoff at 15:15")
                    self._close_all("AUTO_SQUAREOFF", ts)
            return

        # Phase 1 hard cutoff 11:30
        if t >= dtime(*CFG["cutoff_time"]) and not self._squareoff_done:
            with self._lock:
                if not self._squareoff_done:
                    log.info("[NIFTY_EXPIRY_STRADDLE] 11:30 cutoff — closing all")
                    self._close_all("CUTOFF_EXIT", ts)
            return

        with self._lock:
            if not self._position_open:
                return

            if self._leg_ce and not self._leg_ce.closed and token == self._leg_ce.token:
                self._check_leg_sl(self._leg_ce, price, "CE", ts)
            elif self._leg_pe and not self._leg_pe.closed and token == self._leg_pe.token:
                self._check_leg_sl(self._leg_pe, price, "PE", ts)

            if self._position_open:
                self._check_combined_exit(ts)

    # ─────────────────────────────────────────────────────────────────────────
    def _try_entry(self, spot: float, ts: datetime):
        self._entry_attempted = True

        # PCR gate (bypass when PCR not yet warm)
        pcr = self._pm.pcr if self._pm else None
        if pcr is not None:
            if not (CFG["pcr_low"] <= pcr <= CFG["pcr_high"]):
                log.info(
                    f"[NIFTY_EXPIRY_STRADDLE] PCR={pcr:.3f} outside "
                    f"[{CFG['pcr_low']}, {CFG['pcr_high']}] — entry skipped"
                )
                return

        atm = get_atm_strike(spot, step=NIFTY_STRIKE_STEP)
        ce_tok, ce_sym = self._instruments.get_option_token(atm, "CE", self._expiry_date)
        pe_tok, pe_sym = self._instruments.get_option_token(atm, "PE", self._expiry_date)

        if not ce_tok or not pe_tok:
            log.warning(f"[NIFTY_EXPIRY_STRADDLE] ATM={atm} tokens not found — abort")
            return

        for tok in (ce_tok, pe_tok):
            if tok not in self._subscribed:
                self.subscribe_option(tok)
                self._subscribed.add(tok)

        ce_ltp = self.get_price(ce_tok)
        pe_ltp = self.get_price(pe_tok)

        if ce_ltp is None or pe_ltp is None:
            log.warning("[NIFTY_EXPIRY_STRADDLE] LTP unavailable at 9:20 — abort")
            return

        combined = ce_ltp + pe_ltp
        if combined < CFG["min_combined_premium"]:
            log.info(
                f"[NIFTY_EXPIRY_STRADDLE] Combined premium {combined:.1f} < "
                f"{CFG['min_combined_premium']} — entry skipped"
            )
            return

        log.info(
            f"[NIFTY_EXPIRY_STRADDLE] ENTRY | ATM={atm} "
            f"CE={ce_sym}@{ce_ltp:.1f} PE={pe_sym}@{pe_ltp:.1f} "
            f"combined={combined:.1f} PCR={pcr}"
        )

        if not self._acquire_slot():
            log.warning("[NIFTY_EXPIRY_STRADDLE] Slot busy — entry skipped")
            return

        mode = "LIVE" if LIVE_MODE else "PAPER"

        # Sell CE leg
        ce_res = self._place_sell(ce_sym, ce_tok, CFG["quantity"], ce_ltp)
        if ce_res is None:
            log.error("[NIFTY_EXPIRY_STRADDLE] CE SELL failed — aborting")
            self._release_slot()
            return

        # Sell PE leg
        pe_res = self._place_sell(pe_sym, pe_tok, CFG["quantity"], pe_ltp)
        if pe_res is None:
            log.error("[NIFTY_EXPIRY_STRADDLE] PE SELL failed — buying back CE")
            self._place_buy(ce_sym, ce_tok, CFG["quantity"], ce_ltp)
            self._release_slot()
            return

        ce_oid, ce_fill = ce_res
        pe_oid, pe_fill = pe_res

        with self._lock:
            self._leg_ce          = _Leg(ce_tok, ce_sym, ce_fill, ce_oid, CFG["quantity"])
            self._leg_pe          = _Leg(pe_tok, pe_sym, pe_fill, pe_oid, CFG["quantity"])
            self._combined_credit = ce_fill + pe_fill
            self._position_open   = True

        log.info(
            f"[NIFTY_EXPIRY_STRADDLE] ✓ STRADDLE OPEN [{mode}] | "
            f"CE@{ce_fill:.1f} PE@{pe_fill:.1f} credit={self._combined_credit:.1f} "
            f"qty={CFG['quantity']}"
        )

        _csv_append(CFG["entry_csv"], {
            "date"            : ts.strftime("%Y-%m-%d"),
            "time"            : ts.strftime("%H:%M:%S"),
            "mode"            : mode,
            "atm"             : atm,
            "ce_symbol"       : ce_sym,
            "ce_credit"       : round(ce_fill, 2),
            "pe_symbol"       : pe_sym,
            "pe_credit"       : round(pe_fill, 2),
            "combined_credit" : round(self._combined_credit, 2),
            "qty"             : CFG["quantity"],
            "pcr"             : round(pcr, 3) if pcr else "",
            "target_gain"     : round(CFG["target_pct"] * self._combined_credit, 2),
            "ce_sl_level"     : round(ce_fill * CFG["leg_sl_mult"], 2),
            "pe_sl_level"     : round(pe_fill * CFG["leg_sl_mult"], 2),
        })

    # ─────────────────────────────────────────────────────────────────────────
    def _check_leg_sl(self, leg: _Leg, ltp: float, side: str, ts: datetime):
        """Check per-leg SL. Caller holds lock."""
        sl_level = leg.credit * CFG["leg_sl_mult"]
        if ltp >= sl_level:
            log.warning(
                f"[NIFTY_EXPIRY_STRADDLE] {side} SL HIT | "
                f"credit={leg.credit:.1f} ltp={ltp:.1f} sl={sl_level:.1f}"
            )
            self._close_leg(leg, ltp, f"{side}_SL", ts)

    # ─────────────────────────────────────────────────────────────────────────
    def _check_combined_exit(self, ts: datetime):
        """Check combined target and combined MTM SL. Caller holds lock."""
        if not self._position_open or self._squareoff_done:
            return

        leg_ce = self._leg_ce
        leg_pe = self._leg_pe

        ce_ltp = (self.get_price(leg_ce.token)
                  if leg_ce and not leg_ce.closed else
                  (leg_ce.credit if leg_ce else 0.0))
        pe_ltp = (self.get_price(leg_pe.token)
                  if leg_pe and not leg_pe.closed else
                  (leg_pe.credit if leg_pe else 0.0))

        if ce_ltp is None or pe_ltp is None:
            return

        combined_now = (ce_ltp or 0) + (pe_ltp or 0)
        gain = self._combined_credit - combined_now

        target = CFG["target_pct"] * self._combined_credit
        if gain >= target:
            log.info(
                f"[NIFTY_EXPIRY_STRADDLE] TARGET HIT | "
                f"gain={gain:.1f} >= target={target:.1f}"
            )
            self._close_all("TARGET", ts)
            return

        loss = -gain
        combined_sl = CFG["combined_sl_mult"] * self._combined_credit
        if loss >= combined_sl:
            log.warning(
                f"[NIFTY_EXPIRY_STRADDLE] COMBINED MTM SL | "
                f"loss={loss:.1f} >= limit={combined_sl:.1f}"
            )
            self._close_all("COMBINED_MTM_SL", ts)

    # ─────────────────────────────────────────────────────────────────────────
    def _close_leg(self, leg: _Leg, ltp: float, reason: str, ts: datetime) -> float:
        """Buy back one short leg. Caller holds lock. Returns leg P&L in Rs."""
        if leg.closed:
            return 0.0

        res = self._place_buy(leg.symbol, leg.token, leg.qty, ltp)
        if res is None:
            log.error(
                f"[NIFTY_EXPIRY_STRADDLE] BUY-BACK FAILED {leg.symbol} | "
                "MANUAL SQUAREOFF NEEDED"
            )
            return 0.0

        _, close_price = res
        pnl_rs = (leg.credit - close_price) * leg.qty
        leg.closed = True
        self._daily_pnl   += pnl_rs
        self._trade_count += 1

        mode = "LIVE" if LIVE_MODE else "PAPER"
        log.info(
            f"[NIFTY_EXPIRY_STRADDLE] LEG CLOSED [{mode}] | "
            f"{leg.symbol} reason={reason} "
            f"credit={leg.credit:.1f} close={close_price:.1f} "
            f"pnl={pnl_rs:+.0f}Rs daily={self._daily_pnl:+.0f}Rs"
        )

        _csv_append(CFG["exit_csv"], {
            "date"        : ts.strftime("%Y-%m-%d"),
            "time"        : ts.strftime("%H:%M:%S"),
            "mode"        : mode,
            "leg"         : leg.symbol,
            "reason"      : reason,
            "credit"      : round(leg.credit, 2),
            "close_price" : round(close_price, 2),
            "qty"         : leg.qty,
            "pnl_leg_rs"  : round(pnl_rs, 2),
            "daily_pnl"   : round(self._daily_pnl, 2),
        })

        # Release slot when both legs are closed
        ce_closed = self._leg_ce is None or self._leg_ce.closed
        pe_closed = self._leg_pe is None or self._leg_pe.closed
        if ce_closed and pe_closed:
            self._position_open = False
            self._release_slot()
            log.info(
                f"[NIFTY_EXPIRY_STRADDLE] ALL LEGS CLOSED | "
                f"credit={self._combined_credit:.1f} "
                f"daily_pnl={self._daily_pnl:+.0f}Rs"
            )

        return pnl_rs

    # ─────────────────────────────────────────────────────────────────────────
    def _close_all(self, reason: str, ts: datetime):
        """Close all open legs. Caller must hold lock OR call with lock acquired."""
        if self._squareoff_done:
            return
        self._squareoff_done = True

        log.info(f"[NIFTY_EXPIRY_STRADDLE] CLOSING ALL | reason={reason}")

        if self._leg_ce and not self._leg_ce.closed:
            ltp = self.get_price(self._leg_ce.token) or self._leg_ce.credit
            self._close_leg(self._leg_ce, ltp, reason, ts)

        if self._leg_pe and not self._leg_pe.closed:
            ltp = self.get_price(self._leg_pe.token) or self._leg_pe.credit
            self._close_leg(self._leg_pe, ltp, reason, ts)

    # ─────────────────────────────────────────────────────────────────────────
    def eod_summary(self):
        log.info(
            f"[NIFTY_EXPIRY_STRADDLE] ══ EOD ══ "
            f"trades={self._trade_count} daily_pnl={self._daily_pnl:+.0f}Rs"
        )
        if self._position_open and not self._squareoff_done:
            log.warning("[NIFTY_EXPIRY_STRADDLE] ⚠ Position open at EOD — force closing")
            with self._lock:
                self._close_all("EOD_FORCE_CLOSE", _now_ist())
