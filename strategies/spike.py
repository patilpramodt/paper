"""
strategies/spike.py  (fixed)

Fix applied:
  - pre_market() now subscribes ATM CE+PE even when prev_close is None.
    Uses prev_last5m_close as fallback for strike calculation.
    Both days had valid prev_last5m data — this ensures options are
    pre-subscribed and priced before 9:15 AM first tick arrives.
  - on_tick() subscribes ATM options on the very first market tick
    if pre-subscription failed, giving them ~1s to receive a price.

All other logic unchanged.
"""

import csv
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

from core.base_strategy import BaseStrategy
from core.candle import SecondCandleBuilder

log = logging.getLogger("strategy.spike")

# IST FIX: GitHub Actions runners are UTC
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    return datetime.now(tz=_IST).replace(tzinfo=None)

CFG = {
    "quantity"               : 30,
    "start_time"             : dtime(9, 15),
    "spike_exit_time"        : dtime(9, 30),
    "close_time"             : dtime(15, 15),
    "initial_sl_buffer"      : 120,
    "trail_trigger_pts"      : 50,
    "trail_distance"         : 25,
    "doji_threshold"         : 0.10,
    "bucket_sec"             : 8,
    "min_candles_before_mom" : 2,
    "csv_file"               : "spike_trades.csv",
}


class SpikeStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "SPIKE"

    def __init__(self, market_hub):
        super().__init__(market_hub)

        self._index_8s     = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._opt_8s       = None

        self._gap_direction  : Optional[str] = None
        self._gap_filter_done: bool = False
        self._market_opened  : bool = False

        self._pre_ce_token   : Optional[int] = None
        self._pre_pe_token   : Optional[int] = None
        self._pre_ce_sym     : Optional[str] = None
        self._pre_pe_sym     : Optional[str] = None

        self._trade          = None
        self._trade_done     : bool = False
        self._today_pnl      : float = 0.0
        self._completed      : list  = []
        self._pending_entry  = None   # FIX: holds gap entry waiting for first real tick

        self._prev_body_high : Optional[float] = None
        self._prev_body_low  : Optional[float] = None
        self._prev_last5m_high  : Optional[float] = None
        self._prev_last5m_low   : Optional[float] = None
        self._prev_last5m_close : Optional[float] = None
        self._expiry_date    = None
        self._instruments    = None

        self._lock           = threading.Lock()

    # ── Pre-market ────────────────────────────────────────────────────────────

    def pre_market(self, pm, instruments) -> bool:
        from core.instruments import get_atm_strike

        now = _now_ist().time()  # FIX: was datetime.now() — UTC on GitHub
        if now >= CFG["spike_exit_time"]:
            log.warning(f"[{self.name}] Started after spike window — skipping today.")
            self._trade_done = True
            return True

        self._instruments       = instruments
        self._prev_body_high    = pm.prev_body_high
        self._prev_body_low     = pm.prev_body_low
        self._prev_last5m_high  = pm.prev_last5m_high
        self._prev_last5m_low   = pm.prev_last5m_low
        self._prev_last5m_close = pm.prev_last5m_close
        self._expiry_date       = pm.expiry_date

        log.info(f"[{self.name}] Pre-market | "
                 f"body=[{self._prev_body_low}  {self._prev_body_high}] "
                 f"prev_close={pm.prev_close}")

        # ── Determine best reference price for ATM strike ─────────────────────
        # FIX: use prev_last5m_close as fallback when prev_close is None.
        # prev_last5m data is fetched from 5-min history and is more reliable
        # at 9:08 AM than the daily candle call which often times out.
        ref_price = pm.prev_close or pm.prev_last5m_close

        if ref_price is None:
            log.warning(f"[{self.name}] No prev_close and no prev_last5m_close — "
                        f"cannot pre-subscribe options. Will attempt on first tick.")
            return True

        # ── Pre-subscribe ATM CE + PE ─────────────────────────────────────────
        strike = get_atm_strike(ref_price)
        log.info(f"[{self.name}] Token lookup | strike={strike} "
                 f"expiry={pm.expiry_date} ref_price={ref_price:.2f} "
                 f"(source={'prev_close' if pm.prev_close else 'prev_last5m_close'})")

        ce_tok, ce_sym = instruments.get_option_token(strike, "CE", pm.expiry_date)
        pe_tok, pe_sym = instruments.get_option_token(strike, "PE", pm.expiry_date)

        self._pre_ce_token = ce_tok
        self._pre_ce_sym   = ce_sym
        self._pre_pe_token = pe_tok
        self._pre_pe_sym   = pe_sym

        if ce_tok:
            self.subscribe_option(ce_tok)
            log.info(f"[{self.name}] Pre-subscribed CE: {ce_sym} ({ce_tok})")
        else:
            log.error(f"[{self.name}] CE token not found | strike={strike} expiry={pm.expiry_date}")

        if pe_tok:
            self.subscribe_option(pe_tok)
            log.info(f"[{self.name}] Pre-subscribed PE: {pe_sym} ({pe_tok})")
        else:
            log.error(f"[{self.name}] PE token not found | strike={strike} expiry={pm.expiry_date}")

        return True

    # ── Tick handlers ─────────────────────────────────────────────────────────

    def on_tick(self, price: float, ts: datetime, tick_ts: datetime):
        t = ts.time()

        if t < CFG["start_time"] or t > CFG["close_time"]:
            return

        if not self._market_opened and t >= CFG["start_time"]:
            self._market_opened = True
            log.info(f"[{self.name}] Market open tick received: {price:.2f}")

            # FIX: If pre-subscription failed (no ref price at 9:08), subscribe
            # ATM options right now using the actual open price.
            # This gives ~1-2 ticks for the option price to arrive before entry.
            if self._pre_ce_token is None or self._pre_pe_token is None:
                self._subscribe_atm_on_open(price)

        closed_8s = self._index_8s.feed_tick(price, tick_ts)

        if not self._gap_filter_done and self._market_opened:
            self._determine_gap_direction(price)
            self._gap_filter_done = True

            if self._gap_direction in ("CE", "PE"):
                self._attempt_gap_entry(price, ts)
            return

        if (self._gap_direction == "BOTH" and
                not self._trade_done and
                self._trade is None and
                closed_8s is not None):
            self._check_2candle_signal(closed_8s, price, ts)

        if self._trade and self._trade["state"] == "OPEN":
            if t >= CFG["spike_exit_time"]:
                opt_price = self.get_price(self._trade["token"]) or self._trade["entry"]
                self._do_exit(opt_price, "SPIKE_WINDOW_END", ts)
                return

    def on_candle(self, candle: dict, ts: datetime):
        pass

    def on_option_tick(self, token: int, price: float, ts: datetime, tick_ts: datetime = None):
        # FIX: resolve pending gap entry — triggered when first real post-9:15 tick arrives
        # for the expected option token (price was stale/pre-open when entry was attempted)
        if self._pending_entry and token == self._pending_entry["token"] and not self._trade:
            p = self._pending_entry
            self._pending_entry = None
            log.info(f"[{self.name}] Pending entry resolved — first live tick for {p['sym']} "
                     f"@ {price:.2f} (was stale at entry signal time)")
            # Now call _build_entry again; price_ts is now ≥ 9:15 so it will proceed
            self._build_entry(p["sym"], p["token"], p["signal"], ts, p["reason"])
            return

        if self._trade and token == self._trade.get("token"):
            if self._opt_8s is None:
                self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])

            use_ts = tick_ts if tick_ts else ts
            closed_opt_8s = self._opt_8s.feed_tick(price, use_ts)

            if self._trade["state"] == "OPEN":
                if price > self._trade["highest_seen"]:
                    self._trade["highest_seen"] = price

                new_sl = self._compute_trailing_sl(
                    self._trade["entry"], self._trade["highest_seen"], self._trade["sl"]
                )
                if new_sl > self._trade["sl"]:
                    log.info(f"[{self.name}] TSL: {self._trade['sl']:.0f} → {new_sl:.0f} "
                             f"(highest={self._trade['highest_seen']:.0f})")
                    self._trade["sl"] = new_sl

                if price <= self._trade["sl"]:
                    self._do_exit(self._trade["sl"], "SL", ts)
                    return

    # ── FIX: subscribe ATM tokens on first open tick if pre-subscription failed ──

    def _subscribe_atm_on_open(self, open_price: float):
        from core.instruments import get_atm_strike
        strike = get_atm_strike(open_price)
        expiry = self._expiry_date
        log.info(f"[{self.name}] Late-subscribing ATM options on open tick | "
                 f"strike={strike} expiry={expiry} open={open_price:.2f}")

        ce_tok, ce_sym = self._instruments.get_option_token(strike, "CE", expiry)
        pe_tok, pe_sym = self._instruments.get_option_token(strike, "PE", expiry)

        if ce_tok and self._pre_ce_token is None:
            self.subscribe_option(ce_tok)
            self._pre_ce_token = ce_tok
            self._pre_ce_sym   = ce_sym
            log.info(f"[{self.name}] Late-subscribed CE: {ce_sym} ({ce_tok})")

        if pe_tok and self._pre_pe_token is None:
            self.subscribe_option(pe_tok)
            self._pre_pe_token = pe_tok
            self._pre_pe_sym   = pe_sym
            log.info(f"[{self.name}] Late-subscribed PE: {pe_sym} ({pe_tok})")

    # ── Gap direction ─────────────────────────────────────────────────────────

    def _determine_gap_direction(self, open_price: float):
        h5 = self._prev_last5m_high
        l5 = self._prev_last5m_low

        if h5 is not None and l5 is not None:
            log.info(f"[{self.name}] Gap ref → prev last 5-min: "
                     f"H={h5:.0f}  L={l5:.0f}  Today open={open_price:.0f}")
            if open_price > h5:
                self._gap_direction = "CE"
                log.info(f"[{self.name}]  GAP UP: open={open_price:.0f} > last5m_high={h5:.0f}  → CE")
            elif open_price < l5:
                self._gap_direction = "PE"
                log.info(f"[{self.name}]  GAP DOWN: open={open_price:.0f} < last5m_low={l5:.0f}  → PE")
            else:
                self._gap_direction = "BOTH"
                log.info(f"[{self.name}]  NO GAP: open inside [{l5:.0f}–{h5:.0f}] → 2-candle signal")
            return

        log.warning(f"[{self.name}] Last 5-min candle data unavailable — falling back to daily body")
        bh, bl = self._prev_body_high, self._prev_body_low

        if bh is None or bl is None:
            self._gap_direction = "BOTH"
            log.warning(f"[{self.name}] No gap reference at all — defaulting to BOTH")
            return

        if open_price > bh:
            self._gap_direction = "CE"
            log.info(f"[{self.name}]  GAP UP (fallback): open={open_price:.0f} > body_high={bh:.0f}")
        elif open_price < bl:
            self._gap_direction = "PE"
            log.info(f"[{self.name}]  GAP DOWN (fallback): open={open_price:.0f} < body_low={bl:.0f}")
        else:
            self._gap_direction = "BOTH"
            log.info(f"[{self.name}]  NO GAP (fallback): open inside [{bl:.0f}–{bh:.0f}]")

    def _attempt_gap_entry(self, index_price: float, ts: datetime):
        if self._trade_done or self._trade:
            return

        signal = self._gap_direction

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.warning(f"[{self.name}] Gap entry fallback lookup | signal={signal} "
                        f"strike={strike} expiry={expiry} spot={index_price:.2f}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No option token for gap {signal} entry | "
                      f"spot={index_price:.2f} expiry={self._expiry_date}")
            return

        self._build_entry(sym, token, signal, ts, reason=f"gap_{signal.lower()}")

    # ── 2-candle signal ───────────────────────────────────────────────────────

    def _check_2candle_signal(self, latest_closed: dict, index_price: float, ts: datetime):
        last_two = self._index_8s.last_n_closed(2)
        if len(last_two) < 2:
            return

        c1, c2 = last_two[-2], last_two[-1]
        signal = self._check_signal(c1, c2)
        if not signal:
            return

        log.info(f"[{self.name}] 2-candle signal: {signal} at {ts.strftime('%H:%M:%S')}")

        if signal == "CE" and self._pre_ce_token:
            sym, token = self._pre_ce_sym, self._pre_ce_token
        elif signal == "PE" and self._pre_pe_token:
            sym, token = self._pre_pe_sym, self._pre_pe_token
        else:
            from core.instruments import get_atm_strike
            strike = get_atm_strike(index_price)
            expiry = self._expiry_date
            log.info(f"[{self.name}] Fallback token lookup | signal={signal} "
                     f"strike={strike} expiry={expiry}")
            token, sym = self._instruments.get_option_token(strike, signal, expiry)

        if not token or not sym:
            log.error(f"[{self.name}] No token for {signal} — trade SKIPPED")
            return

        self.subscribe_option(token)
        self._build_entry(sym, token, signal, ts, reason="2x8s_signal")

    # ── Entry / Exit ──────────────────────────────────────────────────────────

    def _build_entry(self, sym: str, token: int, signal: str, ts: datetime, reason: str):
        opt_price = self.get_price(token)
        if not opt_price or opt_price <= 0:
            log.warning(f"[{self.name}] No live price for {sym} — retrying in 2s...")
            import time; time.sleep(2)
            opt_price = self.get_price(token)
        if not opt_price or opt_price <= 0:
            log.warning(f"[{self.name}] No live price for {sym} after retry — cannot enter")
            return

        # FIX: reject stale pre-open prices
        # Options receive indicative ticks during 9:00-9:15 pre-open at theoretical
        # (prev-close-based) prices. If the index gaps, these prices are completely
        # wrong. Only use a price that arrived AT or AFTER market open (9:15:00).
        price_ts = self.get_price_ts(token)
        market_open_today = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        if price_ts is None or price_ts < market_open_today:
            log.warning(
                f"[{self.name}] Stale pre-open price for {sym} "
                f"(price={opt_price:.2f} priced_at={price_ts}) — "
                f"waiting for first post-9:15 tick"
            )
            # Store pending entry; will be triggered by next on_option_tick
            self._pending_entry = {
                "sym": sym, "token": token, "signal": signal, "ts": ts, "reason": reason
            }
            return

        sl = opt_price - CFG["initial_sl_buffer"]
        self._opt_8s = SecondCandleBuilder(seconds=CFG["bucket_sec"])
        self._trade = {
            "state": "OPEN", "symbol": sym, "token": token, "signal": signal,
            "entry": opt_price, "sl": sl, "highest_seen": opt_price,
            "entry_time": ts, "gap_direction": self._gap_direction,
        }

        log.info(f"[{self.name}]  ENTRY {sym} @ {opt_price:.0f} | "
                 f"SL={sl:.0f} | Trail kicks at {opt_price + CFG['trail_trigger_pts']:.0f} "
                 f"| Reason={reason}")

        self._log_csv({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "symbol": sym,
            "action": "ENTRY", "price": opt_price, "sl": sl,
            "status": "OPEN", "pnl": 0, "reason": reason,
            "gap_direction": self._gap_direction,
        })

    def _do_exit(self, exit_price: float, reason: str, ts: datetime):
        t = self._trade
        if not t or t["state"] != "OPEN":
            return
        t["state"] = "CLOSED"
        pnl = (exit_price - t["entry"]) * CFG["quantity"]
        self._today_pnl += pnl
        self._trade_done = True

        log.info(f"[{self.name}]  EXIT [{reason}] {t['symbol']} @ {exit_price:.0f} "
                 f"| PnL {pnl:.0f} | Today {self._today_pnl:.0f}")

        self._log_csv({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "symbol": t["symbol"],
            "action": "EXIT", "price": exit_price, "sl": t["sl"],
            "status": "CLOSED", "pnl": round(pnl, 2), "reason": reason,
            "gap_direction": t["gap_direction"],
        })
        self._completed.append({**t, "exit_price": exit_price, "exit_reason": reason, "pnl": pnl})

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_signal(self, c1: dict, c2: dict) -> Optional[str]:
        if self._is_doji(c1) or self._is_doji(c2):
            return None
        if c1["close"] > c1["open"] and c2["close"] > c2["open"]:
            return "CE"
        if c1["close"] < c1["open"] and c2["close"] < c2["open"]:
            return "PE"
        return None

    @staticmethod
    def _is_doji(c: dict) -> bool:
        rng = c["high"] - c["low"]
        if rng == 0:
            return True
        return abs(c["close"] - c["open"]) / rng < CFG["doji_threshold"]

    @staticmethod
    def _compute_trailing_sl(entry: float, highest: float, current_sl: float) -> float:
        profit = highest - entry
        if profit >= CFG["trail_trigger_pts"]:
            new_sl = highest - CFG["trail_distance"]
            return max(new_sl, current_sl)
        return current_sl

    def _is_momentum_dead(self, signal: str, entry_time: datetime) -> bool:
        return False  # DISABLED — exit managed by trailing SL and force-exit at 9:30

    def _log_csv(self, row: dict):
        fname  = CFG["csv_file"]
        exists = os.path.isfile(fname)
        fields = ["timestamp", "symbol", "action", "price",
                  "sl", "status", "pnl", "reason", "gap_direction"]
        with open(fname, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def eod_summary(self):
        log.info(f"\n[{self.name}] {'='*50}")
        log.info(f"[{self.name}] END OF DAY")
        log.info(f"[{self.name}] Gap direction  : {self._gap_direction}")
        log.info(f"[{self.name}] Trade executed : {'Yes' if self._trade_done else 'No'}")
        for t in self._completed:
            log.info(f"[{self.name}]   {t['symbol']} {t['exit_reason']} "
                     f"entry={t['entry']:.0f} exit={t['exit_price']:.0f} PnL={t['pnl']:.0f}")
        log.info(f"[{self.name}] Today PnL      : {self._today_pnl:.0f}")
        log.info(f"[{self.name}] {'='*50}\n")
