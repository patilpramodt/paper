"""
strategies/orb30_rvol_strategy.py

ORB30 "stocks in play" variants. PAPER ONLY. Same marks / entry / candle SL as
ORB30 (strategies/orb30_strategy.py) with two changes:

  FILTER   trade a stock only if its first 30-min candle (09:15-09:45) volume
           is >= 2x the average of the same candle over the previous 14
           sessions (relative volume, Zarattini/Barbon/Aziz 2024).
           Stocks only — index spot has no volume. The first breakout of the
           day decides: below the threshold, the stock is done for the day.

  EXIT     ORB30_RVOL        no target — candle SL or 15:15 square-off
           ORB30_RVOL_TRAIL  no target — candle SL, plus a Rs1000 step lock on
                             option P&L: peak +1000 -> stop at entry (Rs0),
                             +2000 -> +1000, +3000 -> +2000 ...  (gross, bid side)

  Backtest (tools/backtest_orb30_exits.py entries, 2026-09-25..10-09, 18 trades):
    RVOL +Rs6.6k, RVOL_TRAIL +Rs3.6k vs live ORB30 -Rs1.21L on 167 trades.
    On spot over 87 days, rvol>=2 breakouts averaged +0.24R vs -0.01R for all.
"""

import logging
import threading
import time
from datetime import datetime, time as dtime, timedelta

from strategies.orb30_strategy import CFG as BASE_CFG, ORB30Strategy, _now_ist

CFG = {
    "rvol_min":        2.0,
    "rvol_lookback":   14,          # prior sessions averaged
    "rvol_min_days":   10,          # fewer prior sessions -> no rvol -> no trade
    "history_days":    30,          # calendar days fetched at pre-market
    "rvol_fetch_time": dtime(9, 46),
    "trail_step_rs":   1000.0,
}


class ORB30RvolStrategy(ORB30Strategy):

    _log          = logging.getLogger("strategy.orb30_rvol")
    CSV_FILE      = "orb30_rvol_trades.csv"
    TARGET_RS     = None
    TRADE_INDICES = False
    TRAIL_STEP_RS = None

    _FIELDS = ORB30Strategy._FIELDS + ["rvol", "peak_pnl"]

    def __init__(self, market_hub):
        super().__init__(market_hub)
        self._avg_first_vol = {}   # sym -> avg 09:15 30-min candle volume, prior sessions
        self._rvol          = {}   # sym -> today's relative volume
        self._rvol_started  = False

    @property
    def name(self) -> str:
        return "ORB30_RVOL"

    def _exit_desc(self) -> str:
        trail = f" + Rs{self.TRAIL_STEP_RS:.0f} step trail" if self.TRAIL_STEP_RS else ""
        return f"rvol>={CFG['rvol_min']} | no target{trail}"

    # ── relative volume ──────────────────────────────────────────────────────

    def pre_market(self, premarket_data, instruments, index_stores=None) -> bool:
        if not super().pre_market(premarket_data, instruments, index_stores):
            return False
        self._load_volume_history()
        if _now_ist().time() >= CFG["rvol_fetch_time"]:      # mid-session restart
            self._rvol_started = True
            self._fetch_today_rvol()
        return True

    def _first_candles(self, tok: int, start: datetime, end: datetime) -> dict:
        """date -> volume of the 09:15 30-min candle."""
        kite = self._hub.kite
        try:
            raw = kite.historical_data(tok, start.strftime("%Y-%m-%d %H:%M:%S"),
                                       end.strftime("%Y-%m-%d %H:%M:%S"), "30minute")
        finally:
            time.sleep(0.35)   # Kite historical API: 3 requests/second
        return {r["date"].date(): r.get("volume", 0) or 0
                for r in raw if r["date"].time() == BASE_CFG["session_open"]}

    def _load_volume_history(self):
        if getattr(self._hub, "kite", None) is None:
            self._log.error(f"[{self.name}] no kite handle — no rvol, no trades today")
            return
        today = _now_ist().date()
        start = datetime.combine(today - timedelta(days=CFG["history_days"]), dtime(9, 0))
        end   = datetime.combine(today, dtime(9, 0))
        for sym, st in self._inst.items():
            try:
                vols = self._first_candles(st["token"], start, end)
            except Exception as e:
                self._log.warning(f"[{self.name}] volume history failed for {sym}: {e}")
                continue
            prior = [v for d, v in sorted(vols.items()) if d < today and v > 0][-CFG["rvol_lookback"]:]
            if len(prior) >= CFG["rvol_min_days"]:
                self._avg_first_vol[sym] = sum(prior) / len(prior)
        self._log.info(f"[{self.name}] volume history for {len(self._avg_first_vol)}/{len(self._inst)} stocks")

    def _fetch_today_rvol(self):
        today = _now_ist().date()
        start = datetime.combine(today, BASE_CFG["session_open"])
        end   = _now_ist()
        for sym, avg in self._avg_first_vol.items():
            try:
                v = self._first_candles(self._inst[sym]["token"], start, end).get(today)
            except Exception as e:
                self._log.warning(f"[{self.name}] today's volume failed for {sym}: {e}")
                continue
            if v:
                self._rvol[sym] = round(v / avg, 2)
        hot = sorted(((r, s) for s, r in self._rvol.items() if r >= CFG["rvol_min"]), reverse=True)
        self._log.info(f"[{self.name}] rvol for {len(self._rvol)} stocks | in play (>= {CFG['rvol_min']}): "
                       + (", ".join(f"{s} {r:.1f}" for r, s in hot) or "none"))

    def _heartbeat(self, ts: datetime):
        if not self._rvol_started and ts.time() >= CFG["rvol_fetch_time"]:
            # Off the tick thread: ~21 history calls at 3/s. First signal is at 10:15.
            self._rvol_started = True
            threading.Thread(target=self._fetch_today_rvol, name="orb30-rvol", daemon=True).start()
        super()._heartbeat(ts)

    def _arm_entry(self, sym: str, side: str, spot: float, sl_spot: float, ts: datetime):
        rvol = self._rvol.get(sym)
        if rvol is None or rvol < CFG["rvol_min"]:
            # rvol is fixed for the day, so the first breakout decides.
            self._inst[sym]["traded"] = True
            self._log.info(f"[{self.name}] {sym} skipped — rvol "
                           f"{'n/a' if rvol is None else f'{rvol:.2f}'} < {CFG['rvol_min']}")
            return
        self._log.info(f"[{self.name}] {sym} rvol {rvol:.2f} — in play")
        super()._arm_entry(sym, side, spot, sl_spot, ts)

    # ── exits ────────────────────────────────────────────────────────────────

    def _check_target(self, sym: str, ltp: float, ts: datetime):
        tr = self._positions.get(sym)
        if tr is None:
            return
        pnl = (max(0.05, ltp - tr["spread"] / 2.0) - tr["entry"]) * tr["qty"]
        tr["peak_pnl"] = max(tr.get("peak_pnl", 0.0), pnl)
        step = self.TRAIL_STEP_RS
        if step and tr["peak_pnl"] >= step:
            lock = (tr["peak_pnl"] // step - 1) * step
            if pnl <= lock:
                self._exit(sym, "TRAIL", ts, ltp)

    def _row(self, tr: dict) -> dict:
        row = super()._row(tr)
        row.update(rvol=self._rvol.get(tr["sym"], ""), peak_pnl=round(tr.get("peak_pnl", 0.0), 2))
        return row


class ORB30RvolTrailStrategy(ORB30RvolStrategy):

    _log          = logging.getLogger("strategy.orb30_rvol_trail")
    CSV_FILE      = "orb30_rvol_trail_trades.csv"
    TRAIL_STEP_RS = CFG["trail_step_rs"]

    @property
    def name(self) -> str:
        return "ORB30_RVOL_TRAIL"
