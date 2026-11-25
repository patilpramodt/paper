"""
core/vix_tracker.py

Intraday India VIX context for the stock-option scanners (RT, FLOW).
LOGGING ONLY — nothing here gates or changes a trade.

WHY (2026-11-25 study, every RT/FLOW paper trade Nov 2–25, 14 sessions):
  VIX change from 09:15 to 09:45 was the strongest morning predictor of the
  scanners' day P&L out of 20 features tested (FLOW Spearman +0.90, RT +0.49).
                                     days   FLOW net    RT net
      VIX up >= ~1.8% by 09:45          7    +96k (7/7)  +35k
      otherwise                         7    -40k (0/7)  -45k
  Per trade (VIX >= +2% above its 09:15 value at entry): FLOW kept 68 trades
  +68k, skipped 48 -12k. Plausible mechanism: both scanners BUY options, so
  rising IV is a tailwind and an IV crush (e.g. gap-up days) a headwind.
  Threshold was picked after looking at the data and core/option_context.py
  found nothing for VIX on Oct 7-9 — this logging is the out-of-sample test.

Columns added to every signal and trade row (see VIX_FIELDS):
  vix                current INDIA VIX
  vix_open           first VIX seen at/after 09:15 (same definition as study)
  vix_chg_open_pct   % change vs vix_open
  vix_chg_30m_pct    % change vs the value 30 minutes earlier
  vix_0945_chg_pct   % change 09:15 -> 09:45, frozen at 09:45 (blank before)
  vix_trigger        1 if vix_0945_chg_pct >= TRIGGER_PCT, 0 if not, blank before 09:45
  vix_open_src       live | snapshot (recovered from recorder after a restart)
Trade EXIT rows also get vix_entry and vix_chg_trade_pct (entry -> exit).

Data: VIX comes from the Kite WebSocket token 264969 via hub.subscribe_passive,
which caches last_price without routing ticks to any strategy. If the bot
starts after 09:15, vix_open / the 09:45 value are recovered from the tick
recorder's data/ticks/<date>/snapshot_1m.csv when available.
"""

import csv
import logging
import os
from datetime import datetime, time as dtime, timedelta

log = logging.getLogger("core.vix_tracker")

VIX_TOKEN    = 264969
OPEN_TIME    = dtime(9, 15)
TRIGGER_TIME = dtime(9, 45)
TRIGGER_PCT  = 2.0

VIX_FIELDS = [
    "vix", "vix_open", "vix_chg_open_pct", "vix_chg_30m_pct",
    "vix_0945_chg_pct", "vix_trigger", "vix_open_src",
]
VIX_TRADE_FIELDS = ["vix_entry", "vix_chg_trade_pct"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pct(a, b):
    return round((a / b - 1) * 100, 3) if a and b else ""


class VixTracker:
    def __init__(self, hub, owner: str):
        self._hub     = hub
        self._owner   = owner
        self._open    = None
        self._src     = ""
        self._v0945   = None
        self._minutes = {}          # "HH:MM" -> last VIX seen in that minute
        self._announced = False

    def start(self):
        try:
            self._hub.subscribe_passive([VIX_TOKEN])
        except Exception as e:
            log.warning(f"[{self._owner}] VIX passive subscribe failed: {e}")

    # ── called from the strategy heartbeat (~1/s) ─────────────────────────────
    def update(self, ts: datetime):
        t = ts.time()
        if t < OPEN_TIME:
            return
        v = self._hub.last_price(VIX_TOKEN)
        if not v:
            return
        if self._open is None:
            late = t >= (datetime.combine(ts.date(), OPEN_TIME) + timedelta(minutes=2)).time()
            if late and self._recover(ts):
                pass
            else:
                self._open, self._src = v, "live"
            log.info(f"[{self._owner}] VIX open={self._open:.2f} (src={self._src})")
        self._minutes[ts.strftime("%H:%M")] = v
        if self._v0945 is None and t >= TRIGGER_TIME:
            self._v0945 = self._minutes.get("09:45") or v
        if self._v0945 is not None and not self._announced:
            self._announced = True
            chg = _pct(self._v0945, self._open)
            on = chg != "" and chg >= TRIGGER_PCT
            log.info(
                f"[{self._owner}] VIX 09:45 check: open={self._open:.2f} 09:45={self._v0945:.2f} "
                f"chg={chg}% -> trigger {'ON' if on else 'OFF'} (>= {TRIGGER_PCT}%; log only, not gating)"
            )

    def _recover(self, ts: datetime) -> bool:
        """Bot started late: take 09:15 / 09:45 VIX from the recorder snapshot."""
        p = os.path.join(_ROOT, "data", "ticks", ts.strftime("%Y-%m-%d"), "snapshot_1m.csv")
        try:
            with open(p, newline="") as f:
                for r in csv.DictReader(f):
                    hm = r["ts"][11:16]
                    if not r.get("vix"):
                        continue
                    v = float(r["vix"])
                    if hm >= "09:15":
                        self._minutes.setdefault(hm, v)
                        if self._open is None:
                            self._open = v
                    if hm == "09:45" and self._v0945 is None:
                        self._v0945 = v
        except Exception as e:
            log.warning(f"[{self._owner}] VIX snapshot recovery failed ({p}): {e}")
        if self._open is not None:
            self._src = "snapshot"
            return True
        return False

    # ── row values ────────────────────────────────────────────────────────────
    def current(self):
        return self._hub.last_price(VIX_TOKEN)

    def _ago(self, ts: datetime, minutes: int):
        key = (ts - timedelta(minutes=minutes)).strftime("%H:%M")
        earlier = [k for k in self._minutes if k <= key]
        if earlier:
            return self._minutes[max(earlier)]
        return self._open

    def fields(self, ts: datetime) -> dict:
        try:
            v = self.current()
            chg0945 = _pct(self._v0945, self._open) if self._v0945 else ""
            return {
                "vix": round(v, 2) if v else "",
                "vix_open": round(self._open, 2) if self._open else "",
                "vix_chg_open_pct": _pct(v, self._open),
                "vix_chg_30m_pct": _pct(v, self._ago(ts, 30)) if self._open else "",
                "vix_0945_chg_pct": chg0945,
                "vix_trigger": "" if chg0945 == "" else int(chg0945 >= TRIGGER_PCT),
                "vix_open_src": self._src,
            }
        except Exception as e:      # logging must never break trading
            log.warning(f"[{self._owner}] VIX fields error: {e}")
            return {}

    def summary(self) -> str:
        v = self.current()
        chg = _pct(self._v0945, self._open) if self._v0945 else ""
        trig = "n/a" if chg == "" else ("ON" if chg >= TRIGGER_PCT else "OFF")
        return (f"open={self._open or 'n/a'} 09:45={self._v0945 or 'n/a'} chg0945={chg or 'n/a'}% "
                f"trigger={trig} last={v or 'n/a'} chg_day={_pct(v, self._open) or 'n/a'}% src={self._src or 'n/a'}")
