"""
core/entry_gate.py

Market-context entry gates + spot-ATR bracket exit for the stock-option
scanners. RT and FLOW each use their OWN gate (chosen 2026-09-28 by
replaying every live paper trade on real Sep-expiry option minute candles
from Kite, ATM strike from that day's spot, real spreads and costs):

  "rs_range"        (RT)   stock outperforms NIFTY from the open by
                           >= rs_min_pct in the signal's direction AND the
                           day's high-low range so far is >= range_min_atr x
                           1-min ATR14 (an active stock, not a dead one).
                           RT live trades, spot exit: 485/1201 kept,
                           +Rs 92k vs -Rs 190k actual; positive in both
                           halves and on 8/13 days; +Rs 41k without its best
                           day; keeps 47% of all winning trades.

  "rs_off_extreme"  (FLOW) RS >= rs_min_pct AND price is >= ext_min_atr x
                           1-min ATR14 back from the day's high (UP) / low
                           (DOWN) — a strong stock that has pulled back.
                           FLOW live trades, spot exit: 54/456 kept,
                           +Rs 58k vs -Rs 115k actual; positive in both
                           halves and on 4/6 days; +Rs 19k without its best
                           day. Keeps only 18% of winners — no FLOW rule that
                           kept a majority of winners was profitable.

  "pullback"               the earlier trend + RS + 15-min pullback gate
                           (kept for comparison; no longer the default).

All values are signed so positive = in the signal's direction.
"""

import logging
from collections import deque
from datetime import datetime, time as dtime, timedelta
from typing import Optional

log = logging.getLogger("core.entry_gate")

NIFTY_TOKEN = 256265

GATE_DEFAULTS = {
    "mode":              "rs_range",
    "rs_min_pct":        0.10,
    "range_min_atr":     12.0,   # rs_range
    "ext_min_atr":       3.0,    # rs_off_extreme
    "trend_min_pct":     0.30,   # pullback
    "pull_min_pct":      0.15,   # pullback
    "pull_lookback_min": 15,     # pullback
}


class EntryGate:
    def __init__(self, hub, cfg: Optional[dict] = None):
        self._hub   = hub
        self.cfg    = {**GATE_DEFAULTS, **(cfg or {})}
        self._day   = None
        self._opens = {}          # sym -> day open ("NIFTY" for the index)
        self._hod   = {}
        self._lod   = {}
        self._hist  = {}          # sym -> deque[[minute, open, high, low, close]]
        self._next_fetch = None

    # ── feed ─────────────────────────────────────────────────────────────────
    def update(self, sym: str, price: float, ts: datetime):
        """Call on every spot tick."""
        if ts.date() != self._day:
            self._day, self._next_fetch = ts.date(), None
            self._opens, self._hod, self._lod, self._hist = {}, {}, {}, {}
        if ts.time() < dtime(9, 15):
            return
        self._opens.setdefault(sym, price)   # fallback; replaced by Kite ohlc
        self._hod[sym] = max(self._hod.get(sym, price), price)
        self._lod[sym] = min(self._lod.get(sym, price), price)
        minute = ts.replace(second=0, microsecond=0)
        h = self._hist.setdefault(sym, deque(maxlen=120))
        if h and h[-1][0] == minute:
            b = h[-1]
            b[2], b[3], b[4] = max(b[2], price), min(b[3], price), price
        else:
            h.append([minute, price, price, price, price])

    def _refresh_day_stats(self, now: datetime):
        """True day open/high/low from Kite once per day — a mid-session
        restart would otherwise start the day's stats from the first tick."""
        if self._next_fetch is not None and now < self._next_fetch:
            return
        kite = getattr(self._hub, "kite", None)
        if kite is None:
            return
        try:
            keys = [f"NSE:{s}" for s in self._hist] + ["NSE:NIFTY 50"]
            for k, v in kite.ohlc(keys).items():
                o = v.get("ohlc") or {}
                sym = "NIFTY" if k == "NSE:NIFTY 50" else k[4:]
                if o.get("open"):
                    self._opens[sym] = float(o["open"])
                if sym != "NIFTY" and o.get("high") and o.get("low"):
                    self._hod[sym] = max(self._hod.get(sym, 0.0), float(o["high"]))
                    self._lod[sym] = min(self._lod.get(sym, float("inf")), float(o["low"]))
            self._next_fetch = now + timedelta(hours=24)
        except Exception as e:
            log.warning(f"ohlc fetch failed ({e}) — using first-seen prices for day stats")
            self._next_fetch = now + timedelta(minutes=5)

    # ── indicators ───────────────────────────────────────────────────────────
    def atr14(self, sym: str) -> Optional[float]:
        """Mean 1-min true range over the last 14 completed minutes."""
        h = list(self._hist.get(sym) or [])[:-1]
        if len(h) < 6:
            return None
        h = h[-15:]
        trs = [max(b[2] - b[3], abs(b[2] - a[4]), abs(b[3] - a[4])) for a, b in zip(h, h[1:])]
        return sum(trs) / len(trs) if trs else None

    def atr5(self, sym: str) -> Optional[float]:
        """Mean of the rolling 5-minute high-low range over the last 14
        minutes — the ATR the exit backtest used (SL 2x / TP 3x of this)."""
        h = list(self._hist.get(sym) or [])
        if len(h) < 9:
            return None
        rngs = []
        for j in range(max(4, len(h) - 14), len(h)):
            w = h[j - 4:j + 1]
            rngs.append(max(b[2] for b in w) - min(b[3] for b in w))
        return sum(rngs) / len(rngs) if rngs else None

    def _price_ago(self, sym: str, ts: datetime, minutes: int) -> Optional[float]:
        h = self._hist.get(sym)
        target = ts.replace(second=0, microsecond=0) - timedelta(minutes=minutes)
        if not h or h[0][0] > target:
            return None
        px = None
        for b in h:
            if b[0] > target:
                break
            px = b[4]
        return px

    # ── decision ─────────────────────────────────────────────────────────────
    def check(self, sym: str, side: str, price: float, ts: datetime):
        """Returns (ok, block_reason, meta)."""
        self._refresh_day_stats(ts)
        c      = self.cfg
        sg     = 1 if side == "UP" else -1
        s_open = self._opens.get(sym)
        n_open = self._opens.get("NIFTY")
        n_px   = self._hub.last_price(NIFTY_TOKEN)
        if not (s_open and n_open and n_px):
            return False, "gate_no_data", {}

        dret = sg * (price / s_open - 1) * 100
        rs   = dret - sg * (n_px / n_open - 1) * 100
        meta = {"g_dret": round(dret, 3), "g_rs": round(rs, 3)}
        if rs < c["rs_min_pct"]:
            return False, f"rs={rs:.2f}", meta

        mode = c["mode"]
        if mode == "pullback":
            ago = self._price_ago(sym, ts, c["pull_lookback_min"])
            if not ago:
                return False, "gate_no_data", meta
            pull = sg * (price / ago - 1) * 100
            meta["g_pull"] = round(pull, 3)
            if dret < c["trend_min_pct"]:
                return False, f"trend={dret:.2f}", meta
            if pull > -c["pull_min_pct"]:
                return False, f"pull={pull:.2f}", meta
            return True, "", meta

        atr = self.atr14(sym)
        hod, lod = self._hod.get(sym), self._lod.get(sym)
        if not (atr and hod and lod):
            return False, "gate_no_data", meta
        if mode == "rs_range":
            rng = (hod - lod) / atr
            meta["g_range_atr"] = round(rng, 1)
            if rng < c["range_min_atr"]:
                return False, f"range={rng:.1f}atr", meta
            return True, "", meta
        if mode == "rs_off_extreme":
            ext = ((hod - price) if sg > 0 else (price - lod)) / atr
            meta["g_ext_atr"] = round(ext, 1)
            if ext < c["ext_min_atr"]:
                return False, f"ext={ext:.1f}atr", meta
            return True, "", meta
        return False, f"unknown_mode={mode}", meta


# ── spot-ATR bracket exit (shared by RT and FLOW) ─────────────────────────────
# SL 2x / TP 3x atr5 on the STOCK price, held to target/stop/EOD. On both
# strategies' gated live trades (real option candles) this beat the 60/90-min
# time caps, the RT ladder, FLOW's flat TP 300 / SL 1900, and the same bracket
# WITH the 30-min "exit if losing" check (RT +92k vs +72k with it, FLOW +58k vs
# +49k; win rate 47%/56% vs 34%/44%) — so time_min and t30_min are off.
SPOT_EXIT_DEFAULTS = {
    "sl_atr":      2.0,
    "tp_atr":      3.0,
    "time_min":    None,    # minutes; None = hold to EOD
    "t30_min":     None,    # minutes; None = no "exit if losing" check
    "backstop_rs": 5000.0,  # disaster stop on the premium only
}


def spot_bracket(side: str, spot: float, atr: float, cfg: dict) -> dict:
    sg = 1 if side == "UP" else -1
    return {
        "entry_spot": spot,
        "sl_spot":    round(spot - sg * cfg["sl_atr"] * atr, 2),
        "tp_spot":    round(spot + sg * cfg["tp_atr"] * atr, 2),
        "atr5":       atr,
    }


def spot_exit_reason(tr: dict, spot: Optional[float], unreal: float,
                     age_min: float, cfg: dict) -> Optional[str]:
    """Exit reason for a spot-bracket trade, or None to keep holding.
    Mutates tr["t30_done"]."""
    if unreal <= -cfg["backstop_rs"]:
        return "SL_BACKSTOP"
    if spot:
        up = tr["side"] == "UP"
        if (spot <= tr["sl_spot"]) if up else (spot >= tr["sl_spot"]):
            return "SL_SPOT"
        if (spot >= tr["tp_spot"]) if up else (spot <= tr["tp_spot"]):
            return "TP_SPOT"
    if cfg.get("t30_min") and not tr.get("t30_done") and age_min >= cfg["t30_min"]:
        tr["t30_done"] = True
        if unreal < 0:
            return "T30_EXIT"
    if cfg.get("time_min") and age_min >= cfg["time_min"]:
        return "TIME_STOP"
    return None
