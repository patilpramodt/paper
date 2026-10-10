"""
core/option_context.py

Option-chain confirmation gate for the stock-option scanners, built on the
live PCR / OI history TickRecorder keeps (hub.recorder.ctx_nifty / ctx_stock).

A trade is taken only when the option market agrees with its direction:

  "pcr"        NIFTY ATM±10 OI PCR on the trade's side of 1.0 — above for a
               CE (puts being written under the market), below for a PE.
  "nifty_dir"  NIFTY above its day open for a CE, below for a PE.
  "oi_flow"    the stock's own near-ATM OI on the OPPOSITE side rose over the
               last oi_lookback_min — put writing under a CE buy, call writing
               over a PE buy.

WHY (2026-10-10 study — every scanner trade Oct 7-9 replayed on the recorded
chain, real option bid/ask, spot 2x/3x ATR bracket, no T45):

                                  trades   Rs/trade
    all four scanners                369      -221
    pcr + nifty_dir                  175      +439   (FLOW 3/3 days positive)
    pcr + nifty_dir + oi_flow        131      +644   (RT and FLOW positive all 3 days)
    rejected by all three            238      -700

Signed NIFTY PCR and NIFTY's own direction are 0.82 rank-correlated, but
neither alone was as good as both. Thresholds were not tuned: PCR edges of
0 / 0.05 / 0.10 were all positive. VIX (level, intraday change), stock IV
level/rank, ATM skew, option bid/ask imbalance and the stock's own PCR were
tested the same way and are NOT used — none separated winners consistently.

This is THREE sessions of option data (the recorder started 2026-10-07); it
cannot be checked on the 13-month spot history. Treat the paper record as
the test. Missing context (recorder off, warm-up) blocks the trade.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("core.option_context")

NIFTY_TOKEN = 256265

OPT_CTX_DEFAULTS = {
    "pcr":             True,
    "pcr_edge":        0.0,    # PCR must clear 1 +/- this
    "nifty_dir":       True,
    "oi_flow":         True,
    "oi_lookback_min": 15,
    "max_age_min":     3,      # a PCR / OI print older than this counts as missing
}


class OptionContext:
    def __init__(self, hub, cfg: Optional[dict] = None):
        self._hub = hub
        self.cfg = {**OPT_CTX_DEFAULTS, **(cfg or {})}
        self._day = None
        self._nifty_open = None
        self._next_fetch = None
        self._warned = False

    def _open(self, now: datetime) -> Optional[float]:
        if now.date() != self._day:
            self._day, self._nifty_open, self._next_fetch = now.date(), None, None
        if self._nifty_open or (self._next_fetch and now < self._next_fetch):
            return self._nifty_open
        kite = getattr(self._hub, "kite", None)
        try:
            o = kite.ohlc(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["ohlc"]["open"]
            self._nifty_open = float(o) if o else None
        except Exception as e:
            log.warning(f"NIFTY open fetch failed ({e}) — retry in 5 min")
        if not self._nifty_open:
            self._next_fetch = now + timedelta(minutes=5)
        return self._nifty_open

    def check(self, sym: str, side: str, ts: datetime):
        """Returns (ok, block_reason, meta)."""
        c, sg, meta = self.cfg, (1 if side == "UP" else -1), {}
        rec = getattr(self._hub, "recorder", None)
        if rec is None:
            if not self._warned:
                log.error("option context needs the TickRecorder — every entry is blocked")
                self._warned = True
            return False, "ctx_no_recorder", meta
        stale = ts - timedelta(minutes=c["max_age_min"])

        if c["pcr"]:
            hist = tuple(rec.ctx_nifty)
            if not hist or hist[-1][0] < stale:
                return False, "ctx_no_pcr", meta
            pcr = hist[-1][1]
            meta["c_pcr"] = round(pcr, 3)
            if sg * (pcr - 1.0) <= c["pcr_edge"]:
                return False, f"pcr={pcr:.2f}", meta

        if c["nifty_dir"]:
            n_open, n_px = self._open(ts), self._hub.last_price(NIFTY_TOKEN)
            if not (n_open and n_px):
                return False, "ctx_no_nifty", meta
            n_ret = (n_px / n_open - 1) * 100
            meta["c_nifty_pct"] = round(n_ret, 3)
            if sg * n_ret <= 0:
                return False, f"nifty={n_ret:+.2f}%", meta

        if c["oi_flow"]:
            hist = tuple(rec.ctx_stock.get(sym) or ())
            if not hist or hist[-1][0] < stale:
                return False, "ctx_no_oi", meta
            now_t, ce, pe = hist[-1]
            then = [h for h in hist if h[0] <= now_t - timedelta(minutes=c["oi_lookback_min"])]
            if not then:
                return False, "ctx_oi_warmup", meta
            _, ce0, pe0 = then[-1]
            opp, opp0 = (pe, pe0) if sg > 0 else (ce, ce0)
            chg = (opp / opp0 - 1) * 100 if opp0 else 0.0
            meta["c_opp_oi_pct"] = round(chg, 2)
            if chg <= 0:
                return False, f"opp_oi={chg:+.2f}%", meta

        return True, "", meta
