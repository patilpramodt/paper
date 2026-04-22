"""
core/pcr_kite.py

WsPCR — Put/Call Ratio from Zerodha WebSocket OI data.

Zero external libraries. Zero NSE HTTP calls. Zero extra Zerodha subscriptions
beyond what MarketHub already needs for trading.

HOW IT WORKS
─────────────
Zerodha KiteTicker sends open_interest (oi) in every MODE_FULL tick for
option tokens. MarketHub already subscribes all option tokens in MODE_FULL.
The market_hub.py patch stores tick["oi"] per token in hub._last_oi.
WsPCR reads hub.last_oi(token) for ATM±range CE and PE strikes and sums:
    PCR = total_PE_OI / total_CE_OI

This gives the same PCR that NSE publishes, computed from live WS data,
refreshed on every premarket refresh cycle without any HTTP round-trips.

USAGE (wired in t.py)
──────────────────────
    ws_pcr = WsPCR(hub, instruments, pm.expiry_date, spot_range=1000, step=100)
    ws_pcr.setup()    # subscribe CE+PE for ATM±1000pt (42 tokens, deduped with BB_STOCH)
    pm.start_live_refresh(hub._done, vix_interval=300, pcr_interval=600, ws_pcr=ws_pcr)

USAGE (in premarket refresh loop)
───────────────────────────────────
    pcr = ws_pcr.compute_pcr()   # reads from hub cache — instant, no I/O

WHY BETTER THAN NSE HTTP
──────────────────────────
  • No session management, no bot-detection, no soft-block risk
  • PCR is always current (reflects every tick, not a stale scrape)
  • No 8-second wait for OptionStream to warm up
  • Works inside GitHub Actions (no outbound HTTP to NSE needed)
  • Sub-millisecond compute_pcr() — can be called every refresh cycle
"""

import logging
from datetime import date
from typing import Optional

log = logging.getLogger("core.pcr_kite")


class WsPCR:
    """
    Compute PCR from Zerodha WebSocket OI data streamed in MODE_FULL.

    Parameters
    ──────────
    hub         : MarketHub instance  (subscribe() + last_oi() + last_price())
    instruments : InstrumentStore instance  (get_option_token())
    expiry_date : weekly expiry date (from PreMarketData.expiry_date)
    spot_range  : subscribe ATM ± this many index points  (default 1000)
    step        : BankNifty strike spacing                (default 100)
    min_active  : minimum tokens with OI>0 needed per side before returning PCR
    """

    def __init__(
        self,
        hub,
        instruments,
        expiry_date: date,
        spot_range: int = 1000,
        step: int = 100,
        min_active: int = 5,
    ):
        self._hub         = hub
        self._instruments = instruments
        self._expiry      = expiry_date
        self._range       = spot_range
        self._step        = step
        self._min_active  = min_active

        # Lists of subscribed tokens by type
        self._ce_tokens: list[int] = []
        self._pe_tokens: list[int] = []

        # ATM center used for last subscription batch
        # Used to detect when spot has moved enough to need fresh tokens
        self._last_subscribed_atm: Optional[int] = None

        self._setup_done = False

    # ── Reference spot ────────────────────────────────────────────────────────

    def _current_spot(self) -> float:
        """Return live BankNifty spot from hub's WebSocket price cache."""
        price = self._hub.last_price(self._hub._index_token)
        return float(price) if price else 0.0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Subscribe CE + PE tokens for ATM ± spot_range.
        Called once from t.py after hub.load_kite() and instruments.load().

        If the WebSocket isn't connected yet and spot is unavailable,
        subscriptions are deferred to the first compute_pcr() call.
        Deduplication is handled by MarketHub.subscribe() which is a no-op
        for already-subscribed tokens, so overlap with BB_STOCH ATM tokens
        is free.
        """
        spot = self._current_spot()
        if spot <= 0:
            log.info(
                "[WsPCR] setup() called before first WS tick — "
                "subscriptions deferred to first compute_pcr()"
            )
            self._setup_done = False
            return

        self._subscribe_around(spot)
        self._setup_done = True
        log.info(
            f"[WsPCR] setup() complete | "
            f"CE={len(self._ce_tokens)} PE={len(self._pe_tokens)} tokens subscribed | "
            f"range=±{self._range}pt step={self._step}"
        )

    # ── Core subscription helper ──────────────────────────────────────────────

    def _subscribe_around(self, spot: float):
        """
        Subscribe CE + PE tokens for all strikes in ATM ± range.

        BankNifty has 100pt strike spacing. With default range=1000 and step=100:
            offsets = [-1000, -900, ..., 0, ..., 900, 1000] = 21 strikes
            tokens  = 21 CE + 21 PE = 42 total
        Most of these overlap with BB_STOCH's ATM±400 subscriptions — zero
        extra bandwidth since MarketHub.subscribe() deduplicates.
        """
        from core.instruments import get_atm_strike

        atm = get_atm_strike(spot)
        self._last_subscribed_atm = atm

        subscribed_new = 0
        offsets = range(-self._range, self._range + self._step, self._step)

        for offset in offsets:
            strike = atm + offset
            for opt_type, token_list in (("CE", self._ce_tokens), ("PE", self._pe_tokens)):
                tok, sym = self._instruments.get_option_token(
                    strike, opt_type, self._expiry
                )
                if not tok:
                    log.debug(f"[WsPCR] Token not found: {strike}{opt_type} {self._expiry}")
                    continue
                if tok in token_list:
                    continue   # already subscribed
                token_list.append(tok)
                self._hub.subscribe(tok)   # no-op if already subscribed by BB_STOCH etc.
                subscribed_new += 1

        log.info(
            f"[WsPCR] Subscribed {subscribed_new} new tokens around "
            f"ATM={atm} (spot={spot:.0f}) | "
            f"CE={len(self._ce_tokens)} PE={len(self._pe_tokens)} total"
        )

    # ── Compute PCR ───────────────────────────────────────────────────────────

    def compute_pcr(self) -> Optional[float]:
        """
        Read OI from hub's WebSocket OI cache and return PCR = PE_OI / CE_OI.

        This is a pure in-memory read — no HTTP, no I/O, sub-millisecond.
        Returns None when:
          - Spot is not yet available (pre-9:15)
          - Fewer than min_active tokens have OI > 0 on either side
            (OI ticks haven't arrived yet, or expiry mismatch)
          - CE total OI is zero

        Called by premarket.py refresh loop every pcr_interval seconds.
        """
        # Lazy init: subscribe tokens on first call if setup() was too early
        if not self._setup_done:
            spot = self._current_spot()
            if spot > 0:
                self._subscribe_around(spot)
                self._setup_done = True
                log.info("[WsPCR] Deferred setup complete on first compute_pcr()")
            else:
                log.warning("[WsPCR] compute_pcr(): spot not yet available — returning None")
                return None

        # Dynamic range refresh if spot has moved significantly
        self._refresh_tokens_if_needed()

        # Sum OI across all subscribed tokens
        total_ce_oi = sum(self._hub.last_oi(t) for t in self._ce_tokens)
        total_pe_oi = sum(self._hub.last_oi(t) for t in self._pe_tokens)

        # Count tokens with non-zero OI (data quality gate)
        active_ce = sum(1 for t in self._ce_tokens if self._hub.last_oi(t) > 0)
        active_pe = sum(1 for t in self._pe_tokens if self._hub.last_oi(t) > 0)

        log.debug(
            f"[WsPCR] CE_OI={total_ce_oi:,} ({active_ce}/{len(self._ce_tokens)} tokens) | "
            f"PE_OI={total_pe_oi:,} ({active_pe}/{len(self._pe_tokens)} tokens)"
        )

        # Quality gate: need enough tokens with live OI on both sides
        if active_ce < self._min_active or active_pe < self._min_active:
            log.warning(
                f"[WsPCR] Not enough OI data yet: "
                f"CE={active_ce}/{len(self._ce_tokens)} "
                f"PE={active_pe}/{len(self._pe_tokens)} active — "
                f"need {self._min_active} each side. "
                f"OI ticks still arriving (check MODE_FULL subscription)"
            )
            return None

        if total_ce_oi == 0:
            log.warning("[WsPCR] CE total OI is 0 — returning None")
            return None

        pcr = round(total_pe_oi / total_ce_oi, 3)

        log.info(
            f"[WsPCR] PCR={pcr} | "
            f"PE_OI={total_pe_oi:,}  CE_OI={total_ce_oi:,} | "
            f"active: CE={active_ce} PE={active_pe} | "
            f"total tokens: CE={len(self._ce_tokens)} PE={len(self._pe_tokens)}"
        )
        return pcr

    # ── Dynamic range refresh ─────────────────────────────────────────────────

    def _refresh_tokens_if_needed(self):
        """
        Extend token coverage if spot has moved beyond half the range from
        the last subscribed ATM center.

        Example: range=1000, ATM at setup=50000, spot now=51100.
          Half-range = 500. Drift = |51100 - 50000| = 1100 > 500.
          → Subscribe fresh tokens around ATM=51100.
          Old tokens remain subscribed (they won't generate OI so don't
          distort the PCR — they just sit at OI=0 in the sum, harmless
          since the quality gate filters on active token count).
        """
        spot = self._current_spot()
        if spot <= 0 or not self._last_subscribed_atm:
            return

        from core.instruments import get_atm_strike
        current_atm = get_atm_strike(spot)
        drift = abs(current_atm - self._last_subscribed_atm)

        if drift > self._range // 2:
            log.info(
                f"[WsPCR] Spot moved {drift}pt from last ATM center "
                f"({self._last_subscribed_atm} → {current_atm}) — "
                f"extending token coverage"
            )
            self._subscribe_around(spot)

    # ── Summary / diagnostics ─────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return current PCR state as a dict — useful for logging/debugging."""
        total_ce = sum(self._hub.last_oi(t) for t in self._ce_tokens)
        total_pe = sum(self._hub.last_oi(t) for t in self._pe_tokens)
        active_ce = sum(1 for t in self._ce_tokens if self._hub.last_oi(t) > 0)
        active_pe = sum(1 for t in self._pe_tokens if self._hub.last_oi(t) > 0)
        pcr = round(total_pe / total_ce, 3) if total_ce > 0 else None
        return {
            "pcr"        : pcr,
            "ce_oi"      : total_ce,
            "pe_oi"      : total_pe,
            "ce_tokens"  : len(self._ce_tokens),
            "pe_tokens"  : len(self._pe_tokens),
            "active_ce"  : active_ce,
            "active_pe"  : active_pe,
            "setup_done" : self._setup_done,
            "atm_center" : self._last_subscribed_atm,
        }

    def log_summary(self):
        """Log a one-line diagnostic summary."""
        s = self.summary()
        log.info(
            f"[WsPCR] Summary | PCR={s['pcr']} | "
            f"CE_OI={s['ce_oi']:,} PE_OI={s['pe_oi']:,} | "
            f"tokens CE={s['ce_tokens']} PE={s['pe_tokens']} | "
            f"active CE={s['active_ce']} PE={s['active_pe']} | "
            f"ATM_center={s['atm_center']}"
        )
