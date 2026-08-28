"""
core/instruments.py

Fetched ONCE at startup. Shared by all strategies.
No strategy calls kite.instruments() directly  they ask this module.

Provides:
   get_option_token(strike, opt_type, expiry_date)  (token, symbol)
   get_atm_strike(spot)  nearest 100-multiple
   get_nearest_expiry(df, date)  nearest actual expiry from NFO data
"""

import logging
from datetime import date, timezone, timedelta

_IST = timezone(timedelta(hours=5, minutes=30))
def _today_ist(): return __import__('datetime').datetime.now(tz=_IST).date()

import pandas as pd

log = logging.getLogger("core.instruments")


def get_nearest_expiry(df: "pd.DataFrame", for_date: date) -> date:
    """
    Returns the nearest expiry on or after for_date from the live NFO
    instruments DataFrame.

    BankNifty no longer has weekly expiry (removed by SEBI in 2024).
    It now has monthly expiry only. Never use hardcoded weekday arithmetic --
    always derive the expiry from what Kite actually has in its NFO list.

    Args:
        df:       The instruments DataFrame from InstrumentStore._df
        for_date: The reference date (usually date.today())

    Returns:
        Nearest available expiry date >= for_date.

    Raises:
        ValueError if no future expiries found (instruments not loaded yet).
    """
    future = sorted(d for d in df["expiry"].dt.date.unique() if d >= for_date)
    if not future:
        raise ValueError(
            f"No future expiries found in NFO instruments on {for_date}. "
            "Instruments may not be loaded yet."
        )
    return future[0]


def get_atm_strike(spot: float, step: int = 100) -> int:
    return int(round(spot / step) * step)


class InstrumentStore:
    """
    Loaded once at startup via load(kite).
    All strategies call get_option_token() on the shared instance.
    Zero extra API calls after initial load.
    """

    def __init__(self):
        self._df   = None
        self._root = "BANKNIFTY"

    def load(self, kite, option_root: str = "BANKNIFTY"):
        self._root = option_root
        log.info(" Loading NFO instruments (once for all strategies)...")
        raw      = kite.instruments("NFO")
        df       = pd.DataFrame(raw)
        df       = df[df["name"] == option_root].copy()
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
        df["strike"] = df["strike"].astype(float)
        self._df = df
        avail    = sorted(df["expiry"].dt.date.unique())
        log.info(f" {len(df)} {option_root} contracts | available expiries: {avail[:8]}")

    def get_option_token(
        self,
        strike: int,
        opt_type: str,      # "CE" or "PE"
        expiry_date: date,
    ) -> tuple[int | None, str | None]:
        """
        Find Kite instrument_token for a contract.
        Tries 100, 200, 300 if exact strike not listed.
        Returns (None, None) if expiry is historical / not in NFO list.
        """
        if self._df is None:
            log.error("InstrumentStore not loaded. Call load(kite) first.")
            return None, None

        for adj in [strike, strike+100, strike-100,
                    strike+200, strike-200, strike+300, strike-300]:
            mask = (
                (self._df["strike"]          == float(adj)) &
                (self._df["instrument_type"] == opt_type) &
                (self._df["expiry"].dt.date  == expiry_date)
            )
            hit = self._df[mask]
            if not hit.empty:
                if adj != strike:
                    log.warning(f"  ATM {strike} not found  using {adj}")
                r = hit.iloc[0]
                return int(r["instrument_token"]), str(r["tradingsymbol"])

        return None, None   # historical expiry  caller uses B-S fallback

    def get_nearest_expiry_token(
        self,
        spot: float,
        opt_type: str,
    ) -> tuple[int | None, str | None]:
        """
        Get token for nearest expiry ATM option.
        Used by Spike strategy which wants the front-week contract.
        """
        if self._df is None:
            return None, None

        strike     = get_atm_strike(spot)
        today      = _today_ist()
        future_exp = self._df[self._df["expiry"].dt.date >= today]["expiry"].unique()
        if len(future_exp) == 0:
            return None, None

        nearest = sorted(future_exp)[0].date()
        return self.get_option_token(strike, opt_type, nearest)


# ═══════════════════════════════════════════════════════════════════════════════
# StockOptionStore — multi-underlying NFO option store (STOCK_OPT_SCANNER)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY A SECOND STORE
# ------------------
# InstrumentStore.load() filters the NFO dump down to ONE `name` (BANKNIFTY or
# NIFTY) and hard-codes a 100-point strike step. Neither assumption survives
# contact with stock options:
#
#   • ~180 different underlyings, each with its own strike step
#     (RELIANCE ~20, SBIN ~10, HDFCBANK ~20, TATAMOTORS ~10 ...)
#   • each with its own LOT SIZE (RELIANCE 500, SBIN 1500, TATAMOTORS 1425 ...)
#     — this is the single most important field. A rupee profit target is
#     meaningless without it: 1 premium point on RELIANCE is Rs 500, on
#     SBIN it is Rs 1500.
#   • monthly expiry only (SEBI removed stock-option weeklies)
#
# The strike step is DERIVED from the listed strikes for that underlying, never
# assumed — the same defensive rule get_nearest_expiry() already follows for
# expiries.
#
# One kite.instruments("NFO") call, shared. Also resolves the NSE equity token
# for each underlying so the scanner can subscribe the spot for its signal.

# Index roots to exclude — these have their own dedicated strategies/stores.
_INDEX_ROOTS = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50",
    "SENSEX", "BANKEX", "SENSEX50",
}


class StockOptionStore:
    """
    Loads every NFO stock option chain once, plus the NSE equity token for
    each underlying.

        store = StockOptionStore()
        store.load(kite, universe=["RELIANCE", "SBIN", ...])

        store.lot_size("SBIN")                 -> 1500
        store.spot_token("SBIN")               -> 779521
        store.atm_strike("SBIN", 812.4)        -> 810
        store.get_option("SBIN", 810, "CE")    -> (token, tradingsymbol, lot_size)
    """

    def __init__(self):
        self._df        = None          # NFO options for the universe
        self._universe  = []            # resolved, tradable underlyings
        self._lot       = {}            # symbol -> lot_size
        self._step      = {}            # symbol -> strike step
        self._expiry    = {}            # symbol -> nearest expiry date
        self._spot_tok  = {}            # symbol -> NSE equity instrument_token

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, kite, universe: list[str], nfo_raw=None, nse_raw=None):
        """
        universe : list of underlying names as they appear in the NFO dump
                   ("RELIANCE", "HDFCBANK", ...). Names not present in the
                   dump (delisted / removed from F&O) are dropped with a
                   warning rather than raising — the scanner just trades a
                   smaller universe.
        nfo_raw / nse_raw : optional pre-fetched kite.instruments() results,
                   to avoid repeating the (large, rate-limited) download.
        """
        want = {u.upper().strip() for u in universe} - _INDEX_ROOTS

        log.info(f" Loading NFO stock options for {len(want)} underlyings...")
        raw = nfo_raw if nfo_raw is not None else kite.instruments("NFO")
        df  = pd.DataFrame(raw)

        df = df[
            (df["name"].isin(want)) &
            (df["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if df.empty:
            log.error(" StockOptionStore: no matching stock option contracts found.")
            self._df = df
            return

        df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
        df["strike"] = df["strike"].astype(float)
        self._df = df

        today = _today_ist()
        for sym, g in df.groupby("name"):
            future = sorted(d for d in g["expiry"].dt.date.unique() if d >= today)
            if not future:
                log.warning(f"  {sym}: no future expiry in NFO dump — skipped")
                continue
            exp = future[0]

            # Lot size is per-underlying and revised by NSE periodically.
            # ALWAYS read it from the live dump — never hard-code it. A stale
            # lot size silently misstates every P&L number by its ratio
            # (the nifty_directional 75-vs-65 bug, but per stock).
            lots = g.loc[g["expiry"].dt.date == exp, "lot_size"].dropna().unique()
            if len(lots) == 0:
                log.warning(f"  {sym}: no lot_size — skipped")
                continue

            # Derive the strike step from what is actually listed.
            strikes = sorted(g.loc[g["expiry"].dt.date == exp, "strike"].unique())
            step = self._derive_step(strikes)
            if step is None:
                log.warning(f"  {sym}: could not derive strike step — skipped")
                continue

            self._lot[sym]    = int(lots[0])
            self._step[sym]   = step
            self._expiry[sym] = exp
            self._universe.append(sym)

        self._universe.sort()
        missing = sorted(want - set(self._universe))
        if missing:
            log.warning(f"  Not in NFO dump / skipped: {missing}")

        self._load_spot_tokens(kite, nse_raw)

        log.info(
            f" StockOptionStore ready: {len(self._universe)} underlyings, "
            f"{len(self._df)} contracts | e.g. "
            + ", ".join(
                f"{s}(lot={self._lot[s]},step={self._step[s]})"
                for s in self._universe[:4]
            )
        )

    @staticmethod
    def _derive_step(strikes: list[float]):
        """Modal gap between consecutive listed strikes."""
        if len(strikes) < 3:
            return None
        gaps = [
            round(strikes[i + 1] - strikes[i], 2)
            for i in range(len(strikes) - 1)
            if strikes[i + 1] > strikes[i]
        ]
        if not gaps:
            return None
        step = max(set(gaps), key=gaps.count)
        return step if step > 0 else None

    def _load_spot_tokens(self, kite, nse_raw=None):
        """
        Resolve NSE equity instrument_token per underlying. The scanner
        subscribes these for its direction signal — a stock EQ tick carries
        real last_traded_quantity and volume_traded, unlike an index token.
        """
        try:
            raw = nse_raw if nse_raw is not None else kite.instruments("NSE")
            eq  = pd.DataFrame(raw)
            eq  = eq[(eq["segment"] == "NSE") & (eq["instrument_type"] == "EQ")]
            m   = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
            for sym in self._universe:
                tok = m.get(sym)
                if tok:
                    self._spot_tok[sym] = int(tok)
                else:
                    log.warning(f"  {sym}: no NSE EQ token found")
        except Exception as e:
            log.error(f" StockOptionStore: NSE instrument load failed: {e}")

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def universe(self) -> list[str]:
        """Underlyings that resolved fully (chain + lot + step + spot token)."""
        return [s for s in self._universe if s in self._spot_tok]

    def lot_size(self, sym: str) -> int:
        return int(self._lot.get(sym, 0))

    def strike_step(self, sym: str):
        return self._step.get(sym)

    def expiry(self, sym: str):
        return self._expiry.get(sym)

    def spot_token(self, sym: str):
        return self._spot_tok.get(sym)

    def symbol_for_token(self, token: int):
        """Reverse lookup: NSE equity token -> underlying name."""
        for s, t in self._spot_tok.items():
            if t == token:
                return s
        return None

    def days_to_expiry(self, sym: str) -> int:
        exp = self._expiry.get(sym)
        return (exp - _today_ist()).days if exp else 999

    def atm_strike(self, sym: str, spot: float):
        step = self._step.get(sym)
        if not step or not spot:
            return None
        return round(round(spot / step) * step, 2)

    def get_option(self, sym: str, strike: float, opt_type: str):
        """
        Returns (instrument_token, tradingsymbol, lot_size) for the nearest
        expiry, or (None, None, 0). Walks out one strike step at a time if the
        exact strike is not listed.
        """
        if self._df is None or sym not in self._expiry:
            return None, None, 0

        exp  = self._expiry[sym]
        step = self._step[sym]
        base = self._df[
            (self._df["name"] == sym) &
            (self._df["instrument_type"] == opt_type) &
            (self._df["expiry"].dt.date == exp)
        ]
        if base.empty:
            return None, None, 0

        for k in (0, 1, -1, 2, -2, 3, -3):
            adj = round(strike + k * step, 2)
            hit = base[base["strike"] == float(adj)]
            if not hit.empty:
                r = hit.iloc[0]
                if k != 0:
                    log.warning(f"  {sym} strike {strike} not listed → using {adj}")
                return int(r["instrument_token"]), str(r["tradingsymbol"]), int(r["lot_size"])

        return None, None, 0
