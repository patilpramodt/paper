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
from datetime import date

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
        today      = date.today()
        future_exp = self._df[self._df["expiry"].dt.date >= today]["expiry"].unique()
        if len(future_exp) == 0:
            return None, None

        nearest = sorted(future_exp)[0].date()
        return self.get_option_token(strike, opt_type, nearest)
