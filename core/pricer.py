"""
core/pricer.py

Black-Scholes option pricer  shared utility for all strategies.
Used as fallback when live option price is unavailable
(historical expiries not in Kite NFO list).
"""

import math
import numpy as np
from datetime import datetime
from scipy.stats import norm


RISK_FREE = 0.065   # 6.5% Indian risk-free rate


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str = "call") -> float:
    """
    Black-Scholes formula.
    S     = spot price
    K     = strike price
    T     = time to expiry in years (e.g., 6 hours = 6/8760)
    r     = risk-free rate (0.065 for India)
    sigma = implied volatility (0.18 = 18%)
    opt   = "call" or "put"
    """
    if T <= 1e-8:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if opt == "call":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def option_premium(
    spot: float,
    strike: int,
    entry_ts: datetime,
    expiry_dt: datetime,
    direction: str,     # "CE" or "PE"
    iv: float,
) -> float:
    """
    Compute B-S premium at a given moment.
    T = time remaining to expiry in years.
    """
    T   = max((expiry_dt - entry_ts).total_seconds() / (365 * 86400), 1e-8)
    opt = "call" if direction == "CE" else "put"
    return round(bs_price(spot, strike, T, RISK_FREE, iv, opt), 2)


def pick_iv(dte_days: int, range_width: float) -> float:
    """
    DTE-aware IV selection for B-S fallback.
    0DTE options have extreme gamma  use higher IV.
    Wide OR days = volatile = IV also higher.
    """
    if   dte_days == 0: base = 0.30
    elif dte_days <= 2: base = 0.24
    elif dte_days <= 4: base = 0.18
    else:               base = 0.14

    if   range_width > 350: base *= 1.20
    elif range_width < 180: base *= 0.90

    return round(min(base, 0.60), 4)
