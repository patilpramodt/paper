"""
core/pcr_kite.py
Kite WebSocket PCR — uses the SAME kite session already
created by auto_login / MarketHub. No separate credentials needed.
"""

import time
import logging
from typing import Any

log = logging.getLogger("core.pcr_kite")


def fetch_kite_pcr(
    kite,
    symbols: list[str],
    expiry: str,
    percentage_filter: float = 10.0,
) -> dict[str, dict]:
    try:
        from expressoptionchain.option_stream import OptionStream
        from expressoptionchain.option_chain import OptionChainFetcher
    except ImportError:
        log.error("[KitePCR] express-option-chain not installed. "
                  "Run: pip install express-option-chain redis")
        return {}

    secrets = {
        "api_key":      kite.api_key,
        "api_secret":   "",
        "access_token": kite.access_token,
    }

    criteria = {
        "name": "percentage",
        "properties": {"value": percentage_filter},
    }

    log.info(f"[KitePCR] Starting OptionStream — {symbols}  expiry={expiry}")
    try:
        stream = OptionStream(symbols, secrets, expiry=expiry, criteria=criteria)
        stream.start(threaded=True)
    except Exception as e:
        log.error(f"[KitePCR] OptionStream failed to start: {e}")
        return {}

    log.info("[KitePCR] Waiting 8s for first ticks...")
    time.sleep(8)

    fetcher = OptionChainFetcher()
    try:
        chains = fetcher.get_option_chains(symbols)
    except Exception as e:
        log.error(f"[KitePCR] get_option_chains failed: {e}")
        return {}

    results = {}
    for chain in chains:
        if not chain:
            continue
        r = _calc_pcr(chain)
        key = f"NFO:{r['symbol']}"
        results[key] = r
        log.info(
            f"[KitePCR] {r['symbol']:12s}  "
            f"OI-PCR={r['pcr_oi']}  "
            f"Vol-PCR={r['pcr_volume']}  "
            f"ATM={r['atm_strike']}  "
            f"CE_OI={r['total_call_oi']:,}  PE_OI={r['total_put_oi']:,}"
        )
    return results


def _calc_pcr(chain: dict[str, Any]) -> dict:
    symbol     = chain.get("trading_symbol", "UNKNOWN")
    underlying = float(chain.get("underlying_value") or 0)

    all_strikes = []
    for strikes in chain.get("expiry", {}).values():
        all_strikes.extend(strikes)

    total_call_oi = total_put_oi = 0
    total_call_vol = total_put_vol = 0
    per_strike = []

    for row in all_strikes:
        strike   = row.get("strike_price", 0)
        ce       = row.get("ce") or {}
        pe       = row.get("pe") or {}
        call_oi  = ce.get("oi", 0) or 0
        put_oi   = pe.get("oi", 0) or 0
        call_vol = ce.get("volume", 0) or 0
        put_vol  = pe.get("volume", 0) or 0

        total_call_oi  += call_oi
        total_put_oi   += put_oi
        total_call_vol += call_vol
        total_put_vol  += put_vol

        per_strike.append({
            "strike":     strike,
            "call_oi":    call_oi,
            "put_oi":     put_oi,
            "call_vol":   call_vol,
            "put_vol":    put_vol,
            "pcr_oi":     round(put_oi  / call_oi,  4) if call_oi  else None,
            "pcr_volume": round(put_vol / call_vol, 4) if call_vol else None,
        })

    per_strike.sort(key=lambda x: x["strike"])

    atm_strike = None
    if underlying and per_strike:
        atm_strike = min(per_strike, key=lambda x: abs(x["strike"] - underlying))["strike"]

    return {
        "symbol":            symbol,
        "underlying_value":  underlying,
        "atm_strike":        atm_strike,
        "total_call_oi":     total_call_oi,
        "total_put_oi":      total_put_oi,
        "pcr_oi":            round(total_put_oi  / total_call_oi,  4) if total_call_oi  else None,
        "total_call_volume": total_call_vol,
        "total_put_volume":  total_put_vol,
        "pcr_volume":        round(total_put_vol / total_call_vol, 4) if total_call_vol else None,
        "per_strike":        per_strike,
    }
