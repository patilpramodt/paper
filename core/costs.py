"""
core/costs.py

Shared execution-cost model — bid/ask spread, slippage and brokerage.

WHY THIS FILE EXISTS
────────────────────
OrderRouter._paper_fill() returns the raw LTP as the fill price for BOTH
the buy and the sell. That means a paper round-trip is priced at
    pnl = exit_ltp - entry_ltp
with ZERO cost. In reality every round-trip on an NFO index option pays,
at minimum:

  1. Half the bid-ask spread on entry (you lift the offer)
  2. Half the bid-ask spread on exit  (you hit the bid)
  3. Brokerage + STT + exchange txn charges + GST + stamp duty

Audit of this repo (2026-08): only orb_v2.py, bb_stoch_strategy.py and
bb_stoch_nifty_strategy.py model any of this. SPIKE, SPIKE_NIFTY,
SCALPER_V7, SMART_HEDGE, BANKNIFTY_EXPIRY_MOMENTUM, NIFTY_DIRECTIONAL,
NIFTY_FUT_DIRECTIONAL and all four CANDLE_BREAKOUT strategies compute
    pnl = (sell_price - entry) * qty
with no cost at all.

That is not a rounding error. For NIFTY_CANDLE_BREAKOUT the trailing lock
is +5 premium points and the round-trip cost below is ~3.5-5 points
equivalent — i.e. the entire modelled edge. Any paper result from those
strategies is systematically optimistic and cannot be compared against a
live P&L curve until this is applied.

USAGE
─────
    from core.costs import round_trip_cost_pts, entry_fill, exit_fill

    # at entry
    fill = entry_fill(ltp)                      # ltp + half-spread

    # at exit
    px   = exit_fill(ltp)                       # ltp - half-spread
    pnl  = (px - fill) * qty - fixed_costs_rs(qty, fill, px)

Or, if you only want a single number to subtract:

    net_pts = gross_pts - round_trip_cost_pts(entry_ltp, qty)

CALIBRATION
───────────
Numbers below are for liquid near-ATM NIFTY / BANKNIFTY weekly and monthly
options in normal conditions. They are deliberately conservative — it is
far better for a paper backtest to under-promise. Widen SPREAD_PCT if you
trade far-OTM strikes, 0DTE in the last hour, or size beyond top-of-book.
"""

# ── Bid-ask spread model ─────────────────────────────────────────────────────
# Real spreads on liquid index options are bounded by tick size and
# market-maker competition, NOT by premium magnitude — a BankNifty ATM at
# premium 900 does not have a 27-point spread. So: a modest percentage WITH
# an absolute cap and an absolute floor (tick size is 0.05, but you rarely
# get filled inside a 0.5pt spread on a market order).
SPREAD_PCT      = 0.012   # 1.2% of premium
SPREAD_CAP_PTS  = 8.0     # never assume wider than this on a liquid strike
SPREAD_MIN_PTS  = 0.50    # never assume tighter than this

# ── Statutory + broker charges, per leg, as of 2026 ──────────────────────────
# Zerodha F&O option: flat Rs 20 per executed order.
BROKERAGE_PER_ORDER_RS = 20.0
# STT: 0.1% of premium on the SELL side only (options).
STT_SELL_PCT           = 0.001
# NSE exchange transaction charge on options: ~0.0495% of premium, both legs.
EXCHANGE_TXN_PCT       = 0.000495
# GST 18% on (brokerage + exchange txn charge).
GST_PCT                = 0.18
# SEBI turnover fee, negligible but included for completeness.
SEBI_PCT               = 0.000001
# Stamp duty 0.003% on the BUY side only.
STAMP_BUY_PCT          = 0.00003


def estimate_spread(ltp: float) -> float:
    """Estimated full bid-ask spread in premium points for a given LTP."""
    if ltp is None or ltp <= 0:
        return 0.0
    raw = ltp * SPREAD_PCT
    return round(min(max(raw, SPREAD_MIN_PTS), SPREAD_CAP_PTS), 2)


def half_spread(ltp: float) -> float:
    """Cost of crossing the market on one leg, in premium points."""
    return round(estimate_spread(ltp) / 2.0, 2)


def entry_fill(ltp: float) -> float:
    """
    Realistic BUY fill: you lift the offer, so you pay above the mid/LTP.
    Use this instead of the raw LTP when constructing a paper position.
    """
    if ltp is None or ltp <= 0:
        return ltp
    return round(ltp + half_spread(ltp), 2)


def exit_fill(ltp: float) -> float:
    """
    Realistic SELL fill: you hit the bid, so you receive below the mid/LTP.
    Never returns a negative price.
    """
    if ltp is None or ltp <= 0:
        return ltp
    return round(max(0.05, ltp - half_spread(ltp)), 2)


def fixed_costs_rs(qty: int, entry_px: float, exit_px: float) -> float:
    """
    Total statutory + brokerage charges in rupees for one round-trip
    (one BUY order + one SELL order) of `qty` units.
    """
    if not qty or qty <= 0:
        return 0.0
    buy_turnover  = max(0.0, entry_px) * qty
    sell_turnover = max(0.0, exit_px) * qty
    turnover      = buy_turnover + sell_turnover

    brokerage = BROKERAGE_PER_ORDER_RS * 2
    stt       = sell_turnover * STT_SELL_PCT
    exch      = turnover * EXCHANGE_TXN_PCT
    sebi      = turnover * SEBI_PCT
    stamp     = buy_turnover * STAMP_BUY_PCT
    gst       = (brokerage + exch) * GST_PCT

    return round(brokerage + stt + exch + sebi + stamp + gst, 2)


def round_trip_cost_pts(entry_ltp: float, qty: int) -> float:
    """
    Total round-trip cost expressed in PREMIUM POINTS per unit, so it can be
    subtracted straight from a points-based P&L or compared against an SL/TP
    distance.

    This is the number to sanity-check a strategy's reward target against.
    If your trailing lock is +5 points and this returns 4.2, the strategy
    has no edge no matter how good the entry signal is.
    """
    if not entry_ltp or entry_ltp <= 0 or not qty or qty <= 0:
        return 0.0
    spread_pts = estimate_spread(entry_ltp)            # both legs combined
    charges_rs = fixed_costs_rs(qty, entry_ltp, entry_ltp)
    return round(spread_pts + (charges_rs / qty), 2)


def net_pnl_rs(entry_px: float, exit_px: float, qty: int) -> float:
    """
    Net rupee P&L for a long-option round-trip, after charges.
    `entry_px` / `exit_px` should already include the half-spread
    (i.e. come from entry_fill() / exit_fill()).
    """
    gross = (exit_px - entry_px) * qty
    return round(gross - fixed_costs_rs(qty, entry_px, exit_px), 2)