# ==============================================================================
# config.py  Scalper V7 | BankNifty Paper Trading
#
# V6 additions vs V5:
#   [OK] ATR gate (block dead/sideways markets)
#   [OK] ATR-scaled SL and TP (adapts to session volatility)
#   [OK] ATR-scaled MACD threshold (no more fixed 0.4)
#   [OK] RSI Z-score normalization (adaptive thresholds)
#   [OK] RSI slope (catch momentum start, not middle)
#   [OK] 5-candle structure breakout filter
#   [OK] 3-bar MACD histogram expansion
#   [OK] Regime classifier (trending vs sideways via ATR)
#   [OK] 14:30 session block (dangerous reversal zone)
#   [OK] Directional consecutive-loss blocker
#   [OK] Expiry day special rules (Thursday)
#   [OK] Option LTP sanity checks (spread, premium bounds)
#
# V7 changes vs V6:
#   [OK] RSI_ZSCORE_BULL_TREND = 0.4  — relaxed Rz on confirmed trend days
#   [OK] ATR_VOL_RATIO_MIN lowered 0.70 -> 0.50  — smooth glide-trends block fix
#   [OK] ATR_VOL_RATIO_LUNCH_MIN lowered 0.90 -> 0.70
#   [OK] VWAP_FILTER_ENABLED = False  — paused; unvalidated, not in block logs
#   [OK] rsi_acc removed from entry combine (signal_logic.py) — plateau blocks
# ==============================================================================

# ------------------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------------------
TOKEN_FILE      = "token.json"

# ------------------------------------------------------------------------------
# Instrument
# ------------------------------------------------------------------------------
INDEX_SYMBOL    = "NSE:NIFTY BANK"
ROOT            = "BANKNIFTY"
STRIKE_STEP     = 100
EXCHANGE_INDEX  = "NSE"
EXCHANGE_NFO    = "NFO"

# ------------------------------------------------------------------------------
# Candle Settings
# ------------------------------------------------------------------------------
CANDLE_1M_INTERVAL  = "minute"
CANDLE_5M_INTERVAL  = "5minute"
CANDLE_1M_FETCH     = 120       # fetch extra for Z-score rolling window
CANDLE_5M_FETCH     = 80
CANDLE_1M_USE       = 80        # use last 80 bars (EMA50 needs 50+, Z needs 20+)
CANDLE_5M_USE       = 50

# ------------------------------------------------------------------------------
# Session Windows
# ------------------------------------------------------------------------------
MARKET_OPEN         = (9,  15)
SESSION_START       = (9,  30)   # No entries before this (was 9:45, missed early momentum)
LUNCH_START         = (12, 30)
LUNCH_END           = (13, 30)
AFTERNOON_BLOCK     = (14, 30)   # Block NEW entries (dangerous reversal zone)
AUTO_SQUAREOFF      = (15,  0)   # Force close all positions
EXPIRY_ENTRY_CUTOFF = (12,  0)   # On expiry day -- only trade morning session

# ------------------------------------------------------------------------------
#
#  V7 NEW FILTER 1 -- ATR VOL-RATIO (Volatility Expansion Gate)
#
# Current ATR must be > rolling mean ATR by this multiplier.
# Meaning: volatility must be EXPANDING vs its own recent baseline.
# Different from V6's ATR_MIN_PCT which checks absolute ATR level.
# V6: "is ATR big enough?"  V7 also asks: "is ATR rising?"
#
# CHANGE v7: lowered 0.70 -> 0.50 (non-lunch), 0.90 -> 0.70 (lunch).
# On sustained trending days spot glides smoothly -- ATR contracts slightly
# relative to the rolling mean even though price is genuinely trending.
# 0.70 was causing spurious atr_vol_ratio blocks on valid trend entries.
# 0.50 still blocks truly dead/flat markets while allowing smooth trends.
ATR_VOL_RATIO_WINDOW    = 20        # Rolling bars to compute mean ATR
ATR_VOL_RATIO_MIN       = 0.50      # CHANGED: was 0.70 -- smooth trends were over-blocked
ATR_VOL_RATIO_LUNCH_MIN = 0.70      # CHANGED: was 0.90

# ------------------------------------------------------------------------------
#
#  V7 NEW FILTER 2 -- RSI ACCELERATION (2nd Derivative)
#
# First derivative (slope) tells direction of momentum.
# Second derivative (acceleration) tells if momentum is SPEEDING UP.
# acc = (RSI[-1] - RSI[-2]) - (RSI[-2] - RSI[-3])
#
# NOTE: rsi_acc is still COMPUTED in indicators.py and LOGGED in signal_logic.py
# but is NO LONGER part of the entry combine gate (removed in signal_logic.py).
# On sustained trends RSI slope plateaus -> acc goes negative -> mid-trend blocks.
# RSI slope (1st derivative) already ensures momentum direction; acc is redundant.
RSI_ACC_MIN             = 0.0       # Kept for logging/future use; not in combine gate

# ------------------------------------------------------------------------------
#
#  V7 NEW FILTER 3 -- VWAP (Liquidity Context)
#
# VWAP is the only non-price-derivative signal in the system.
# Directly addresses the collinearity problem (EMA+MACD+RSI are all
# derived from price smoothing and thus correlated).
# VWAP is volume-weighted -- adds orthogonal information.
# For CE: price must be ABOVE VWAP (buying above average cost = bullish)
# For PE: price must be BELOW VWAP (selling below average cost = bearish)
#
# CHANGE v7: PAUSED (False). Not appearing in block logs -- not a blocker today.
# Will re-enable once we have 5+ trade sample to validate it doesn't filter
# valid entries on trend days. Set back to True to re-enable.
VWAP_FILTER_ENABLED     = False     # CHANGED: was True -- paused for paper validation
VWAP_BUFFER_PTS         = 5.0      # Allow entry within N pts of VWAP
#                                    (0 = strict, 5 = some tolerance near VWAP)

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 1 -- ATR GATE (Dead Market Blocker)
#
# ATR as % of spot must exceed this to allow ANY entries.
# BankNifty at 50000: 0.08% = 40 pts ATR minimum
# Below this = market not moving enough to hit your target
ATR_MIN_PCT         = 0.025      # lowered: was 0.030 (0.030 blocks normal-vol BankNifty days)
# NOTE: At BankNifty ~61000, 0.025% = ~15 pts ATR required.
# Old value was 0.08 (49 pts) which blocked all trades in low-vol markets.
# Typical BankNifty ATR in normal markets: 20-50 pts (0.03-0.08%).
# Raise back toward 0.06-0.08 on high-vol days if desired.
ATR_STRONG_PCT      = 0.12      # Above this = high volatility (reduce SL buffer)

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 2 -- ATR-SCALED SL & TP
#
# SL and TP adapt to current session volatility.
# Maintains minimum 1:1.6 RR at all times.
ATR_SL_MULTIPLIER   = 0.75      # SL  = max(SL_MIN, ATR * 0.75)
ATR_TP_MULTIPLIER   = 1.30      # TP  = max(TP_MIN, ATR * 1.30)
SL_MIN_POINTS       = 4.0       # Hard floor -- never below 4 pts SL
SL_MAX_POINTS       = 9.0       # Hard ceiling -- never above 9 pts SL
TP_MIN_POINTS       = 7.0       # Hard floor -- never below 7 pts TP
TP_MAX_POINTS       = 18.0      # Hard ceiling -- cap on moonshot targets

# Trailing SL
TRAIL_ARM           = 5.0       # Move SL to BE when profit >= this
TRAIL_STEP          = 2.0       # Trail every N points after arm

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 3 -- REGIME CLASSIFIER
#
# EMA gap must be meaningful RELATIVE to ATR.
# Prevents entering on tiny crosses during low-vol regimes.
REGIME_EMA_ATR_RATIO    = 0.25  # ema_gap / atr14 must be >= this on 5-min (lowered: was 0.40 -- blocked all normal-trend days)
REGIME_EMA_ATR_RATIO_1M = 0.15  # ema_gap / atr14 on 1-min (lowered: was 0.25)

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 4 -- RSI Z-SCORE
#
# Instead of fixed RSI>58, use Z-score (adaptive to the day's range)
RSI_ZSCORE_WINDOW   = 20        # Rolling window for mean/std
RSI_ZSCORE_BULL     = 0.8       # Z > +0.8 for bullish (RSI above its own avg)
RSI_ZSCORE_BEAR     = -0.8      # Z < -0.8 for bearish
# On sustained trend days the Z-score rolling mean drifts up with RSI,
# compressing Rz back toward 0 even though price is genuinely trending.
# Use a relaxed threshold when 5-min regime is confirmed TRENDING (not lunch).
RSI_ZSCORE_BULL_TREND = 0.4     # relaxed Rz for bull trend (was causing 155 rsi_z_bull blocks)
RSI_ZSCORE_BEAR_TREND = -0.4    # relaxed Rz for bear trend
RSI_ZSCORE_EXTREME  = 2.0       # Z > +2.0 = exhaustion (block entry)

# Fallback raw RSI thresholds (used when Z-score window not full)
RSI_BULL_RAW        = 58
RSI_BEAR_RAW        = 42
RSI_OVERBOUGHT      = 72
RSI_OVERSOLD        = 28

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 5 -- RSI SLOPE
#
# RSI must be rising (for CE) or falling (for PE).
# 3-bar slope smoother than 1-bar.
RSI_SLOPE_LOOKBACK  = 3         # RSI[now] - RSI[3 bars ago]
RSI_SLOPE_MIN       = 1.0       # lowered: was 1.5 (1.5 too strict in slow trends)

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 6 -- MACD EXPANSION (3-bar)
#
# Histogram must be expanding for N consecutive bars.
MACD_EXPANSION_BARS = 2         # 2 consecutive expanding bars (3 causes too many misses)
# ATR-scaled MACD threshold (replaces fixed 0.40)
MACD_ATR_RATIO      = 0.008     # macd_hist must be > ATR * 0.008
MACD_HIST_FLOOR     = 0.25      # Absolute minimum even on low ATR days
MACD_HIST_CAP       = 2.0       # Ignore if ATR-scaled threshold is absurdly high
MACD_SLOPE_MIN      = 0.05      # Slope must be positive (expanding)

# Standard MACD periods
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL_PERIOD  = 9

# ------------------------------------------------------------------------------
#
#  V6 CORE FILTER 7 -- STRUCTURE BREAKOUT
#
# Price must break above/below N-candle range before entry.
STRUCTURE_LOOKBACK  = 3         # Number of candles to define the range (lowered: was 5 -- too strict on ranging days)
# For CE: close > max(high, last 5 candles)
# For PE: close < min(low,  last 5 candles)

# ------------------------------------------------------------------------------
# EMA Periods
# ------------------------------------------------------------------------------
EMA_FAST_PERIOD     = 20
EMA_SLOW_PERIOD     = 50

# 5-min trend -- use ATR-relative gap (REGIME_EMA_ATR_RATIO)
EMA_GAP_5M_FIXED    = 4.0       # Fallback fixed gap if ATR unavailable

# ------------------------------------------------------------------------------
# Signal Persistence
# ------------------------------------------------------------------------------
PERSISTENCE         = 1         # lowered: was 2 (2 consecutive bars = 2 min delay, misses moves)

# ------------------------------------------------------------------------------
# Lunch Window -- Extra Strict
# ------------------------------------------------------------------------------
LUNCH_ATR_MULTIPLIER    = 1.3   # Require 30% higher ATR than base minimum
LUNCH_RSI_ZSCORE_BULL   = 1.2   # Stricter Z-score during lunch
LUNCH_RSI_ZSCORE_BEAR   = -1.2
LUNCH_MACD_ATR_RATIO    = 0.014 # Stricter MACD ratio
LUNCH_EMA_ATR_RATIO_5M  = 0.60  # Stricter 5-min regime

# ------------------------------------------------------------------------------
# Trade Parameters
# ------------------------------------------------------------------------------
QUANTITY            = 35        # BankNifty lot size as of Nov 2024
SLIPPAGE_POINTS     = 1.5       # Simulated per side

# ------------------------------------------------------------------------------
#
#  V6 RISK -- DIRECTIONAL CONSECUTIVE LOSS BLOCKER
#
CONSEC_LOSS_BLOCK_COUNT = 2     # Block direction after N consecutive losses
CONSEC_LOSS_BLOCK_MINS  = 20    # Block for this many minutes

# ------------------------------------------------------------------------------
#
#  V6 RISK -- EXPIRY DAY RULES (Thursday)
#
EXPIRY_WEEKDAY          = 3     # 0=Mon, 3=Thu (BankNifty weekly expiry)
EXPIRY_TP_REDUCTION     = 0.75  # TP = normal_TP * 0.75 (theta erodes target)
EXPIRY_USE_ITM_OFFSET   = 100   # Use 1-strike ITM (better liquidity, less theta)

# ------------------------------------------------------------------------------
#
#  V6 RISK -- OPTION SANITY CHECKS
#
OPTION_LTP_MIN          = 8.0   # Reject if premium < 8 pts (illiquid)
OPTION_LTP_MAX_PCT      = 2.5   # Reject if premium > 2.5% of spot (wrong strike)
OPTION_SPREAD_MAX_PCT   = 30.0  # Reject if spread > 30% of target (too expensive)
ESTIMATED_SPREAD_PCT    = 0.03  # Estimate spread as 3% of LTP

# ------------------------------------------------------------------------------
# General Risk
# ------------------------------------------------------------------------------
MAX_DAILY_LOSS      = 5000      # Rs daily loss circuit breaker
MAX_TRADES_DAY      = 6
MAX_TRADES_PER_HOUR = 2         # NEW V7: cap entries per rolling 60-min window
POST_SL_COOLDOWN    = 600       # Seconds (10 min) after any SL hit
EXIT_COOLDOWN       = 3

# ------------------------------------------------------------------------------
# Loop & Cache
# ------------------------------------------------------------------------------
LOOP_SLEEP          = 0.5
SPOT_CACHE_TTL      = 5
WS_CONNECT_WAIT     = 5

# ------------------------------------------------------------------------------
# Output Files
# ------------------------------------------------------------------------------
ENTRY_LOG       = "logs/paper_entry.csv"
EXIT_LOG        = "logs/paper_exit.csv"
SIGNAL_LOG      = "logs/signal_log.csv"
DAILY_SUMMARY   = "logs/daily_summary.csv"
STATE_FILE      = "logs/state.json"
APP_LOG         = "logs/app.log"

