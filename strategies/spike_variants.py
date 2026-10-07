"""
strategies/spike_variants.py

Entry-logic test variants of SPIKE (BankNifty) and SPIKE_NIFTY (Nifty 50).
They run alongside the originals in paper mode and share their SL, trailing
SL, cost model, max_trades_day and EOD exit — only the entry rule differs,
so the CSVs can be compared like for like.

  SPIKE          (strategies/spike.py)  first 2 consecutive same-colour 10s
                                        candles → CE/PE           spike_trades.csv
  SPIKE_5S       colour of 9:15:00 → 9:15:05 move → CE (green) /
                 PE (red); flat = no trade                       spike_5s_trades.csv
  SPIKE_GAP2M    gap days only (open outside prev last-5m range);
                 at 9:17:00 the 2-min colour must match the gap
                 direction → enter, else no trade                spike_gap2m_trades.csv

SPIKE_NIFTY_5S / SPIKE_NIFTY_GAP2M apply the same two rules to Nifty 50
(spike_nifty_5s_trades.csv, spike_nifty_gap2m_trades.csv).

All variants evaluate exactly once per day and are always paper.
"""

from strategies.spike import SpikeStrategy
from strategies.spike_nifty import SpikeNiftyStrategy


class _PaperOnly:
    @property
    def LIVE_MODE(self) -> bool:
        return False


class SpikeColor5sStrategy(_PaperOnly, SpikeStrategy):
    ENTRY_MODE    = "COLOR_5S"
    STRATEGY_NAME = "SPIKE_5S"
    CSV_FILE      = "spike_5s_trades.csv"
    OPEN_WAIT_SEC = 5


class SpikeGap2mStrategy(_PaperOnly, SpikeStrategy):
    ENTRY_MODE    = "GAP_2M"
    STRATEGY_NAME = "SPIKE_GAP2M"
    CSV_FILE      = "spike_gap2m_trades.csv"
    OPEN_WAIT_SEC = 120


# ── NIFTY 50 (same rules, SPIKE_NIFTY's SL / trail / qty) ─────────────────────

class SpikeNiftyColor5sStrategy(_PaperOnly, SpikeNiftyStrategy):
    ENTRY_MODE    = "COLOR_5S"
    STRATEGY_NAME = "SPIKE_NIFTY_5S"
    CSV_FILE      = "spike_nifty_5s_trades.csv"
    OPEN_WAIT_SEC = 5


class SpikeNiftyGap2mStrategy(_PaperOnly, SpikeNiftyStrategy):
    ENTRY_MODE    = "GAP_2M"
    STRATEGY_NAME = "SPIKE_NIFTY_GAP2M"
    CSV_FILE      = "spike_nifty_gap2m_trades.csv"
    OPEN_WAIT_SEC = 120
