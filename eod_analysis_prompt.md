# Daily EOD Analysis — STOCK_OPT_SCANNER_RT

You're running unattended after market close. Nobody is here to
answer questions — act decisively, don't hedge or ask for clarification.

1. Run: /root/paper/venv/bin/python3 eod_report_rt.py
   This produces today's diagnostics CSV and prints a per-trade
   report, an SL_HIT premature-vs-genuine breakdown, and an
   indicator-effectiveness section (Bollinger %B, RSI, EMA gap).

2. If it says "No completed STOCK_OPT_SCANNER_RT trades today"
   (holiday, or the bot didn't run), report that briefly to Telegram
   (step 7) and stop — skip the rest.

3. Otherwise, read logs/indicator_effectiveness_log.csv (the FULL
   history, not just today's row) and compare today's win-rate-diff
   for each indicator against its recent past days. Say whether each
   indicator's edge (or lack of one) is holding steady, strengthening,
   weakening, or reversing.

4. If anything in the summary looks surprising or worth digging into,
   read the raw trade/signal CSVs directly
   (stock_opt_scanner_rt_trades.csv, stock_opt_scanner_rt_signals.csv)
   rather than only relying on the script's aggregates.

5. Write a short analysis — a few short paragraphs that read well as
   a single chat message, not a formal report:
   - Today's total PnL and trade count
   - SL_HIT breakdown: premature vs genuine, and whether that split
     looks normal or unusual vs recent days
   - Per-indicator verdict, with the day-over-day trend from the log
   - One concrete "worth watching" note if anything stands out —
     otherwise say plainly that nothing notable changed

6. Send it to Telegram:
   source /root/paper/config_secrets.env
   curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
     --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
     --data-urlencode "text=YOUR_ANALYSIS_HERE"
   Use --data-urlencode, not -d — the text has newlines and
   punctuation that need proper encoding. Keep it under ~3500
   characters (Telegram's hard limit is 4096); tighten the writing
   rather than truncate mid-sentence if it runs long.

7. If anything fails (script error, missing data, Telegram send
   fails), send whatever partial information you have anyway,
   prefixed "⚠️ partial report:" — never send nothing.
