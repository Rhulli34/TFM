"""One-shot gap ingestion: 2026-07-20 to today.

Reports actual date ranges returned by Finnhub per ticker so we can detect
if the free tier's lookback limit prevents reaching recent dates.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datetime import date

START = "2026-07-20"
END   = date.today().strftime("%Y-%m-%d")

from src.config import DEFAULT_TICKERS
from src.data.ingest import run_ingest

print("Financial News Radar — Gap Ingestion")
print(f"Tickers : {DEFAULT_TICKERS}")
print(f"Period  : {START} -> {END}\n")

summary = run_ingest(start_date=START, end_date=END)

print("\n" + "=" * 72)
print("GAP INGESTION SUMMARY")
print("=" * 72)
print(f"{'Ticker':<8} {'Fetched':>8} {'Inserted':>9} {'Finnhub from':>13} {'Finnhub to':>11}")
print("-" * 72)
total_inserted = 0
for ticker in DEFAULT_TICKERS:
    if ticker not in summary:
        continue
    info = summary[ticker]
    n   = info["n_articles"]
    ins = info["n_inserted"]
    dmin = info["date_min"] or "—"
    dmax = info["date_max"] or "—"
    total_inserted += ins
    print(f"{ticker:<8} {n:>8} {ins:>9} {dmin:>13} {dmax:>11}")

print("-" * 72)
print(f"{'TOTAL':<8} {'':>8} {total_inserted:>9}")
print(f"\nRequested window : {START} -> {END}")
print("If 'Finnhub to' < today, the free tier does not reach recent dates.")
