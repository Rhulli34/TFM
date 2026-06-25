"""Historical ingestion: ~1 year for the full DEFAULT_TICKERS portfolio.

Run once to seed the database; idempotent (safe to re-run).
Progress and cost prints come from ingest.py / sentiment.py.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_TICKERS
from src.data.ingest import run_ingest

START = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
END   = date.today().strftime("%Y-%m-%d")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 65)
    print("INGESTION SUMMARY")
    print("=" * 65)

    total_articles = 0
    for ticker, info in summary.items():
        n = info["n_articles"]
        total_articles += n
        if info["date_min"]:
            span = f"{info['date_min']} -> {info['date_max']}"
        else:
            span = "no data"

        sent = info["sentiment"]
        neg = sent.get("negative", 0)
        neu = sent.get("neutral", 0)
        pos = sent.get("positive", 0)
        tot = neg + neu + pos or 1

        print(f"\n{ticker:6s}  n={n:4d}  [{span}]")
        print(
            f"         neg={neg:3d} ({neg/tot*100:4.0f}%)  "
            f"neu={neu:3d} ({neu/tot*100:4.0f}%)  "
            f"pos={pos:3d} ({pos/tot*100:4.0f}%)"
        )

    print(f"\nTotal: {total_articles} articles across {len(summary)} tickers")
    print("DB   : data/processed/radar.db")


if __name__ == "__main__":
    print("Financial News Radar — Historical Ingestion")
    print(f"Tickers : {DEFAULT_TICKERS}")
    print(f"Period  : {START} -> {END}\n")

    summary = run_ingest(start_date=START, end_date=END)
    print_summary(summary)
