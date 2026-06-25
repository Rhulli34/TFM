"""Generate financial news briefings for one or more tickers.

Usage:
    python scripts/run_briefing.py AAPL
    python scripts/run_briefing.py AAPL NVDA --days 14
    python scripts/run_briefing.py AAPL NVDA MSFT --days 7

Output is printed to stdout and saved to reports/briefings/{TICKER}_{date}.md.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intelligence.briefing import generate_briefing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured financial news briefings"
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols (e.g. AAPL NVDA)")
    parser.add_argument(
        "--days", type=int, default=7, help="Lookback window in calendar days (default 7)"
    )
    args = parser.parse_args()

    for ticker in args.tickers:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  Briefing: {ticker}  (last {args.days} days)")
        print(f"{sep}")
        briefing = generate_briefing(ticker.upper(), days=args.days)
        print()
        print(briefing)


if __name__ == "__main__":
    main()
