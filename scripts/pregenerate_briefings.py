"""Pre-generate briefings for all catalog tickers and store them in the DB.

Run this locally (GPU + OPENAI_API_KEY available) before rebuilding the Docker
image.  Production only reads the cached results — no LLM call at request time.

The window is anchored to the newest article date in the snapshot, not today,
so the briefing content is always within the data that exists in the DB.

Usage:
    python scripts/pregenerate_briefings.py [--days N] [--tickers A,B,C]

Options:
    --days N          Lookback window in days (default: 7).
    --tickers A,B,C   Comma-separated list; defaults to all catalog tickers.
    --db PATH         Path to radar.db (default: data/processed/radar.db).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.data.store import DB_PATH, get_briefing, init_db, save_briefing
from src.intelligence.briefing import generate_briefing


def _catalog(db_path: Path) -> list[tuple[str, str]]:
    """Return [(ticker, max_article_date)] for tickers with is_relevant=1 news."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ticker, MAX(substr(datetime, 1, 10)) AS max_date
            FROM news
            WHERE is_relevant = 1
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days",    type=int, default=7,
                        help="Lookback window in days (default: 7)")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated tickers; default: all catalog tickers")
    parser.add_argument("--db",      type=str, default=str(DB_PATH),
                        help="Path to radar.db")
    args = parser.parse_args()

    db_path = Path(args.db)
    init_db(db_path)

    all_tickers = _catalog(db_path)
    if not all_tickers:
        print("No catalog tickers found (no is_relevant=1 news in DB). Exiting.")
        sys.exit(1)

    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        tickers = [(t, d) for t, d in all_tickers if t in wanted]
    else:
        tickers = all_tickers

    print(f"Pre-generating briefings: {len(tickers)} tickers, days={args.days}\n")

    ok = err = 0
    for ticker, max_date_str in tickers:
        as_of = date.fromisoformat(max_date_str)
        from_date = as_of - timedelta(days=args.days)
        print(f"[{ticker}] window {from_date} to {as_of} (newest article: {max_date_str})")

        try:
            markdown = generate_briefing(ticker, days=args.days, db_path=db_path, as_of=as_of)
            generated_at = datetime.utcnow().isoformat() + "Z"
            save_briefing(ticker, args.days, generated_at, markdown, db_path)
            print(f"  OK saved ({len(markdown)} chars)\n")
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            err += 1

    print(f"Done. {ok} OK / {err} errors.")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
