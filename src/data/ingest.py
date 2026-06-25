"""News and price ingestion pipeline.

Fetches company news from Finnhub (monthly chunks to stay within free-tier
reliability), classifies sentiment with the fine-tuned DeBERTa model, fetches
OHLCV prices from yfinance, and persists everything to SQLite via store.py.
Idempotent: re-running never duplicates rows.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from src.config import DATA_PROCESSED, DEFAULT_TICKERS
from src.data.news import get_company_news
from src.data.prices import get_prices
from src.data.relevance import is_relevant
from src.data.store import DB_PATH, init_db, upsert_news, upsert_prices
from src.models.sentiment import predict_sentiment

# Stay comfortably under Finnhub free-tier limit (60 req/min)
_FINNHUB_SLEEP = 1.1


def _month_ranges(start: date, end: date) -> list[tuple[str, str]]:
    """Return (from_str, to_str) pairs covering [start, end] month by month."""
    ranges: list[tuple[str, str]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        month_start = max(cur, start)
        if cur.month == 12:
            next_month = date(cur.year + 1, 1, 1)
        else:
            next_month = date(cur.year, cur.month + 1, 1)
        month_end = min(next_month - timedelta(days=1), end)
        ranges.append((month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        cur = next_month
    return ranges


def run_ingest(
    tickers: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> dict:
    """Ingest news + prices for *tickers* over a date range.

    Args:
        tickers:    Ticker symbols. Defaults to DEFAULT_TICKERS.
        start_date: "YYYY-MM-DD". Defaults to 365 days ago.
        end_date:   "YYYY-MM-DD". Defaults to today.
        db_path:    SQLite file path.

    Returns:
        Per-ticker summary dict with article counts, date ranges, sentiment dist.
    """
    tickers = tickers or DEFAULT_TICKERS
    today = date.today()
    end = date.fromisoformat(end_date) if end_date else today
    start = date.fromisoformat(start_date) if start_date else (today - timedelta(days=365))

    init_db(db_path)
    months = _month_ranges(start, end)
    total_calls = len(tickers) * len(months)

    print(f"Portfolio : {tickers}")
    print(f"Period    : {start} -> {end}  ({len(months)} months)")
    print(f"Finnhub   : {total_calls} calls planned  (sleep={_FINNHUB_SLEEP}s)\n")

    summary: dict = {}
    call_idx = 0

    for ticker in tickers:
        print(f"[{ticker}]", flush=True)
        raw_articles: list[dict] = []

        # ── Fetch news month-by-month ─────────────────────────────────────────
        for from_str, to_str in months:
            try:
                batch = get_company_news(ticker, from_str, to_str)
                raw_articles.extend(batch)
            except Exception as exc:
                print(f"  WARN {ticker} {from_str}..{to_str}: {exc}", flush=True)
            call_idx += 1
            # Sleep between calls, but not after the very last one
            if call_idx < total_calls:
                time.sleep(_FINNHUB_SLEEP)

        # De-duplicate within this ticker's batch (same url+headline may appear
        # in overlapping monthly queries for edge-case dates)
        seen: set[tuple] = set()
        articles: list[dict] = []
        for a in raw_articles:
            key = (a.get("url", ""), a.get("headline", ""))
            if key not in seen:
                seen.add(key)
                articles.append(a)

        print(f"  {len(articles)} unique articles fetched", flush=True)

        # ── Relevance flag + Sentiment classification ─────────────────────────
        if articles:
            for a in articles:
                a["is_relevant"] = 1 if is_relevant(
                    a.get("headline"), a.get("summary"), ticker
                ) else 0
                a["relevance_method"] = "keyword"

            texts = [
                f"{a['headline']} {a.get('summary', '')}".strip()
                for a in articles
            ]
            preds = predict_sentiment(texts)
            for a, (label, score) in zip(articles, preds):
                a["sentiment_label"] = label
                a["sentiment_score"] = round(score, 4)

        # ── Persist news ──────────────────────────────────────────────────────
        inserted_news = upsert_news(articles, ticker, db_path)
        print(f"  {inserted_news} new rows inserted into news", flush=True)

        # ── Fetch and persist prices ──────────────────────────────────────────
        inserted_prices = 0
        try:
            price_end = (end + timedelta(days=1)).strftime("%Y-%m-%d")
            price_df = get_prices(ticker, start.strftime("%Y-%m-%d"), price_end)
            inserted_prices = upsert_prices(price_df, ticker, db_path)
            print(f"  {inserted_prices} price rows upserted", flush=True)
        except Exception as exc:
            print(f"  WARN prices for {ticker}: {exc}", flush=True)

        # ── Per-ticker summary ────────────────────────────────────────────────
        sent_dist: dict[str, int] = {}
        datetimes: list[str] = []
        for a in articles:
            lbl = a.get("sentiment_label", "unknown")
            sent_dist[lbl] = sent_dist.get(lbl, 0) + 1
            dt = a.get("datetime", "")
            if dt and dt != "N/A":
                datetimes.append(dt)

        summary[ticker] = {
            "n_articles": len(articles),
            "n_inserted": inserted_news,
            "n_prices": inserted_prices,
            "date_min": min(datetimes)[:10] if datetimes else None,
            "date_max": max(datetimes)[:10] if datetimes else None,
            "sentiment": sent_dist,
        }

    return summary
