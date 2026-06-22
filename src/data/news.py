"""Fetch company news from the Finnhub company-news endpoint."""

from __future__ import annotations

import requests
from datetime import datetime
from typing import Any

from src.config import NEWS_API_KEY, FINNHUB_BASE_URL


def get_company_news(
    ticker: str,
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """Retrieve news articles for *ticker* from Finnhub.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        start:  Start date "YYYY-MM-DD", inclusive.
        end:    End date   "YYYY-MM-DD", inclusive.

    Returns:
        List of dicts, each with keys:
          datetime (str, ISO-like), headline, summary, source, url.

    Raises:
        EnvironmentError: if NEWS_API_KEY is not set.
        requests.HTTPError: on non-200 responses.
    """
    if not NEWS_API_KEY:
        raise EnvironmentError(
            "NEWS_API_KEY is not set. Add your Finnhub token to .env."
        )

    url = f"{FINNHUB_BASE_URL}/company-news"
    params = {
        "symbol": ticker,
        "from": start,
        "to": end,
        "token": NEWS_API_KEY,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    raw: list[dict] = response.json()

    articles = []
    for item in raw:
        ts = item.get("datetime", 0)
        dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"
        articles.append(
            {
                "datetime": dt_str,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
            }
        )

    return articles
