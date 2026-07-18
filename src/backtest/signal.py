"""Daily sentiment signal from classified news.

Assigns each is_relevant=1 article to a trading day:
  - Articles before 20:00 UTC count for that calendar day.
  - Articles at or after 20:00 UTC count for the next calendar day.
  - Non-trading days (weekends/holidays) roll forward to the next date
    present in the prices table.

Net sentiment per (ticker, trading_day):
  raw      = mean(pos→+1, neu→0, neg→-1)
  weighted = mean(label_value * sentiment_score)

Also produces 3-day and 5-day rolling means. Trading days with no news
receive a zero signal (neutral fill).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.store import DB_PATH

_LABEL_VALUE: dict[str, float] = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_MARKET_CLOSE_UTC = 20  # approx. 4 pm ET (EDT); slightly early in winter


def build_signal(
    tickers: list[str] | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Build daily sentiment signal DataFrame.

    Args:
        tickers: Ticker symbols to include. None means all tickers in DB.
        db_path: SQLite file path.

    Returns:
        DataFrame sorted by (ticker, date) with columns:
          ticker, date, raw, weighted, n_articles,
          raw_ma3, raw_ma5, weighted_ma3, weighted_ma5.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if tickers:
        ph = ",".join("?" * len(tickers))
        news_rows = conn.execute(
            f"SELECT ticker, datetime, sentiment_label, sentiment_score "
            f"FROM news WHERE is_relevant=1 AND ticker IN ({ph})",
            tickers,
        ).fetchall()
        price_rows = conn.execute(
            f"SELECT ticker, date FROM prices WHERE ticker IN ({ph}) ORDER BY ticker, date",
            tickers,
        ).fetchall()
    else:
        news_rows = conn.execute(
            "SELECT ticker, datetime, sentiment_label, sentiment_score "
            "FROM news WHERE is_relevant=1"
        ).fetchall()
        price_rows = conn.execute(
            "SELECT ticker, date FROM prices ORDER BY ticker, date"
        ).fetchall()
    conn.close()

    # Build trading calendar per ticker
    td_by_ticker: dict[str, list[str]] = {}
    for row in price_rows:
        td_by_ticker.setdefault(row["ticker"], []).append(row["date"])
    td_set_by_ticker = {t: set(days) for t, days in td_by_ticker.items()}

    # Assign each article to a trading day
    records: list[dict] = []
    for art in news_rows:
        ticker = art["ticker"]
        label  = art["sentiment_label"]
        score  = float(art["sentiment_score"] or 0.5)
        if label not in _LABEL_VALUE:
            continue

        dt_str = (art["datetime"] or "").replace(" UTC", "")
        try:
            dt = pd.Timestamp(dt_str)
        except Exception:
            continue

        # After-hours articles count for the next calendar day
        cal_date = dt.date()
        if dt.hour >= _MARKET_CLOSE_UTC:
            cal_date = (dt + pd.Timedelta(days=1)).date()

        # Snap forward to the next actual trading day
        td_set = td_set_by_ticker.get(ticker, set())
        td_list = td_by_ticker.get(ticker, [])
        if not td_list:
            continue
        max_td = max(td_list)
        snapped = cal_date.isoformat()
        while snapped not in td_set:
            if snapped > max_td:
                break
            cal_date = (pd.Timestamp(snapped) + pd.Timedelta(days=1)).date()
            snapped = cal_date.isoformat()
        if snapped > max_td:
            continue

        label_val = _LABEL_VALUE[label]
        records.append({
            "ticker": ticker,
            "date": snapped,
            "label_val": label_val,
            "weighted_val": label_val * score,
        })

    # Aggregate to (ticker, date)
    if records:
        raw_df = pd.DataFrame(records)
        agg = (
            raw_df.groupby(["ticker", "date"])
            .agg(
                raw=("label_val", "mean"),
                weighted=("weighted_val", "mean"),
                n_articles=("label_val", "count"),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(columns=["ticker", "date", "raw", "weighted", "n_articles"])

    # Build complete calendar (one row per trading day per ticker)
    full_rows = [
        {"ticker": t, "date": d}
        for t, days in td_by_ticker.items()
        for d in days
    ]
    full_df = pd.DataFrame(full_rows)
    merged = full_df.merge(agg, on=["ticker", "date"], how="left")
    merged["raw"]        = merged["raw"].fillna(0.0)
    merged["weighted"]   = merged["weighted"].fillna(0.0)
    merged["n_articles"] = merged["n_articles"].fillna(0).astype(int)

    # Rolling means per ticker (sorted chronologically)
    parts: list[pd.DataFrame] = []
    for ticker, grp in merged.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        for col in ("raw", "weighted"):
            grp[f"{col}_ma3"] = grp[col].rolling(3, min_periods=1).mean()
            grp[f"{col}_ma5"] = grp[col].rolling(5, min_periods=1).mean()
        parts.append(grp)

    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
