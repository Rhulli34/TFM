"""SQLite persistence layer for news and prices.

Tables:
  news   — one row per article, deduped by sha256(url||headline)[:16].
  prices — one row per (ticker, date), upserted on re-run.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from src.config import DATA_PROCESSED

DB_PATH = DATA_PROCESSED / "radar.db"

_DDL = """
CREATE TABLE IF NOT EXISTS news (
    id               TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    datetime         TEXT NOT NULL,
    headline         TEXT NOT NULL,
    summary          TEXT,
    source           TEXT,
    url              TEXT,
    sentiment_label  TEXT,
    sentiment_score  REAL
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker);
CREATE INDEX IF NOT EXISTS idx_news_dt     ON news(datetime);

CREATE TABLE IF NOT EXISTS prices (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,
    article_id   TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    datetime     TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    entities     TEXT,
    description  TEXT NOT NULL,
    is_material  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_dt     ON events(datetime);

CREATE TABLE IF NOT EXISTS briefings (
    ticker       TEXT NOT NULL,
    days         INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    markdown     TEXT NOT NULL,
    PRIMARY KEY (ticker, days)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables and indexes if they don't already exist."""
    with _connect(db_path) as conn:
        conn.executescript(_DDL)
        # Migrations: add columns introduced after initial schema
        cols = {row[1] for row in conn.execute("PRAGMA table_info(news)").fetchall()}
        if "is_relevant" not in cols:
            conn.execute("ALTER TABLE news ADD COLUMN is_relevant INTEGER DEFAULT NULL")
        if "relevance_method" not in cols:
            conn.execute("ALTER TABLE news ADD COLUMN relevance_method TEXT DEFAULT NULL")


def news_id(url: str, headline: str) -> str:
    """Stable 16-char hex ID used as primary key for deduplication."""
    raw = f"{url}||{headline}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def upsert_news(articles: list[dict[str, Any]], ticker: str,
                db_path: Path = DB_PATH) -> int:
    """INSERT OR IGNORE article rows. Returns count of newly inserted rows."""
    if not articles:
        return 0
    rows = [
        (
            news_id(a.get("url", ""), a.get("headline", "")),
            ticker,
            a.get("datetime", ""),
            a.get("headline", ""),
            a.get("summary", ""),
            a.get("source", ""),
            a.get("url", ""),
            a.get("sentiment_label"),
            a.get("sentiment_score"),
            a.get("is_relevant"),
            a.get("relevance_method"),
        )
        for a in articles
    ]
    with _connect(db_path) as conn:
        cur = conn.executemany(
            """INSERT OR IGNORE INTO news
               (id, ticker, datetime, headline, summary, source, url,
                sentiment_label, sentiment_score, is_relevant, relevance_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return cur.rowcount


def upsert_prices(price_df, ticker: str, db_path: Path = DB_PATH) -> int:
    """INSERT OR REPLACE OHLCV rows from a yfinance DataFrame."""
    rows = []
    for date_idx, row in price_df.iterrows():
        date_str = str(date_idx)[:10]  # "YYYY-MM-DD"
        rows.append((
            ticker,
            date_str,
            float(row.get("Open") or 0),
            float(row.get("High") or 0),
            float(row.get("Low") or 0),
            float(row.get("Close") or 0),
            float(row.get("Volume") or 0),
        ))
    with _connect(db_path) as conn:
        cur = conn.executemany(
            """INSERT OR REPLACE INTO prices
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return cur.rowcount


def query_news(ticker: str | None = None,
               db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return news rows, optionally filtered by ticker."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if ticker:
            rows = conn.execute(
                "SELECT * FROM news WHERE ticker=? ORDER BY datetime DESC", (ticker,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM news ORDER BY datetime DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def query_news_window(
    ticker: str,
    from_date: str,
    to_date: str,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Return news for *ticker* within [from_date, to_date] (YYYY-MM-DD, inclusive).

    Datetime is stored as "YYYY-MM-DD HH:MM UTC"; prefix comparison on the first
    10 characters gives correct date-level filtering.
    """
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM news
               WHERE ticker = ?
                 AND substr(datetime, 1, 10) >= ?
                 AND substr(datetime, 1, 10) <= ?
               ORDER BY datetime DESC""",
            (ticker, from_date, to_date),
        ).fetchall()
    return [dict(r) for r in rows]


_EVENT_SIM_THRESHOLD = 0.85


def _dedup_event_list(items: list[dict]) -> list[dict]:
    """Remove near-duplicate events within the same (article_id, event_type) group.

    Two events are duplicates when SequenceMatcher similarity >= _EVENT_SIM_THRESHOLD.
    The longer description is kept; on equal length the first occurrence wins.
    Guarantees at least one survivor per input group.
    """
    from difflib import SequenceMatcher

    if len(items) == 1:
        return items
    descs = [it["description"] for it in items]
    drop: set[int] = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if SequenceMatcher(None, descs[i], descs[j]).ratio() >= _EVENT_SIM_THRESHOLD:
                if len(descs[i]) >= len(descs[j]):
                    drop.add(j)
                else:
                    drop.add(i)
    return [it for k, it in enumerate(items) if k not in drop]


def upsert_events(events: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Insert events, skipping near-duplicate descriptions for the same article+type.

    Step 1 — deduplicate within the incoming batch (same article_id + event_type,
              similarity >= _EVENT_SIM_THRESHOLD).
    Step 2 — for each candidate, check existing DB rows for the same article_id +
              event_type. If a near-duplicate exists and is longer, skip the new one.
              If the new one is longer, delete the existing row and insert the new one.

    Returns count of newly inserted rows.
    """
    from difflib import SequenceMatcher
    from itertools import groupby

    if not events:
        return 0

    # Step 1: dedup within the batch
    key = lambda e: (e["article_id"], e["event_type"])
    deduped: list[dict] = []
    for _, grp in groupby(sorted(events, key=key), key=key):
        deduped.extend(_dedup_event_list(list(grp)))

    inserted = 0
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Step 2: check against existing DB rows per (article_id, event_type)
        for (art_id, ev_type), grp in groupby(
            sorted(deduped, key=key), key=key
        ):
            new_items = list(grp)
            existing = conn.execute(
                "SELECT id, description FROM events WHERE article_id=? AND event_type=?",
                (art_id, ev_type),
            ).fetchall()

            to_insert: list[dict] = []
            for new in new_items:
                dup_found = False
                for ex in existing:
                    sim = SequenceMatcher(
                        None, new["description"], ex["description"]
                    ).ratio()
                    if sim >= _EVENT_SIM_THRESHOLD:
                        dup_found = True
                        if len(new["description"]) > len(ex["description"]):
                            # New is longer: replace existing (delete + insert)
                            conn.execute("DELETE FROM events WHERE id=?", (ex["id"],))
                            to_insert.append(new)
                        # else: existing is longer or equal — discard new silently
                        break
                if not dup_found:
                    to_insert.append(new)

            for e in to_insert:
                conn.execute(
                    """INSERT OR IGNORE INTO events
                       (id, article_id, ticker, datetime, event_type, entities, description, is_material)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        e["id"],
                        e["article_id"],
                        e["ticker"],
                        e["datetime"],
                        e["event_type"],
                        e.get("entities"),
                        e["description"],
                        int(e.get("is_material", 1)),
                    ),
                )
                inserted += conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]

    return inserted


def populate_relevance(db_path: Path = DB_PATH) -> dict[str, dict[str, int]]:
    """Compute and store is_relevant for all news rows.

    Uses TICKER_KEYWORDS from relevance.py. Safe to re-run (idempotent).

    Returns:
        Per-ticker counts: {ticker: {"total": N, "relevant": R, "irrelevant": I}}
    """
    from src.data.relevance import is_relevant

    with _connect(db_path) as conn:
        rows = conn.execute("SELECT id, ticker, headline, summary FROM news").fetchall()

        updates: list[tuple[int, str]] = []
        for row_id, ticker, headline, summary in rows:
            flag = 1 if is_relevant(headline, summary, ticker) else 0
            updates.append((flag, row_id))

        conn.executemany(
            "UPDATE news SET is_relevant=?, relevance_method='keyword' WHERE id=?",
            updates,
        )

    # Build summary from the updates we just applied
    stats: dict[str, dict[str, int]] = {}
    for flag, row_id in updates:
        pass  # need ticker — re-query is simpler

    with _connect(db_path) as conn:
        for ticker_row in conn.execute(
            "SELECT ticker, is_relevant, COUNT(*) FROM news GROUP BY ticker, is_relevant"
        ).fetchall():
            t, rel, cnt = ticker_row
            if t not in stats:
                stats[t] = {"total": 0, "relevant": 0, "irrelevant": 0}
            stats[t]["total"] += cnt
            if rel == 1:
                stats[t]["relevant"] += cnt
            else:
                stats[t]["irrelevant"] += cnt
    return stats


def query_news_relevant(
    ticker: str,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Return only is_relevant=1 news rows for *ticker*, newest first."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM news WHERE ticker=? AND is_relevant=1 ORDER BY datetime DESC",
            (ticker,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_briefing(
    ticker: str,
    days: int,
    generated_at: str,
    markdown: str,
    db_path: Path = DB_PATH,
) -> None:
    """Upsert a pre-generated briefing. Safe to re-run (idempotent)."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO briefings (ticker, days, generated_at, markdown)
               VALUES (?, ?, ?, ?)""",
            (ticker, days, generated_at, markdown),
        )


def get_briefing(ticker: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Return the most recently generated briefing for *ticker*, or None."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT ticker, days, generated_at, markdown
               FROM briefings WHERE ticker = ?
               ORDER BY generated_at DESC LIMIT 1""",
            (ticker,),
        ).fetchone()
    return dict(row) if row else None


def query_events(ticker: str | None = None, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return event rows, optionally filtered by ticker."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if ticker:
            rows = conn.execute(
                "SELECT * FROM events WHERE ticker=? ORDER BY datetime DESC", (ticker,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM events ORDER BY datetime DESC").fetchall()
    return [dict(r) for r in rows]
