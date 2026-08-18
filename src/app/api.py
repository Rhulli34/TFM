"""FastAPI backend for the Financial News Radar.

Endpoints
---------
GET /health                        Liveness probe.
GET /catalog                       All companies available in the DB.
GET /portfolio?tickers=A,B,...     Per-ticker summary + alert flag.
GET /company/{ticker}              30-day sentiment series, recent news, events.
GET /company/{ticker}/briefing     Markdown briefing via gpt-4o-mini (F4 pipeline).

Run with:
    uvicorn src.app.api:app --reload
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.app.settings import get_settings

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_CFG = get_settings()

# Best-effort company names; any ticker not listed falls back to its symbol.
_TICKER_NAMES: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "NVDA":  "NVIDIA Corp.",
    "MSFT":  "Microsoft Corp.",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "GOOGL": "Alphabet Inc.",
    "TSLA":  "Tesla Inc.",
    "JPM":   "JPMorgan Chase & Co.",
    "XOM":   "Exxon Mobil Corp.",
    "PFE":   "Pfizer Inc.",
    "GS":    "Goldman Sachs Group Inc.",
    "QCOM":  "Qualcomm Inc.",
}

# Alert threshold: mean net sentiment below this triggers an alert.
_ALERT_THRESHOLD = -0.3


# ── DB helper ─────────────────────────────────────────────────────────────────

@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_CFG.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _catalog_tickers() -> set[str]:
    """Return the set of tickers that have is_relevant=1 news in the DB."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM news WHERE is_relevant=1"
        ).fetchall()
    return {r["ticker"] for r in rows}


def _snapshot_max_date() -> date:
    """Return the most recent article date in the DB.

    All lookback windows use this as their anchor so the portfolio view stays
    populated when the DB is a historical snapshot (not live).
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT MAX(substr(datetime, 1, 10)) AS max_date FROM news"
        ).fetchone()
    raw = row["max_date"] if row and row["max_date"] else date.today().isoformat()
    return date.fromisoformat(raw)


def _require_ticker(ticker: str) -> None:
    if ticker not in _catalog_tickers():
        raise HTTPException(status_code=404, detail=f"Ticker {ticker!r} not found in database.")


# ── Pydantic models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    env: str
    db_path: str


class CatalogItem(BaseModel):
    ticker: str
    name: str
    news_count: int    # total is_relevant=1 articles
    date_from: str     # oldest article date (YYYY-MM-DD)
    date_to: str       # newest article date (YYYY-MM-DD)


class PortfolioItem(BaseModel):
    ticker: str
    name: str
    news_count_7d: int
    net_sentiment_7d: float   # mean(pos=+1, neu=0, neg=-1) over last 7 days
    alert: bool
    alert_reason: str | None  # human-readable explanation when alert=True


class SentimentPoint(BaseModel):
    date: str
    net_sentiment: float
    n_articles: int


class NewsItem(BaseModel):
    date: str
    headline: str
    sentiment_label: str
    sentiment_score: float
    url: str | None


class EventItem(BaseModel):
    datetime: str
    event_type: str
    description: str
    is_material: bool


class CompanyDetail(BaseModel):
    ticker: str
    name: str
    sentiment_series: list[SentimentPoint]  # daily aggregates, last 30 days
    recent_news: list[NewsItem]             # last 20 relevant articles
    events: list[EventItem]                 # all events for this ticker


class BriefingResponse(BaseModel):
    ticker: str
    markdown: str
    generated_at: str  # ISO-8601 UTC
    days: int | None = None  # lookback window used when pre-generating


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial News Radar",
    description="Sentiment monitoring API for portfolio companies.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CFG.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", env=_CFG.env, db_path=str(_CFG.db_path))


@app.get("/catalog", response_model=list[CatalogItem])
def catalog() -> list[CatalogItem]:
    """Return every ticker that has is_relevant=1 news in the DB.

    The list grows automatically as the DB is populated with new companies —
    no code changes needed to support a larger universe.
    """
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT
              ticker,
              COUNT(*)                          AS news_count,
              MIN(substr(datetime, 1, 10))      AS date_from,
              MAX(substr(datetime, 1, 10))      AS date_to
            FROM news
            WHERE is_relevant = 1
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()

    return [
        CatalogItem(
            ticker=r["ticker"],
            name=_TICKER_NAMES.get(r["ticker"], r["ticker"]),
            news_count=r["news_count"],
            date_from=r["date_from"],
            date_to=r["date_to"],
        )
        for r in rows
    ]


@app.get("/portfolio", response_model=list[PortfolioItem])
def portfolio(
    tickers: str | None = Query(
        None,
        description="Comma-separated ticker list, e.g. 'AAPL,NVDA,JPM'. "
                    "Omit to use the full catalog.",
    ),
    days: int = Query(
        7,
        ge=1,
        le=30,
        description="Lookback window in calendar days (typically 7, 14, or 30).",
    ),
) -> list[PortfolioItem]:
    """Per-ticker 7-day summary and alert flag for a user-chosen portfolio.

    Unknown tickers (not in catalog) are silently skipped so that a single
    bad ticker does not break the whole request.
    """
    known = _catalog_tickers()

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        ticker_list = [t for t in ticker_list if t in known]
    else:
        ticker_list = sorted(known)

    since = (_snapshot_max_date() - timedelta(days=days)).isoformat()
    results: list[PortfolioItem] = []

    with _db() as conn:
        for ticker in ticker_list:
            # 7-day aggregated sentiment
            agg = conn.execute(
                """
                SELECT
                  COUNT(*) AS n,
                  AVG(CASE sentiment_label
                    WHEN 'positive' THEN  1.0
                    WHEN 'neutral'  THEN  0.0
                    WHEN 'negative' THEN -1.0
                    ELSE 0.0 END) AS net
                FROM news
                WHERE ticker = ?
                  AND is_relevant = 1
                  AND substr(datetime, 1, 10) >= ?
                """,
                (ticker, since),
            ).fetchone()

            n = agg["n"] or 0
            net = round(agg["net"] or 0.0, 4)

            # Material legal events from is_relevant=1 articles in last 7 days.
            # Only 'legal' events are used: legal = lawsuit/investigation/fine, reliably
            # negative. 'guidance' and 'macro' are excluded because guidance is often
            # positive and macro events are generic market commentary, not company-specific.
            legal_rows = conn.execute(
                """
                SELECT e.description
                FROM events e
                JOIN news n ON n.id = e.article_id
                WHERE e.ticker = ?
                  AND e.is_material = 1
                  AND e.event_type = 'legal'
                  AND substr(e.datetime, 1, 10) >= ?
                  AND n.is_relevant = 1
                ORDER BY e.datetime DESC
                LIMIT 3
                """,
                (ticker, since),
            ).fetchall()
            legal_count = len(legal_rows)

            alert = (net < _ALERT_THRESHOLD) or (legal_count > 0)
            reason: str | None = None
            if alert:
                parts = []
                if net < _ALERT_THRESHOLD:
                    parts.append(
                        f"Sentimiento muy negativo: {net:+.2f} de media"
                        f" en {n} noticias (últimos {days} días)"
                    )
                if legal_count > 0:
                    first_desc = legal_rows[0]["description"][:160]
                    label = (
                        "Evento legal detectado"
                        if legal_count == 1
                        else f"{legal_count} eventos legales detectados"
                    )
                    parts.append(f"{label}: {first_desc}")
                reason = ". ".join(parts)

            results.append(
                PortfolioItem(
                    ticker=ticker,
                    name=_TICKER_NAMES.get(ticker, ticker),
                    news_count_7d=n,
                    net_sentiment_7d=net,
                    alert=alert,
                    alert_reason=reason,
                )
            )

    return results


@app.get("/company/{ticker}", response_model=CompanyDetail)
def company_detail(ticker: str) -> CompanyDetail:
    """30-day daily sentiment series, last 20 relevant articles, and all events."""
    ticker = ticker.upper()
    _require_ticker(ticker)

    since_30 = (_snapshot_max_date() - timedelta(days=30)).isoformat()

    with _db() as conn:
        series_rows = conn.execute(
            """
            SELECT
              substr(datetime, 1, 10) AS day,
              AVG(CASE sentiment_label
                WHEN 'positive' THEN  1.0
                WHEN 'neutral'  THEN  0.0
                WHEN 'negative' THEN -1.0
                ELSE 0.0 END) AS net_sentiment,
              COUNT(*) AS n_articles
            FROM news
            WHERE ticker = ?
              AND is_relevant = 1
              AND substr(datetime, 1, 10) >= ?
            GROUP BY day
            ORDER BY day
            """,
            (ticker, since_30),
        ).fetchall()

        news_rows = conn.execute(
            """
            SELECT substr(datetime, 1, 10) AS day, headline,
                   sentiment_label, sentiment_score, url
            FROM news
            WHERE ticker = ?
              AND is_relevant = 1
            ORDER BY datetime DESC
            LIMIT 20
            """,
            (ticker,),
        ).fetchall()

        event_rows = conn.execute(
            """
            SELECT datetime, event_type, description, is_material
            FROM events
            WHERE ticker = ?
            ORDER BY datetime DESC
            """,
            (ticker,),
        ).fetchall()

    return CompanyDetail(
        ticker=ticker,
        name=_TICKER_NAMES.get(ticker, ticker),
        sentiment_series=[
            SentimentPoint(
                date=r["day"],
                net_sentiment=round(r["net_sentiment"], 4),
                n_articles=r["n_articles"],
            )
            for r in series_rows
        ],
        recent_news=[
            NewsItem(
                date=r["day"],
                headline=r["headline"],
                sentiment_label=r["sentiment_label"] or "neutral",
                sentiment_score=float(r["sentiment_score"] or 0.5),
                url=r["url"],
            )
            for r in news_rows
        ],
        events=[
            EventItem(
                datetime=r["datetime"],
                event_type=r["event_type"],
                description=r["description"],
                is_material=bool(r["is_material"]),
            )
            for r in event_rows
        ],
    )


@app.get("/company/{ticker}/briefing", response_model=BriefingResponse)
def company_briefing(ticker: str) -> BriefingResponse:
    """Return the pre-generated briefing for *ticker* from the DB.

    Briefings are produced offline by scripts/pregenerate_briefings.py and
    stored in the 'briefings' table inside radar.db.  Production never calls
    the OpenAI API at request time — it only serves the cached result.
    """
    ticker = ticker.upper()
    _require_ticker(ticker)

    from src.data.store import get_briefing

    row = get_briefing(ticker)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pre-generated briefing for {ticker!r}. "
                "Run scripts/pregenerate_briefings.py to generate one."
            ),
        )
    return BriefingResponse(
        ticker=ticker,
        markdown=row["markdown"],
        generated_at=row["generated_at"],
        days=row["days"],
    )


# ── Static frontend (served last so API routes take priority) ──────────────────

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the SPA entry point."""
    return FileResponse(_FRONTEND_DIR / "index.html")

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
