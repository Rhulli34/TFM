"""Briefing generator: fetches recent news, extracts events, retrieves top
articles via RAG, and synthesises a structured markdown report with gpt-4o-mini.

Entry point: generate_briefing(ticker, days=7)
Output saved to reports/briefings/{ticker}_{YYYY-MM-DD}.md
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.config import REPORTS_DIR
from src.data.store import DB_PATH, init_db, query_news_window, upsert_events
from src.intelligence.events import extract_events
from src.intelligence.retrieval import build_index, retrieve

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

_BRIEFING_DIR = REPORTS_DIR / "briefings"

_SYNTHESIS_SYSTEM = """\
You are a financial research analyst writing concise briefings for a portfolio management desk.

Write a structured markdown briefing for {ticker} covering {from_date} to {to_date}.

Use EXACTLY this structure (keep the ## headers verbatim):

## Executive Summary
Two to three sentences on the most important developments in the period.

## Key Events
Bulleted list. Each bullet: [event_type] one-sentence description.
Include ALL extracted events provided. If none, write "No material events identified."

## Sentiment Distribution
One line with the figures (e.g. "Negative 20% | Neutral 42% | Positive 38%").
One sentence of interpretation relevant to the portfolio desk.

## Risk Flags
Bulleted list of negative, legal, or guidance-cut events only.
If none, write "None identified."

## Top Source Articles
List up to 5 of the most relevant articles: headline, date (YYYY-MM-DD), sentiment label.

Be factual, concise, and investment-focused. Do not add sections beyond the above five.
"""


def _sentiment_stats(articles: list[dict]) -> dict[str, tuple[int, float]]:
    counts: dict[str, int] = {"negative": 0, "neutral": 0, "positive": 0}
    for a in articles:
        lbl = a.get("sentiment_label") or "neutral"
        counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values()) or 1
    return {k: (v, round(v / total * 100, 1)) for k, v in counts.items()}


def generate_briefing(
    ticker: str,
    days: int = 7,
    db_path: Path = DB_PATH,
    as_of: date | None = None,
) -> str:
    """Generate a structured news briefing for *ticker* over the last *days*.

    Pipeline:
      1. Fetch articles in window from SQLite.
      2. Extract structured events with gpt-4o-mini and persist to DB.
      3. Build FAISS index; retrieve top-10 most relevant articles.
      4. Synthesise markdown briefing with gpt-4o-mini.
      5. Save to reports/briefings/{ticker}_{date}.md.

    Args:
        ticker:  Ticker symbol (e.g. "AAPL").
        days:    Calendar-day lookback window (default 7).
        db_path: SQLite file path.
        as_of:   Reference date for the window end. Defaults to today.
                 Pass the snapshot's max article date when pre-generating
                 so the window is anchored to the data, not the current date.

    Returns:
        Markdown string of the full briefing.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    init_db(db_path)

    to_date = as_of if as_of is not None else date.today()
    from_date = to_date - timedelta(days=days)
    from_str = from_date.isoformat()
    to_str = to_date.isoformat()

    # ── 1. Fetch window articles (is_relevant=1 only) ────────────────────────
    # Filtering here prevents irrelevant articles (e.g. Finnhub generic market
    # news attributed to multiple tickers) from generating spurious events.
    all_articles = query_news_window(ticker, from_str, to_str, db_path)
    articles = [a for a in all_articles if a.get("is_relevant") == 1]
    if not articles:
        return (
            f"# Financial News Briefing: {ticker}\n\n"
            f"No relevant news found for {from_str} to {to_str}."
        )
    print(
        f"  {len(articles)} relevant articles in window "
        f"[{from_str} -> {to_str}] ({len(all_articles)} total)",
        flush=True,
    )

    # ── 2. Event extraction ───────────────────────────────────────────────────
    print("  Extracting events...", flush=True)
    events = extract_events(articles, ticker)
    material = [e for e in events if e["is_material"]]
    print(f"  {len(events)} events extracted ({len(material)} material)", flush=True)
    if events:
        upsert_events(events, db_path)

    # ── 3. RAG retrieval ──────────────────────────────────────────────────────
    index, indexed = build_index(articles)
    rag_query = f"major events earnings guidance risks opportunities for {ticker}"
    top_articles = retrieve(index, indexed, rag_query, k=10)

    # ── 4. Assemble synthesis prompt ──────────────────────────────────────────
    stats = _sentiment_stats(articles)
    neg_c, neg_p = stats["negative"]
    neu_c, neu_p = stats["neutral"]
    pos_c, pos_p = stats["positive"]

    events_block = "\n".join(
        f"- [{e['event_type']}] {e['description']}" for e in material
    ) or "No material events extracted."

    top_block = "\n".join(
        f"- {a['headline']} ({a['datetime'][:10]}, {a.get('sentiment_label', '?')})"
        for a in top_articles[:10]
    )

    user_msg = (
        f"Ticker: {ticker}\n"
        f"Period: {from_str} to {to_str}\n"
        f"Total articles analysed: {len(articles)}\n\n"
        f"Sentiment: negative={neg_c} ({neg_p}%), "
        f"neutral={neu_c} ({neu_p}%), positive={pos_c} ({pos_p}%)\n\n"
        f"Extracted events:\n{events_block}\n\n"
        f"Top relevant articles (by semantic similarity):\n{top_block}\n"
    )

    system = (
        _SYNTHESIS_SYSTEM
        .replace("{ticker}", ticker)
        .replace("{from_date}", from_str)
        .replace("{to_date}", to_str)
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    body = resp.choices[0].message.content

    # ── 5. Assemble final markdown and save ───────────────────────────────────
    header = (
        f"# Financial News Briefing: {ticker}\n"
        f"**Period:** {from_str} to {to_str}  \n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"**Articles analysed:** {len(articles)}  \n\n"
    )
    markdown = header + body

    _BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BRIEFING_DIR / f"{ticker}_{to_str}.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"  Saved -> {out_path.relative_to(ROOT)}", flush=True)

    return markdown
