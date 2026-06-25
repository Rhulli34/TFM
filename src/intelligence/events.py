"""Event extraction from news articles using gpt-4o-mini.

Each article is analysed for structured financial events (earnings, M&A, legal, etc.).
Articles are sent in batches of _BATCH_SIZE to reduce API calls.
Results are returned as flat dicts ready for upsert_events().
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
_BATCH_SIZE = 10
_RETRY_SLEEP = 2.0

VALID_EVENT_TYPES = frozenset({
    "earnings", "guidance", "m_and_a", "leadership",
    "legal", "product", "analyst", "macro", "other",
})

_SYSTEM_PROMPT = """\
You are a financial event extractor for a news monitoring system.

Given a list of financial news articles about {ticker}, extract structured financial events \
from each one.

For each article return an object with:
  - article_index: the 0-based index of the article in the input list
  - events: list of events (empty list [] if no material event found), each event with:
      - event_type: one of [earnings, guidance, m_and_a, leadership, legal, product, analyst, macro, other]
      - entities:    list of company or person names most relevant to the event
      - description: one concise sentence describing the event and its financial significance
      - is_material: true if this could affect an investment decision; false otherwise

Respond ONLY with valid JSON:
{"results": [{"article_index": 0, "events": [...]}, ...]}

Rules:
- Do not fabricate events not supported by the text.
- Generic market commentary with no company-specific content -> events: []
- One article may produce multiple events (e.g. earnings + guidance).
"""


def _event_id(article_id: str, event_type: str, description: str) -> str:
    raw = f"{article_id}||{event_type}||{description}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _parse_batch_response(raw: str, batch: list[dict]) -> list[dict]:
    """Parse gpt-4o-mini JSON response into a flat list of event dicts."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    events: list[dict] = []
    for result in data.get("results", []):
        idx = result.get("article_index")
        if idx is None or not isinstance(idx, int) or idx >= len(batch):
            continue
        article = batch[idx]
        for ev in result.get("events", []):
            ev_type = ev.get("event_type", "other")
            if ev_type not in VALID_EVENT_TYPES:
                ev_type = "other"
            desc = str(ev.get("description", ""))[:500]
            if not desc:
                continue
            events.append({
                "id": _event_id(article["id"], ev_type, desc),
                "article_id": article["id"],
                "ticker": article["ticker"],
                "datetime": article.get("datetime", ""),
                "event_type": ev_type,
                "entities": json.dumps(ev.get("entities") or []),
                "description": desc,
                "is_material": 1 if ev.get("is_material") else 0,
            })
    return events


def extract_events(articles: list[dict], ticker: str) -> list[dict]:
    """Extract financial events from *articles* using gpt-4o-mini.

    Args:
        articles: News rows from the DB (must have: id, headline, summary, datetime, ticker).
        ticker:   Ticker symbol injected into the system prompt for context.

    Returns:
        Flat list of event dicts ready for store.upsert_events().
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    system = _SYSTEM_PROMPT.replace("{ticker}", ticker)
    all_events: list[dict] = []

    for i in range(0, len(articles), _BATCH_SIZE):
        batch = articles[i: i + _BATCH_SIZE]
        user_msg = json.dumps(
            [
                {
                    "article_index": j,
                    "headline": a.get("headline", ""),
                    "summary": (a.get("summary") or "")[:300],
                }
                for j, a in enumerate(batch)
            ],
            ensure_ascii=False,
        )

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content
                batch_events = _parse_batch_response(raw, batch)
                all_events.extend(batch_events)
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  WARN events batch {i // _BATCH_SIZE}: {exc}", flush=True)
                else:
                    time.sleep(_RETRY_SLEEP)

    return all_events
