"""Hybrid ticker-relevance classifier: keyword pre-filter + LLM second pass.

Stage 1 (keyword): fast, free — marks is_relevant=1 on direct mention of
  company name, ticker, CEO, brands, etc.
Stage 2 (LLM): gpt-4o-mini on the keyword=0 residual — rescues articles that
  refer to the company without naming it literally (e.g. "the chipmaker").

relevance_method column records which stage decided each row:
  'keyword' = keyword stage was decisive (either true or false)
  'llm'     = LLM was consulted (keyword said no; LLM had the final word)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Word/phrase matches are applied case-insensitively as substrings.
# Order within each list does not matter.
TICKER_KEYWORDS: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook", "app store", "ios "],
    "NVDA": ["nvidia", "nvda", "jensen huang", "geforce", "cuda"],
    "MSFT": ["microsoft", "msft", "satya nadella", "azure", "copilot", "windows ", "xbox"],
    "AMZN": ["amazon", "amzn", "aws", "jeff bezos", "andy jassy", "prime "],
    "META": [
        "meta platforms", "meta's", "meta stock", " meta ", "(meta)", "nasdaq:meta",
        "facebook", "instagram", "whatsapp", "reels", "mark zuckerberg", "zuckerberg",
    ],
    "GOOGL": [
        "google", "alphabet", "googl", " goog", "waymo", "youtube",
        "sundar pichai", "deepmind", "gemini",
    ],
    "TSLA": ["tesla", "tsla", "elon musk", "cybertruck", "model s", "model 3", "model y", "powerwall"],
    "JPM":  ["jpmorgan", "jp morgan", "j.p. morgan", " jpm", "jamie dimon", "chase bank"],
    "XOM":  ["exxon", "exxonmobil", " xom", "darren woods"],
    "PFE":  ["pfizer", " pfe", "biontech", "paxlovid", "comirnaty"],
}


# Human-readable company names injected into the LLM prompt
COMPANY_NAMES: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "NVDA":  "NVIDIA Corp",
    "MSFT":  "Microsoft Corp",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "GOOGL": "Alphabet Inc. (Google)",
    "TSLA":  "Tesla Inc.",
    "JPM":   "JPMorgan Chase & Co.",
    "XOM":   "Exxon Mobil Corp",
    "PFE":   "Pfizer Inc.",
}

_LLM_SYSTEM = """\
You are a financial news relevance classifier for a portfolio monitoring system.

Task: for each article in the list, decide whether it MAINLY covers {company} ({ticker}).

Rules:
- true  → {company} or its direct business/products/results is the primary subject
- false → generic market commentary, industry overview, or a different company is the focus
          (mentioning {company} briefly in passing does NOT make it true)

Respond ONLY with valid JSON:
{{"results": [{{"article_index": 0, "relevant": true}}, ...]}}
One entry per input article, in the same order.
"""

_LLM_BATCH_SIZE = 20
_LLM_RETRY_SLEEP = 2.0
_PRICE_INPUT_PER_1M  = 0.150  # gpt-4o-mini
_PRICE_OUTPUT_PER_1M = 0.600


def is_relevant(headline: str | None, summary: str | None, ticker: str) -> bool:
    """Return True if *headline* or *summary* mentions the company for *ticker*.

    Args:
        headline: Article headline string (may be None).
        summary:  Article summary string (may be None).
        ticker:   Uppercase ticker symbol (must be a key in TICKER_KEYWORDS).

    Returns:
        True if any keyword for the ticker appears in the combined text.
        False if ticker not in TICKER_KEYWORDS or no keyword matches.
    """
    keywords = TICKER_KEYWORDS.get(ticker.upper())
    if not keywords:
        return False
    text = (f"{headline or ''} {summary or ''}").lower()
    return any(kw.lower() in text for kw in keywords)


# ── LLM second-pass ────────────────────────────────────────────────────────────

def _llm_classify_batch(
    articles: list[dict],
    ticker: str,
    client,
) -> tuple[list[tuple[str, bool]], int, int]:
    """Call gpt-4o-mini on one batch of articles.

    Args:
        articles: Dicts with keys: id, headline, summary.
        ticker:   Ticker symbol.
        client:   openai.OpenAI instance.

    Returns:
        (results, input_tokens, output_tokens)
        results = list of (article_id, is_relevant_bool)
    """
    company = COMPANY_NAMES.get(ticker.upper(), ticker)
    system = _LLM_SYSTEM.replace("{company}", company).replace("{ticker}", ticker)

    user_msg = json.dumps(
        [
            {
                "article_index": i,
                "headline": a.get("headline", ""),
                "summary": (a.get("summary") or "")[:300],
            }
            for i, a in enumerate(articles)
        ],
        ensure_ascii=False,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            inp_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens

            data = json.loads(raw)
            results: list[tuple[str, bool]] = []
            for item in data.get("results", []):
                idx = item.get("article_index")
                if idx is None or not isinstance(idx, int) or idx >= len(articles):
                    continue
                results.append((articles[idx]["id"], bool(item.get("relevant", False))))
            return results, inp_tok, out_tok

        except Exception as exc:
            if attempt == 2:
                print(f"  WARN LLM batch failed: {exc}", flush=True)
                return [], 0, 0
            time.sleep(_LLM_RETRY_SLEEP)

    return [], 0, 0


def run_llm_pilot(
    db_path: Path,
    n: int = 50,
    update_db: bool = True,
) -> dict:
    """Run LLM relevance check on *n* randomly sampled keyword=0 articles.

    Samples evenly across the 10 working tickers. Updates is_relevant and
    relevance_method in the DB for those rows (unless update_db=False).

    Args:
        db_path:   SQLite file path.
        n:         Number of articles to pilot (default 50).
        update_db: If True, persist LLM decisions to DB.

    Returns:
        Dict with keys: results (list of row dicts with llm_decision),
        input_tokens, output_tokens, cost_usd, extrapolated_total_cost_usd,
        total_dubious (all keyword=0 rows in DB).
    """
    import sqlite3
    from dotenv import load_dotenv
    from openai import OpenAI

    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Count total dubious per working ticker
    working_tickers = list(TICKER_KEYWORDS.keys())
    ticker_placeholder = ",".join("?" * len(working_tickers))
    total_dubious_rows = conn.execute(
        f"SELECT COUNT(*) FROM news WHERE is_relevant=0 AND ticker IN ({ticker_placeholder})",
        working_tickers,
    ).fetchone()[0]

    # Sample ~n/10 per ticker (balanced), is_relevant=0
    per_ticker = max(1, n // len(working_tickers))
    sample_rows: list[dict] = []
    for ticker in working_tickers:
        rows = conn.execute(
            "SELECT id, ticker, headline, summary FROM news "
            "WHERE ticker=? AND is_relevant=0 ORDER BY RANDOM() LIMIT ?",
            (ticker, per_ticker),
        ).fetchall()
        sample_rows.extend([dict(r) for r in rows])

    conn.close()

    # Group by ticker for batching
    by_ticker: dict[str, list[dict]] = {}
    for row in sample_rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    all_results: list[dict] = []
    total_in_tok = 0
    total_out_tok = 0

    for ticker, articles in by_ticker.items():
        for i in range(0, len(articles), _LLM_BATCH_SIZE):
            batch = articles[i: i + _LLM_BATCH_SIZE]
            decisions, in_tok, out_tok = _llm_classify_batch(batch, ticker, client)
            total_in_tok += in_tok
            total_out_tok += out_tok
            for art_id, llm_rel in decisions:
                art = next(a for a in batch if a["id"] == art_id)
                all_results.append({
                    "id": art_id,
                    "ticker": art["ticker"],
                    "headline": art.get("headline", ""),
                    "summary": (art.get("summary") or "")[:120],
                    "llm_relevant": llm_rel,
                })

    # Persist to DB
    if update_db and all_results:
        conn2 = sqlite3.connect(db_path)
        conn2.executemany(
            "UPDATE news SET is_relevant=?, relevance_method='llm' WHERE id=?",
            [(1 if r["llm_relevant"] else 0, r["id"]) for r in all_results],
        )
        conn2.commit()
        conn2.close()

    # Cost for this pilot
    pilot_cost = (
        total_in_tok / 1_000_000 * _PRICE_INPUT_PER_1M
        + total_out_tok / 1_000_000 * _PRICE_OUTPUT_PER_1M
    )

    # Extrapolate: tokens per article from pilot → total
    n_piloted = len(all_results) or 1
    avg_in  = total_in_tok  / n_piloted
    avg_out = total_out_tok / n_piloted
    extrap_cost = (
        total_dubious_rows * avg_in  / 1_000_000 * _PRICE_INPUT_PER_1M
        + total_dubious_rows * avg_out / 1_000_000 * _PRICE_OUTPUT_PER_1M
    )

    return {
        "n_piloted": n_piloted,
        "total_dubious": total_dubious_rows,
        "results": all_results,
        "input_tokens": total_in_tok,
        "output_tokens": total_out_tok,
        "pilot_cost_usd": round(pilot_cost, 5),
        "avg_input_tokens_per_article": round(avg_in, 1),
        "avg_output_tokens_per_article": round(avg_out, 1),
        "extrapolated_total_cost_usd": round(extrap_cost, 4),
    }
