"""Phase 0 smoke test — verifies that the three data connectors work."""

import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_TICKERS
from src.data.news import get_company_news
from src.data.prices import get_prices
from src.data.corpus import load_corpus

TICKER = DEFAULT_TICKERS[0]  # AAPL

today = date.today()
start_news = (today - timedelta(days=30)).strftime("%Y-%m-%d")
start_prices = (today - timedelta(days=182)).strftime("%Y-%m-%d")
end = today.strftime("%Y-%m-%d")

SECTION = "=" * 60


# ── 1. NEWS ───────────────────────────────────────────────────────
print(f"\n{SECTION}")
print(f"  NEWS  |  {TICKER}  |  {start_news} -> {end}")
print(SECTION)
try:
    articles = get_company_news(TICKER, start_news, end)
    print(f"Total articles fetched: {len(articles)}\n")
    for i, art in enumerate(articles[:3], 1):
        print(f"  [{i}] {art['datetime']}")
        print(f"      {art['headline']}")
        print(f"      source: {art['source']}\n")
except EnvironmentError as exc:
    print(f"[SKIP] {exc}")
except Exception as exc:
    print(f"[ERROR] {exc}")


# ── 2. PRICES ─────────────────────────────────────────────────────
print(f"\n{SECTION}")
print(f"  PRICES  |  {TICKER}  |  {start_prices} -> {end}")
print(SECTION)
try:
    df = get_prices(TICKER, start_prices, end)
    print(f"Shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nLast 3 rows:")
    print(df.tail(3).to_string())
except Exception as exc:
    print(f"[ERROR] {exc}")


# ── 3. CORPUS ─────────────────────────────────────────────────────
print(f"\n{SECTION}")
print("  CORPUS  |  financial_phrasebank / sentences_allagree")
print(SECTION)
try:
    corpus = load_corpus()
    print(f"Total examples: {len(corpus)}")
    print("\nClass distribution:")
    print(corpus["label"].value_counts().to_string())
    print("\nSample rows:")
    print(corpus.head(3).to_string(index=False))
except Exception as exc:
    print(f"[ERROR] {exc}")

print(f"\n{SECTION}")
print("  Smoke test complete.")
print(SECTION)
