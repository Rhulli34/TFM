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
Eres un analista de riesgo financiero. Redacta un briefing en ESPAÑOL sobre {ticker} para el \
periodo {from_date}–{to_date}, a partir de los artículos y eventos que se te dan. Los titulares \
y nombres de fuentes se mantienen en inglés. NO inventes datos: usa solo lo aportado.

Estructura obligatoria (mantén las cabeceras ## exactas):

## Veredicto
2–3 frases: ¿cuál es la situación de la empresa esta semana? Empieza por lo más material. \
Si se ha disparado una alerta, explica en una frase por qué (sentimiento muy negativo u evento legal).

## Termómetro de sentimiento
Distribución Negativo/Neutro/Positivo (%). Una frase de interpretación.

## Temas clave
AGRUPA los artículos por tema, NO listes eventos sueltos. Si varios artículos hablan del MISMO \
hecho, es UN solo tema con el nº de artículos entre paréntesis. Ordena los temas por \
materialidad/riesgo, marcando cada uno con 🔴 ALTO / 🟠 MEDIO / 🟢 BAJO. No uses una categoría \
"otros" como cajón de sastre: reparte esos ítems en temas con sentido. Si dos artículos se \
contradicen, reconcílialo explicando que son catalizadores distintos.

## Señales de riesgo
Los 3–5 riesgos más materiales, ya deduplicados (no repitas el mismo hecho varias veces). \
Si una cifra parece un máximo teórico, indícalo ("hasta X, máximo teórico") en vez de afirmarla \
como si fuera segura. Distingue tono (sentimiento) de hecho (evento): aclara cuándo algo suena \
mal pero no ha pasado nada material, y viceversa.

## Fuentes principales
Tabla markdown con los artículos más relevantes: titular en inglés, fecha (YYYY-MM-DD), sentimiento.

Reglas de rigor:
- Deduplica. Un hecho = una entrada + recuento de artículos entre paréntesis.
- Etiquetas de sentimiento SIEMPRE en español: negativo / neutro / positivo (nunca en inglés).
- Juicio federal contra Meta (si aplica): presenta SIEMPRE la cifra de daños como \
"hasta 1,4 billones de dólares (máximo teórico legal)". Añade que las partes barajan \
~200.000 millones de dólares como cifra realista y que el jurado es consultivo \
(la decisión final corresponde a la jueza). Nunca la presentes como condena esperada.
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
        f"# Informe de noticias: {ticker}\n"
        f"**Periodo:** {from_str} a {to_str}  \n"
        f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"**Artículos analizados:** {len(articles)}  \n\n"
    )
    markdown = header + body

    _BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BRIEFING_DIR / f"{ticker}_{to_str}.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"  Saved -> {out_path.relative_to(ROOT)}", flush=True)

    return markdown
