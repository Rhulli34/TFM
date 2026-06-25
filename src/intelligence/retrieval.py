"""Semantic retrieval over news articles using FAISS + sentence-transformers.

Embeds headline+summary with all-MiniLM-L6-v2 and builds a FAISS IndexFlatIP
(inner product on L2-normalised vectors == cosine similarity) for fast ANN search.
The embedder is loaded once and cached for the process lifetime.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[retrieval] Loading {_MODEL_NAME}...", flush=True)
        _embedder = SentenceTransformer(_MODEL_NAME)
        print("[retrieval] Embedder ready.", flush=True)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings of shape (N, D).

    Args:
        texts: List of strings to embed.

    Returns:
        Float32 numpy array, rows are unit vectors.
    """
    model = _get_embedder()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / np.maximum(norms, 1e-10)).astype("float32")


def build_index(articles: list[dict]) -> tuple:
    """Build a FAISS cosine-similarity index from *articles*.

    Args:
        articles: List of news dicts with 'headline' and 'summary' keys.

    Returns:
        (index, articles) — the FAISS IndexFlatIP and the article list (same order).
    """
    import faiss

    texts = [
        f"{a.get('headline', '')} {a.get('summary', '')}".strip()
        for a in articles
    ]
    embs = embed_texts(texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, articles


def retrieve(index, articles: list[dict], query: str, k: int = 10) -> list[dict]:
    """Return the top-k articles most semantically similar to *query*.

    Args:
        index:    FAISS index built with build_index().
        articles: Article list in the same order as passed to build_index().
        query:    Free-text query string.
        k:        Number of results to return (capped to len(articles)).

    Returns:
        List of article dicts with an added '_similarity' float field,
        sorted by descending similarity.
    """
    k = min(k, len(articles))
    q_emb = embed_texts([query])
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        hit = dict(articles[idx])
        hit["_similarity"] = float(score)
        results.append(hit)
    return results
