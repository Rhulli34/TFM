"""Verify that the train/val/test splits share no leaked sentences.

A test macro-F1 of 0.98 is high enough to demand proof that the model never saw
the exam. This script produces that proof as a machine-readable artefact:

  1. Exact overlap    - raw sentence strings intersected across the three splits.
  2. Normalised overlap - same, after lowercasing and stripping punctuation, so
                        that cosmetic edits cannot hide a repeat.
  3. Near-duplicates  - TF-IDF cosine similarity of every test sentence against
                        every train sentence, reported at several thresholds
                        together with the closest pairs.

The near-duplicate stage runs twice on purpose. Scikit-learn's default token
pattern (\\b\\w\\w+\\b) discards standalone numbers, which makes two boilerplate
earnings sentences that differ only in their figures look identical. The
number-aware pass keeps digits as tokens and shows how much of the measured
similarity was really numeric.

Usage:
    python scripts/leakage_check.py

Writes reports/results/leakage_check.json and prints a summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "reports" / "results"

SPLITS = ("train", "val", "test")
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
TOP_N_PAIRS = 10
CHUNK = 50

# Keeps standalone numbers as tokens, unlike the scikit-learn default.
NUMBER_AWARE_PATTERN = r"(?u)\b\w+\b"


def load_splits() -> dict[str, pd.DataFrame]:
    """Read the three parquet splits written by src/data/splits.py."""
    splits = {}
    for name in SPLITS:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.splits` first."
            )
        splits[name] = pd.read_parquet(path)
    return splits


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_overlap(splits: dict[str, pd.DataFrame], normalized: bool) -> dict:
    """Count identical sentences shared by each pair of splits."""
    sets = {}
    for name, df in splits.items():
        texts = df["text"].map(normalize) if normalized else df["text"]
        sets[name] = set(texts)

    result = {}
    for a, b in (("test", "train"), ("test", "val"), ("val", "train")):
        shared = sets[a] & sets[b]
        result[f"{a}_vs_{b}"] = {
            "n_shared": len(shared),
            "examples": sorted(shared)[:5],
        }
    return result


def near_duplicates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    token_pattern: str | None,
) -> dict:
    """Cosine similarity of each test sentence against its closest train match.

    Args:
        train: training split.
        test:  test split.
        token_pattern: regex passed to TfidfVectorizer. None uses the
            scikit-learn default, which drops standalone numbers.

    Returns:
        Counts above each threshold, distribution summary and the closest pairs.
    """
    kwargs = {"ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1}
    if token_pattern is not None:
        kwargs["token_pattern"] = token_pattern

    vec = TfidfVectorizer(**kwargs)
    vec.fit(list(train["text"]) + list(test["text"]))
    x_train = vec.transform(train["text"])
    x_test = vec.transform(test["text"])

    best_scores = np.zeros(len(test))
    best_indices = np.zeros(len(test), dtype=int)
    for i in range(0, len(test), CHUNK):
        sims = cosine_similarity(x_test[i : i + CHUNK], x_train)
        best_scores[i : i + CHUNK] = sims.max(axis=1)
        best_indices[i : i + CHUNK] = sims.argmax(axis=1)

    counts = {
        f"above_{t:.2f}": int((best_scores > t).sum()) for t in THRESHOLDS
    }
    pct = {
        f"above_{t:.2f}_pct": round(float((best_scores > t).mean() * 100), 2)
        for t in THRESHOLDS
    }

    pairs = []
    for ti in np.argsort(best_scores)[::-1][:TOP_N_PAIRS]:
        tr_i = best_indices[ti]
        pairs.append(
            {
                "cosine": round(float(best_scores[ti]), 4),
                "test_text": test["text"].iloc[ti],
                "train_text": train["text"].iloc[tr_i],
                "test_label": test["label"].iloc[ti],
                "train_label": train["label"].iloc[tr_i],
                "same_label": bool(
                    test["label"].iloc[ti] == train["label"].iloc[tr_i]
                ),
            }
        )

    return {
        "counts": counts,
        "pct_of_test": pct,
        "mean_max_cosine": round(float(best_scores.mean()), 4),
        "median_max_cosine": round(float(np.median(best_scores)), 4),
        "max_cosine": round(float(best_scores.max()), 4),
        "top_pairs": pairs,
    }


def main() -> None:
    splits = load_splits()
    sizes = {name: int(len(df)) for name, df in splits.items()}
    print("Split sizes:", sizes, "total:", sum(sizes.values()))

    print("\n[1/3] Exact overlap...")
    exact = exact_overlap(splits, normalized=False)
    for key, val in exact.items():
        print(f"  {key}: {val['n_shared']}")

    print("\n[2/3] Normalised overlap...")
    normalised = exact_overlap(splits, normalized=True)
    for key, val in normalised.items():
        print(f"  {key}: {val['n_shared']}")

    print("\n[3/3] Near-duplicates (test vs train)...")
    default_tok = near_duplicates(splits["train"], splits["test"], None)
    number_aware = near_duplicates(
        splits["train"], splits["test"], NUMBER_AWARE_PATTERN
    )
    for label, res in (("default", default_tok), ("number-aware", number_aware)):
        print(
            f"  [{label}] >0.95: {res['counts']['above_0.95']}"
            f"  >0.90: {res['counts']['above_0.90']}"
            f"  max: {res['max_cosine']}  median: {res['median_max_cosine']}"
        )

    exact_clean = all(v["n_shared"] == 0 for v in exact.values())
    norm_clean = all(v["n_shared"] == 0 for v in normalised.values())

    payload = {
        "description": (
            "Leakage audit of the Financial PhraseBank splits used to train and "
            "evaluate the sentiment classifier."
        ),
        "split_sizes": sizes,
        "exact_overlap": exact,
        "normalized_overlap": normalised,
        "near_duplicates_default_tokenizer": default_tok,
        "near_duplicates_number_aware": number_aware,
        "verdict": {
            "exact_overlap_clean": exact_clean,
            "normalized_overlap_clean": norm_clean,
            "n_test_near_duplicates_above_095": number_aware["counts"][
                "above_0.95"
            ],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "leakage_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nExact overlap clean      : {exact_clean}")
    print(f"Normalised overlap clean : {norm_clean}")
    print(f"Saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
