"""Create reproducible, stratified train/val/test splits of the Financial PhraseBank corpus.

Usage:
    python -m src.data.splits          # writes to data/processed/
    from src.data.splits import make_splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running as a module from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.corpus import load_corpus

SEED = 42
TRAIN_RATIO = 0.70  # 70 % train
VAL_RATIO = 0.15    # 15 % val
# TEST_RATIO = 0.15  (remainder after train)

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}

_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def make_splits(
    config: str = "sentences_allagree",
    out_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Load corpus, remove exact duplicates, split 70/15/15 (stratified by class).

    Args:
        config:  HuggingFace configuration name for the corpus subset.
        out_dir: Directory where parquet files are written.
                 Defaults to data/processed/.

    Returns:
        (splits, n_duplicates) where splits is a dict with keys
        "train", "val", "test" each holding a DataFrame with columns
        [text, label, label_id], and n_duplicates is how many rows were dropped.
    """
    if out_dir is None:
        out_dir = _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_corpus(config)

    n_before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_duplicates = n_before - len(df)

    df["label_id"] = df["label"].map(LABEL_TO_ID)

    # First cut: train vs (val + test)
    train, tmp = train_test_split(
        df,
        test_size=round(1.0 - TRAIN_RATIO, 10),
        stratify=df["label"],
        random_state=SEED,
    )
    # Second cut: val vs test — equal halves so each is ~15 %
    val, test = train_test_split(
        tmp,
        test_size=0.5,
        stratify=tmp["label"],
        random_state=SEED,
    )

    splits = {"train": train.reset_index(drop=True),
              "val":   val.reset_index(drop=True),
              "test":  test.reset_index(drop=True)}

    for name, split_df in splits.items():
        path = out_dir / f"{name}.parquet"
        split_df[["text", "label", "label_id"]].to_parquet(path, index=False)

    return splits, n_duplicates


def print_split_stats(splits: dict[str, pd.DataFrame], n_duplicates: int) -> None:
    """Print size and class balance for each split."""
    order = ["negative", "neutral", "positive"]
    print(f"\nDuplicates removed before split: {n_duplicates}")
    for name, df in splits.items():
        vc = df["label"].value_counts()
        pct = (vc / len(df) * 100).round(1)
        print(f"\n{name.upper():5s}  n={len(df)}")
        for lbl in order:
            print(f"  {lbl:10s}: {vc.get(lbl, 0):4d}  ({pct.get(lbl, 0.0):.1f}%)")


if __name__ == "__main__":
    splits, n_dups = make_splits()
    print_split_stats(splits, n_dups)
