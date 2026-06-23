"""Baseline sentiment classifier for the Financial PhraseBank corpus.

Approach:
  - TF-IDF (1-2 grams, sublinear TF, min_df=2), fit on train only.
  - Two classifiers: LogisticRegression and LinearSVC, class_weight='balanced'.
  - C swept over a log-spaced grid; best C chosen by macro-F1 on val.
  - Final metrics reported on test set.
  - Primary metric: macro-F1 (robust to the 61/25/14 class imbalance).

Usage:
    python -m src.models.baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, RESULTS_DIR

SEED = 42
CLASS_ORDER = ["negative", "neutral", "positive"]
C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

CLASSIFIERS = {
    "LogisticRegression": lambda c: LogisticRegression(
        C=c, class_weight="balanced", max_iter=1000, random_state=SEED
    ),
    "LinearSVC": lambda c: LinearSVC(
        C=c, class_weight="balanced", max_iter=2000, random_state=SEED
    ),
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    return train, val, test


def build_vectorizer(train_texts: pd.Series) -> TfidfVectorizer:
    """Fit TF-IDF on train texts only."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    vec.fit(train_texts)
    return vec


def tune_on_val(
    clf_factory,
    X_train,
    y_train,
    X_val,
    y_val,
) -> tuple[float, object]:
    """Grid-search C on val; return (best_c, fitted_model_on_train)."""
    best_c, best_score, best_clf = None, -1.0, None
    for c in C_GRID:
        clf = clf_factory(c)
        clf.fit(X_train, y_train)
        score = f1_score(y_val, clf.predict(X_val), average="macro")
        if score > best_score:
            best_score, best_c, best_clf = score, c, clf
    return best_c, best_clf


def evaluate(clf, X, y_true: pd.Series, split_name: str) -> dict:
    """Return metrics dict for a given split."""
    y_pred = clf.predict(X)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    acc = report["accuracy"]
    per_class = {cls: report[cls] for cls in CLASS_ORDER}
    print(f"\n  {split_name.upper()} — macro-F1={macro_f1:.4f}  acc={acc:.4f}")
    for cls in CLASS_ORDER:
        m = per_class[cls]
        print(f"    {cls:10s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}  n={int(m['support'])}")
    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "per_class": {
            cls: {
                "precision": round(per_class[cls]["precision"], 4),
                "recall":    round(per_class[cls]["recall"], 4),
                "f1":        round(per_class[cls]["f1-score"], 4),
                "support":   int(per_class[cls]["support"]),
            }
            for cls in CLASS_ORDER
        },
    }


def save_confusion_matrix(clf, X_test, y_test, name: str) -> None:
    """Save confusion-matrix figure for the best model."""
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {name} (test)", fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"baseline_confusion_matrix_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out.name}")


def run() -> dict:
    """Train, tune, and evaluate both baseline classifiers. Return results dict."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train, val, test = load_splits()

    print("Fitting TF-IDF on train...")
    vec = build_vectorizer(train["text"])
    X_train = vec.transform(train["text"])
    X_val   = vec.transform(val["text"])
    X_test  = vec.transform(test["text"])
    y_train, y_val, y_test = train["label"], val["label"], test["label"]
    print(f"  Vocabulary size: {len(vec.vocabulary_):,} features")

    results = {}
    best_macro_f1 = -1.0
    best_clf_name = None
    best_clf_obj = None

    for name, factory in CLASSIFIERS.items():
        print(f"\n{'='*50}")
        print(f"Classifier: {name}")
        best_c, clf = tune_on_val(factory, X_train, y_train, X_val, y_val)
        print(f"  Best C on val: {best_c}")
        val_metrics  = evaluate(clf, X_val,  y_val,  "val")
        test_metrics = evaluate(clf, X_test, y_test, "test")
        results[name] = {
            "best_c": best_c,
            "val":  val_metrics,
            "test": test_metrics,
        }
        if test_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_metrics["macro_f1"]
            best_clf_name = name
            best_clf_obj = clf

    print(f"\n{'='*50}")
    print(f"WINNER on test macro-F1: {best_clf_name}  (macro-F1={best_macro_f1:.4f})")

    # Confusion matrix and serialized model for the winner
    save_confusion_matrix(best_clf_obj, X_test, y_test, best_clf_name)

    # Also save the vectorizer (needed to transform future inputs)
    model_path = MODELS_DIR / f"baseline_{best_clf_name.lower().replace(' ', '_')}.joblib"
    joblib.dump({"vectorizer": vec, "classifier": best_clf_obj}, model_path)
    print(f"  Model saved -> models/{model_path.name}")

    summary = {
        "winner": best_clf_name,
        "tfidf": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "vocab_size": len(vec.vocabulary_),
        },
        "classifiers": results,
    }
    out_path = RESULTS_DIR / "baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved -> reports/results/baseline.json")

    return summary


if __name__ == "__main__":
    run()
