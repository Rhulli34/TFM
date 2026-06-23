"""Helper: writes notebooks/01_eda_corpus.ipynb programmatically."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def cell(cell_type, source, cell_id):
    base = {"cell_type": cell_type, "id": cell_id, "metadata": {}}
    if cell_type == "markdown":
        base["source"] = [source]
    else:
        base.update({"execution_count": None, "outputs": [], "source": [source]})
    return base


cells = [
    cell("markdown", (
        "# EDA — Financial PhraseBank Corpus\n\n"
        "Phase 1 · Financial News Radar (TFM)\n\n"
        "Goal: understand the corpus before modelling. Covers class distribution, "
        "text-length analysis, duplicate detection, per-class examples, and a "
        "comparison across all four agreement-level configurations."
    ), "aa01"),

    cell("code", """\
import sys, json
from pathlib import Path

ROOT = Path().resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from src.data.corpus import load_corpus

FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"negative": "#d62728", "neutral": "#1f77b4", "positive": "#2ca02c"}
CLASS_ORDER = ["negative", "neutral", "positive"]
print("Paths OK. ROOT =", ROOT)
""", "aa02"),

    cell("markdown", "## 1 · Load corpus (sentences_allagree)", "aa03"),

    cell("code", """\
df = load_corpus("sentences_allagree")
print(f"Raw rows: {len(df)}")
print(df["label"].value_counts())
df.head(3)
""", "aa04"),

    cell("markdown", "## 2 · Class distribution", "aa05"),

    cell("code", """\
vc = df["label"].value_counts().reindex(CLASS_ORDER)
pct = (vc / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(CLASS_ORDER, vc.values,
              color=[COLORS[c] for c in CLASS_ORDER], edgecolor="white", linewidth=0.8)
for bar, count, p in zip(bars, vc.values, pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{count}\\n({p}%)", ha="center", va="bottom", fontsize=10)
ax.set_title("Class Distribution — sentences_allagree", fontweight="bold")
ax.set_xlabel("Sentiment class")
ax.set_ylabel("Count")
ax.set_ylim(0, vc.max() * 1.22)
ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
plt.tight_layout()
fig.savefig(FIGURES / "class_distribution.png", dpi=150)
plt.show()
print("Figure saved.")
""", "aa06"),

    cell("markdown", "## 3 · Text-length distribution", "aa07"),

    cell("code", """\
df["n_words"] = df["text"].str.split().str.len()
df["n_chars"] = df["text"].str.len()

print("=== Word count stats (global) ===")
print(df["n_words"].describe().round(1).to_string())
print("\\n=== Char count stats (global) ===")
print(df["n_chars"].describe().round(1).to_string())
print("\\n=== Word count stats by class ===")
print(df.groupby("label")["n_words"].describe().round(1).to_string())
""", "aa08"),

    cell("code", """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for metric, ax, label in [
    ("n_words", axes[0], "Number of words"),
    ("n_chars",  axes[1], "Number of characters"),
]:
    for cls in CLASS_ORDER:
        subset = df.loc[df["label"] == cls, metric]
        ax.hist(subset, bins=40, alpha=0.55, color=COLORS[cls], label=cls, edgecolor="none")
    ax.set_title(f"Text length ({label})", fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend()
plt.tight_layout()
fig.savefig(FIGURES / "text_length_distribution.png", dpi=150)
plt.show()
print("Figure saved.")
""", "aa09"),

    cell("code", """\
fig, ax = plt.subplots(figsize=(6, 4))
data_by_class = [df.loc[df["label"] == cls, "n_words"].values for cls in CLASS_ORDER]
bp = ax.boxplot(data_by_class, tick_labels=CLASS_ORDER, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2))
for patch, cls in zip(bp["boxes"], CLASS_ORDER):
    patch.set_facecolor(COLORS[cls])
    patch.set_alpha(0.7)
ax.set_title("Word count by class (box plot)", fontweight="bold")
ax.set_xlabel("Sentiment class")
ax.set_ylabel("Number of words")
plt.tight_layout()
fig.savefig(FIGURES / "text_length_boxplot.png", dpi=150)
plt.show()
print("Figure saved.")
""", "aa10"),

    cell("markdown", "## 4 · Duplicate and null check", "aa11"),

    cell("code", """\
n_total  = len(df)
n_nulls  = int(df["text"].isna().sum() + (df["text"].str.strip() == "").sum())
n_exact  = int(df.duplicated(subset="text").sum())
n_unique = n_total - n_exact

print(f"Total rows        : {n_total}")
print(f"Empty / null texts: {n_nulls}")
print(f"Exact duplicates  : {n_exact}")
print(f"Unique texts      : {n_unique}")
""", "aa12"),

    cell("markdown", "## 5 · Sample sentences per class", "aa13"),

    cell("code", """\
rng = np.random.default_rng(42)
for cls in CLASS_ORDER:
    subset = df[df["label"] == cls]
    idx = rng.choice(len(subset), size=min(4, len(subset)), replace=False)
    samples = subset.iloc[idx]["text"].tolist()
    print(f"\\n--- {cls.upper()} ({len(subset)} total) ---")
    for i, s in enumerate(samples, 1):
        print(f"  [{i}] {s}")
""", "aa14"),

    cell("markdown",
         "## 6 · Configuration comparison (allagree / 75agree / 66agree / 50agree)",
         "aa15"),

    cell("code", """\
configs = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]
config_stats = []
for cfg in configs:
    d = load_corpus(cfg)
    n_dup = d.duplicated(subset="text").sum()
    row = {"config": cfg, "total": len(d), "unique": len(d) - n_dup, "duplicates": int(n_dup)}
    for cls in CLASS_ORDER:
        cnt = int((d["label"] == cls).sum())
        row[cls] = cnt
        row[f"{cls}_pct"] = round(cnt / len(d) * 100, 1)
    config_stats.append(row)

cfg_df = pd.DataFrame(config_stats).set_index("config")
print(cfg_df[["total", "unique", "duplicates",
              "negative", "neutral", "positive",
              "negative_pct", "neutral_pct", "positive_pct"]].to_string())
""", "aa16"),

    cell("code", """\
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(configs))
width = 0.25
for i, cls in enumerate(CLASS_ORDER):
    ax.bar(x + i * width, cfg_df[cls].values, width,
           label=cls, color=COLORS[cls], alpha=0.85, edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels([c.replace("sentences_", "") for c in configs], rotation=15)
ax.set_title("Class counts by agreement configuration", fontweight="bold")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES / "config_comparison.png", dpi=150)
plt.show()
print("Figure saved.")
""", "aa17"),

    cell("markdown", "## 7 · Save numeric summary to reports/results/", "aa18"),

    cell("code", """\
summary = {
    "corpus_config": "sentences_allagree",
    "n_total": int(n_total),
    "n_unique": int(n_unique),
    "n_exact_duplicates": int(n_exact),
    "n_nulls_or_empty": int(n_nulls),
    "class_counts": {cls: int(vc[cls]) for cls in CLASS_ORDER},
    "class_pct":    {cls: float(pct[cls]) for cls in CLASS_ORDER},
    "word_count_global": {
        "mean":   round(float(df["n_words"].mean()), 1),
        "median": round(float(df["n_words"].median()), 1),
        "std":    round(float(df["n_words"].std()), 1),
        "min":    int(df["n_words"].min()),
        "max":    int(df["n_words"].max()),
    },
    "char_count_global": {
        "mean":   round(float(df["n_chars"].mean()), 1),
        "median": round(float(df["n_chars"].median()), 1),
        "std":    round(float(df["n_chars"].std()), 1),
        "min":    int(df["n_chars"].min()),
        "max":    int(df["n_chars"].max()),
    },
    "config_comparison": cfg_df[
        ["total", "unique", "duplicates", "negative", "neutral", "positive"]
    ].to_dict(orient="index"),
}

out_path = RESULTS / "eda_corpus.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Results saved to {out_path}")
core = {k: v for k, v in summary.items() if k != "config_comparison"}
print(json.dumps(core, indent=2))
""", "aa19"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "notebooks" / "01_eda_corpus.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out}")
