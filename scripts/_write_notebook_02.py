"""Helper: writes notebooks/02_modeling_comparison.ipynb.

Sections so far: Baseline, Transformer fine-tune, partial comparison.
LLM zero-shot section (cc*) to be added in the next sub-step.
"""
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
    # ── Header ────────────────────────────────────────────────────────────────
    cell("markdown", (
        "# F2 — Modeling Comparison: Baseline vs Transformer vs LLM Zero-Shot\n\n"
        "Phase 2 · Financial News Radar (TFM)\n\n"
        "This notebook accumulates results for the three modelling approaches. "
        "Each section adds one model; the final section compares them head-to-head.\n\n"
        "**Primary metric:** macro-F1 (robust to the 61 / 25 / 14 class imbalance).  \n"
        "**Train/val/test:** 1 581 / 339 / 339 samples (70/15/15 stratified, seed 42)."
    ), "bb01"),

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
from matplotlib.image import imread

FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"negative": "#d62728", "neutral": "#1f77b4", "positive": "#2ca02c"}
CLASS_ORDER = ["negative", "neutral", "positive"]
print("Paths OK. ROOT =", ROOT)
""", "bb02"),

    # ── Section 1: Baseline ──────────────────────────────────────────────────
    cell("markdown", (
        "---\n"
        "## 1 · Baseline: TF-IDF + Classical Classifiers\n\n"
        "Approach: TF-IDF (1-2 grams, sublinear TF, min_df=2) fit exclusively on "
        "train text; two classifiers (LogisticRegression and LinearSVC) with "
        "`class_weight='balanced'`; C swept over a log-spaced grid and chosen by "
        "macro-F1 on val.  \n"
        "Full implementation: `src/models/baseline.py`."
    ), "bb03"),

    cell("code", """\
with open(RESULTS / "baseline.json") as f:
    bl = json.load(f)

winner = bl["winner"]
print(f"Winner: {winner}")
print(f"TF-IDF vocab size: {bl['tfidf']['vocab_size']:,}")
print()

rows = []
for clf_name, data in bl["classifiers"].items():
    rows.append({
        "Classifier": clf_name,
        "Best C": data["best_c"],
        "Val macro-F1":  data["val"]["macro_f1"],
        "Test macro-F1": data["test"]["macro_f1"],
        "Test accuracy":  data["test"]["accuracy"],
    })
summary_df = pd.DataFrame(rows).set_index("Classifier")
summary_df
""", "bb04"),

    cell("code", """\
rows_pc = []
for clf_name, data in bl["classifiers"].items():
    for cls in CLASS_ORDER:
        m = data["test"]["per_class"][cls]
        rows_pc.append({
            "Classifier": clf_name, "Class": cls,
            "Precision": m["precision"], "Recall": m["recall"],
            "F1": m["f1"], "Support": m["support"],
        })
pc_df = pd.DataFrame(rows_pc)
pivot = pc_df.pivot_table(index="Classifier", columns="Class", values="F1")[CLASS_ORDER]
print("Per-class F1 on TEST set:")
print(pivot.round(3).to_string())
""", "bb05"),

    cell("code", """\
clf_names = list(bl["classifiers"].keys())
x = np.arange(len(CLASS_ORDER))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
for i, clf_name in enumerate(clf_names):
    f1s = [bl["classifiers"][clf_name]["test"]["per_class"][cls]["f1"] for cls in CLASS_ORDER]
    bars = ax.bar(x + i * width - width / 2, f1s, width,
                  label=clf_name, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_ORDER)
ax.set_ylim(0, 1.08)
ax.set_ylabel("F1-score")
ax.set_title("Baseline — per-class F1 on test set", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES / "baseline_per_class_f1.png", dpi=150)
plt.show()
print("Figure saved.")
""", "bb06"),

    cell("code", """\
cm_path = FIGURES / f"baseline_confusion_matrix_{winner.lower().replace(' ', '_')}.png"
img = imread(str(cm_path))
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(img)
ax.axis("off")
ax.set_title(f"Confusion matrix — {winner} (test)", fontweight="bold")
plt.tight_layout()
plt.show()
""", "bb07"),

    cell("markdown", (
        "### Observations — Baseline\n\n"
        "- **LinearSVC wins** by a narrow margin (test macro-F1 ≈ 0.856 vs 0.855 for LR).\n"
        "- **Neutral** is classified almost perfectly (F1 > 0.93), as expected given its "
        "dominant share (61 %).\n"
        "- **Negative** is the hardest class: F1 ≈ 0.81 despite `class_weight='balanced'`, "
        "reflecting both its small size (n=46 in test) and lexical overlap with neutral.\n"
        "- **Positive** sits in between (F1 ≈ 0.82).\n"
        "- This establishes the **listón** for F2: a fine-tuned DeBERTa-v3-base should clear "
        "macro-F1 > 0.87 to justify the extra complexity."
    ), "bb08"),

    # ── Section 2: Transformer ───────────────────────────────────────────────
    cell("markdown", (
        "---\n"
        "## 2 · Transformer Fine-Tune (DeBERTa-v3-base)\n\n"
        "**Base model:** `microsoft/deberta-v3-base` — general-purpose encoder with NO "
        "task-specific pre-training on Financial PhraseBank.  \n"
        "**Why not ProsusAI/finbert?** FinBERT was fine-tuned on financial sentiment data that "
        "overlaps with Financial PhraseBank → data leakage; excluded on methodological grounds.\n\n"
        "**Setup:** 10 epochs, lr=2e-5, batch=16, float32, class-weighted cross-entropy "
        "(negative×2.49, neutral×0.54, positive×1.32), max_length=128, "
        "best checkpoint by val macro-F1.  \n"
        "Full implementation: `src/models/transformer.py`."
    ), "bb09"),

    cell("code", """\
with open(RESULTS / "transformer.json") as f:
    tr = json.load(f)

print(f"Base model : {tr['base_model']}")
print(f"Training time: {tr['training_time_min']} min")
print(f"Class weights: {tr['class_weights']}")
print()
print(f"VAL  macro-F1 = {tr['val']['macro_f1']:.4f}  acc = {tr['val']['accuracy']:.4f}")
print(f"TEST macro-F1 = {tr['test']['macro_f1']:.4f}  acc = {tr['test']['accuracy']:.4f}")
""", "bb10"),

    cell("code", """\
rows_tr = []
for cls in CLASS_ORDER:
    m = tr["test"]["per_class"][cls]
    rows_tr.append({"Class": cls, "Precision": m["precision"],
                    "Recall": m["recall"], "F1": m["f1"], "Support": m["support"]})
tr_df = pd.DataFrame(rows_tr).set_index("Class")
print("DeBERTa-v3-base — per-class metrics on TEST set:")
print(tr_df.round(4).to_string())
""", "bb11"),

    cell("code", """\
fig, ax = plt.subplots(figsize=(7, 4))
f1s = [tr["test"]["per_class"][cls]["f1"] for cls in CLASS_ORDER]
bars = ax.bar(CLASS_ORDER, f1s, color=[COLORS[c] for c in CLASS_ORDER],
              alpha=0.85, edgecolor="white")
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0.85, 1.03)
ax.set_ylabel("F1-score")
ax.set_title("DeBERTa-v3-base — per-class F1 on test set", fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES / "transformer_per_class_f1.png", dpi=150)
plt.show()
print("Figure saved.")
""", "bb12"),

    cell("code", """\
cm_path = FIGURES / "transformer_confusion_matrix_deberta.png"
img = imread(str(cm_path))
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(img)
ax.axis("off")
ax.set_title("Confusion matrix — DeBERTa-v3-base (test)", fontweight="bold")
plt.tight_layout()
plt.show()
""", "bb13"),

    cell("markdown", (
        "### Observations — Transformer\n\n"
        "- **macro-F1 = 0.981** on test — a +12.5 pp improvement over the baseline (0.856).\n"
        "- **Negative** is the biggest winner: F1 climbs from 0.814 → 0.979 with recall=1.000 "
        "(zero negative misses). This is the class that matters most for the early-warning use case.\n"
        "- **Neutral** and **positive** also improve substantially (0.932 → 0.993 and "
        "0.822 → 0.970 respectively).\n"
        "- Training took only **3 minutes** on an RTX 5060 (float32). The corpus is small enough "
        "that 10 epochs are sufficient for convergence.\n"
        "- The val macro-F1 (0.9928) is slightly above test (0.9806), consistent with "
        "best-checkpoint selection on val — not a sign of meaningful overfitting given the gap is "
        "within normal variance for n=339."
    ), "bb14"),

    # ── Section 3: LLM Zero-Shot ─────────────────────────────────────────────
    cell("markdown", (
        "---\n"
        "## 3 · LLM Zero-Shot (gpt-4o-mini)\n\n"
        "**Approach:** zero-shot — system prompt with class definitions only, no labelled examples.  \n"
        "**Model:** `gpt-4o-mini`, temperature=0 (reproducible).  \n"
        "**Output:** structured JSON `{\"label\": \"...\"}` to guarantee parseable single-label response.  \n"
        "**Cost:** $0.0108 USD for all 339 test sentences (~1 cent).  \n"
        "Full implementation: `src/models/llm_zeroshot.py`."
    ), "bb15"),

    cell("code", """\
with open(RESULTS / "llm_zeroshot.json") as f:
    llm = json.load(f)

print(f"Model    : {llm['model']} ({llm['approach']})")
print(f"Cost     : ${llm['cost_usd']} USD ({llm['total_input_tokens']:,} in + {llm['total_output_tokens']:,} out tokens)")
print(f"Elapsed  : {llm['elapsed_s']}s")
print()
print(f"TEST macro-F1 = {llm['test']['macro_f1']:.4f}   accuracy = {llm['test']['accuracy']:.4f}")
""", "bb20"),

    cell("code", """\
rows_llm = []
for cls in CLASS_ORDER:
    m = llm["test"]["per_class"][cls]
    rows_llm.append({"Class": cls, "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "Support": m["support"]})
llm_df = pd.DataFrame(rows_llm).set_index("Class")
print("gpt-4o-mini zero-shot — per-class metrics on TEST set:")
print(llm_df.round(4).to_string())
""", "bb21"),

    cell("code", """\
cm_path = FIGURES / "llm_zeroshot_confusion_matrix.png"
img = imread(str(cm_path))
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(img)
ax.axis("off")
ax.set_title("Confusion matrix — gpt-4o-mini zero-shot (test)", fontweight="bold")
plt.tight_layout()
plt.show()
""", "bb22"),

    cell("markdown", (
        "### Observations — LLM Zero-Shot\n\n"
        "- **macro-F1 = 0.963** — only 1.8 pp below the fine-tuned DeBERTa, with zero training data.\n"
        "- **Negative recall = 1.000** (identical to DeBERTa): the LLM misses no negative sentences, "
        "which is the most important property for an early-warning system.\n"
        "- **Positive is the weak point** (F1=0.939): 8 of 12 errors are positive→neutral, "
        "reflecting a conservative bias — the LLM treats borderline-positive news (small growth, "
        "capacity expansion announcements, appointments) as neutral.\n"
        "- **Cost: $0.011** for 339 sentences. Scaling to a full real-time system of thousands of "
        "articles per day would make per-sentence cost significant — the fine-tuned transformer "
        "runs locally at zero marginal cost.\n"
        "- **No training needed**: this is the key advantage — immediate deployment on new tasks "
        "without labelled data."
    ), "bb23"),

    # ── Section 4: Final 3-Way Comparison ────────────────────────────────────
    cell("markdown", (
        "---\n"
        "## 4 · Final Comparison — Baseline vs Transformer vs LLM Zero-Shot\n\n"
        "All three approaches evaluated on the same test set (n=339). "
        "Primary metric: macro-F1."
    ), "bb16"),

    cell("code", """\
# ── Build 3-way comparison table ────────────────────────────────────────────
bl_test  = bl["classifiers"][bl["winner"]]["test"]
tr_test  = tr["test"]
llm_test = llm["test"]

rows_cmp = []
for cls in CLASS_ORDER:
    rows_cmp.append({
        "Class":           cls,
        "Baseline F1":     bl_test["per_class"][cls]["f1"],
        "Transformer F1":  tr_test["per_class"][cls]["f1"],
        "LLM Zero-Shot F1": llm_test["per_class"][cls]["f1"],
    })

cmp_df = pd.DataFrame(rows_cmp).set_index("Class")
cmp_df.loc["macro avg"] = {
    "Baseline F1":      bl_test["macro_f1"],
    "Transformer F1":   tr_test["macro_f1"],
    "LLM Zero-Shot F1": llm_test["macro_f1"],
}
print("=== F2 FINAL — Baseline vs Transformer vs LLM Zero-Shot (TEST) ===")
print(cmp_df.round(4).to_string())
""", "bb17"),

    cell("code", """\
# 3-bar comparison chart
labels_x = CLASS_ORDER + ["macro avg"]
bl_f1s   = [bl_test["per_class"][cls]["f1"]  for cls in CLASS_ORDER] + [bl_test["macro_f1"]]
tr_f1s   = [tr_test["per_class"][cls]["f1"]  for cls in CLASS_ORDER] + [tr_test["macro_f1"]]
llm_f1s  = [llm_test["per_class"][cls]["f1"] for cls in CLASS_ORDER] + [llm_test["macro_f1"]]

x = np.arange(len(labels_x))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
bars_bl  = ax.bar(x - width, bl_f1s,  width, label=f"Baseline ({bl['winner']})",
                  color="#aec7e8", edgecolor="white")
bars_tr  = ax.bar(x,          tr_f1s,  width, label="DeBERTa-v3-base (fine-tuned)",
                  color="#1f77b4", edgecolor="white")
bars_llm = ax.bar(x + width,  llm_f1s, width, label="gpt-4o-mini (zero-shot)",
                  color="#ff7f0e", edgecolor="white")

for bars, f1s in [(bars_bl, bl_f1s), (bars_tr, tr_f1s), (bars_llm, llm_f1s)]:
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.set_ylim(0.7, 1.07)
ax.set_ylabel("F1-score")
ax.set_title("F2 Final — Baseline vs Transformer vs LLM Zero-Shot (test set)",
             fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIGURES / "comparison_all_three_models.png", dpi=150)
plt.show()
print("Figure saved.")
""", "bb18"),

    cell("markdown", (
        "## F2 Summary — Phase 2 Final Results\n\n"
        "| Model | Approach | Test macro-F1 | neg F1 | neu F1 | pos F1 | Cost / 339 sent |\n"
        "|-------|----------|:---:|:---:|:---:|:---:|:---:|\n"
        "| TF-IDF + LinearSVC | Classical ML | 0.856 | 0.814 | 0.932 | 0.822 | $0 |\n"
        "| **DeBERTa-v3-base** | **Fine-tuned** | **0.981** | **0.979** | **0.993** | **0.970** | $0 (local, 3 min) |\n"
        "| gpt-4o-mini | Zero-shot | 0.963 | 0.979 | 0.971 | 0.939 | $0.011 |\n\n"
        "**Key findings:**\n"
        "1. Fine-tuning is the clear winner (+12.5 pp over baseline, +1.8 pp over zero-shot LLM).\n"
        "2. The LLM zero-shot achieves 0.963 with no training data — remarkable for a zero-shot system.\n"
        "3. All three models achieve negative recall=1.000 or near-perfect, which is critical for the "
        "early-warning use case (no false negatives on bad news).\n"
        "4. The fine-tuned transformer runs locally at zero marginal cost; the LLM incurs API cost "
        "per inference — relevant for production at scale.\n"
        "5. Positive class is where models diverge most: baseline=0.822, LLM=0.939, transformer=0.970."
    ), "bb19"),
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

out = ROOT / "notebooks" / "02_modeling_comparison.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out}")
