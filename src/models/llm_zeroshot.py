"""LLM zero-shot sentiment classifier using OpenAI API.

Approach:
  - gpt-4o-mini, temperature=0 for reproducibility.
  - Zero-shot: system prompt with class definitions only, no labelled examples.
  - Structured JSON output: {"label": "negative|neutral|positive"}.

Usage:
    python -m src.models.llm_zeroshot            # probe mode: 5 sentences + cost estimate
    python -m src.models.llm_zeroshot --full     # full evaluation (339 test sentences)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS_OUTPUT = 20
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0

# Pricing as of mid-2025 (USD per 1 million tokens)
PRICE_INPUT_PER_1M = 0.150
PRICE_OUTPUT_PER_1M = 0.600

VALID_LABELS = {"negative", "neutral", "positive"}
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_ORDER = ["negative", "neutral", "positive"]

SYSTEM_PROMPT = (
    "You are a financial news sentiment classifier.\n\n"
    "Classify the sentiment of a financial news sentence into exactly one of:\n"
    "- negative: unfavourable financial news — declining profits, losses, layoffs, "
    "missed targets, negative outlook, write-downs, or deteriorating market position.\n"
    "- neutral: factual or administrative with no clear positive/negative financial "
    "implication — routine announcements, appointments, regulatory filings, product "
    "descriptions, flat metrics, or ambiguous information.\n"
    "- positive: favourable financial news — growing profits, new contracts, market "
    "expansion, exceeded targets, positive outlook, or strengthening market position.\n\n"
    'Respond ONLY with JSON: {"label": "<label>"} '
    "where <label> is exactly one of: negative, neutral, positive.\n"
    "No explanation. No extra text."
)

DATA_DIR = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"


# ── API helpers ───────────────────────────────────────────────────────────────

def _get_client():
    """Return an OpenAI client, raising a clear error if key is missing."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set. Add it to .env")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def classify_one(client, sentence: str) -> tuple[str, int, int]:
    """Classify a single sentence. Returns (label, input_tokens, output_tokens)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sentence},
                ],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_OUTPUT,
            )
            raw = response.choices[0].message.content
            usage = response.usage
            parsed = json.loads(raw)
            label = parsed.get("label", "").lower().strip()
            if label not in VALID_LABELS:
                label = "unknown"
            return label, usage.prompt_tokens, usage.completion_tokens
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"  [WARN] Failed after {MAX_RETRIES} attempts: {exc}")
                return "unknown", 0, 0
            time.sleep(RETRY_DELAY_S * attempt)


# ── Probe mode ────────────────────────────────────────────────────────────────

def run_probe(client, test_df: pd.DataFrame) -> None:
    """Classify 5 sentences, print results and cost estimate for full run."""
    sample = test_df.sample(5, random_state=42).reset_index(drop=True)

    total_in = 0
    total_out = 0

    print("=" * 70)
    print(f"PROBE — 5 sentences via {MODEL} (temperature={TEMPERATURE})")
    print("=" * 70)

    for i, row in sample.iterrows():
        label, in_tok, out_tok = classify_one(client, row["text"])
        total_in += in_tok
        total_out += out_tok
        match = "OK" if label == row["label"] else "WRONG"
        print(f"\n[{i+1}] TRUE={row['label']:8s}  PRED={label:8s}  [{match}]")
        print(f"     tokens in={in_tok} out={out_tok}")
        print(f"     TEXT: {row['text'][:110]}")

    avg_in = total_in / 5
    avg_out = total_out / 5
    n_full = len(test_df)
    est_in_tok = avg_in * n_full
    est_out_tok = avg_out * n_full
    est_cost = (
        est_in_tok  / 1_000_000 * PRICE_INPUT_PER_1M
        + est_out_tok / 1_000_000 * PRICE_OUTPUT_PER_1M
    )

    print("\n" + "=" * 70)
    print(f"COST ESTIMATE for full run ({n_full} sentences)")
    print(f"  Avg tokens/call : {avg_in:.0f} input + {avg_out:.0f} output")
    print(f"  Total input  : ~{est_in_tok:,.0f} tokens  @ ${PRICE_INPUT_PER_1M}/1M = ${est_in_tok/1e6*PRICE_INPUT_PER_1M:.4f}")
    print(f"  Total output : ~{est_out_tok:,.0f} tokens  @ ${PRICE_OUTPUT_PER_1M}/1M = ${est_out_tok/1e6*PRICE_OUTPUT_PER_1M:.4f}")
    print(f"  >> Estimated total cost: ${est_cost:.4f} USD")
    print("=" * 70)
    print("\nProbe done. If OK, run with --full to classify all 339 test sentences.")


# ── Full evaluation ───────────────────────────────────────────────────────────

def run_full(client, test_df: pd.DataFrame) -> None:
    """Classify all test sentences, compute metrics, and save outputs."""
    n = len(test_df)
    print(f"Classifying {n} test sentences with {MODEL}…\n")

    labels_pred: list[str] = []
    total_in = total_out = 0
    t0 = time.time()

    for idx, (_, row) in enumerate(test_df.iterrows()):
        label, in_tok, out_tok = classify_one(client, row["text"])
        labels_pred.append(label)
        total_in += in_tok
        total_out += out_tok
        pos = idx + 1
        if pos % 50 == 0 or pos == n:
            elapsed = time.time() - t0
            print(f"  [{pos:3d}/{n}]  elapsed={elapsed:.0f}s  "
                  f"cost_so_far=${(total_in/1e6*PRICE_INPUT_PER_1M + total_out/1e6*PRICE_OUTPUT_PER_1M):.4f}")

    elapsed = time.time() - t0
    actual_cost = (
        total_in  / 1_000_000 * PRICE_INPUT_PER_1M
        + total_out / 1_000_000 * PRICE_OUTPUT_PER_1M
    )

    n_unknown = labels_pred.count("unknown")
    if n_unknown:
        print(f"\n[WARN] {n_unknown} sentences returned 'unknown' — check API errors.")

    y_true_str = test_df["label"].tolist()
    y_true = [LABEL_TO_ID[l] for l in y_true_str]
    y_pred = [LABEL_TO_ID.get(l, -1) for l in labels_pred]

    # Only use valid predictions for metrics
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp != -1]
    yt_v = [p[0] for p in pairs]
    yp_v = [p[1] for p in pairs]

    macro_f1 = f1_score(yt_v, yp_v, average="macro")
    accuracy = np.mean(np.array(yt_v) == np.array(yp_v))
    report = classification_report(
        yt_v, yp_v,
        labels=[0, 1, 2], target_names=CLASS_ORDER,
        output_dict=True, zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — {MODEL} zero-shot on TEST (n={n})")
    print(f"  macro-F1 = {macro_f1:.4f}   accuracy = {accuracy:.4f}")
    print(f"  Cost: ${actual_cost:.4f} USD  "
          f"({total_in:,} in + {total_out:,} out tokens, {elapsed:.0f}s)")
    for cls in CLASS_ORDER:
        m = report[cls]
        print(f"  {cls:10s}: P={m['precision']:.3f}  R={m['recall']:.3f}"
              f"  F1={m['f1-score']:.3f}  n={int(m['support'])}")

    # ── Predictions CSV ───────────────────────────────────────────────────────
    pred_df = test_df[["text", "label"]].copy().reset_index(drop=True)
    pred_df["predicted"] = labels_pred
    pred_df["correct"] = pred_df["label"] == pred_df["predicted"]
    pred_path = RESULTS / "llm_zeroshot_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved: {pred_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(yt_v, yp_v, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {MODEL} zero-shot (test)", fontweight="bold")
    plt.tight_layout()
    cm_path = FIGURES / "llm_zeroshot_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {cm_path}")

    # ── JSON results ──────────────────────────────────────────────────────────
    results = {
        "model": MODEL,
        "approach": "zero-shot",
        "temperature": TEMPERATURE,
        "n_test": n,
        "n_unknown": n_unknown,
        "cost_usd": round(actual_cost, 4),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "elapsed_s": round(elapsed, 1),
        "test": {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"], 4),
                    "f1":        round(report[cls]["f1-score"], 4),
                    "support":   int(report[cls]["support"]),
                }
                for cls in CLASS_ORDER
            },
        },
    }

    results_path = RESULTS / "llm_zeroshot.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved: {results_path}")

    # ── Error inspection ──────────────────────────────────────────────────────
    errors = pred_df[~pred_df["correct"]]
    print(f"\nMisclassifications: {len(errors)} / {n} ({len(errors)/n*100:.1f}%)")
    for _, row in errors.iterrows():
        print(f"  TRUE={row['label']:8s}  PRED={row['predicted']:8s}  {row['text'][:90]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM zero-shot sentiment classifier")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation on all 339 test sentences")
    args = parser.parse_args()

    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"Test set loaded: {len(test_df)} rows")

    client = _get_client()
    print(f"OpenAI client ready. Model: {MODEL}\n")

    if args.full:
        run_full(client, test_df)
    else:
        run_probe(client, test_df)


if __name__ == "__main__":
    main()
