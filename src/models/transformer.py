"""Fine-tune a transformer for financial sentiment classification.

Base model: microsoft/deberta-v3-base — a strong general-purpose encoder with
NO task-specific pre-training on Financial PhraseBank. This ensures a fair
comparison with the baseline.

ProsusAI/finbert is deliberately excluded: it has already been fine-tuned on
financial sentiment data that overlaps with Financial PhraseBank, which would
contaminate the evaluation (data leakage at the pre-training level).

Approach:
  - DeBERTa-v3-base tokenizer, max_length=128 (median text = 21 words)
  - Class-weighted cross-entropy derived from train-set frequencies
  - HuggingFace Trainer; best checkpoint selected by macro-F1 on val
  - Single test-set evaluation at the very end
  - fp16 mixed precision for speed on RTX 5060 (Blackwell)

Usage:
    python -m src.models.transformer

Fallback (OOM): set BATCH_SIZE=8, GRAD_ACCUM_STEPS=2 (same effective batch).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, RESULTS_DIR

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
BATCH_SIZE = 16       # per device; drop to 8 + GRAD_ACCUM_STEPS=2 if OOM
GRAD_ACCUM_STEPS = 1  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_ORDER = ["negative", "neutral", "positive"]

MODEL_OUT_DIR = MODELS_DIR / "deberta-v3-base-finetuned"


# ── Dataset helper ─────────────────────────────────────────────────────────────
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Weighted-loss Trainer ──────────────────────────────────────────────────────
class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-class weights to cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits.float(), labels)  # float32 cast: safe regardless of model dtype
        return (loss, outputs) if return_outputs else loss


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    accuracy = float((preds == labels).mean())
    return {"macro_f1": macro_f1, "accuracy": accuracy}


# ── Helpers ────────────────────────────────────────────────────────────────────
def class_weights_from_train(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights, one per class in LABEL_TO_ID order."""
    n_total = len(train_df)
    n_classes = len(LABEL_TO_ID)
    weights = []
    for cls in CLASS_ORDER:
        count = (train_df["label"] == cls).sum()
        weights.append(n_total / (n_classes * count))
    return torch.tensor(weights, dtype=torch.float)


def evaluate_split(
    trainer: Trainer,
    dataset: SentimentDataset,
    y_true: list[int],
    split_name: str,
) -> dict:
    preds_out = trainer.predict(dataset)
    y_pred = np.argmax(preds_out.predictions, axis=-1).tolist()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=list(LABEL_TO_ID.values()),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    acc = report["accuracy"]
    per_class = {cls: report[cls] for cls in CLASS_ORDER}
    print(f"\n  {split_name.upper()} — macro-F1={macro_f1:.4f}  acc={acc:.4f}")
    for cls in CLASS_ORDER:
        m = per_class[cls]
        print(
            f"    {cls:10s}: P={m['precision']:.3f}  R={m['recall']:.3f}"
            f"  F1={m['f1-score']:.3f}  n={int(m['support'])}"
        )
    return {
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(acc), 4),
        "per_class": {
            cls: {
                "precision": round(float(per_class[cls]["precision"]), 4),
                "recall":    round(float(per_class[cls]["recall"]), 4),
                "f1":        round(float(per_class[cls]["f1-score"]), 4),
                "support":   int(per_class[cls]["support"]),
            }
            for cls in CLASS_ORDER
        },
        "y_pred": y_pred,
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_TO_ID.values()))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix — DeBERTa-v3-base (test)", fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "transformer_confusion_matrix_deberta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    print("Loading splits...")
    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    val_df   = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test_df  = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    y_train = train_df["label_id"].tolist()
    y_val   = val_df["label_id"].tolist()
    y_test  = test_df["label_id"].tolist()

    # 2. Class weights
    weights = class_weights_from_train(train_df)
    print(f"Class weights: { {cls: round(float(w), 3) for cls, w in zip(CLASS_ORDER, weights)} }")

    # 3. Tokenize
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,  # DataCollatorWithPadding handles dynamic padding
        )

    print("Tokenizing...")
    train_enc = tokenize(train_df["text"].tolist())
    val_enc   = tokenize(val_df["text"].tolist())
    test_enc  = tokenize(test_df["text"].tolist())

    train_ds = SentimentDataset(train_enc, y_train)
    val_ds   = SentimentDataset(val_enc,   y_val)
    test_ds  = SentimentDataset(test_enc,  y_test)

    # 4. Model
    print(f"\nLoading model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,  # force float32 regardless of safetensors default
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,   # noqa: deprecated in v5.2 → fine for now
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=False,                 # float32: avoids dtype mismatch between logits and class_weights
        fp16=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,   # Windows: avoid multiprocessing issues
        seed=SEED,
        report_to="none",
        logging_steps=20,
        save_total_limit=1,         # keep only best checkpoint on disk
    )

    # 6. Train
    trainer = WeightedLossTrainer(
        class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining — {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, "
          f"batch={BATCH_SIZE} (accum={GRAD_ACCUM_STEPS}), float32")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    elapsed_min = elapsed / 60
    print(f"\nTraining done in {elapsed_min:.1f} min")

    # 7. Evaluate val and test with best checkpoint
    print("\nEvaluating with best checkpoint...")
    val_metrics  = evaluate_split(trainer, val_ds,  y_val,  "val")
    test_metrics = evaluate_split(trainer, test_ds, y_test, "test")

    # 8. Confusion matrix
    save_confusion_matrix(y_test, test_metrics.pop("y_pred"))
    val_metrics.pop("y_pred", None)

    # 9. Save metrics
    summary = {
        "base_model": BASE_MODEL,
        "hyperparams": {
            "max_length": MAX_LENGTH,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "bf16": False,
            "fp16": False,
        },
        "class_weights": {cls: round(float(w), 4) for cls, w in zip(CLASS_ORDER, weights)},
        "training_time_min": round(elapsed_min, 1),
        "val":  val_metrics,
        "test": test_metrics,
    }
    out_path = RESULTS_DIR / "transformer.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved -> reports/results/transformer.json")

    # 10. Save tokenizer alongside model (needed for inference)
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    print(f"  Model + tokenizer saved -> models/deberta-v3-base-finetuned/")

    return summary


if __name__ == "__main__":
    result = run()
    print(f"\nFINAL TEST macro-F1: {result['test']['macro_f1']}")
