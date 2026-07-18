"""Batch sentiment predictor using the fine-tuned DeBERTa-v3-base from F2.

The model is loaded once (lazy) and cached for the lifetime of the process.
Call predict_sentiment(texts) anywhere; it returns (label, confidence) pairs.

In dev (local checkpoint present) the model is loaded from disk.
In prod (Docker / CI) it is downloaded from the HF Hub on first use and cached
at ~/.cache/huggingface/hub (pre-warmed at build time in the Dockerfile).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

# Public HF Hub repo — used when the local checkpoint is absent (prod / CI).
MODEL_ID = "Rhulli/financial-news-radar-deberta"
# Local checkpoint (dev only — ignored in Docker image).
_LOCAL_CKPT = ROOT / "models" / "deberta-v3-base-finetuned" / "checkpoint-198"

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

_BATCH_SIZE = 32
_MAX_LENGTH = 128


def _model_source() -> str:
    """Return the local checkpoint path if available, otherwise the Hub ID."""
    return str(_LOCAL_CKPT) if _LOCAL_CKPT.exists() else MODEL_ID


class _Predictor:
    """Wraps the DeBERTa model for batch inference."""

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        source = _model_source()
        print(f"[sentiment] Loading model from {source!r}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            torch_dtype=torch.float32,  # explicit float32: avoids safetensors fp16 default
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[sentiment] Ready on {self.device}.", flush=True)

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        with torch.inference_mode():
            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i : i + _BATCH_SIZE]
                enc = self.tokenizer(
                    batch,
                    max_length=_MAX_LENGTH,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for prob_row in probs:
                    pred_id = int(np.argmax(prob_row))
                    results.append((ID_TO_LABEL[pred_id], float(prob_row[pred_id])))
        return results


_predictor: Optional[_Predictor] = None


def predict_sentiment(texts: list[str]) -> list[tuple[str, float]]:
    """Classify *texts* in batch. Model loaded once on first call.

    Args:
        texts: List of strings (headline + summary recommended).

    Returns:
        List of (label, confidence) where label in {negative, neutral, positive}
        and confidence is the softmax probability of the predicted class.
    """
    global _predictor
    if _predictor is None:
        _predictor = _Predictor()
    return _predictor.predict(texts)
