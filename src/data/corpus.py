"""Load the Financial PhraseBank corpus.

The HuggingFace loading script for this dataset has a broken internal URI,
so we download the raw zip file directly via huggingface_hub and parse
the @-separated text format ourselves.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "takala/financial_phrasebank"

# Map HuggingFace config names to the text file inside the zip
_CONFIG_TO_FILE = {
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    "sentences_66agree":  "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_75agree":  "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_50agree":  "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
}


def load_corpus(config: str = "sentences_allagree") -> pd.DataFrame:
    """Download and return the Financial PhraseBank corpus.

    Args:
        config: Agreement-level subset to use.
                Choices: sentences_allagree (highest quality, ~2.2k),
                         sentences_66agree, sentences_75agree,
                         sentences_50agree (~4.8k, full set).

    Returns:
        DataFrame with columns:
          text  (str) — financial sentence.
          label (str) — "negative", "neutral", or "positive".

    Raises:
        KeyError: if *config* is not a valid configuration name.
        huggingface_hub.errors.EntryNotFoundError: if the data file is missing.
    """
    if config not in _CONFIG_TO_FILE:
        raise KeyError(f"Unknown config '{config}'. Choose from: {list(_CONFIG_TO_FILE)}")

    # Download the zip (~1 MB) from HuggingFace; cached after first run
    zip_path: str = hf_hub_download(
        repo_id=_REPO_ID,
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    txt_entry = _CONFIG_TO_FILE[config]
    rows: list[dict] = []

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(txt_entry) as fh:
            # The file is latin-1 encoded; format is "sentence@label\r\n"
            content = fh.read().decode("latin-1")

    for line in content.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"text": sentence.strip(), "label": label.strip()})

    return pd.DataFrame(rows)
