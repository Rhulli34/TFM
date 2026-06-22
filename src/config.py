"""Central configuration: paths, env vars, and project-wide constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Root of the repo (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

# API keys
NEWS_API_KEY: str | None = os.getenv("NEWS_API_KEY")
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")

# Data directories
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Reports
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# Models
MODELS_DIR = ROOT_DIR / "models"

# Tickers used throughout the project
DEFAULT_TICKERS = ["AAPL", "NVDA", "QCOM"]

# Finnhub base URL
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
