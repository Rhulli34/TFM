# Registro de decisiones

Formato: una línea por decisión.
- YYYY-MM-DD — Qué se decidió — Por qué.
- 2026-06-22 — F0 completada: precios=yfinance, noticias=Finnhub company-news endpoint, corpus=Financial PhraseBank v1.0 (takala/financial_phrasebank en HuggingFace, descargado como zip y parseado directamente porque el loading script HF tiene URI rota). Python 3.14 usado en .venv (3.11 no instalado en la máquina); datasets pinneado a 3.6.0; la dependencia efectiva de datasets para el corpus es solo huggingface_hub (hf_hub_download).
