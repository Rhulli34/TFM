# Registro de decisiones

Formato: una línea por decisión.
- YYYY-MM-DD — Qué se decidió — Por qué.
- 2026-06-22 — F0 completada: precios=yfinance, noticias=Finnhub company-news endpoint, corpus=Financial PhraseBank v1.0 (takala/financial_phrasebank en HuggingFace, descargado como zip y parseado directamente porque el loading script HF tiene URI rota). Python 3.12 en .venv (pandas 2.2.x porque pandas 3.x tiene DLLs bloqueadas por WDAC en esta máquina); datasets pinneado a 3.6.0.
