# Proyecto: Financial News Radar (TFM)

## Qué es
Sistema que monitoriza las noticias financieras de una cartera de empresas
cotizadas, clasifica su sentimiento, detecta eventos relevantes y genera un
informe estructurado por empresa. Funciona como alerta temprana de noticias
negativas. Es el TFM del Máster de Big Data, Data Science e IA (UCM).

## Caso de uso
Una mesa de riesgo/research (banco o fondo) que cubre decenas de empresas y no
puede leerse todo: el sistema lee por ella y flaggea lo material. Es una versión
transparente y open de plataformas tipo RavenPack / AlphaSense.

## Arquitectura (dos capas)
1. NÚCLEO DE MODELADO (la parte rigurosa, da los resultados): clasificador de
   sentimiento financiero entrenado y comparado contra baselines y un LLM zero-shot.
2. CAPA DE PRODUCTO (lo que se demuestra): extracción de eventos + RAG +
   generación de informes, servido como app (API + dashboard).

## Stack
- Python 3.12 (.venv) — compatible con PyTorch; pandas 2.2.x (pandas 3.x bloqueado por WDAC en esta máquina)
- PyTorch, HuggingFace Transformers, scikit-learn, pandas, numpy
- sentence-transformers + FAISS para RAG
- LLM vía API (NUNCA local) para extracción de eventos y síntesis — OpenAI gpt-4o-mini (OPENAI_API_KEY en .env)
- FastAPI + Streamlit + Docker
- Datos: Financial PhraseBank v1.0 (corpus, descargado vía hf_hub_download como zip) + noticias vía Finnhub company-news API + precios vía yfinance + filings EDGAR
- datasets pinneado a 3.6.0 (v5 rompió el loading script de financial_phrasebank)
- Almacenamiento: SQLite/DuckDB; FAISS para vectores

## Estructura
- src/data/         ingesta y preproc (corpus, noticias, precios)
- src/models/       baseline, transformer, llm_zeroshot, evaluate
- src/intelligence/ events, retrieval, briefing
- src/backtest/     early_warning
- src/app/          api, dashboard
- notebooks/        EDA y experimentos
- reports/          figuras y resultados (alimentan la memoria y el vídeo)
- docs/             decisiones.md (log cronológico) + la memoria (al FINAL)

## Plan por fases (ESTADO ACTUAL: FASE 6)
- F0 Cimientos: entorno, estructura, fuentes conectadas  ✓ COMPLETADA
- F1 Datos + EDA del corpus  ✓ COMPLETADA
- F2 Modelado y comparativa (baseline vs fine-tune vs zero-shot)  ✓ COMPLETADA
  - Baseline (TF-IDF + LinearSVC): macro-F1=0.856 en test
  - Transformer fine-tune (deberta-v3-base): macro-F1=0.981 en test
  - LLM zero-shot (gpt-4o-mini): macro-F1=0.963 en test, $0.011/339 frases
- F3 Ingesta en vivo + histórico  ✓ COMPLETADA
  - 10 tickers (AAPL,NVDA,MSFT,AMZN,META,GOOGL,TSLA,JPM,XOM,PFE)
  - 27.266 artículos únicos + precios en data/processed/radar.db (SQLite)
  - Sentimiento clasificado con DeBERTa-v3-base (F2) en batch, GPU
  - Nota: GS y QCOM también en la BD (ingesta previa), no se borran
- F4 Eventos + RAG + briefing  ✓ COMPLETADA
  - events.py: extracción con gpt-4o-mini, batch=10, 8 event types → tabla events en SQLite
  - retrieval.py: all-MiniLM-L6-v2 + FAISS IndexFlatIP (cosine similarity)
  - briefing.py: pipeline completo → markdown guardado en reports/briefings/
  - scripts/run_briefing.py: CLI — python scripts/run_briefing.py TICKER [--days N]
- F5 Backtest de alerta temprana  ✓ COMPLETADA
  - signal.py: señal diaria neta (raw + weighted + MA3/MA5), solo is_relevant=1
  - returns.py: retornos forward log a t+1, t+3, t+5 desde precios
  - notebooks/04_backtest.ipynb: lead-lag, event study, caso de estudio TSLA+NVDA
  - Hallazgo clave: señal REACTIVA, no predictiva (correlación máx. en lag=-1)
  - Valor del sistema: cualitativo (flagging de noticias negativas), no cuantitativo
  - Resultados en reports/results/backtest.json, figuras en reports/figures/
- F6 App + despliegue  ← EN CURSO
  - F6.1 Backend FastAPI ✓: settings.py (dev/prod), api.py (5 endpoints)
    - GET /health, /catalog, /portfolio?tickers=, /company/{ticker}, /company/{ticker}/briefing
    - Catálogo dinámico: cualquier ticker con is_relevant=1 en BD aparece automáticamente
    - Alertas: net_sentiment_7d < -0.3 OR evento legal material (is_relevant=1)
    - uvicorn src.app.api:app --reload
  - F6.2 Frontend a medida ✓: src/app/frontend/ (HTML+CSS+JS vanilla), servido desde FastAPI
    - index.html + styles.css (design system Stripe/Linear: Inter, blanco, acento #635BFF)
    - app.js: fetch a API relativa, estado en JS puro, sin framework
    - Vista de cartera: stat cards, segmented control 7/14/30d, chips de filtro, grid 5-col,
      tarjetas con gauge centrado en 0, borde izq. rojo en alertas, last-updated en header
    - Selección vacía → estado vacío explícito (no repobla); tooltip dark en badge de alerta
    - /portfolio acepta ?days=N (1-30); etiquetas dinámicas en tarjetas y stats
    - Escala de color: dead zone ±0.03 (antes ±0.1); -0.07 muestra rojo, no gris
    - Drill-down: 4 tabs (Sentimiento/Noticias/Eventos/Informe), Chart.js (CDN)
    - Pestaña Eventos condicional: se oculta si la empresa no tiene eventos; visible si los tiene
    - api.py sirve GET / → index.html + mount /static → frontend/
    - Arranque único: uvicorn src.app.api:app --reload (puerto 8000, todo junto)
  - F6.2 ✓ COMPLETADA
  - F6.3 Docker Compose ✓ COMPLETADA
    - Dockerfile: python:3.12-slim, torch CPU (inferencia, no GPU en prod)
    - .dockerignore: excluye data/ excepto data/processed/radar.db (baked-in), reports/, notebooks/, models/ completo
    - docker-compose.yml: monta radar.db:ro (override dev) + reports/briefings, env_file .env, healthcheck
    - requirements-prod.txt: sin torch (se instala aparte con CPU wheel), sin streamlit/jupyter/datasets
    - settings.py: docstring actualizado (sin referencia a Streamlit, CORS abierto en ambos modos)
    - README.md: sección Docker con build/run y tabla de variables de entorno
  - F6.4 Snapshot autocontenido + HF Spaces ✓ COMPLETADA
    - radar.db (19 MB) baked-in; usuario no-root UID 1000; puerto 7860 (HF Spaces)
    - WORKDIR /home/user/app; CMD: uvicorn 0.0.0.0:${PORT:-7860}
    - docker-compose.yml: porta ${PORT:-7860}, volúmenes apuntan a /home/user/app/…
    - README.md: frontmatter YAML HF Spaces (title, emoji, sdk: docker, app_port: 7860)
    - api.py: briefing captura EnvironmentError → HTTP 503 (arranca sin OPENAI_API_KEY)
    - Frontend: subtitle "Monitorización de sentimiento…"; header "Demo · datos hasta <fecha>"
  - F6.5 Modelo en HF Hub ✓ COMPLETADA
    - Modelo publicado en https://huggingface.co/Rhulli/financial-news-radar-deberta (público)
    - Solo 4 ficheros de inferencia: config.json, model.safetensors, tokenizer.json, tokenizer_config.json
    - sentiment.py: MODEL_ID="Rhulli/financial-news-radar-deberta"; carga local si existe checkpoint, Hub si no
    - Dockerfile: eliminado COPY del modelo; RUN descarga y cachea en ~/.cache/huggingface en build time
    - Estrategia build-time (no runtime): evita descarga de 750 MB en cada cold-start (Render free duerme)
    - Tamaño imagen: ~2.9 GB (torch CPU 920 MB + deps 648 MB + DeBERTa 754 MB baked vía Hub + radar.db 19 MB)
- F7 Vídeo beca
- F8 Documentación
Trabaja UNA fase cada vez. No saltes de fase sin cerrar el hito de la actual.

## Convenciones
- Código en inglés, docstrings claros (la guía valora la legibilidad del código).
- Funciones modulares, POO donde aporte. Nada de notebooks con listados de datos enormes.
- Reproducible: fija semillas, guarda métricas en reports/results y figuras en reports/figures.
- Cada gráfica o métrica relevante -> GUÁRDALA (la necesito para memoria y vídeo).

## Reglas importantes
- NUNCA commitear datos (data/), modelos (models/) ni secretos. Claves en .env (gitignored); plantilla en .env.example.
- LLM siempre por API, nunca cargar un LLM grande en local (GPU 8GB).
- El fine-tuning del transformer SÍ va en local (modelo pequeño, cabe en 8GB).
- La memoria NO se escribe sobre la marcha. Al cerrar cada fase, apunta una línea en docs/decisiones.md.

## Mantenimiento de este fichero (IMPORTANTE)
Este CLAUDE.md describe el ESTADO ACTUAL del proyecto. Mantenlo al día:
- Antes de terminar tu turno, si has cambiado algo que afecte a la arquitectura,
  la estructura de carpetas, el stack, las convenciones o la fase actual, o has
  tomado una decisión técnica relevante -> ACTUALIZA la sección correspondiente
  de este fichero (edición in-place, sin acumular texto).
- Mantén el fichero conciso (objetivo < 150 líneas). Si una sección crece de más,
  resúmela o muévela a un rule/skill.
- El registro cronológico NO va aquí: añade una línea en docs/decisiones.md
  (fecha + qué se decidió + por qué). Aquí solo queda el estado vigente.
- Cuando actualices este fichero, dilo en tu resumen final.
