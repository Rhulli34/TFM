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
- LLM vía API (NUNCA local) para extracción de eventos y síntesis
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

## Plan por fases (ESTADO ACTUAL: FASE 1)
- F0 Cimientos: entorno, estructura, fuentes conectadas  ✓ COMPLETADA
- F1 Datos + EDA del corpus
- F2 Modelado y comparativa (baseline vs fine-tune vs zero-shot)  <- fase clave
- F3 Ingesta en vivo + histórico
- F4 Eventos + RAG + briefing
- F5 Backtest de alerta temprana
- F6 App + despliegue
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
