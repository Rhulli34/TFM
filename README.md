# Financial News Radar

Sistema de monitorización inteligente de noticias financieras para la alerta
temprana de eventos sobre una cartera de valores.
TFM — Máster en Big Data, Data Science e IA (UCM).

## Qué hace
Lee las noticias de una cartera de empresas cotizadas, clasifica su sentimiento,
detecta eventos relevantes y avisa de lo material/negativo.

## Arquitectura
- Núcleo de modelado: clasificador de sentimiento (baseline vs fine-tune vs LLM zero-shot).
- Capa de producto: extracción de eventos + RAG + generación de informes, como app.

## Stack
Python · PyTorch/Transformers · scikit-learn · FAISS · FastAPI · Streamlit · Docker

## Instalación
- pip install -r requirements.txt
- copia .env.example a .env y rellena tus claves

## Estado
En desarrollo (ver fases en CLAUDE.md).

## Autor
Raúl Moreno Mejías
