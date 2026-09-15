# Financial News Radar

Sistema de monitorización inteligente de noticias financieras para la alerta
temprana de eventos sobre una cartera de valores.
TFM — Máster en Big Data, Data Science e IA (UCM).

**Demo desplegada**: [financial-news-radar.onrender.com](https://financial-news-radar.onrender.com)
(datos hasta la fecha del snapshot; el plan gratuito duerme el contenedor tras un
rato de inactividad, así que la primera carga puede tardar).

## Qué hace
Lee las noticias de una cartera de empresas cotizadas, clasifica su sentimiento,
detecta eventos relevantes y avisa de lo material/negativo.

## Arquitectura
- Núcleo de modelado: clasificador de sentimiento (baseline vs fine-tune vs LLM zero-shot).
- Capa de producto: extracción de eventos + RAG + generación de informes, como app.

## Stack
Python · PyTorch/Transformers · scikit-learn · FAISS · FastAPI · Docker

## Instalación (desarrollo)

```bash
pip install -r requirements.txt   # incluye torch+CUDA para GPU local
cp .env.example .env              # rellena OPENAI_API_KEY
uvicorn src.app.api:app --reload  # API + frontend en http://localhost:8000
```

## Docker (producción)

### Variables de entorno requeridas

| Variable        | Descripción                         | Requerida |
|-----------------|-------------------------------------|-----------|
| `OPENAI_API_KEY`| Clave OpenAI (gpt-4o-mini) para extraer eventos y pre-generar briefings | Solo pipeline offline |
| `APP_ENV`       | `prod` (lo fija docker-compose)     | Auto      |
| `PORT`          | Puerto de escucha (default: `7860`; Render lo inyecta) | No |

`docker compose` lee `.env` automáticamente. La app **desplegada no llama a
OpenAI**: los briefings se pre-generan en local y viajan en la tabla `briefings`
de `radar.db`, así que el contenedor arranca y sirve el dashboard sin
`OPENAI_API_KEY`. La clave solo hace falta para ejecutar el pipeline de datos
(`scripts/pregenerate_briefings.py`, extracción de eventos).

### Construir y arrancar

```bash
# Opción A — Docker Compose (recomendado para dev local)
docker compose up --build          # primera vez
docker compose up -d               # arranques sucesivos (imagen cacheada)

# Opción B — Docker directo (imagen autocontenida, sin volúmenes)
docker build -t fnr-api .
docker run --env-file .env -p 7860:7860 fnr-api

# Opción B con BD más reciente (sobreescribe el snapshot baked-in):
docker run --env-file .env -p 7860:7860 \
  -v $(pwd)/data/processed/radar.db:/home/user/app/data/processed/radar.db:ro \
  fnr-api
```

La app queda disponible en `http://localhost:7860` (o en el puerto que fijes con
`PORT`). En desarrollo sin Docker, `uvicorn` la sirve en `http://localhost:8000`.

### Despliegue

La plataforma de despliegue es **Render** (plan gratuito). Hugging Face Spaces se
descartó al pasar a requerir pago para contenedores Docker; no confundir con
**Hugging Face Hub**, que sí se usa y es donde se aloja el modelo.

### Notas de diseño
- **Imagen autocontenida**: `data/processed/radar.db` se copia dentro de la imagen
  en el build (snapshot histórico), por lo que el contenedor arranca con datos sin
  necesidad de volúmenes. Imprescindible en Render, cuyo plan gratuito no conserva
  ficheros entre reinicios.
- **Modelo desde HF Hub**: el modelo DeBERTa fine-tuneado está publicado en
  [`Rhulli/financial-news-radar-deberta`](https://huggingface.co/Rhulli/financial-news-radar-deberta)
  (público). El `Dockerfile` lo descarga y cachea en `~/.cache/huggingface` durante el
  `docker build`, por lo que el contenedor arranca sin necesitar acceso a red ni token.
- **Dev override**: `docker-compose.yml` monta la BD local encima del snapshot para
  reflejar la última ingesta sin reconstruir la imagen. En dev local, `sentiment.py`
  carga el checkpoint desde `models/deberta-v3-base-finetuned/checkpoint-198/`.
- **torch CPU en el contenedor**: el fine-tuning ya está hecho; en producción solo
  se clasifica texto (unos pocos titulares por petición), lo que la CPU maneja en
  menos de un segundo. La imagen CPU es ~1.5 GB más pequeña que la CUDA equivalente.
- **Briefings pre-generados**: los informes que sirve la app se guardan en la tabla
  `briefings` de `radar.db` y viajan en el snapshot, porque generarlos en vivo agota
  los recursos del plan gratuito (da 502). `reports/briefings/` se monta como volumen
  solo para conservar el markdown que producen las ejecuciones locales.

## Autor
Raúl Moreno Mejías
