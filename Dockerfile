# Financial News Radar — production image
#
# HF Spaces compatible: port 7860, non-root UID 1000, self-contained DB snapshot.
# For local dev, docker-compose.yml overrides the snapshot with the fresh host DB.
#
# Build:   docker build -t fnr-api .
# Run:     docker run --env-file .env -p 7860:7860 fnr-api
# Compose: docker compose up --build

FROM python:3.12-slim

# ── Non-root user (UID 1000, required by HF Spaces) ───────────────────────────
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# libgomp1 is required by faiss-cpu for OpenMP parallelism
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: CPU-only PyTorch ──────────────────────────────────────────────────
# Installed separately so pip resolves the CPU wheel, not the CUDA one from PyPI.
# Fine-tuning is complete; production inference on CPU is fast enough (<1 s per
# batch of headlines).
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: remaining runtime deps ───────────────────────────────────────────
COPY --chown=user:user requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# ── Layer 3: application source ────────────────────────────────────────────────
COPY --chown=user:user src/ ./src/

# ── Layer 4: fine-tuned model (inference files only) ──────────────────────────
# Training artifacts (optimizer.pt, scheduler.pt, rng_state.pth, training_args.bin)
# are excluded via .dockerignore — they add ~600 MB and are never read at runtime.
COPY --chown=user:user models/deberta-v3-base-finetuned/ ./models/deberta-v3-base-finetuned/

# ── Layer 5: SQLite DB snapshot ────────────────────────────────────────────────
# Baked into the image for self-contained deployment.
# docker-compose.yml mounts the host DB on top to override the snapshot in dev.
COPY --chown=user:user data/processed/radar.db ./data/processed/radar.db

# reports/briefings/ receives generated markdown; mount it to persist across restarts.
RUN mkdir -p reports/briefings && chown user:user reports/briefings

# ── Environment ────────────────────────────────────────────────────────────────
ENV APP_ENV=prod \
    HOME=/home/user

USER user

# HF Spaces requires port 7860; override with -e PORT=8000 for other platforms.
EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn src.app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
