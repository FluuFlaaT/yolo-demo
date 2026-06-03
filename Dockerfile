# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# YOLO Demo — Lightweight real-time object detection for edge computing
# Multi-stage build:  build (uv sync)  →  runtime (venv + source only)
# ──────────────────────────────────────────────────────────────────────────────

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Stage 1: Build — resolve & install all dependencies with uv              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
FROM python:3.9-slim AS build

# Install uv (fast Python package manager)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv

WORKDIR /build

# ── Layer 1: dependency manifests (cached aggressively) ────────────────────
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ── Layer 2: application source + configs + helper scripts ─────────────────
COPY src/     src/
COPY configs/ configs/
COPY scripts/ scripts/

# Re-sync so the *current* package is installed into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Stage 2: Runtime — minimal production image                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
FROM python:3.9-slim AS runtime

# ── System libraries required by OpenCV at runtime ─────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (uid 1000) ──────────────────────────────────────────────
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

# ── Copy build artifacts ──────────────────────────────────────────────────
COPY --from=build --chown=appuser:appuser /build/.venv        /opt/venv
COPY --from=build --chown=appuser:appuser /build/src          /opt/app/src
COPY --from=build --chown=appuser:appuser /build/configs       /opt/app/configs
COPY --from=build --chown=appuser:appuser /build/scripts       /opt/app/scripts
COPY --from=build --chown=appuser:appuser /build/pyproject.toml /opt/app/

# ── Environment ───────────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    SERVICE_MODE=api \
    API_PORT=8000 \
    WEBUI_PORT=7860 \
    CORS_ORIGINS="*" \
    LOG_LEVEL=INFO

WORKDIR /opt/app

# ── Model weights volume mount point ──────────────────────────────────────
RUN mkdir -p /models && chown appuser:appuser /models
VOLUME ["/models"]

# ── Network ───────────────────────────────────────────────────────────────
EXPOSE 8000 7860

# ── OCI image labels ──────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="yolo-demo" \
      org.opencontainers.image.description="Lightweight real-time object detection for edge computing" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/nerbai/yolo-demo" \
      org.opencontainers.image.authors="nerbai" \
      org.opencontainers.image.licenses="MIT"

# ── Entrypoint — switches on SERVICE_MODE env var ─────────────────────────
COPY --chown=appuser:appuser <<-"ENTRYSCRIPT" /entrypoint.sh
#!/bin/bash
set -e

MODE="${SERVICE_MODE:-api}"

case "$MODE" in
    api)
        echo "[entrypoint] Starting YOLO Demo API on 0.0.0.0:${API_PORT:-8000}"
        exec yolo-demo api --host 0.0.0.0 --port "${API_PORT:-8000}"
        ;;
    webui)
        echo "[entrypoint] Starting YOLO Demo WebUI on 0.0.0.0:${WEBUI_PORT:-7860}"
        exec yolo-demo webui --host 0.0.0.0 --port "${WEBUI_PORT:-7860}"
        ;;
    *)
        echo "[entrypoint] ERROR: unknown SERVICE_MODE='$MODE'. Use 'api' or 'webui'."
        exit 1
        ;;
esac
ENTRYSCRIPT

RUN chmod +x /entrypoint.sh

# ── Security: drop to unprivileged user BEFORE health check / entrypoint ──
USER appuser

# ── Health check (queries the /health endpoint exposed by FastAPI) ────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
