# syntax=docker/dockerfile:1.9

# ---- build: resolve and install dependencies into a virtualenv ------------------------------------
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
WORKDIR /app

# Dependencies first (cached until pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the project itself, non-editable so the runtime stage needs no source tree.
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


# ---- runtime: minimal image, non-root, no build tools -----------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="opencv-face-mask-detection" \
      org.opencontainers.image.description="Production-grade face mask detection service (YuNet + ONNX Runtime + FastAPI)" \
      org.opencontainers.image.source="https://github.com/chan4kum/opencv-face-mask-detection" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

# pip/setuptools/wheel are not needed at runtime (dependencies are pre-installed in the venv): remove them
# to shrink the attack surface and the vulnerability-scan surface.
RUN python -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true \
 && groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    MD_MODEL_PATH=/app/models/face_detection_yunet_2026may.onnx \
    MD_MASK_MODEL_PATH=/app/models/mask_classifier.onnx \
    MD_ENVIRONMENT=prod

WORKDIR /app
COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --chown=root:root models /app/models

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]

# One process per container; scale horizontally with replicas. SIGTERM triggers graceful drain.
CMD ["uvicorn", "mask_detection.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--no-server-header", "--no-access-log", "--timeout-graceful-shutdown", "25", "--proxy-headers"]
