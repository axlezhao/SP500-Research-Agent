# syntax=docker/dockerfile:1

# --- 1. Build the React dashboard -------------------------------------------------------------
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- 2. Python app: API, research pipeline, built dashboard --------------------------------------
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SP500_HOME=/app \
    DATA_SOURCE=live \
    PORT=8000
WORKDIR /app

# Install with the versions pinned in requirements.txt (as constraints), without dev/Kaggle/Streamlit extras.
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN grep -v '^-e' requirements.txt | sed 's/\[[^]]*\]//' > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt ".[web,llm]"

COPY run_pipeline.py ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
COPY --from=web /web/dist ./web/dist
RUN chmod +x /usr/local/bin/entrypoint \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data models reports \
    && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health')"
ENTRYPOINT ["entrypoint"]
