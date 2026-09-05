# ---- builder: install deps into a clean layer ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- runtime: slim image, non-root user ----
FROM python:3.11-slim AS runtime

RUN groupadd -r committee && useradd -r -g committee committee

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY observability/ ./observability/

RUN mkdir -p /app/traces /app/mlruns \
    && chown -R committee:committee /app

USER committee

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRACE_JSON_DIR=/app/traces \
    PORT=8000

EXPOSE 8000

# Reads $PORT so this same image runs unmodified under docker-compose (fixed port),
# Kubernetes (fixed containerPort), or Cloud Run (injects PORT at runtime).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=3).status==200 else 1)"

CMD ["sh", "-c", "uvicorn committee.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]