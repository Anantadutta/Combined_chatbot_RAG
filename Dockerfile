# NexaCorp RAG assistant - web UI + REST API
#   docker build -t nexacorp-rag .
#   docker run -p 8000:8000 --env-file .env nexacorp-rag        -> http://localhost:8000
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data/raw ./data/raw
COPY eval ./eval

# run as a normal user; the index and uploads live in data/ (mount a volume to keep them)
RUN useradd --create-home appuser && mkdir -p data/index data/uploads && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)"

# ONE worker: local Qdrant locks its folder (set QDRANT_URL to a Qdrant server to scale out).
# The server indexes data/raw on first start if the index is empty.
CMD ["sh", "-c", "uvicorn app.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
