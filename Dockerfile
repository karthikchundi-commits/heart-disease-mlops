# ---- Heart disease risk API: inference image -------------------------------
# Small runtime: only serving deps (no MLflow / plotting), non-root user,
# model artefact baked in so the image is self-contained and immutable.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/heart_model.joblib \
    PORT=8000

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .

COPY models/heart_model.joblib models/model_meta.json models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"

CMD ["sh", "-c", "uvicorn heart.serving:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
