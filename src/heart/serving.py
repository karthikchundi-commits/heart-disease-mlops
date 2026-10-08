"""FastAPI model-serving API.

Endpoints
  POST /predict        one patient  -> prediction + confidence
  POST /predict/batch  list of patients
  GET  /health         liveness  (process up)
  GET  /ready          readiness (model loaded)
  GET  /model-info     model metadata + training metrics
  GET  /metrics        Prometheus exposition format

Run locally:  uvicorn heart.serving:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heart import config

MODEL_PATH = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))
META_PATH = Path(os.getenv("MODEL_META_PATH", MODEL_PATH.with_name("model_meta.json")))


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """One JSON object per line - easy to grep, ship to Loki/ELK, or `kubectl logs`."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("heart.api")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


log = _setup_logging()

# ---------------------------------------------------------------- metrics
REQUESTS = Counter("heart_api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_latency_seconds", "Request latency", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions served", ["outcome"])
PROBABILITY = Histogram("heart_api_disease_probability", "Predicted P(disease)",
                        buckets=[i / 10 for i in range(1, 11)])
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if a model is loaded")


# ---------------------------------------------------------------- schemas
class Patient(BaseModel):
    """Clinical inputs. Fields not always measured (ca, thal, slope, ...) are optional:
    the model pipeline imputes them exactly as it did during training."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 289, "fbs": 0, "restecg": 2,
        "thalach": 135, "exang": 1, "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float | None = Field(None, ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float | None = Field(None, ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] | None = Field(None, description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] | None = Field(None, description="Resting ECG result")
    thalach: float | None = Field(None, ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] | None = Field(None, description="Exercise-induced angina")
    oldpeak: float | None = Field(None, ge=-5, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] | None = Field(None, description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed defect, 7 reversible defect")


class Prediction(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_band: Literal["low", "moderate", "high"]
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# ---------------------------------------------------------------- app
STATE: dict = {"model": None, "meta": {}}


def load_model() -> None:
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["meta"] = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    MODEL_LOADED.set(1)
    log.info("model loaded", extra={"extra_fields": {
        "path": str(MODEL_PATH), "model": STATE["meta"].get("model_name")}})


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        MODEL_LOADED.set(0)
        log.error("model file not found", extra={"extra_fields": {"path": str(MODEL_PATH)}})
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan,
              description="Predicts presence of heart disease from UCI clinical features.")


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        if path != "/metrics":
            log.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _risk_band(p: float) -> str:
    return "low" if p < 0.35 else "moderate" if p < 0.65 else "high"


def _predict(patients: list[Patient]) -> list[Prediction]:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = pd.DataFrame([p.model_dump() for p in patients])[config.FEATURES].astype(float)
    proba = model.predict_proba(frame)[:, 1]
    threshold = STATE["meta"].get("threshold", 0.5)
    out = []
    for p in proba:
        pred = int(p >= threshold)
        PREDICTIONS.labels("disease" if pred else "no_disease").inc()
        PROBABILITY.observe(p)
        out.append(Prediction(
            prediction=pred, label="heart disease" if pred else "no heart disease",
            confidence=round(float(p if pred else 1 - p), 4), probability_disease=round(float(p), 4),
            risk_band=_risk_band(p), model_name=STATE["meta"].get("model_name", "unknown"),
            model_version=STATE["meta"].get("model_version", "unknown")))
    return out


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    result = _predict([patient])[0]
    log.info("prediction", extra={"extra_fields": {
        "prediction": result.prediction, "probability": result.probability_disease}})
    return result


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=_predict(req.patients))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model": STATE["meta"].get("model_name")}


@app.get("/model-info")
def model_info() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return STATE["meta"]


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
