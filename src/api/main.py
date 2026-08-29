"""FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   - liveness/readiness probe
    POST /predict  - multipart image upload -> class label + probabilities
    GET  /metrics  - Prometheus exposition format

Logging: every request is logged as a structured JSON line (method, path,
status, latency, and for /predict: filename/content-type/label/probability).
Raw image bytes are never logged.
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pythonjsonlogger import jsonlogger

from src.api.schemas import HealthResponse, PredictionResponse
from src.models.predict import load_model, predict_image_bytes

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))

# ---- structured logging -----------------------------------------------
logger = logging.getLogger("catsdogs-api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logger.handlers = [_handler]
logger.propagate = False

# ---- prometheus metrics -------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "Request latency in seconds", ["path"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["label"]
)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if MODEL_PATH.exists():
        _model = load_model(MODEL_PATH)
        logger.info("model_loaded", extra={"model_path": str(MODEL_PATH)})
    else:
        logger.warning("model_not_found", extra={"model_path": str(MODEL_PATH)})
    yield


app = FastAPI(title="Cats vs Dogs Inference Service", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(latency)
    logger.info(
        "request_handled",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency * 1000, 2),
        },
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    start = time.perf_counter()
    result = predict_image_bytes(_model, image_bytes)
    latency_ms = (time.perf_counter() - start) * 1000

    PREDICTION_COUNT.labels(result["label"]).inc()
    logger.info(
        "prediction_made",
        extra={
            "image_filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_bytes),
            "label": result["label"],
            "probabilities": result["probabilities"],
            "latency_ms": round(latency_ms, 2),
        },
    )

    return PredictionResponse(
        label=result["label"],
        probabilities=result["probabilities"],
        latency_ms=round(latency_ms, 2),
    )
