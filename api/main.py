"""
TruthLens FastAPI Application Entrypoint
Provides REST endpoints for fake news detection, explainability, and model benchmarks.
"""

import sys
import logging
import os
import io
import csv
import threading
import time
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    CompareRequest,
    CredibilityResponse,
    HealthResponse,
    MetricsResponse
)
from api.service import PredictionService
from api.analytics import get_analytics, log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TruthLens - Fake News Detection API",
    description="NLP-powered real-time Fake News Detection API with LIME explainability.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration. Set ALLOWED_ORIGINS to a comma-separated
# list in production when the frontend calls the API directly.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

rate_limit_requests = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "60")))
rate_limit_window_seconds = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
request_history = defaultdict(deque)
request_history_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_requests_by_ip(request, call_next):
    """Apply a lightweight in-memory sliding-window limit per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with request_history_lock:
        timestamps = request_history[client_ip]
        while timestamps and now - timestamps[0] >= rate_limit_window_seconds:
            timestamps.popleft()
        if len(timestamps) >= rate_limit_requests:
            retry_after = max(1, int(rate_limit_window_seconds - (now - timestamps[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)
    return await call_next(request)

# Initialize Prediction Service
service = PredictionService()


@app.get("/", tags=["General"])
def root():
    return {
        "project": "TruthLens - Fake News Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict-batch",
            "predict_compare": "POST /predict-compare",
            "metrics": "GET /metrics",
            "analytics": "GET /analytics",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API and model service health status."""
    is_ready = service.is_ready()
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        version="1.0.0",
        pipeline_type="TF-IDF + Calibrated Linear SVM (Champion)"
    )


@app.get("/metrics", tags=["Benchmarks"])
def get_model_metrics():
    """Retrieve benchmark metrics for baseline and transformer models."""
    metrics = service.get_metrics()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark metrics not found. Run model training first."
        )
    return metrics


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_news(request: PredictionRequest):
    """
    Predict whether a news article or headline is Real or Fake,
    providing a confidence score and token-level LIME explainability.
    """
    try:
        result = service.predict_and_explain(
            text=request.text,
            num_explanation_words=request.num_explanation_words
        )
        if request.source_url:
            result["credibility"] = service.check_credibility(request.source_url)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["error"]
            )
        log_prediction(
            request.text,
            result["label"],
            result["confidence_score"],
            "Linear SVM",
            result.get("credibility", {}).get("domain"),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal prediction error occurred."
        )


@app.get("/credibility", response_model=CredibilityResponse, tags=["Source Credibility"])
def check_source_credibility(url: str):
    """Check a URL hostname against the curated source registry."""
    try:
        return service.check_credibility(url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/analytics", tags=["Analytics"])
def analytics():
    """Return aggregate prediction activity and frequently flagged keywords."""
    try:
        return get_analytics()
    except (sqlite3.Error, OSError) as exc:
        logger.exception("Analytics database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics are temporarily unavailable. Please retry shortly.",
        ) from exc


@app.post("/predict-compare", tags=["Prediction"])
def compare_predictions(request: CompareRequest):
    """Compare the same text across the classical and transformer models."""
    try:
        predictions = service.predict_compare(request.text)
        if len(predictions) < 3:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="All comparison model artifacts are not available.",
            )
        for prediction in predictions:
            log_prediction(
                request.text,
                prediction["label"],
                prediction["confidence_score"],
                prediction["model_name"],
            )
        return {"text": request.text, "predictions": predictions}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model comparison failed")
        raise HTTPException(status_code=500, detail="Model comparison failed.")


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Predict every row in a CSV containing a required `text` column."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    try:
        contents = await file.read()
        rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))))
        if not rows or "text" not in rows[0]:
            raise HTTPException(status_code=400, detail="CSV must contain a text column.")
        if len(rows) > 500:
            raise HTTPException(status_code=413, detail="Batch uploads are limited to 500 rows.")

        output = io.StringIO()
        fieldnames = list(rows[0].keys()) + ["label", "confidence_score"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            text = (row.get("text") or "").strip()
            if len(text) < 10 or len(text.split()) < 3:
                row.update({"label": "Invalid", "confidence_score": "0"})
            else:
                prediction = service.predict_basic(text)
                log_prediction(text, prediction["label"], prediction["confidence_score"], "Linear SVM")
                row.update({
                    "label": prediction["label"],
                    "confidence_score": prediction["confidence_score"],
                })
            writer.writerow(row)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=truthlens-predictions.csv"},
        )
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.")
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
