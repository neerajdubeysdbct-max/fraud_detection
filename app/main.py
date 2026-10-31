"""FastAPI service for the fraud detection model.

Run from the project root:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000  (frontend)  or  http://127.0.0.1:8000/docs  (API docs).

Environment variables:
    FRAUD_MODEL_DIR  folder holding xgb_fraud*.json / model_meta*.json   (default: ./models)
    FRAUD_USE_TUNED  "1" (default) = tuned model, "0" = baseline model
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .fraud_predictor import FraudPredictor, InvalidTransactionError
from .schemas import (BatchItem, BatchRequest, BatchResponse, ModelInfo,
                      PredictionOut, Transaction)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraud_api")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("FRAUD_MODEL_DIR", ROOT / "models"))
USE_TUNED = os.getenv("FRAUD_USE_TUNED", "1") == "1"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_lock = threading.Lock()  # the in-memory destination history is shared mutable state


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = FraudPredictor.from_artifacts(MODEL_DIR, tuned=USE_TUNED)
        log.info("ready: %r", app.state.predictor)
    except Exception:
        log.exception("could not load model artifacts from %s", MODEL_DIR)
        app.state.predictor = None
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


def get_predictor(request: Request) -> FraudPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, f"Model not loaded. Check artifacts in {MODEL_DIR}")
    return predictor


@app.exception_handler(InvalidTransactionError)
async def invalid_txn_handler(_: Request, exc: InvalidTransactionError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health(request: Request):
    return {"status": "ok" if request.app.state.predictor else "model_not_loaded"}


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request):
    p = get_predictor(request)
    meta = p._meta
    return ModelInfo(
        model_name=p.model_name, feature_set=p.feature_set, features=p.features,
        threshold=p.threshold, destinations_tracked=len(p.history),
        validation_metrics=meta.get("validation_metrics", {}),
        test_metrics=meta.get("test_metrics", {}),
    )


@app.post("/predict", response_model=PredictionOut)
def predict(txn: Transaction, request: Request, explain: bool = True, record_history: bool = False):
    """Score one transaction. `record_history=true` counts it toward the destination's history."""
    p = get_predictor(request)
    with _lock:
        result = p.predict(txn.model_dump(), record_history=record_history, explain=explain)
    out = result.to_dict()
    out["reasons"] = [{"feature": f, "value": v, "contribution": c} for f, v, c in result.reasons]
    return out


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request):
    p = get_predictor(request)
    frame = pd.DataFrame([t.model_dump() for t in body.transactions])
    with _lock:
        scored = p.predict_batch(frame)
    results = [BatchItem(fraud_probability=float(a), is_fraud=bool(b))
               for a, b in zip(scored["fraud_probability"], scored["is_fraud"])]
    return BatchResponse(count=len(results), flagged=sum(r.is_fraud for r in results), results=results)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
