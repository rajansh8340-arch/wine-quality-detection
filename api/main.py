import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.schema import (
    WineFeatures, WineBatchInput, PredictionResult,
    BatchPredictionResponse, ModelMetadataResponse
)
from src.model import WineQualityModelPipeline
from src.explainability import get_feature_drivers
from src.config import MODEL_PATH, METRICS_PATH, FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Wine Quality AI - Production Inference Service",
    description="High-performance machine learning inference API for red wine sensory classification.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model state
model_pipeline: WineQualityModelPipeline = None
metrics_data: Dict[str, Any] = {}

@app.on_event("startup")
def load_model_on_startup():
    global model_pipeline, metrics_data
    try:
        model_pipeline = WineQualityModelPipeline.load(MODEL_PATH)
        logger.info(f"Loaded trained model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_pipeline = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "Wine Quality AI Inference Microservice",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health", tags=["Monitoring"])
def health_check():
    if model_pipeline is None or model_pipeline.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or ready for inference"
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_expected": len(FEATURE_NAMES)
    }

@app.get("/metrics", response_model=ModelMetadataResponse, tags=["Monitoring"])
def get_model_metrics():
    if not metrics_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics data not found"
        )
    return ModelMetadataResponse(
        model_name="RandomForestClassifier (Balanced)",
        version="1.0.0",
        accuracy=metrics_data.get("accuracy", 0.0),
        precision=metrics_data.get("precision", 0.0),
        recall=metrics_data.get("recall", 0.0),
        f1_score=metrics_data.get("f1_score", 0.0),
        roc_auc=metrics_data.get("roc_auc", 0.0),
        training_samples=metrics_data.get("training_samples", 0)
    )

@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict(wine: WineFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    feature_dict = wine.to_feature_dict()
    pred_class, prob_bad, prob_good = model_pipeline.predict_one(feature_dict)
    
    is_good = bool(pred_class == 1)
    confidence = round(prob_good * 100 if is_good else prob_bad * 100, 2)
    label = "Good Quality Wine" if is_good else "Bad Quality Wine"
    
    # Feature drivers for explainability
    drivers = get_feature_drivers(feature_dict, model_pipeline.model, FEATURE_NAMES)

    return PredictionResult(
        is_good_quality=is_good,
        label=label,
        confidence=confidence,
        probability_good=round(prob_good, 4),
        probability_bad=round(prob_bad, 4),
        key_drivers=drivers
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: WineBatchInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    results: List[PredictionResult] = []
    good_count = 0
    bad_count = 0

    for sample in batch.samples:
        feat_dict = sample.to_feature_dict()
        pred_class, prob_bad, prob_good = model_pipeline.predict_one(feat_dict)
        is_good = bool(pred_class == 1)
        if is_good:
            good_count += 1
        else:
            bad_count += 1

        confidence = round(prob_good * 100 if is_good else prob_bad * 100, 2)
        results.append(
            PredictionResult(
                is_good_quality=is_good,
                label="Good Quality Wine" if is_good else "Bad Quality Wine",
                confidence=confidence,
                probability_good=round(prob_good, 4),
                probability_bad=round(prob_bad, 4)
            )
        )

    return BatchPredictionResponse(
        total_samples=len(batch.samples),
        good_quality_count=good_count,
        bad_quality_count=bad_count,
        predictions=results
    )
