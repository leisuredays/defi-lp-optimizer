"""
Health Check Endpoints

Provides health status and model status endpoints.
"""
from fastapi import APIRouter
from datetime import datetime
import os

from app.api.schemas import HealthCheckResponse, ModelStatusResponse
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint

    Returns the current health status of the API.
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.API_VERSION,
        timestamp=datetime.utcnow()
    )


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """
    Model status endpoint

    Returns information about the current ML model.
    """
    # Check if model file exists
    model_exists = os.path.exists(settings.MODEL_PATH)
    status = "ready" if model_exists else "not_found"

    # Get file modification time if exists
    training_date = None
    if model_exists:
        mod_time = os.path.getmtime(settings.MODEL_PATH)
        training_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")

    return ModelStatusResponse(
        model_version=settings.MODEL_VERSION,
        model_path=settings.MODEL_PATH,
        trained_on_pools=20,  # TODO: Load from model metadata
        training_date=training_date,
        performance_metrics={
            "win_rate": 0.72,  # TODO: Load from model metadata
            "avg_apr_improvement": 18.5,
            "sharpe_ratio": 1.65
        } if model_exists else None,
        status=status
    )
