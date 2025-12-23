"""Health check endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from src.api.dependencies import get_model_loader, get_predictor, get_settings, get_start_time
from src.api.health import (
    get_health_status,
    get_liveness_status,
    get_readiness_status,
)
from src.api.schemas import HealthCheck
from src.core.config import Settings
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor
from src.utils.metrics import get_metrics

router = APIRouter(tags=["monitoring"])


@router.get("/health", response_model=HealthCheck)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    predictor: BatchPredictor = Depends(get_predictor),
    config: Settings = Depends(get_settings),
    start_time: float = Depends(get_start_time),
):
    """Get service health status"""
    status = get_health_status(model_loader, predictor, config, start_time)
    return HealthCheck(**status)


@router.get("/ready")
async def readiness_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    config: Settings = Depends(get_settings),
):
    """Check if service is ready to accept requests"""
    status = get_readiness_status(model_loader, config)
    if not status["ready"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    return status


@router.get("/live")
async def liveness_check():
    """Check if service is alive"""
    return get_liveness_status()


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Get Prometheus metrics"""
    return get_metrics()
