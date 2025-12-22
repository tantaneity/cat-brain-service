import time

from src.core.config import Settings
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor


def get_health_status(
    model_loader: ModelLoader, predictor: BatchPredictor, config: Settings, start_time: float
) -> dict:
    model_loaded = False
    model_version = config.MODEL_VERSION

    try:
        model = model_loader.get_model(model_version)
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    cache_available = predictor.cache.is_available() if predictor.cache else False

    uptime = time.time() - start_time

    status = "healthy" if model_loaded else "degraded"

    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_version": model_version,
        "cache_available": cache_available,
        "uptime": round(uptime, 2),
    }


def get_readiness_status(model_loader: ModelLoader, config: Settings) -> dict:
    try:
        model = model_loader.get_model(config.MODEL_VERSION)
        if model is not None:
            return {"ready": True}
    except Exception:
        pass

    return {"ready": False}


def get_liveness_status() -> dict:
    return {"alive": True}
