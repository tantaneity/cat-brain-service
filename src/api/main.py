import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.api.health import (
    get_health_status,
    get_liveness_status,
    get_readiness_status,
)
from src.api.middleware import LoggingMiddleware, RequestIdMiddleware
from src.api.schemas import (
    BatchCatActions,
    BatchCatStates,
    CatAction,
    CatState,
    ErrorResponse,
    HealthCheck,
    ModelInfo,
)
from src.core.config import Settings, settings
from src.core.environment import CatAction as CatActionEnum
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor
from src.utils.logger import get_logger, setup_logger
from src.utils.metrics import MetricsMiddleware, get_metrics

logger = get_logger(__name__)

ACTION_NAMES: dict[int, str] = {
    CatActionEnum.IDLE: "idle",
    CatActionEnum.MOVE_TO_FOOD: "move_to_food",
    CatActionEnum.MOVE_TO_TOY: "move_to_toy",
    CatActionEnum.SLEEP: "sleep",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logger(settings.LOG_LEVEL)
    logger.info("starting_service", version="1.0.0")

    model_loader = ModelLoader(settings)

    try:
        model_loader.load_model(settings.MODEL_VERSION)
    except FileNotFoundError:
        logger.warning("model_not_found", version=settings.MODEL_VERSION)

    predictor = BatchPredictor(model_loader, settings)
    predictor.start()

    app.state.settings = settings
    app.state.model_loader = model_loader
    app.state.predictor = predictor
    app.state.start_time = time.time()

    logger.info("service_started")

    yield

    predictor.stop()
    logger.info("service_stopped")


app = FastAPI(
    title="Cat Brain Service",
    description="ML microservice for cat decision making",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIdMiddleware)


def get_predictor() -> BatchPredictor:
    return app.state.predictor


def get_model_loader() -> ModelLoader:
    return app.state.model_loader


def get_settings() -> Settings:
    return app.state.settings


@app.post("/predict", response_model=CatAction, responses={500: {"model": ErrorResponse}})
async def predict(
    state: CatState,
    predictor: BatchPredictor = Depends(get_predictor),
):
    try:
        obs = np.array(
            [state.hunger, state.energy, state.distance_to_food, state.distance_to_toy],
            dtype=np.float32,
        )
        action = await predictor.predict_single(obs)
        return CatAction(action=action, action_name=ACTION_NAMES.get(action))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchCatActions, responses={500: {"model": ErrorResponse}})
async def predict_batch(
    batch: BatchCatStates,
    predictor: BatchPredictor = Depends(get_predictor),
):
    try:
        observations = [
            np.array(
                [s.hunger, s.energy, s.distance_to_food, s.distance_to_toy],
                dtype=np.float32,
            )
            for s in batch.states
        ]
        actions = await predictor.predict_batch(observations)
        return BatchCatActions(actions=actions)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[str])
async def list_models(model_loader: ModelLoader = Depends(get_model_loader)):
    return model_loader.list_versions()


@app.get("/models/{version}", response_model=ModelInfo, responses={404: {"model": ErrorResponse}})
async def get_model_info(
    version: str,
    model_loader: ModelLoader = Depends(get_model_loader),
):
    info = model_loader.get_model_info(version)
    if info.get("trained_at") == "unknown":
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")
    return ModelInfo(**info)


@app.get("/health", response_model=HealthCheck)
async def health_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    predictor: BatchPredictor = Depends(get_predictor),
    config: Settings = Depends(get_settings),
):
    status = get_health_status(model_loader, predictor, config, app.state.start_time)
    return HealthCheck(**status)


@app.get("/ready")
async def readiness_check(
    model_loader: ModelLoader = Depends(get_model_loader),
    config: Settings = Depends(get_settings),
):
    status = get_readiness_status(model_loader, config)
    if not status["ready"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    return status


@app.get("/live")
async def liveness_check():
    return get_liveness_status()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return get_metrics()
