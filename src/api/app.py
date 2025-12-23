
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api.middleware import LoggingMiddleware, RequestIdMiddleware
from src.api.routes import api_router
from src.core.config import settings
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor
from src.services.cat_service import CatService
from src.training.trainer import CatBrainTrainer
from src.utils.action_history import ActionHistory
from src.utils.logger import get_logger, setup_logger
from src.utils.metrics import MetricsMiddleware

logger = get_logger(__name__)


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

    trainer = CatBrainTrainer(settings)
    action_history = ActionHistory()
    cat_service = CatService(trainer, model_loader, action_history)


    app.state.settings = settings
    app.state.model_loader = model_loader
    app.state.predictor = predictor
    app.state.trainer = trainer
    app.state.action_history = action_history
    app.state.cat_service = cat_service
    app.state.start_time = time.time()

    logger.info("service_started")

    yield


    predictor.stop()
    logger.info("service_stopped")


def create_app() -> FastAPI:

    app = FastAPI(
        title="Cat Brain Service",
        description="ML microservice for cat decision making",
        version="1.0.0",
        lifespan=lifespan,
    )


    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIdMiddleware)


    app.include_router(api_router)

    return app
