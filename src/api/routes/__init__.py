
from fastapi import APIRouter

from src.api.routes import cats, models, monitoring, predictions, learning, jump

api_router = APIRouter()

api_router.include_router(predictions.router)
api_router.include_router(cats.router)
api_router.include_router(models.router)
api_router.include_router(monitoring.router)
api_router.include_router(learning.router)
api_router.include_router(jump.router)
