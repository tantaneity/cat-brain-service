
from fastapi import Request

from src.core.config import Settings
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor
from src.services.cat_service import CatService
from src.training.trainer import CatBrainTrainer
from src.utils.action_history import ActionHistory


def get_predictor(request: Request) -> BatchPredictor:
    return request.app.state.predictor


def get_model_loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_trainer(request: Request) -> CatBrainTrainer:
    return request.app.state.trainer


def get_action_history(request: Request) -> ActionHistory:
    return request.app.state.action_history


def get_cat_service(request: Request) -> CatService:
    return request.app.state.cat_service


def get_start_time(request: Request) -> float:
    return request.app.state.start_time
