
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_model_loader
from src.api.schemas import ErrorResponse, ModelInfo
from src.inference.model_loader import ModelLoader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[str])
async def list_models(model_loader: ModelLoader = Depends(get_model_loader)):

    return model_loader.list_versions()


@router.get("/{version}", response_model=ModelInfo, responses={404: {"model": ErrorResponse}})
async def get_model_info(
    version: str,
    model_loader: ModelLoader = Depends(get_model_loader),
):

    info = model_loader.get_model_info(version)
    if info.get("trained_at") == "unknown":
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")
    return ModelInfo(**info)
