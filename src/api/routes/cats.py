"""Cat management endpoints"""
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_cat_service
from src.api.schemas import CatInfo, CreateCatRequest, CreateCatResponse, ErrorResponse
from src.services.cat_service import CatAlreadyExistsError, CatNotFoundError, CatService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/cats", tags=["cats"])


@router.post("", response_model=CreateCatResponse, responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
async def create_cat(
    request: CreateCatRequest,
    cat_service: CatService = Depends(get_cat_service),
):
    """Create a new cat with a brain initialized from the default model"""
    try:
        result = cat_service.create_cat(request.cat_id, request.personality.value)
        return CreateCatResponse(**result)
    except CatAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("cat_creation_error", cat_id=request.cat_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{cat_id}", response_model=CatInfo, responses={404: {"model": ErrorResponse}})
async def get_cat_info(
    cat_id: str,
    cat_service: CatService = Depends(get_cat_service),
):
    """Get information about a specific cat"""
    try:
        info = cat_service.get_cat_info(cat_id)
        return CatInfo(**info)
    except CatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
