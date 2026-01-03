from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.services.jump_learning_service import JumpLearningService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/jump", tags=["jump-learning"])

jump_service = JumpLearningService()


class JumpForceRequest(BaseModel):
    cat_id: str
    target_id: str
    height_diff: float
    distance: float


class JumpForceResponse(BaseModel):
    force_multiplier: float
    from_memory: bool


class JumpResultRequest(BaseModel):
    cat_id: str
    target_id: str
    height_diff: float
    distance: float
    force_used: float
    success: bool


class JumpResultResponse(BaseModel):
    new_force_multiplier: float
    message: str


@router.post("/predict", response_model=JumpForceResponse)
async def predict_jump_force(request: JumpForceRequest):
    memories = jump_service.get_all_memories(request.cat_id)
    from_memory = request.target_id in memories
    
    force = jump_service.get_jump_force(
        cat_id=request.cat_id,
        target_id=request.target_id,
        height_diff=request.height_diff,
        distance=request.distance
    )
    
    return JumpForceResponse(
        force_multiplier=force,
        from_memory=from_memory
    )


@router.post("/result", response_model=JumpResultResponse)
async def record_jump_result(request: JumpResultRequest):
    new_force = jump_service.record_jump_result(
        cat_id=request.cat_id,
        target_id=request.target_id,
        height_diff=request.height_diff,
        distance=request.distance,
        force_used=request.force_used,
        success=request.success
    )
    
    message = "learned" if request.success else "adjusting"
    
    return JumpResultResponse(
        new_force_multiplier=new_force,
        message=message
    )


@router.get("/memory/{cat_id}")
async def get_jump_memory(cat_id: str):
    memories = jump_service.get_all_memories(cat_id)
    return {
        "cat_id": cat_id,
        "targets": memories,
        "total_targets": len(memories)
    }


@router.delete("/memory/{cat_id}/{target_id}")
async def reset_target_memory(cat_id: str, target_id: str):
    success = jump_service.reset_target_memory(cat_id, target_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"status": "reset", "target_id": target_id}
