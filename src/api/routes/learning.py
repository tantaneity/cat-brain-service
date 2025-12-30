from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List

from src.api.schemas import ObservationSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["learning"])


class ExperienceSchema(BaseModel):
    state: ObservationSchema
    action: int
    reward: float
    next_state: ObservationSchema
    done: bool


class SubmitExperienceRequest(BaseModel):
    cat_id: str
    state: ObservationSchema
    action: int
    reward: float
    next_state: ObservationSchema
    done: bool


class SubmitExperienceBatchRequest(BaseModel):
    cat_id: str
    experiences: List[ExperienceSchema]


@router.post("/experience")
async def submit_experience(request: Request, data: SubmitExperienceRequest):
    logger.info("submit_experience", cat_id=data.cat_id, reward=data.reward)
    
    experience_buffer = getattr(request.app.state, 'experience_buffer', {})
    
    if data.cat_id not in experience_buffer:
        experience_buffer[data.cat_id] = []
    
    experience_buffer[data.cat_id].append({
        'state': [
            data.state.hunger,
            data.state.energy,
            data.state.distance_to_food,
            data.state.distance_to_toy,
            data.state.mood,
            data.state.lazy_score,
            data.state.foodie_score,
            data.state.playful_score,
        ],
        'action': data.action,
        'reward': data.reward,
        'next_state': [
            data.next_state.hunger,
            data.next_state.energy,
            data.next_state.distance_to_food,
            data.next_state.distance_to_toy,
            data.next_state.mood,
            data.next_state.lazy_score,
            data.next_state.foodie_score,
            data.next_state.playful_score,
        ],
        'done': data.done,
    })
    
    request.app.state.experience_buffer = experience_buffer
    
    return {"status": "ok"}


@router.post("/experience/batch")
async def submit_experience_batch(request: Request, data: SubmitExperienceBatchRequest):
    logger.info("submit_experience_batch", cat_id=data.cat_id, count=len(data.experiences))
    
    experience_buffer = getattr(request.app.state, 'experience_buffer', {})
    
    if data.cat_id not in experience_buffer:
        experience_buffer[data.cat_id] = []
    
    for exp in data.experiences:
        experience_buffer[data.cat_id].append({
            'state': [
                exp.state.hunger,
                exp.state.energy,
                exp.state.distance_to_food,
                exp.state.distance_to_toy,
                exp.state.mood,
                exp.state.lazy_score,
                exp.state.foodie_score,
                exp.state.playful_score,
            ],
            'action': exp.action,
            'reward': exp.reward,
            'next_state': [
                exp.next_state.hunger,
                exp.next_state.energy,
                exp.next_state.distance_to_food,
                exp.next_state.distance_to_toy,
                exp.next_state.mood,
                exp.next_state.lazy_score,
                exp.next_state.foodie_score,
                exp.next_state.playful_score,
            ],
            'done': exp.done,
        })
    
    request.app.state.experience_buffer = experience_buffer
    
    if len(experience_buffer[data.cat_id]) >= 100:
        logger.info("experience_buffer_full", cat_id=data.cat_id, size=len(experience_buffer[data.cat_id]))
    
    return {"status": "ok", "total_experiences": len(experience_buffer[data.cat_id])}
