
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_contextual_engine, get_predictor
from src.api.observation_builder import build_observation
from src.api.schemas import (
    BatchCatActions,
    BatchCatStates,
    CatAction,
    CatState,
    ErrorResponse,
)
from src.core.environment import CatAction as CatActionEnum
from src.inference.predictor import BatchPredictor
from src.services.contextual_engine import ContextualBehaviorEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])

ACTION_NAMES: dict[int, str] = {
    CatActionEnum.IDLE: "idle",
    CatActionEnum.MOVE_TO_FOOD: "move_to_food",
    CatActionEnum.MOVE_TO_TOY: "move_to_toy",
    CatActionEnum.SLEEP: "sleep",
    CatActionEnum.GROOM: "groom",
    CatActionEnum.PLAY: "play",
    CatActionEnum.EXPLORE: "explore",
    CatActionEnum.MEOW_AT_BOWL: "meow_at_bowl",
}


@router.post("", response_model=CatAction, responses={500: {"model": ErrorResponse}})
async def predict(
    state: CatState,
    predictor: BatchPredictor = Depends(get_predictor),
    contextual_engine: ContextualBehaviorEngine = Depends(get_contextual_engine),
):
    try:
        obs = build_observation(state)
        
        base_action = await predictor.predict_single(
            obs,
            cat_id=state.cat_id,
            personality=state.personality.value,
        )
        
        result = contextual_engine.process_action(
            base_action=base_action,
            state=state,
            cat_id=state.cat_id,
        )
        
        return CatAction(
            action=result["action"],
            action_name=ACTION_NAMES.get(result["action"]),
            emotion=result["emotional_state"].primary_emotion.value,
            emotion_intensity=result["emotional_state"].intensity.value,
            mood_change=result["mood_delta"],
            arousal_level=result["emotional_state"].arousal_level,
            animation_hint=result["animation_hint"],
            sound_hint=result["sound_hint"],
            reaction_triggered=result["reaction_triggered"],
            emotion_axes=result.get("emotion_axes"),
            visual_layers=result.get("visual_layers"),
            visual_primary=result.get("visual_primary"),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    except Exception as e:
        logger.error("prediction_error", error=str(e), cat_id=state.cat_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("_batch", response_model=BatchCatActions, responses={500: {"model": ErrorResponse}})
async def predict_batch(
    batch: BatchCatStates,
    predictor: BatchPredictor = Depends(get_predictor),
):
    try:
        payload = [
            (build_observation(state), state.cat_id, state.personality.value)
            for state in batch.states
        ]
        actions = await predictor.predict_batch(payload)
        return BatchCatActions(actions=actions)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    except Exception as e:
        logger.error("batch_prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
