
import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_predictor
from src.api.schemas import (
    BatchCatActions,
    BatchCatStates,
    CatAction,
    CatState,
    ErrorResponse,
)
from src.core.environment import CatAction as CatActionEnum
from src.inference.predictor import BatchPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])

ACTION_NAMES: dict[int, str] = {
    CatActionEnum.IDLE: "idle",
    CatActionEnum.MOVE_TO_FOOD: "move_to_food",
    CatActionEnum.MOVE_TO_TOY: "move_to_toy",
    CatActionEnum.SLEEP: "sleep",
}


@router.post("", response_model=CatAction, responses={500: {"model": ErrorResponse}})
async def predict(
    state: CatState,
    predictor: BatchPredictor = Depends(get_predictor),
):
    try:
        obs = np.array(
            [
                state.hunger,
                state.energy,
                state.distance_to_food,
                state.distance_to_toy,
                state.mood,
                state.lazy_score,
                state.foodie_score,
                state.playful_score,
            ],
            dtype=np.float32,
        )
        action = await predictor.predict_single(
            obs,
            cat_id=state.cat_id,
            personality=state.personality.value,
        )
        return CatAction(action=action, action_name=ACTION_NAMES.get(action))
    except FileNotFoundError as e:
        detail = f"Model not found for cat '{state.cat_id}'" if state.cat_id else "Default model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    except Exception as e:
        logger.error("prediction_error", error=str(e), cat_id=state.cat_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("_batch", response_model=BatchCatActions, responses={500: {"model": ErrorResponse}})
async def predict_batch(
    batch: BatchCatStates,
    predictor: BatchPredictor = Depends(get_predictor),
):
    try:
        observations = [
            np.array(
                [
                    s.hunger,
                    s.energy,
                    s.distance_to_food,
                    s.distance_to_toy,
                    s.mood,
                    s.lazy_score,
                    s.foodie_score,
                    s.playful_score,
                ],
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
