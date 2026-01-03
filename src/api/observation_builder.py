
import numpy as np

from src.api.schemas import CatState
from src.core.environment import ObservationIndex


def build_observation(state: CatState) -> np.ndarray:
    obs = np.zeros(len(ObservationIndex), dtype=np.float32)
    
    obs[ObservationIndex.HUNGER] = state.hunger
    obs[ObservationIndex.ENERGY] = state.energy
    obs[ObservationIndex.DISTANCE_FOOD] = state.distance_to_food
    obs[ObservationIndex.DISTANCE_TOY] = state.distance_to_toy
    obs[ObservationIndex.MOOD] = state.mood
    obs[ObservationIndex.LAZY_SCORE] = state.lazy_score
    obs[ObservationIndex.FOODIE_SCORE] = state.foodie_score
    obs[ObservationIndex.PLAYFUL_SCORE] = state.playful_score
    obs[ObservationIndex.IS_BOWL_EMPTY] = 1.0 if state.is_bowl_empty else 0.0
    obs[ObservationIndex.IS_BOWL_TIPPED] = 1.0 if state.is_bowl_tipped else 0.0
    
    return obs
