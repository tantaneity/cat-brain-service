from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.environment import EnvConstants


class CatState(BaseModel):
    hunger: float = Field(
        ge=0,
        le=EnvConstants.MAX_HUNGER,
        description="Cat hunger level",
    )
    energy: float = Field(
        ge=0,
        le=EnvConstants.MAX_ENERGY,
        description="Cat energy level",
    )
    distance_to_food: float = Field(
        ge=0,
        le=EnvConstants.MAX_DISTANCE,
        description="Distance to food",
    )
    distance_to_toy: float = Field(
        ge=0,
        le=EnvConstants.MAX_DISTANCE,
        description="Distance to toy",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "hunger": 50.0,
                "energy": 70.0,
                "distance_to_food": 3.5,
                "distance_to_toy": 7.2,
            }
        }
    }


class CatAction(BaseModel):
    action: int = Field(
        ge=0,
        le=EnvConstants.NUM_ACTIONS - 1,
        description="Action: 0=idle, 1=move_to_food, 2=move_to_toy, 3=sleep",
    )
    action_name: Optional[str] = Field(
        default=None,
        description="Human readable action name",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": 1,
                "action_name": "move_to_food",
            }
        }
    }


class BatchCatStates(BaseModel):
    states: List[CatState] = Field(description="List of cat states")


class BatchCatActions(BaseModel):
    actions: List[int] = Field(description="List of actions")


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    total_timesteps: int
    mean_reward: float
    algorithm: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    cache_available: bool
    uptime: float


class ErrorResponse(BaseModel):
    detail: str
