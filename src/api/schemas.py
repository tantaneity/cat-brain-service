from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.core.environment import EnvConstants


class CatPersonality(str, Enum):
    BALANCED = "balanced"
    LAZY = "lazy"
    FOODIE = "foodie"
    PLAYFUL = "playful"


class CatState(BaseModel):
    cat_id: Optional[str] = Field(
        default=None,
        description="Unique cat identifier (uses default brain if not provided)",
    )
    personality: CatPersonality = Field(
        default=CatPersonality.BALANCED,
        description="Cat personality type affecting decision making",
    )
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
                "cat_id": "whiskers_123",
                "personality": "lazy",
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

class CreateCatRequest(BaseModel):
    cat_id: str = Field(
        description="Unique identifier for the cat",
        min_length=1,
        max_length=100,
    )
    personality: CatPersonality = Field(
        default=CatPersonality.BALANCED,
        description="Initial personality type for the cat",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "cat_id": "whiskers_123",
                "personality": "playful",
            }
        }
    }


class CreateCatResponse(BaseModel):
    cat_id: str
    personality: str
    brain_path: str
    created_at: str
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "cat_id": "whiskers_123",
                "personality": "playful",
                "brain_path": "models/cats/whiskers_123/latest/cat_brain.zip",
                "created_at": "2025-12-23T10:30:00",
                "message": "Cat brain created successfully from default model",
            }
        }
    }


class CatInfo(BaseModel):
    cat_id: str
    model_path: str
    created_at: Optional[str] = None
    total_actions: int = 0