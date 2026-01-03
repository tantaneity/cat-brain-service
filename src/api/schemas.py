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
    distance_to_bed: float = Field(
        default=50.0,
        ge=0,
        le=EnvConstants.MAX_DISTANCE,
        description="Distance to bed/sleep spot",
    )
    mood: float = Field(
        default=50.0,
        ge=0,
        le=EnvConstants.MAX_MOOD,
        description="Cat mood level (0-100, affects playfulness)",
    )
    lazy_score: float = Field(
        default=50.0,
        ge=0,
        le=100.0,
        description="How lazy the cat has become (personality drift)",
    )
    foodie_score: float = Field(
        default=50.0,
        ge=0,
        le=100.0,
        description="How food-focused the cat has become (personality drift)",
    )
    playful_score: float = Field(
        default=50.0,
        ge=0,
        le=100.0,
        description="How playful the cat has become (personality drift)",
    )
    is_bowl_empty: Optional[bool] = Field(
        default=False,
        description="Whether the food bowl is empty",
    )
    is_bowl_tipped: Optional[bool] = Field(
        default=False,
        description="Whether the food bowl is tipped over",
    )
    
    player_nearby: bool = Field(
        default=False,
        description="Whether player is near the cat",
    )
    player_distance: float = Field(
        default=100.0,
        ge=0,
        le=100.0,
        description="Distance to player",
    )
    is_being_petted: bool = Field(
        default=False,
        description="Whether player is currently petting the cat",
    )
    is_player_calling: bool = Field(
        default=False,
        description="Whether player is calling the cat",
    )
    loud_noise_level: float = Field(
        default=0.0,
        ge=0,
        le=1.0,
        description="Environmental noise level (0=quiet, 1=very loud)",
    )
    new_toy_appeared: bool = Field(
        default=False,
        description="Whether a new toy just appeared",
    )
    food_bowl_refilled: bool = Field(
        default=False,
        description="Whether food bowl was just refilled",
    )
    sudden_movement: bool = Field(
        default=False,
        description="Whether there was sudden movement nearby",
    )
    time_of_day: str = Field(
        default="afternoon",
        description="Time of day: morning, afternoon, evening, night",
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
                "mood": 65.0,
                "lazy_score": 60.0,
                "foodie_score": 45.0,
                "playful_score": 30.0,
                "is_bowl_empty": False,
                "is_bowl_tipped": False,
                "player_nearby": True,
                "player_distance": 15.5,
                "is_being_petted": False,
                "is_player_calling": False,
                "loud_noise_level": 0.0,
                "new_toy_appeared": False,
                "food_bowl_refilled": False,
                "sudden_movement": False,
                "time_of_day": "afternoon",
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
    
    emotion: Optional[str] = Field(
        default=None,
        description="Current emotional state",
    )
    emotion_intensity: Optional[str] = Field(
        default=None,
        description="Intensity of emotion: subtle, moderate, strong, intense",
    )
    mood_change: float = Field(
        default=0.0,
        description="Change in mood from this action",
    )
    arousal_level: Optional[float] = Field(
        default=None,
        description="Cat arousal level (0-1, affects energy/alertness)",
    )
    animation_hint: Optional[str] = Field(
        default=None,
        description="Suggested animation for Unity",
    )
    sound_hint: Optional[str] = Field(
        default=None,
        description="Suggested sound effect",
    )
    reaction_triggered: bool = Field(
        default=False,
        description="Whether a reaction to stimulus was triggered",
    )
    behavior_pattern: Optional[str] = Field(
        default=None,
        description="Active behavior pattern",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": 4,
                "action_name": "groom",
                "emotion": "content",
                "emotion_intensity": "moderate",
                "mood_change": 5.0,
                "arousal_level": 0.3,
                "animation_hint": "purr",
                "sound_hint": "purr_soft",
                "reaction_triggered": True,
                "behavior_pattern": "lazy_sunday",
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


ObservationSchema = CatState