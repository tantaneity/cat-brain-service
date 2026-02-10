from dataclasses import dataclass
from enum import Enum
from typing import Optional
import random

import numpy as np

from src.core.emotions import EmotionType, EmotionalState, BehaviorIntensity


class StimulusType(Enum):
    PLAYER_APPROACH = "player_approach"
    PLAYER_PET = "player_pet"
    PLAYER_CALL = "player_call"
    LOUD_NOISE = "loud_noise"
    NEW_TOY = "new_toy"
    FOOD_REFILL = "food_refill"
    DOOR_OPEN = "door_open"
    SUDDEN_MOVEMENT = "sudden_movement"
    UNKNOWN_PERSON = "unknown_person"


@dataclass
class Stimulus:
    type: StimulusType
    intensity: float
    timestamp: float = 0.0


@dataclass
class ReactionModifier:
    action_override: Optional[int] = None
    action_probabilities: Optional[dict[int, float]] = None
    mood_delta: float = 0.0
    energy_delta: float = 0.0
    arousal_boost: float = 0.0
    animation_hint: Optional[str] = None
    sound_hint: Optional[str] = None
    probability: float = 1.0
    reaction_emotion: Optional[EmotionType] = None
    reaction_intensity: Optional[BehaviorIntensity] = None
    reaction_duration: float = 0.0


class ReactionSystem:
    
    REACTION_RULES = {
        (StimulusType.PLAYER_PET, EmotionType.HAPPY): ReactionModifier(
            action_probabilities={4: 0.6, 0: 0.3},
            mood_delta=15.0,
            animation_hint="purr",
            sound_hint="purr",
            probability=0.85,
        ),
        (StimulusType.PLAYER_PET, EmotionType.CONTENT): ReactionModifier(
            action_probabilities={4: 0.5, 0: 0.4},
            mood_delta=10.0,
            animation_hint="purr",
            sound_hint="purr_soft",
            probability=0.7,
        ),
        (StimulusType.PLAYER_PET, EmotionType.GRUMPY): ReactionModifier(
            action_probabilities={0: 0.5},
            mood_delta=-5.0,
            animation_hint="tail_flick",
            sound_hint="meow_annoyed",
            probability=0.6,
        ),
        (StimulusType.PLAYER_PET, EmotionType.SLEEPY): ReactionModifier(
            action_probabilities={0: 0.7, 3: 0.2},
            mood_delta=-2.0,
            animation_hint="ear_twitch",
            probability=0.8,
        ),
        (StimulusType.LOUD_NOISE, EmotionType.CONTENT): ReactionModifier(
            action_probabilities={0: 0.7},
            mood_delta=-15.0,
            energy_delta=-5.0,
            arousal_boost=0.3,
            animation_hint="startle",
            probability=0.9,
        ),
        (StimulusType.LOUD_NOISE, EmotionType.ANXIOUS): ReactionModifier(
            action_probabilities={0: 0.5},
            mood_delta=-25.0,
            energy_delta=-10.0,
            arousal_boost=0.5,
            animation_hint="hide",
            sound_hint="hiss",
            probability=0.95,
        ),
        (StimulusType.LOUD_NOISE, EmotionType.SCARED): ReactionModifier(
            action_override=0,
            mood_delta=-30.0,
            arousal_boost=0.7,
            animation_hint="run_hide",
            probability=1.0,
        ),
        (StimulusType.NEW_TOY, EmotionType.PLAYFUL): ReactionModifier(
            action_probabilities={2: 0.7, 5: 0.2},
            mood_delta=20.0,
            arousal_boost=0.3,
            animation_hint="excited",
            sound_hint="meow_excited",
            probability=0.8,
        ),
        (StimulusType.NEW_TOY, EmotionType.CURIOUS): ReactionModifier(
            action_probabilities={6: 0.5, 2: 0.3},
            mood_delta=10.0,
            arousal_boost=0.2,
            animation_hint="investigate",
            probability=0.7,
        ),
        (StimulusType.NEW_TOY, EmotionType.SLEEPY): ReactionModifier(
            action_probabilities={0: 0.6, 3: 0.3},
            mood_delta=2.0,
            animation_hint="lazy_look",
            probability=0.3,
        ),
        (StimulusType.FOOD_REFILL, EmotionType.HUNGRY): ReactionModifier(
            action_probabilities={1: 0.8, 7: 0.1},
            mood_delta=25.0,
            arousal_boost=0.4,
            animation_hint="run_to_food",
            sound_hint="meow_happy",
            probability=0.95,
        ),
        (StimulusType.FOOD_REFILL, EmotionType.DEMANDING): ReactionModifier(
            action_probabilities={1: 0.9},
            mood_delta=20.0,
            arousal_boost=0.5,
            animation_hint="rush_food",
            sound_hint="meow_urgent",
            probability=1.0,
        ),
        (StimulusType.FOOD_REFILL, EmotionType.CONTENT): ReactionModifier(
            action_probabilities={1: 0.4, 0: 0.4},
            mood_delta=5.0,
            animation_hint="casual_approach",
            probability=0.5,
        ),
        (StimulusType.PLAYER_CALL, EmotionType.AFFECTIONATE): ReactionModifier(
            action_probabilities={0: 0.5},
            mood_delta=10.0,
            animation_hint="come_running",
            sound_hint="meow_response",
            probability=0.8,
        ),
        (StimulusType.PLAYER_CALL, EmotionType.PLAYFUL): ReactionModifier(
            action_probabilities={5: 0.4, 0: 0.3},
            mood_delta=12.0,
            animation_hint="playful_approach",
            sound_hint="chirp",
            probability=0.7,
        ),
        (StimulusType.PLAYER_CALL, EmotionType.GRUMPY): ReactionModifier(
            action_probabilities={0: 0.7},
            mood_delta=-3.0,
            animation_hint="ignore",
            probability=0.6,
        ),
        (StimulusType.PLAYER_APPROACH, EmotionType.AFFECTIONATE): ReactionModifier(
            action_probabilities={0: 0.5, 4: 0.3},
            mood_delta=8.0,
            animation_hint="rub_legs",
            sound_hint="purr",
            probability=0.7,
        ),
        (StimulusType.PLAYER_APPROACH, EmotionType.SCARED): ReactionModifier(
            action_probabilities={0: 0.8},
            mood_delta=-10.0,
            animation_hint="back_away",
            probability=0.8,
        ),
        (StimulusType.SUDDEN_MOVEMENT, EmotionType.ANXIOUS): ReactionModifier(
            action_probabilities={0: 0.7},
            mood_delta=-12.0,
            arousal_boost=0.3,
            animation_hint="alert",
            probability=0.8,
        ),
    }
    
    FALLBACK_REACTIONS = {
        StimulusType.PLAYER_PET: ReactionModifier(
            mood_delta=5.0,
            animation_hint="acknowledge",
            probability=0.5,
        ),
        StimulusType.LOUD_NOISE: ReactionModifier(
            mood_delta=-10.0,
            arousal_boost=0.2,
            animation_hint="ears_back",
            probability=0.7,
        ),
        StimulusType.NEW_TOY: ReactionModifier(
            mood_delta=5.0,
            animation_hint="glance",
            probability=0.4,
        ),
        StimulusType.FOOD_REFILL: ReactionModifier(
            mood_delta=8.0,
            animation_hint="look_at_food",
            probability=0.6,
        ),
    }
    
    @staticmethod
    def get_reaction(
        stimulus: Stimulus,
        emotional_state: EmotionalState,
    ) -> Optional[ReactionModifier]:
        key = (stimulus.type, emotional_state.primary_emotion)
        
        reaction = ReactionSystem.REACTION_RULES.get(key)
        
        if not reaction:
            reaction = ReactionSystem.FALLBACK_REACTIONS.get(stimulus.type)
        
        if not reaction:
            return None
        
        adjusted_probability = reaction.probability * stimulus.intensity
        
        if random.random() > adjusted_probability:
            return None
        
        return reaction
    
    @staticmethod
    def apply_reaction(
        base_action: int,
        reaction: ReactionModifier,
    ) -> int:
        if reaction.action_override is not None:
            return reaction.action_override
        
        if reaction.action_probabilities:
            if random.random() < 0.8:
                actions = list(reaction.action_probabilities.keys())
                probabilities = list(reaction.action_probabilities.values())
                
                total = sum(probabilities)
                normalized_probs = [p / total for p in probabilities]
                
                return np.random.choice(actions, p=normalized_probs)
        
        return base_action
