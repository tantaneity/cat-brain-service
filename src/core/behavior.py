import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.environment import CatAction


@dataclass
class BehaviorPattern:
    name: str
    actions: list[int]
    weights: list[float]
    min_mood: float = 0.0
    max_mood: float = 100.0
    min_energy: float = 0.0
    time_preference: Optional[str] = None


class BehaviorLibrary:
    
    PATTERNS = {
        "zoomies": BehaviorPattern(
            name="zoomies",
            actions=[CatAction.EXPLORE, CatAction.PLAY, CatAction.MOVE_TO_TOY],
            weights=[0.5, 0.3, 0.2],
            min_mood=60.0,
            min_energy=60.0,
        ),
        "lazy_sunday": BehaviorPattern(
            name="lazy_sunday",
            actions=[CatAction.SLEEP, CatAction.IDLE, CatAction.GROOM],
            weights=[0.5, 0.3, 0.2],
            max_mood=50.0,
            time_preference="afternoon",
        ),
        "midnight_madness": BehaviorPattern(
            name="midnight_madness",
            actions=[CatAction.EXPLORE, CatAction.MEOW_AT_BOWL, CatAction.PLAY],
            weights=[0.4, 0.3, 0.3],
            min_energy=40.0,
            time_preference="night",
        ),
        "morning_routine": BehaviorPattern(
            name="morning_routine",
            actions=[CatAction.MEOW_AT_BOWL, CatAction.GROOM, CatAction.MOVE_TO_FOOD],
            weights=[0.4, 0.3, 0.3],
            time_preference="morning",
        ),
        "food_obsession": BehaviorPattern(
            name="food_obsession",
            actions=[CatAction.MOVE_TO_FOOD, CatAction.MEOW_AT_BOWL, CatAction.IDLE],
            weights=[0.6, 0.3, 0.1],
            min_mood=30.0,
        ),
    }
    
    @staticmethod
    def get_random_quirk_action(mood: float, energy: float) -> Optional[int]:
        quirks = [
            (CatAction.GROOM, 0.05),
            (CatAction.EXPLORE, 0.03),
            (CatAction.MEOW_AT_BOWL, 0.02),
        ]
        
        if mood > 70:
            quirks.append((CatAction.PLAY, 0.04))
        
        if energy < 40:
            quirks.append((CatAction.SLEEP, 0.06))
        
        for action, probability in quirks:
            if random.random() < probability:
                return action
        
        return None


class StochasticBehavior:
    
    @staticmethod
    def add_noise_to_prediction(
        base_action: int,
        confidence: float = 0.8,
        mood: float = 50.0,
    ) -> int:
        randomness = 0.2
        
        if mood < 30:
            randomness += 0.1
        elif mood > 80:
            randomness += 0.05
        
        effective_randomness = randomness * (1 - confidence)
        
        if random.random() < effective_randomness:
            available_actions = list(range(8))
            available_actions.remove(base_action)
            return random.choice(available_actions)
        
        return base_action
    
    @staticmethod
    def should_change_mind(
        current_action: int,
        new_action: int,
        mood: float,
        personality_playful: float,
    ) -> bool:
        if current_action == new_action:
            return False
        
        base_chance = 0.05
        
        if mood > 70:
            base_chance += 0.05
        
        playfulness_factor = personality_playful / 100 * 0.1
        base_chance += playfulness_factor
        
        return random.random() < base_chance
    
    @staticmethod
    def get_attention_span_modifier(
        energy: float,
        mood: float,
        personality_lazy: float,
    ) -> float:
        base_span = 1.0
        
        if energy < 30:
            base_span *= 0.7
        elif energy > 70:
            base_span *= 1.2
        
        if mood < 40:
            base_span *= 0.8
        
        lazy_factor = (personality_lazy / 100) * 0.3
        base_span *= (1 - lazy_factor)
        
        return base_span
    
    @staticmethod
    def introduce_distraction(base_action: int, environment_richness: float = 0.5) -> int:
        distraction_chance = environment_richness * 0.1
        
        if random.random() < distraction_chance:
            distraction_actions = [
                CatAction.EXPLORE,
                CatAction.IDLE,
                CatAction.GROOM,
            ]
            return random.choice(distraction_actions)
        
        return base_action


class CatMemory:
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.recent_actions: list[int] = []
        self.recent_moods: list[float] = []
        self.recent_rewards: list[float] = []
        self.interaction_count: int = 0
        self.last_pet_time: Optional[float] = None
        self.last_food_time: Optional[float] = None
    
    def record_action(self, action: int, mood: float, reward: float = 0.0):
        self.recent_actions.append(action)
        self.recent_moods.append(mood)
        self.recent_rewards.append(reward)
        
        if len(self.recent_actions) > self.capacity:
            self.recent_actions.pop(0)
            self.recent_moods.pop(0)
            self.recent_rewards.pop(0)
    
    def get_recent_activity_level(self) -> float:
        if not self.recent_actions:
            return 0.0
        
        active_actions = [
            CatAction.MOVE_TO_FOOD,
            CatAction.MOVE_TO_TOY,
            CatAction.PLAY,
            CatAction.EXPLORE,
        ]
        
        recent_window = self.recent_actions[-10:]
        active_count = sum(1 for a in recent_window if a in active_actions)
        
        return active_count / len(recent_window)
    
    def get_action_diversity(self) -> float:
        if not self.recent_actions:
            return 0.0
        
        recent_window = self.recent_actions[-10:]
        unique_actions = len(set(recent_window))
        
        return unique_actions / 8.0
    
    def is_repeating_behavior(self) -> bool:
        if len(self.recent_actions) < 5:
            return False
        
        last_five = self.recent_actions[-5:]
        return len(set(last_five)) <= 2
    
    def record_interaction(self, interaction_type: str, timestamp: float):
        self.interaction_count += 1
        
        if interaction_type == "pet":
            self.last_pet_time = timestamp
        elif interaction_type == "food":
            self.last_food_time = timestamp
