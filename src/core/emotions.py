from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class EmotionType(Enum):
    CONTENT = "content"
    HAPPY = "happy"
    EXCITED = "excited"
    PLAYFUL = "playful"
    CURIOUS = "curious"
    RELAXED = "relaxed"
    SLEEPY = "sleepy"
    HUNGRY = "hungry"
    GRUMPY = "grumpy"
    ANNOYED = "annoyed"
    SCARED = "scared"
    ANXIOUS = "anxious"
    AFFECTIONATE = "affectionate"
    DEMANDING = "demanding"


class BehaviorIntensity(Enum):
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    INTENSE = "intense"


@dataclass
class EmotionalState:
    primary_emotion: EmotionType
    intensity: BehaviorIntensity
    mood_value: float
    arousal_level: float
    
    valence: float
    
    def to_dict(self) -> dict:
        return {
            "emotion": self.primary_emotion.value,
            "intensity": self.intensity.value,
            "mood": round(self.mood_value, 2),
            "arousal": round(self.arousal_level, 2),
            "valence": round(self.valence, 2),
        }


class EmotionEngine:
    
    EMOTION_THRESHOLDS = {
        EmotionType.SCARED: {"mood_max": 30, "arousal_min": 0.7},
        EmotionType.ANXIOUS: {"mood_max": 40, "arousal_min": 0.5},
        EmotionType.GRUMPY: {"mood_max": 35, "hunger_min": 60, "energy_max": 40},
        EmotionType.ANNOYED: {"mood_max": 45, "arousal_min": 0.4},
        EmotionType.HUNGRY: {"hunger_min": 70, "mood_max": 50},
        EmotionType.DEMANDING: {"hunger_min": 75, "energy_min": 40},
        EmotionType.SLEEPY: {"energy_max": 30, "mood_min": 30},
        EmotionType.RELAXED: {"mood_min": 60, "energy_min": 40, "arousal_max": 0.3},
        EmotionType.CONTENT: {"mood_min": 55, "arousal_max": 0.4},
        EmotionType.AFFECTIONATE: {"mood_min": 70, "arousal_max": 0.5},
        EmotionType.CURIOUS: {"mood_min": 50, "arousal_min": 0.4, "arousal_max": 0.7},
        EmotionType.PLAYFUL: {"mood_min": 65, "energy_min": 50, "arousal_min": 0.5},
        EmotionType.EXCITED: {"mood_min": 75, "arousal_min": 0.7},
        EmotionType.HAPPY: {"mood_min": 70, "energy_min": 45},
    }
    
    @staticmethod
    def calculate_arousal(
        hunger: float,
        energy: float,
        recent_activity: float,
        noise_level: float = 0.0,
    ) -> float:
        hunger_arousal = (100 - hunger) / 100 * 0.3
        energy_arousal = energy / 100 * 0.4
        activity_arousal = recent_activity * 0.2
        noise_arousal = noise_level * 0.1
        
        return np.clip(
            hunger_arousal + energy_arousal + activity_arousal + noise_arousal,
            0.0,
            1.0,
        )
    
    @staticmethod
    def determine_emotion(
        mood: float,
        hunger: float,
        energy: float,
        arousal: float,
    ) -> EmotionType:
        valence = (mood / 100 - 0.5) * 2
        
        for emotion, thresholds in EmotionEngine.EMOTION_THRESHOLDS.items():
            matches = True
            
            if "mood_min" in thresholds and mood < thresholds["mood_min"]:
                matches = False
            if "mood_max" in thresholds and mood > thresholds["mood_max"]:
                matches = False
            if "hunger_min" in thresholds and hunger < thresholds["hunger_min"]:
                matches = False
            if "energy_min" in thresholds and energy < thresholds["energy_min"]:
                matches = False
            if "energy_max" in thresholds and energy > thresholds["energy_max"]:
                matches = False
            if "arousal_min" in thresholds and arousal < thresholds["arousal_min"]:
                matches = False
            if "arousal_max" in thresholds and arousal > thresholds["arousal_max"]:
                matches = False
            
            if matches:
                return emotion
        
        return EmotionType.CONTENT
    
    @staticmethod
    def calculate_intensity(
        mood: float,
        arousal: float,
        hunger: float,
        energy: float,
    ) -> BehaviorIntensity:
        intensity_score = 0.0
        
        if hunger > 80 or hunger < 20:
            intensity_score += 0.3
        if energy < 20 or energy > 80:
            intensity_score += 0.2
        
        intensity_score += arousal * 0.3
        
        mood_deviation = abs(mood - 50) / 50
        intensity_score += mood_deviation * 0.2
        
        if intensity_score > 0.75:
            return BehaviorIntensity.INTENSE
        elif intensity_score > 0.5:
            return BehaviorIntensity.STRONG
        elif intensity_score > 0.25:
            return BehaviorIntensity.MODERATE
        else:
            return BehaviorIntensity.SUBTLE
    
    @staticmethod
    def get_emotional_state(
        mood: float,
        hunger: float,
        energy: float,
        recent_activity: float = 0.0,
        noise_level: float = 0.0,
    ) -> EmotionalState:
        arousal = EmotionEngine.calculate_arousal(
            hunger, energy, recent_activity, noise_level
        )
        
        emotion = EmotionEngine.determine_emotion(mood, hunger, energy, arousal)
        
        intensity = EmotionEngine.calculate_intensity(mood, arousal, hunger, energy)
        
        valence = (mood / 100 - 0.5) * 2
        
        return EmotionalState(
            primary_emotion=emotion,
            intensity=intensity,
            mood_value=mood,
            arousal_level=arousal,
            valence=valence,
        )
