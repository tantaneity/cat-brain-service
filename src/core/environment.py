from enum import IntEnum
from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CatAction(IntEnum):
    IDLE = 0
    MOVE_TO_FOOD = 1
    MOVE_TO_TOY = 2
    SLEEP = 3
    GROOM = 4
    PLAY = 5
    EXPLORE = 6
    MEOW_AT_BOWL = 7


class ObservationIndex(IntEnum):
    HUNGER = 0
    ENERGY = 1
    DISTANCE_FOOD = 2
    DISTANCE_TOY = 3
    MOOD = 4
    LAZY_SCORE = 5
    FOODIE_SCORE = 6
    PLAYFUL_SCORE = 7
    IS_BOWL_EMPTY = 8
    IS_BOWL_TIPPED = 9


class EnvConstants:
    MAX_HUNGER: float = 100.0
    MAX_ENERGY: float = 100.0
    MAX_DISTANCE: float = 100.0
    MIN_DISTANCE: float = 5.0

    HUNGER_PER_STEP: float = 1.0
    ENERGY_PER_STEP: float = 0.5
    HUNGER_DEGRADATION_FACTOR: float = 0.015
    ENERGY_DEGRADATION_FACTOR: float = 0.012
    FOOD_HUNGER_REDUCTION: float = 30.0
    SLEEP_ENERGY_GAIN: float = 10.0
    MOVE_DISTANCE: float = 1.0

    HUNGRY_THRESHOLD: float = 70.0
    TIRED_THRESHOLD: float = 30.0
    CRITICAL_TIRED_THRESHOLD: float = 15.0
    

    INEFFICIENT_EATING_THRESHOLD: float = 40.0
    PLAY_HUNGER_THRESHOLD: float = 60.0
    PLAY_ENERGY_THRESHOLD: float = 40.0
    GOOD_MOOD_THRESHOLD: float = 70.0
    GOOD_MOOD_PLAY_BONUS: float = 1.0

    MAX_MOOD: float = 100.0
    MOOD_DECAY_BASE: float = 0.05
    MOOD_DECAY_SCALE: float = 0.001
    MOOD_REWARD_SCALE: float = 3.0
    MOOD_HISTORY_WINDOW: int = 10

    PERSONALITY_DRIFT_RATE: float = 0.05
    LAZY_DRIFT_REDUCTION: float = 0.7
    PLAYFUL_DRIFT_REDUCTION: float = 0.5
    MAX_PERSONALITY_SCORE: float = 100.0
    MIN_PERSONALITY_SCORE: float = 0.0

    REWARD_STEP: float = 0.5
    REWARD_EAT_HUNGRY: float = 10.0
    REWARD_SLEEP_TIRED: float = 8.0
    REWARD_SLEEP_CRITICAL: float = 15.0
    REWARD_PLAY: float = 2.0
    REWARD_DEATH: float = -50.0
    PENALTY_LOW_ENERGY: float = -0.5
    PENALTY_INEFFICIENT_ACTION: float = -1.0

    INIT_HUNGER_MIN: float = 20.0
    INIT_HUNGER_MAX: float = 50.0
    INIT_ENERGY_MIN: float = 40.0
    INIT_ENERGY_MAX: float = 70.0

    MAX_STEPS: int = 1000
    NUM_ACTIONS: int = 8


class CatEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,  # hunger
                0.0,  # energy
                0.0,  # distance_to_food
                0.0,  # distance_to_toy
                0.0,  # mood
                EnvConstants.MIN_PERSONALITY_SCORE,  # lazy_score
                EnvConstants.MIN_PERSONALITY_SCORE,  # foodie_score
                EnvConstants.MIN_PERSONALITY_SCORE,  # playful_score
                0.0,  # is_bowl_empty
                0.0,  # is_bowl_tipped
            ]),
            high=np.array([
                EnvConstants.MAX_HUNGER,
                EnvConstants.MAX_ENERGY,
                EnvConstants.MAX_DISTANCE,
                EnvConstants.MAX_DISTANCE,
                EnvConstants.MAX_MOOD,
                EnvConstants.MAX_PERSONALITY_SCORE,  # lazy_score
                EnvConstants.MAX_PERSONALITY_SCORE,  # foodie_score
                EnvConstants.MAX_PERSONALITY_SCORE,  # playful_score
                1.0,  # is_bowl_empty
                1.0,  # is_bowl_tipped
            ]),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(EnvConstants.NUM_ACTIONS)

        self.hunger: float = 50.0
        self.energy: float = 50.0
        self.distance_to_food: float = 5.0
        self.distance_to_toy: float = 5.0
        self.mood: float = 50.0
        self.is_bowl_empty: bool = False
        self.is_bowl_tipped: bool = False
        self.recent_rewards: list[float] = []
        self.personality_scores: dict[str, float] = {
            "lazy": 50.0,
            "foodie": 50.0,
            "playful": 50.0,
        }
        self.steps: int = 0
        self.max_steps: int = EnvConstants.MAX_STEPS

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.hunger = self.np_random.uniform(
            EnvConstants.INIT_HUNGER_MIN,
            EnvConstants.INIT_HUNGER_MAX,
        )
        self.energy = self.np_random.uniform(
            EnvConstants.INIT_ENERGY_MIN,
            EnvConstants.INIT_ENERGY_MAX,
        )
        self.distance_to_food = self.np_random.uniform(
            EnvConstants.MIN_DISTANCE,
            EnvConstants.MAX_DISTANCE,
        )
        self.distance_to_toy = self.np_random.uniform(
            EnvConstants.MIN_DISTANCE,
            EnvConstants.MAX_DISTANCE,
        )
        self.mood = 50.0
        self.is_bowl_empty = self.np_random.random() < 0.2
        self.is_bowl_tipped = self.np_random.random() < 0.1
        self.recent_rewards = []
        self.personality_scores = {
            "lazy": 50.0,
            "foodie": 50.0,
            "playful": 50.0,
        }
        self.steps = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [
                self.hunger,
                self.energy,
                self.distance_to_food,
                self.distance_to_toy,
                self.mood,
                self.personality_scores["lazy"],
                self.personality_scores["foodie"],
                self.personality_scores["playful"],
                1.0 if self.is_bowl_empty else 0.0,
                1.0 if self.is_bowl_tipped else 0.0,
            ],
            dtype=np.float32,
        )

    def _respawn_distance(self) -> float:
        return self.np_random.uniform(
            EnvConstants.MIN_DISTANCE,
            EnvConstants.MAX_DISTANCE,
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.steps += 1
        reward = EnvConstants.REWARD_STEP
        
        critical_energy = self.energy < EnvConstants.CRITICAL_TIRED_THRESHOLD
        low_energy = self.energy < EnvConstants.TIRED_THRESHOLD
        very_hungry = self.hunger < 20.0
        hungry = self.hunger < 30.0

        hunger_deficit = EnvConstants.MAX_HUNGER - self.hunger
        hunger_multiplier = 1.0 + (hunger_deficit / EnvConstants.MAX_HUNGER) * EnvConstants.HUNGER_DEGRADATION_FACTOR
        self.hunger -= EnvConstants.HUNGER_PER_STEP * hunger_multiplier

        energy_deficit = EnvConstants.MAX_ENERGY - self.energy
        energy_multiplier = 1.0 + (energy_deficit / EnvConstants.MAX_ENERGY) * EnvConstants.ENERGY_DEGRADATION_FACTOR
        self.energy -= EnvConstants.ENERGY_PER_STEP * energy_multiplier

        if action == CatAction.MOVE_TO_FOOD:
            if critical_energy:
                reward -= 20.0
            elif low_energy:
                reward -= 10.0
            
            self.distance_to_food -= EnvConstants.MOVE_DISTANCE
            if self.distance_to_food <= 0:
                if self.is_bowl_empty or self.is_bowl_tipped:
                    reward -= 10.0
                elif very_hungry:
                    reward += EnvConstants.REWARD_EAT_HUNGRY * 1.5
                elif hungry:
                    reward += EnvConstants.REWARD_EAT_HUNGRY
                elif self.hunger > 60.0:
                    reward += EnvConstants.PENALTY_INEFFICIENT_ACTION
                
                if not self.is_bowl_empty and not self.is_bowl_tipped:
                    self.hunger = min(EnvConstants.MAX_HUNGER, self.hunger + EnvConstants.FOOD_HUNGER_REDUCTION)
                self.distance_to_food = self._respawn_distance()

        elif action == CatAction.MOVE_TO_TOY:
            if critical_energy or low_energy or very_hungry:
                reward -= 20.0
            else:
                self.distance_to_toy -= EnvConstants.MOVE_DISTANCE
                if self.distance_to_toy <= 0:
                    if self.hunger > 40.0 and self.energy > EnvConstants.PLAY_ENERGY_THRESHOLD:
                        reward += EnvConstants.REWARD_PLAY * 2.0
                    self.distance_to_toy = self._respawn_distance()

        elif action == CatAction.SLEEP:
            energy_before = self.energy
            
            self.energy = min(
                EnvConstants.MAX_ENERGY,
                self.energy + EnvConstants.SLEEP_ENERGY_GAIN,
            )
            
            actual_energy_gain = self.energy - energy_before
            
            if critical_energy:
                reward += EnvConstants.REWARD_SLEEP_CRITICAL * 2.0
                reward += actual_energy_gain * 2.0
            elif self.energy < EnvConstants.CRITICAL_TIRED_THRESHOLD:
                reward += EnvConstants.REWARD_SLEEP_CRITICAL
                reward += actual_energy_gain * 1.5
            elif self.energy < EnvConstants.TIRED_THRESHOLD:
                reward += EnvConstants.REWARD_SLEEP_TIRED
                reward += actual_energy_gain * 1.0
            elif self.energy < 60.0:
                reward += actual_energy_gain * 0.3
            else:
                reward -= 15.0
            
            if very_hungry:
                reward -= 20.0
        
        elif action == CatAction.GROOM:
            if critical_energy or very_hungry:
                reward -= 5.0
            else:
                reward += 3.0
                self.mood = min(EnvConstants.MAX_MOOD, self.mood + 10.0)
                if self.mood > EnvConstants.GOOD_MOOD_THRESHOLD:
                    reward += 2.0
        
        elif action == CatAction.PLAY:
            if critical_energy or very_hungry:
                reward -= 10.0
            elif low_energy or hungry:
                reward -= 5.0
            else:
                self.energy -= 5.0
                reward += EnvConstants.REWARD_PLAY * 2.0
                self.mood = min(EnvConstants.MAX_MOOD, self.mood + 15.0)
                if self.mood > EnvConstants.GOOD_MOOD_THRESHOLD:
                    reward += EnvConstants.GOOD_MOOD_PLAY_BONUS * 3.0
        
        elif action == CatAction.EXPLORE:
            if critical_energy or very_hungry:
                reward -= 8.0
            elif low_energy:
                reward -= 3.0
            else:
                self.energy -= 2.0
                reward += 1.5
                self.mood = min(EnvConstants.MAX_MOOD, self.mood + 5.0)
        
        elif action == CatAction.IDLE:
            if critical_energy:
                reward -= 20.0
            elif very_hungry:
                reward -= 15.0
            elif low_energy or hungry:
                reward -= 10.0
            else:
                reward += 1.0

        elif action == CatAction.MEOW_AT_BOWL:
            if self.is_bowl_empty and very_hungry:
                reward += 15.0
                self.mood = min(EnvConstants.MAX_MOOD, self.mood + 5.0)
            elif self.is_bowl_tipped and very_hungry:
                reward += 12.0
                self.mood = min(EnvConstants.MAX_MOOD, self.mood + 3.0)
            elif self.is_bowl_empty or self.is_bowl_tipped:
                reward += 5.0
            else:
                reward -= 5.0

        if self.energy < EnvConstants.CRITICAL_TIRED_THRESHOLD:
            reward += EnvConstants.PENALTY_LOW_ENERGY * 3.0
        elif self.energy < EnvConstants.TIRED_THRESHOLD:
            reward += EnvConstants.PENALTY_LOW_ENERGY

        self.hunger = np.clip(self.hunger, 0.0, EnvConstants.MAX_HUNGER)
        self.energy = np.clip(self.energy, 0.0, EnvConstants.MAX_ENERGY)


        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > EnvConstants.MOOD_HISTORY_WINDOW:
            self.recent_rewards.pop(0)
        
        avg_recent_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
        mood_delta = avg_recent_reward * EnvConstants.MOOD_REWARD_SCALE
        

        mood_decay = EnvConstants.MOOD_DECAY_BASE + (self.mood * EnvConstants.MOOD_DECAY_SCALE)
        self.mood += float(mood_delta) - mood_decay
        self.mood = np.clip(self.mood, 0.0, EnvConstants.MAX_MOOD)
        

        if action == CatAction.MOVE_TO_TOY and self.mood > EnvConstants.GOOD_MOOD_THRESHOLD:
            reward += EnvConstants.GOOD_MOOD_PLAY_BONUS


        drift_rate = EnvConstants.PERSONALITY_DRIFT_RATE
        
        if action == CatAction.SLEEP or action == CatAction.IDLE:

            self.personality_scores["lazy"] = min(
                EnvConstants.MAX_PERSONALITY_SCORE,
                self.personality_scores["lazy"] + drift_rate
            )

            self.personality_scores["playful"] = max(
                EnvConstants.MIN_PERSONALITY_SCORE,
                self.personality_scores["playful"] - drift_rate * EnvConstants.PLAYFUL_DRIFT_REDUCTION
            )
        
        if action == CatAction.MOVE_TO_FOOD:

            self.personality_scores["foodie"] = min(
                EnvConstants.MAX_PERSONALITY_SCORE,
                self.personality_scores["foodie"] + drift_rate
            )
        
        if action == CatAction.MOVE_TO_TOY:

            self.personality_scores["playful"] = min(
                EnvConstants.MAX_PERSONALITY_SCORE,
                self.personality_scores["playful"] + drift_rate
            )
            self.personality_scores["lazy"] = max(
                EnvConstants.MIN_PERSONALITY_SCORE,
                self.personality_scores["lazy"] - drift_rate * EnvConstants.LAZY_DRIFT_REDUCTION
            )

        terminated = False
        truncated = False

        if self.hunger >= EnvConstants.MAX_HUNGER:
            reward += EnvConstants.REWARD_DEATH
            terminated = True

        if self.energy <= 0:
            reward += EnvConstants.REWARD_DEATH / 2
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self) -> None:
        if self.render_mode == "human":
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(
                "env_state",
                step=self.steps,
                hunger=round(self.hunger, 1),
                energy=round(self.energy, 1),
                food_dist=round(self.distance_to_food, 1),
                toy_dist=round(self.distance_to_toy, 1),
            )
