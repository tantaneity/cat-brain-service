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


class EnvConstants:
    MAX_HUNGER: float = 100.0
    MAX_ENERGY: float = 100.0
    MAX_DISTANCE: float = 10.0
    MIN_DISTANCE: float = 5.0

    HUNGER_PER_STEP: float = 1.0
    ENERGY_PER_STEP: float = 0.5
    FOOD_HUNGER_REDUCTION: float = 30.0
    SLEEP_ENERGY_GAIN: float = 10.0
    MOVE_DISTANCE: float = 1.0

    HUNGRY_THRESHOLD: float = 70.0
    TIRED_THRESHOLD: float = 30.0
    CRITICAL_TIRED_THRESHOLD: float = 15.0

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
    NUM_ACTIONS: int = 4


class CatEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([
                EnvConstants.MAX_HUNGER,
                EnvConstants.MAX_ENERGY,
                EnvConstants.MAX_DISTANCE,
                EnvConstants.MAX_DISTANCE,
            ]),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(EnvConstants.NUM_ACTIONS)

        self.hunger: float = 50.0
        self.energy: float = 50.0
        self.distance_to_food: float = 5.0
        self.distance_to_toy: float = 5.0
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
        self.steps = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [self.hunger, self.energy, self.distance_to_food, self.distance_to_toy],
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

        self.hunger += EnvConstants.HUNGER_PER_STEP
        self.energy -= EnvConstants.ENERGY_PER_STEP

        if action == CatAction.MOVE_TO_FOOD:
            self.distance_to_food -= EnvConstants.MOVE_DISTANCE
            if self.distance_to_food <= 0:
                if self.hunger > EnvConstants.HUNGRY_THRESHOLD:
                    reward += EnvConstants.REWARD_EAT_HUNGRY
                elif self.hunger < 40:
                    reward += EnvConstants.PENALTY_INEFFICIENT_ACTION
                self.hunger = max(0.0, self.hunger - EnvConstants.FOOD_HUNGER_REDUCTION)
                self.distance_to_food = self._respawn_distance()

        elif action == CatAction.MOVE_TO_TOY:
            self.distance_to_toy -= EnvConstants.MOVE_DISTANCE
            if self.distance_to_toy <= 0:
                if self.hunger < 60 and self.energy > 40:
                    reward += EnvConstants.REWARD_PLAY
                self.distance_to_toy = self._respawn_distance()

        elif action == CatAction.SLEEP:
            if self.energy < EnvConstants.CRITICAL_TIRED_THRESHOLD:
                reward += EnvConstants.REWARD_SLEEP_CRITICAL
            elif self.energy < EnvConstants.TIRED_THRESHOLD:
                reward += EnvConstants.REWARD_SLEEP_TIRED
            self.energy = min(
                EnvConstants.MAX_ENERGY,
                self.energy + EnvConstants.SLEEP_ENERGY_GAIN,
            )

        if self.energy < EnvConstants.TIRED_THRESHOLD:
            reward += EnvConstants.PENALTY_LOW_ENERGY

        self.hunger = np.clip(self.hunger, 0.0, EnvConstants.MAX_HUNGER)
        self.energy = np.clip(self.energy, 0.0, EnvConstants.MAX_ENERGY)

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
