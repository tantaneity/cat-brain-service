import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CatEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([100.0, 100.0, 10.0, 10.0]),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        self.action_names = {0: "idle", 1: "move_to_food", 2: "move_to_toy", 3: "sleep"}

        self.hunger = 50.0
        self.energy = 50.0
        self.distance_to_food = 5.0
        self.distance_to_toy = 5.0
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.hunger = self.np_random.uniform(20.0, 50.0)
        self.energy = self.np_random.uniform(40.0, 70.0)
        self.distance_to_food = self.np_random.uniform(5.0, 10.0)
        self.distance_to_toy = self.np_random.uniform(5.0, 10.0)
        self.steps = 0

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array(
            [self.hunger, self.energy, self.distance_to_food, self.distance_to_toy],
            dtype=np.float32,
        )

    def step(self, action):
        self.steps += 1
        reward = 0.1

        self.hunger += 1.0
        self.energy -= 0.5

        if action == 1:  # move_to_food
            self.distance_to_food -= 1.0
            if self.distance_to_food <= 0:
                if self.hunger > 70:
                    reward += 10.0
                self.hunger = max(0.0, self.hunger - 30.0)
                self.distance_to_food = self.np_random.uniform(5.0, 10.0)

        elif action == 2:  # move_to_toy
            self.distance_to_toy -= 1.0
            if self.distance_to_toy <= 0:
                self.distance_to_toy = self.np_random.uniform(5.0, 10.0)

        elif action == 3:  # sleep
            self.energy = min(100.0, self.energy + 5.0)
            if self.energy < 30:
                reward += 5.0

        self.hunger = np.clip(self.hunger, 0.0, 100.0)
        self.energy = np.clip(self.energy, 0.0, 100.0)

        terminated = False
        truncated = False

        if self.hunger >= 100:
            reward -= 100.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            print(
                f"Step: {self.steps}, Hunger: {self.hunger:.1f}, Energy: {self.energy:.1f}, "
                f"Food dist: {self.distance_to_food:.1f}, Toy dist: {self.distance_to_toy:.1f}"
            )
