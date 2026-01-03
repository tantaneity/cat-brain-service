from gymnasium.spaces import Box, Discrete

from src.core.environment import CatEnvironment


class TestCatEnvironment:
    def test_observation_space(self):
        env = CatEnvironment()
        assert env.observation_space.shape == (4,)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.low.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert env.observation_space.high.tolist() == [100.0, 100.0, 10.0, 10.0]

    def test_action_space(self):
        env = CatEnvironment()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 4

    def test_reset(self):
        env = CatEnvironment()
        obs, info = env.reset(seed=42)

        assert obs.shape == (11,)
        assert 20.0 <= obs[0] <= 50.0  # hunger
        assert 40.0 <= obs[1] <= 70.0  # energy
        assert 5.0 <= obs[2] <= 100.0  # distance_to_food
        assert 5.0 <= obs[3] <= 100.0  # distance_to_toy
        assert 5.0 <= obs[4] <= 100.0  # distance_to_bed
        assert env.steps == 0

    def test_hunger_increases_each_step(self):
        env = CatEnvironment()
        env.reset(seed=42)

        initial_hunger = env.hunger
        env.step(0)  # idle

        assert env.hunger == initial_hunger + 1.0

    def test_energy_decreases_each_step(self):
        env = CatEnvironment()
        env.reset(seed=42)

        initial_energy = env.energy
        env.step(0)  # idle

        assert env.energy == initial_energy - 0.5

    def test_move_to_food_decreases_distance(self):
        env = CatEnvironment()
        env.reset(seed=42)

        initial_distance = env.distance_to_food
        env.step(1)  # move_to_food

        assert env.distance_to_food == initial_distance - 1.0

    def test_move_to_toy_decreases_distance(self):
        env = CatEnvironment()
        env.reset(seed=42)

        initial_distance = env.distance_to_toy
        env.step(2)  # move_to_toy

        assert env.distance_to_toy == initial_distance - 1.0

    def test_sleep_increases_energy(self):
        env = CatEnvironment()
        env.reset(seed=42)
        env.energy = 50.0

        env.step(3)  # sleep

        assert env.energy == 50.0 - 0.5 + 5.0  # -0.5 from step, +5 from sleep

    def test_eating_when_hungry_gives_reward(self):
        env = CatEnvironment()
        env.reset(seed=42)
        env.hunger = 80.0
        env.distance_to_food = 1.0

        _, reward, _, _, _ = env.step(1)  # move_to_food

        assert float(reward) >= 10.0

    def test_death_on_max_hunger(self):
        env = CatEnvironment()
        env.reset(seed=42)
        env.hunger = 99.0

        _, reward, terminated, _, _ = env.step(0)

        assert terminated is True
        assert float(reward) < 0

    def test_truncation_on_max_steps(self):
        env = CatEnvironment()
        env.reset(seed=42)
        env.steps = 999
        env.hunger = 0.0

        _, _, terminated, truncated, _ = env.step(0)

        assert terminated is False
        assert truncated is True

    def test_valid_actions(self):
        env = CatEnvironment()
        env.reset(seed=42)

        for action in range(4):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (4,)
            assert isinstance(reward, (int, float))
            if terminated or truncated:
                env.reset()
