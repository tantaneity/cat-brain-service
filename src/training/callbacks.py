import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals.get("rewards", [0])[0]

        if self.locals.get("dones", [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            logger.info(
                "training_progress",
                timesteps=self.num_timesteps,
                mean_reward=float(mean_reward),
                episodes=len(self.episode_rewards),
            )

        return True


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, "logger") and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            if "train/loss" in logs:
                self.losses.append(logs["train/loss"])


def get_training_callbacks(
    checkpoint_path: str, checkpoint_freq: int = 10000, log_freq: int = 1000
) -> list:
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="cat_brain_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    reward_callback = RewardLoggingCallback(log_freq=log_freq)
    metrics_callback = TrainingMetricsCallback()

    return [checkpoint_callback, reward_callback, metrics_callback]
