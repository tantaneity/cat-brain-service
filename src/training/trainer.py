import json
import os
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from src.core.config import Settings
from src.core.environment import CatEnvironment
from src.training.callbacks import get_training_callbacks
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatBrainTrainer:
    def __init__(self, config: Settings):
        self.config = config
        self.model_path = Path(config.MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train(self, total_timesteps: int = None) -> PPO:
        if total_timesteps is None:
            total_timesteps = self.config.TOTAL_TIMESTEPS

        logger.info("training_started", total_timesteps=total_timesteps)

        env = CatEnvironment()

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=str(self.model_path / "tensorboard"),
        )

        checkpoint_path = str(self.model_path / "checkpoints")
        callbacks = get_training_callbacks(checkpoint_path)
        callback_list = CallbackList(callbacks)

        model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)

        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        mean_reward = self._evaluate_model(model, env)
        self.save_model(model, version, total_timesteps, mean_reward)

        logger.info("training_completed", version=version, mean_reward=mean_reward)

        env.close()
        return model

    def _evaluate_model(self, model: PPO, env: CatEnvironment, n_episodes: int = 10) -> float:
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
        return sum(rewards) / len(rewards)

    def save_model(self, model: PPO, version: str, timesteps: int, mean_reward: float):
        version_path = self.model_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        model_file = version_path / "cat_brain.zip"
        model.save(str(model_file))

        metadata = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "total_timesteps": timesteps,
            "mean_reward": mean_reward,
            "algorithm": "PPO",
        }

        metadata_file = version_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        latest_link = self.model_path / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            if os.name == "nt":
                if latest_link.is_dir():
                    os.rmdir(latest_link)
                else:
                    latest_link.unlink()
            else:
                latest_link.unlink()

        if os.name == "nt":
            import shutil

            if latest_link.exists():
                shutil.rmtree(latest_link)
            shutil.copytree(version_path, latest_link)
        else:
            latest_link.symlink_to(version_path.name)

        logger.info("model_saved", version=version, path=str(version_path))


def main():
    config = Settings()
    trainer = CatBrainTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
