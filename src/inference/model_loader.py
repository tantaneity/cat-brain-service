import json
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO

from src.core.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXCLUDED_DIRS: set[str] = {"checkpoints", "tensorboard"}


class ModelLoader:
    def __init__(self, config: Settings):
        self.config = config
        self.model_path = Path(config.MODEL_PATH)
        self.models: dict[str, PPO] = {}
        self.metadata_cache: dict[str, dict] = {}

    def load_model(self, version: str = "latest") -> PPO:
        if version in self.models:
            return self.models[version]

        version_path = self.model_path / version
        model_file = version_path / "cat_brain.zip"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        model = PPO.load(str(model_file))
        self.models[version] = model

        logger.info("model_loaded", version=version, path=str(model_file))

        return model

    def get_model(self, version: str = "latest") -> Optional[PPO]:
        if version in self.models:
            return self.models[version]
        return None

    def get_model_info(self, version: str = "latest") -> dict:
        if version in self.metadata_cache:
            return self.metadata_cache[version]

        version_path = self.model_path / version
        metadata_file = version_path / "metadata.json"

        if not metadata_file.exists():
            return {
                "version": version,
                "trained_at": "unknown",
                "total_timesteps": 0,
                "mean_reward": 0.0,
            }

        with open(metadata_file) as f:
            metadata = json.load(f)

        self.metadata_cache[version] = metadata
        return metadata

    def list_versions(self) -> list[str]:
        versions: list[str] = []

        if not self.model_path.exists():
            return versions

        for item in self.model_path.iterdir():
            if item.is_dir() and (item / "cat_brain.zip").exists():
                if item.name not in EXCLUDED_DIRS:
                    versions.append(item.name)

        return sorted(versions, reverse=True)

    def unload_model(self, version: str) -> None:
        if version in self.models:
            del self.models[version]
            logger.info("model_unloaded", version=version)

    def reload_model(self, version: str) -> PPO:
        self.unload_model(version)
        if version in self.metadata_cache:
            del self.metadata_cache[version]
        return self.load_model(version)
