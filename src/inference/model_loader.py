import json
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO

from src.core.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXCLUDED_DIRS: set[str] = {"checkpoints", "tensorboard"}
DEFAULT_BRAIN_KEY: str = "__default__"


class ModelLoader:
    def __init__(self, config: Settings):
        self.config = config
        self.model_path = Path(config.MODEL_PATH)
        self.models: dict[str, PPO] = {}
        self.metadata_cache: dict[str, dict] = {}
        self.default_version = config.MODEL_VERSION

    def load_model(self, version: str = "latest", cat_id: Optional[str] = None) -> PPO:
        model_key = self._get_model_key(version)
        
        if model_key in self.models:
            return self.models[model_key]

        version_path = self.model_path / version
        model_file = version_path / "cat_brain.zip"

        if not model_file.exists():
            logger.error(
                "model_not_found",
                version=version,
                cat_id=cat_id,
                path=str(model_file),
            )
            raise FileNotFoundError(
                f"Model not found at {model_file}. "
                f"Please train a model first or check MODEL_PATH and MODEL_VERSION settings."
            )

        try:
            model = PPO.load(str(model_file))
            self.models[model_key] = model

            logger.info(
                "model_loaded",
                version=version,
                cat_id=cat_id,
                path=str(model_file),
            )

            return model
        except Exception as e:
            logger.error(
                "model_load_failed",
                version=version,
                cat_id=cat_id,
                path=str(model_file),
                error=str(e),
            )
            raise

    def _get_model_key(self, version: str) -> str:
        return f"{DEFAULT_BRAIN_KEY}:{version}"

    def get_model(self, version: str = "latest", cat_id: Optional[str] = None) -> Optional[PPO]:
        model_key = self._get_model_key(version)
        if model_key in self.models:
            return self.models[model_key]
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
    
    def load_model_for_cat(self, cat_id: str) -> PPO:
        return self.load_model("latest")
