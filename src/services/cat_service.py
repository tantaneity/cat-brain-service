import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.inference.model_loader import ModelLoader
from src.training.trainer import CatBrainTrainer
from src.utils.action_history import ActionHistory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatAlreadyExistsError(Exception):

    pass


class CatNotFoundError(Exception):

    pass


class CatService:

    
    def __init__(
        self,
        trainer: CatBrainTrainer,
        model_loader: ModelLoader,
        action_history: ActionHistory,
    ):
        self.trainer = trainer
        self.model_loader = model_loader
        self.action_history = action_history
    
    def create_cat(self, cat_id: str, personality: str) -> dict:

        cat_brain_path = self._get_cat_brain_path(cat_id)
        
        if cat_brain_path.exists():
            raise CatAlreadyExistsError(
                f"Cat '{cat_id}' already exists. Use a different cat_id or delete the existing cat first."
            )
        
        brain_path = self.trainer.create_cat_brain(cat_id)
        
        return {
            "cat_id": cat_id,
            "personality": personality,
            "brain_path": str(brain_path),
            "created_at": datetime.now().isoformat(),
            "message": "Cat brain created successfully from default model",
        }
    
    def get_cat_info(self, cat_id: str) -> dict:

        cat_brain_path = self._get_cat_brain_path(cat_id)
        
        if not cat_brain_path.exists():
            raise CatNotFoundError(f"Cat '{cat_id}' not found")
        
        metadata_path = cat_brain_path.parent / "metadata.json"
        created_at = None
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                created_at = metadata.get("created_at")
        
        stats = self.action_history.get_history_stats(cat_id)
        
        return {
            "cat_id": cat_id,
            "model_path": str(cat_brain_path),
            "created_at": created_at,
            "total_actions": stats["total_actions"],
        }
    
    def cat_exists(self, cat_id: str) -> bool:

        return self._get_cat_brain_path(cat_id).exists()
    
    def _get_cat_brain_path(self, cat_id: str) -> Path:

        return self.model_loader.model_path / "cats" / cat_id / "latest" / "cat_brain.zip"
    
    def reload_cat_brain(self, cat_id: str) -> None:
        cat_brain_path = self._get_cat_brain_path(cat_id)
        
        if not cat_brain_path.exists():
            raise CatNotFoundError(f"Cat '{cat_id}' not found")
        
        self.model_loader.load_model_for_cat(cat_id)
        logger.info("cat_brain_reloaded", cat_id=cat_id)
