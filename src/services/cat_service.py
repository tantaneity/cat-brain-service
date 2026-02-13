
from src.inference.model_loader import ModelLoader
from src.training.trainer import CatBrainTrainer
from src.services.cat_profile_store import CatProfileStore
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
        profile_store: CatProfileStore,
    ):
        self.trainer = trainer
        self.model_loader = model_loader
        self.action_history = action_history
        self.profile_store = profile_store
    
    def create_cat(self, cat_id: str, personality: str) -> dict:

        if self.profile_store.profile_exists(cat_id):
            raise CatAlreadyExistsError(
                f"Cat '{cat_id}' already exists. Use a different cat_id or delete the existing cat first."
            )

        profile = self.profile_store.create_profile(cat_id, personality)
        profile_path = self.profile_store.get_profile_path(cat_id)

        return {
            "cat_id": cat_id,
            "personality": personality,
            "brain_path": str(profile_path),
            "created_at": profile.created_at,
            "message": "Cat profile created successfully from base model",
        }
    
    def get_cat_info(self, cat_id: str) -> dict:

        profile = self.profile_store.get_profile(cat_id)
        if not profile:
            raise CatNotFoundError(f"Cat '{cat_id}' not found")
        
        stats = self.action_history.get_history_stats(cat_id)
        
        return {
            "cat_id": cat_id,
            "model_path": str(self.profile_store.get_profile_path(cat_id)),
            "created_at": profile.created_at,
            "total_actions": stats["total_actions"],
        }

    def get_profile_summary(self, cat_id: str) -> dict:

        profile = self.profile_store.get_profile(cat_id)
        if not profile:
            raise CatNotFoundError(f"Cat '{cat_id}' not found")
        
        return {
            "cat_id": profile.cat_id,
            "personality": profile.personality,
            "created_at": profile.created_at,
            "seed": profile.seed,
            "modifiers": profile.modifiers,
        }
    
    def cat_exists(self, cat_id: str) -> bool:

        return self.profile_store.profile_exists(cat_id)
    
    def reload_cat_brain(self, cat_id: str) -> None:
        if not self.profile_store.profile_exists(cat_id):
            raise CatNotFoundError(f"Cat '{cat_id}' not found")
        
        self.model_loader.reload_model(self.model_loader.default_version)
        logger.info("cat_brain_reloaded", cat_id=cat_id)
