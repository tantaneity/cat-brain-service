import json
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CatProfile:
    cat_id: str
    personality: str
    created_at: str
    seed: int
    modifiers: dict

    @staticmethod
    def from_dict(data: dict) -> "CatProfile":
        return CatProfile(
            cat_id=data.get("cat_id", ""),
            personality=data.get("personality", "balanced"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            seed=int(data.get("seed", 0)),
            modifiers=data.get("modifiers", {}),
        )

    def to_dict(self) -> dict:
        return {
            "cat_id": self.cat_id,
            "personality": self.personality,
            "created_at": self.created_at,
            "seed": self.seed,
            "modifiers": self.modifiers,
        }


class CatProfileStore:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, CatProfile] = {}

    def profile_exists(self, cat_id: str) -> bool:
        return self._get_profile_path(cat_id).exists()

    def get_profile(self, cat_id: str) -> Optional[CatProfile]:
        cached = self._cache.get(cat_id)
        if cached:
            return cached
        profile_path = self._get_profile_path(cat_id)
        if not profile_path.exists():
            return None
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = CatProfile.from_dict(data)
            self._cache[cat_id] = profile
            return profile
        except Exception as e:
            logger.error("cat_profile_load_failed", cat_id=cat_id, error=str(e))
            return None

    def create_profile(self, cat_id: str, personality: str) -> CatProfile:
        profile_path = self._get_profile_path(cat_id)
        if profile_path.exists():
            raise FileExistsError(f"Profile for {cat_id} already exists")

        profile_path.parent.mkdir(parents=True, exist_ok=True)
        seed = self._seed_from_cat_id(cat_id)
        modifiers = self._generate_modifiers(seed)
        profile = CatProfile(
            cat_id=cat_id,
            personality=personality,
            created_at=datetime.now().isoformat(),
            seed=seed,
            modifiers=modifiers,
        )
        self._save_profile(profile, profile_path)
        self._cache[cat_id] = profile
        logger.info("cat_profile_created", cat_id=cat_id, path=str(profile_path))
        return profile

    def ensure_profile(self, cat_id: str, personality: str) -> CatProfile:
        existing = self.get_profile(cat_id)
        if existing:
            return existing
        return self.create_profile(cat_id, personality)

    def get_profile_path(self, cat_id: str) -> Path:
        return self._get_profile_path(cat_id)

    def _save_profile(self, profile: CatProfile, path: Path) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("cat_profile_save_failed", cat_id=profile.cat_id, error=str(e))
            raise

    def _get_profile_path(self, cat_id: str) -> Path:
        return self.base_path / cat_id / "profile.json"

    def _seed_from_cat_id(self, cat_id: str) -> int:
        digest = hashlib.sha256(cat_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _generate_modifiers(self, seed: int) -> dict:
        rng = random.Random(seed)
        def scale(min_val: float = 0.92, max_val: float = 1.08) -> float:
            return round(rng.uniform(min_val, max_val), 3)

        return {
            "hunger": scale(),
            "energy": scale(),
            "distance_food": scale(),
            "distance_toy": scale(),
            "distance_bed": scale(),
            "mood": scale(),
            "lazy_score": scale(),
            "foodie_score": scale(),
            "playful_score": scale(),
        }
