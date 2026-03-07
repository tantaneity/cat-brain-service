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
    version: int = 1

    @staticmethod
    def from_dict(data: dict) -> "CatProfile":
        return CatProfile(
            cat_id=data.get("cat_id", ""),
            personality=data.get("personality", "balanced"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            seed=int(data.get("seed", 0)),
            modifiers=data.get("modifiers", {}),
            version=int(data.get("version", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "cat_id": self.cat_id,
            "personality": self.personality,
            "created_at": self.created_at,
            "seed": self.seed,
            "modifiers": self.modifiers,
            "version": self.version,
        }


class CatProfileStore:
    CURRENT_PROFILE_VERSION = 2

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
            profile = self._upgrade_profile_if_needed(profile, profile_path)
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
        modifiers = self._generate_modifiers(seed, personality)
        profile = CatProfile(
            cat_id=cat_id,
            personality=personality,
            created_at=datetime.now().isoformat(),
            seed=seed,
            modifiers=modifiers,
            version=self.CURRENT_PROFILE_VERSION,
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

    def _generate_modifiers(self, seed: int, personality: str) -> dict:
        rng = random.Random(seed)
        profile = self._personality_centers(personality)
        return {
            "hunger": self._sample(rng, profile["hunger"], 0.08),
            "energy": self._sample(rng, profile["energy"], 0.1),
            "distance_food": self._sample(rng, profile["distance_food"], 0.1),
            "distance_toy": self._sample(rng, profile["distance_toy"], 0.12),
            "distance_bed": self._sample(rng, profile["distance_bed"], 0.08),
            "mood": self._sample(rng, profile["mood"], 0.08),
            "lazy_score": self._sample(rng, profile["lazy_score"], 0.16),
            "foodie_score": self._sample(rng, profile["foodie_score"], 0.16),
            "playful_score": self._sample(rng, profile["playful_score"], 0.16),
        }

    def _upgrade_profile_if_needed(self, profile: CatProfile, profile_path: Path) -> CatProfile:
        if profile.version >= self.CURRENT_PROFILE_VERSION:
            return profile

        upgraded = CatProfile(
            cat_id=profile.cat_id,
            personality=profile.personality,
            created_at=profile.created_at,
            seed=profile.seed,
            modifiers=self._generate_modifiers(profile.seed, profile.personality),
            version=self.CURRENT_PROFILE_VERSION,
        )
        self._save_profile(upgraded, profile_path)
        logger.info(
            "cat_profile_upgraded",
            cat_id=profile.cat_id,
            from_version=profile.version,
            to_version=self.CURRENT_PROFILE_VERSION,
        )
        return upgraded

    def _sample(
        self,
        rng: random.Random,
        center: float,
        spread: float,
        min_value: float = 0.6,
        max_value: float = 1.4,
    ) -> float:
        return round(max(min_value, min(max_value, rng.uniform(center - spread, center + spread))), 3)

    def _personality_centers(self, personality: str) -> dict[str, float]:
        key = (personality or "balanced").lower()
        presets = {
            "balanced": {
                "hunger": 1.0,
                "energy": 1.0,
                "distance_food": 1.0,
                "distance_toy": 1.0,
                "distance_bed": 1.0,
                "mood": 1.0,
                "lazy_score": 1.0,
                "foodie_score": 1.0,
                "playful_score": 1.0,
            },
            "lazy": {
                "hunger": 0.94,
                "energy": 1.16,
                "distance_food": 1.02,
                "distance_toy": 1.18,
                "distance_bed": 0.9,
                "mood": 0.98,
                "lazy_score": 1.28,
                "foodie_score": 0.92,
                "playful_score": 0.72,
            },
            "foodie": {
                "hunger": 1.18,
                "energy": 0.92,
                "distance_food": 0.78,
                "distance_toy": 1.1,
                "distance_bed": 1.0,
                "mood": 0.98,
                "lazy_score": 0.9,
                "foodie_score": 1.28,
                "playful_score": 0.86,
            },
            "playful": {
                "hunger": 0.92,
                "energy": 1.06,
                "distance_food": 1.08,
                "distance_toy": 0.74,
                "distance_bed": 1.06,
                "mood": 1.08,
                "lazy_score": 0.74,
                "foodie_score": 0.9,
                "playful_score": 1.3,
            },
        }
        return presets.get(key, presets["balanced"])
