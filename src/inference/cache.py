import hashlib
from typing import Optional

import numpy as np

from src.core.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

HASH_DECIMALS: int = 3
HASH_LENGTH: int = 16
CACHE_KEY_PREFIX: str = "cat_action"


class PredictionCache:
    def __init__(self, config: Settings):
        self.enabled = config.CACHE_ENABLED
        self.ttl = config.CACHE_TTL
        self.redis = None

        if self.enabled:
            try:
                import redis

                self.redis = redis.from_url(config.REDIS_URL, decode_responses=True)
                self.redis.ping()
                logger.info("cache_connected", redis_url=config.REDIS_URL)
            except Exception as e:
                logger.warning("cache_connection_failed", error=str(e))
                self.redis = None
                self.enabled = False

    def _hash_observation(self, observation: np.ndarray) -> str:
        rounded = np.round(observation, decimals=HASH_DECIMALS)
        obs_bytes = rounded.tobytes()
        return hashlib.sha256(obs_bytes).hexdigest()[:HASH_LENGTH]

    async def get(self, observation: np.ndarray) -> Optional[int]:
        if not self.enabled or self.redis is None:
            return None

        try:
            key = f"{CACHE_KEY_PREFIX}:{self._hash_observation(observation)}"
            cached = self.redis.get(key)
            if cached is not None:
                return int(cached)
        except Exception as e:
            logger.warning("cache_get_error", error=str(e))

        return None

    async def set(self, observation: np.ndarray, action: int) -> None:
        if not self.enabled or self.redis is None:
            return

        try:
            key = f"{CACHE_KEY_PREFIX}:{self._hash_observation(observation)}"
            self.redis.setex(key, self.ttl, str(action))
        except Exception as e:
            logger.warning("cache_set_error", error=str(e))

    def is_available(self) -> bool:
        return self.enabled and self.redis is not None
