import asyncio
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.config import Settings, PERSONALITY_CONFIG
from src.core.environment import EnvConstants, ObservationIndex
from src.inference.cache import PredictionCache
from src.inference.model_loader import ModelLoader
from src.utils.action_history import ActionHistory
from src.utils.logger import get_logger

logger = get_logger(__name__)

QUEUE_TIMEOUT: float = 1.0


class PersonalityModifier:
    """Modifies observation based on cat personality
    
    Only modifies hunger, energy, and distances.
    Mood and personality scores remain unchanged.
    """
    
    @staticmethod
    def apply(observation: np.ndarray, personality: str) -> np.ndarray:
        if personality not in PERSONALITY_CONFIG:
            return observation
        
        mods = PERSONALITY_CONFIG[personality]
        modified = observation.copy()
        

        modified[ObservationIndex.HUNGER] *= mods["hunger"]
        modified[ObservationIndex.ENERGY] *= mods["energy"]
        modified[ObservationIndex.DISTANCE_FOOD] *= mods["distance_food"]
        modified[ObservationIndex.DISTANCE_TOY] *= mods["distance_toy"]
        modified[ObservationIndex.DISTANCE_BED] *= mods.get("distance_bed", 1.0)
        

        

        modified[ObservationIndex.HUNGER] = np.clip(
            modified[ObservationIndex.HUNGER], 0, EnvConstants.MAX_HUNGER
        )
        modified[ObservationIndex.ENERGY] = np.clip(
            modified[ObservationIndex.ENERGY], 0, EnvConstants.MAX_ENERGY
        )
        modified[ObservationIndex.DISTANCE_FOOD] = np.clip(
            modified[ObservationIndex.DISTANCE_FOOD], 0, EnvConstants.MAX_DISTANCE
        )
        modified[ObservationIndex.DISTANCE_TOY] = np.clip(
            modified[ObservationIndex.DISTANCE_TOY], 0, EnvConstants.MAX_DISTANCE
        )
        modified[ObservationIndex.DISTANCE_BED] = np.clip(
            modified[ObservationIndex.DISTANCE_BED], 0, EnvConstants.MAX_DISTANCE
        )
        
        return modified


@dataclass
class PredictionRequest:
    observation: np.ndarray
    future: asyncio.Future[int]


class BatchPredictor:
    def __init__(
        self,
        model_loader: ModelLoader,
        config: Settings,
        action_history: Optional[ActionHistory] = None,
    ):
        self.model_loader = model_loader
        self.config = config
        self.action_history = action_history
        self.batch_size = config.BATCH_SIZE
        self.batch_timeout = config.BATCH_TIMEOUT

        self.batch_queue: asyncio.Queue[PredictionRequest] = asyncio.Queue()
        self.cache = PredictionCache(config)
        self._processor_task: Optional[asyncio.Task[None]] = None
        self._running = False

    def start(self) -> None:
        if self._processor_task is None or self._processor_task.done():
            self._running = True
            self._processor_task = asyncio.create_task(self._batch_processor())
            logger.info("batch_processor_started")

    def stop(self) -> None:
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            logger.info("batch_processor_stopped")

    async def predict_single(
        self,
        observation: np.ndarray,
        cat_id: Optional[str] = None,
        personality: str = "balanced",
        use_cache: bool = True,
    ) -> int:
        modified_obs = PersonalityModifier.apply(observation, personality)
        
        if use_cache:
            cached = await self.cache.get(modified_obs)
            if cached is not None:
                return cached

        model = self.model_loader.get_model(self.config.MODEL_VERSION, cat_id)
        if model is None:
            model = self.model_loader.load_model(self.config.MODEL_VERSION, cat_id)

        action, _ = model.predict(modified_obs, deterministic=True)
        action_int = int(action)

        if use_cache:
            await self.cache.set(modified_obs, action_int)
        

        if self.action_history and cat_id:
            self.action_history.log_action(cat_id, modified_obs, action_int)

        return action_int

    async def predict_batch(self, observations: list[np.ndarray]) -> list[int]:
        tasks = [self.predict_single(obs) for obs in observations]
        return await asyncio.gather(*tasks)

    async def _batch_processor(self) -> None:
        while self._running:
            try:
                requests: list[PredictionRequest] = []

                try:
                    first_request = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=QUEUE_TIMEOUT,
                    )
                    requests.append(first_request)
                except asyncio.TimeoutError:
                    continue

                deadline = asyncio.get_event_loop().time() + self.batch_timeout

                while len(requests) < self.batch_size:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break

                    try:
                        request = await asyncio.wait_for(
                            self.batch_queue.get(), timeout=remaining
                        )
                        requests.append(request)
                    except asyncio.TimeoutError:
                        break

                if requests:
                    await self._process_batch(requests)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_processor_error", error=str(e))

    async def _process_batch(self, requests: list[PredictionRequest]) -> None:
        try:
            observations = np.array([r.observation for r in requests])

            actions = await asyncio.to_thread(self._predict_sync, observations)

            for request, action in zip(requests, actions):
                if not request.future.done():
                    request.future.set_result(int(action))

        except Exception as e:
            logger.error("batch_prediction_error", error=str(e))
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)

    def _predict_sync(self, observations: np.ndarray) -> np.ndarray:
        model = self.model_loader.get_model(self.config.MODEL_VERSION)
        if model is None:
            model = self.model_loader.load_model(self.config.MODEL_VERSION)

        actions, _ = model.predict(observations, deterministic=True)
        return actions
