import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.config import Settings
from src.inference.model_loader import ModelLoader
from src.inference.predictor import BatchPredictor


@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.MODEL_PATH = "./test_models"
    settings.MODEL_VERSION = "test"
    settings.BATCH_SIZE = 4
    settings.BATCH_TIMEOUT = 0.05
    settings.CACHE_ENABLED = False
    return settings


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = (np.array([1]), None)
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    loader = MagicMock(spec=ModelLoader)
    loader.get_model.return_value = mock_model
    loader.load_model.return_value = mock_model
    return loader


class TestBatchPredictor:
    @pytest.mark.asyncio
    async def test_predict_single(self, mock_model_loader, mock_settings, mock_model):
        mock_model.predict.return_value = (np.array([2]), None)

        predictor = BatchPredictor(mock_model_loader, mock_settings)
        predictor.start()

        await asyncio.sleep(0.01)

        obs = np.array([50.0, 50.0, 5.0, 5.0], dtype=np.float32)
        action = await predictor.predict_single(obs)

        assert action == 2
        predictor.stop()

    @pytest.mark.asyncio
    async def test_predict_batch(self, mock_model_loader, mock_settings, mock_model):
        mock_model.predict.return_value = (np.array([1, 2, 3]), None)

        predictor = BatchPredictor(mock_model_loader, mock_settings)
        predictor.start()

        await asyncio.sleep(0.01)

        observations = [
            np.array([50.0, 50.0, 5.0, 5.0], dtype=np.float32),
            np.array([70.0, 30.0, 2.0, 8.0], dtype=np.float32),
            np.array([20.0, 80.0, 7.0, 3.0], dtype=np.float32),
        ]

        actions = await predictor.predict_batch(observations)

        assert len(actions) == 3
        predictor.stop()

    @pytest.mark.asyncio
    async def test_batch_timeout(self, mock_model_loader, mock_settings, mock_model):
        mock_settings.BATCH_SIZE = 100
        mock_settings.BATCH_TIMEOUT = 0.02
        mock_model.predict.return_value = (np.array([1]), None)

        predictor = BatchPredictor(mock_model_loader, mock_settings)
        predictor.start()

        await asyncio.sleep(0.01)

        obs = np.array([50.0, 50.0, 5.0, 5.0], dtype=np.float32)

        start = asyncio.get_event_loop().time()
        action = await predictor.predict_single(obs)
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 1.0
        assert action == 1
        predictor.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_model_loader, mock_settings):
        predictor = BatchPredictor(mock_model_loader, mock_settings)

        assert predictor._processor_task is None

        predictor.start()
        assert predictor._processor_task is not None
        assert predictor._running is True

        predictor.stop()
        assert predictor._running is False
