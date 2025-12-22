from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()

    async def mock_predict_single(obs):
        return 1

    async def mock_predict_batch(observations):
        return [1] * len(observations)

    predictor.predict_single = mock_predict_single
    predictor.predict_batch = mock_predict_batch
    predictor.cache = MagicMock()
    predictor.cache.is_available.return_value = False

    return predictor


@pytest.fixture
def mock_model_loader():
    loader = MagicMock()
    loader.get_model.return_value = MagicMock()
    loader.list_versions.return_value = ["20231201_120000", "latest"]
    loader.get_model_info.return_value = {
        "version": "latest",
        "trained_at": "2023-12-01T12:00:00",
        "total_timesteps": 100000,
        "mean_reward": 150.5,
        "algorithm": "PPO",
    }
    return loader


@pytest.fixture
def client(mock_predictor, mock_model_loader):
    app.state.predictor = mock_predictor
    app.state.model_loader = mock_model_loader
    app.state.settings = MagicMock()
    app.state.settings.MODEL_VERSION = "latest"
    app.state.start_time = 0

    return TestClient(app, raise_server_exceptions=False)


class TestPredictEndpoint:
    def test_predict_success(self, client):
        response = client.post(
            "/predict",
            json={
                "hunger": 50.0,
                "energy": 70.0,
                "distance_to_food": 3.5,
                "distance_to_toy": 7.2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert 0 <= data["action"] <= 3

    def test_predict_invalid_hunger(self, client):
        response = client.post(
            "/predict",
            json={
                "hunger": 150.0,
                "energy": 70.0,
                "distance_to_food": 3.5,
                "distance_to_toy": 7.2,
            },
        )

        assert response.status_code == 422

    def test_predict_missing_field(self, client):
        response = client.post(
            "/predict",
            json={"hunger": 50.0, "energy": 70.0},
        )

        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_predict_batch_success(self, client):
        response = client.post(
            "/predict_batch",
            json={
                "states": [
                    {"hunger": 50.0, "energy": 70.0, "distance_to_food": 3.5, "distance_to_toy": 7.2},
                    {"hunger": 80.0, "energy": 30.0, "distance_to_food": 1.0, "distance_to_toy": 5.0},
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "actions" in data
        assert len(data["actions"]) == 2

    def test_predict_batch_empty(self, client):
        response = client.post("/predict_batch", json={"states": []})

        assert response.status_code == 200
        data = response.json()
        assert data["actions"] == []


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_model_info(self, client):
        response = client.get("/models/latest")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "trained_at" in data
        assert "total_timesteps" in data
        assert "mean_reward" in data


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime" in data

    def test_liveness_check(self, client):
        response = client.get("/live")

        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
