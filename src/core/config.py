from pydantic_settings import BaseSettings


PERSONALITY_CONFIG = {
    "balanced": {"hunger": 1.0, "energy": 1.0, "distance_food": 1.0, "distance_toy": 1.0},
    "lazy": {"hunger": 0.8, "energy": 1.5, "distance_food": 1.1, "distance_toy": 0.7},
    "foodie": {"hunger": 1.4, "energy": 0.7, "distance_food": 0.7, "distance_toy": 1.3},
    "playful": {"hunger": 0.7, "energy": 0.9, "distance_food": 1.2, "distance_toy": 0.6},
}


class PPOConfig:
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2


class TrainingConfig:
    CHECKPOINT_FREQ: int = 10000
    LOG_FREQ: int = 1000
    EVAL_EPISODES: int = 10
    REWARD_WINDOW: int = 100


class Settings(BaseSettings):
    MODEL_PATH: str = "./models"
    MODEL_VERSION: str = "latest"

    TOTAL_TIMESTEPS: int = 100_000

    BATCH_SIZE: int = 32
    BATCH_TIMEOUT: float = 0.1

    CACHE_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 300

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
