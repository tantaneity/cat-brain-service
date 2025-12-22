from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = "./models"
    MODEL_VERSION: str = "latest"

    TOTAL_TIMESTEPS: int = 100_000

    BATCH_SIZE: int = 32
    BATCH_TIMEOUT: float = 0.1

    CACHE_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 60

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
