import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    project_name: str
    port_number: int
    limiter_requests: int

    # LLM API keys
    gemini_api_key: str
    openrouter_api_key: str

    # LLM settings
    model: str
    async_max_workers: int
    cache: bool
    num_retries: int
    max_tokens: int


@lru_cache
def get_config() -> Settings:
    if os.environ.get("DOCKER_ENV"):
        return Settings()
    else:
        env_file = Path().resolve() / ".env"
        return Settings(_env_file=env_file)
