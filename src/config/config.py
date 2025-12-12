"""
Configuration for Movie Recommendation System
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    # Project settings
    project_name: str = "Movie Recommendation System"
    port_number: int = 8000
    limiter_requests: int = 100

    # Base directories
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"

    # Server settings
    host: str = "0.0.0.0"
    debug: bool = True

    # API keys (chỉ dùng OpenRouter)
    openrouter_api_key: str

    # LLM settings (DSPy)
    model: str
    async_max_workers: int
    cache: bool
    num_retries: int
    max_tokens: int

    # NextItNet model configuration
    nextitnet_num_items: int
    nextitnet_embedding_dim: int
    nextitnet_num_blocks: int
    nextitnet_kernel_size: int
    nextitnet_dilations: list[int] = [1, 2, 4, 1, 2, 4]  # Default dilation pattern
    nextitnet_dropout: float

    # Recommendation settings
    default_top_k: int
    max_history_length: int
    min_history_length: int

    # Default active model
    default_model: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = ""
        case_sensitive = False


@lru_cache
def get_config() -> Settings:
    """Get cached configuration instance."""
    if os.environ.get("DOCKER_ENV"):
        return Settings()
    else:
        env_file = Path().resolve() / ".env"
        if env_file.exists():
            return Settings(_env_file=env_file)
        return Settings()
