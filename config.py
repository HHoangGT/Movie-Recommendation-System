"""
Configuration for Production Deployment
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for the production deployment."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    # API Keys (from environment)
    openrouter_api_key: str | None = None

    # LLM Model
    model: str = "google/gemini-2.0-flash-001"

    # Model settings
    default_model: str = "llm"  # 'nextitnet' or 'llm'

    def __post_init__(self):
        # Load from environment
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = os.environ.get("MODEL", "openrouter/google/gemini-3.0-pro-preview")
        self.port = int(os.environ.get("PORT", 5000))
        self.debug = os.environ.get("DEBUG", "true").lower() == "true"
