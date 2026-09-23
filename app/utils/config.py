"""
app/utils/config.py
-------------------
Central configuration module.

Loads all settings from environment variables (via .env file).
Never exposes secrets through API responses or logs.
"""

import os
from dotenv import load_dotenv

# Load .env into the process environment (safe no-op if .env doesn't exist)
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Application
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "false").lower() == "true"
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # LLM — will be used in later phases
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def is_llm_configured(self) -> bool:
        """Return True only if an LLM API key has been provided."""
        return bool(self.LLM_API_KEY)


# Singleton — import this wherever config is needed
settings = Settings()
