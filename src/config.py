from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Resolve the `.env` file once and only load it when we can actually read it.
# Sandboxed CI environments sometimes block file access, which would otherwise
# crash Pydantic before the app has a chance to start.
_ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"
_ENV_FILE = _ENV_FILE_PATH if (_ENV_FILE_PATH.exists() and os.access(_ENV_FILE_PATH, os.R_OK)) else None


class Settings(BaseSettings):
    mongodb_uri: str = Field(env="MONGODB_URI")
    mongodb_db: str = Field(env="MONGODB_DB", default="california_procurement")
    mongodb_collection: str = Field(env="MONGODB_COLLECTION", default="purchase_orders")

    primary_llm_api_key: str | None = Field(default=None, env="PRIMARY_LLM_API_KEY")
    primary_llm_model: str = Field(default="grok-4-1-fast-non-reasoning", env="PRIMARY_LLM_MODEL")
    primary_llm_temperature: float = Field(default=0.1, env="PRIMARY_LLM_TEMPERATURE")
    primary_llm_base_url: str | None = Field(default=None, env="PRIMARY_LLM_BASE_URL")

    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def llm_api_key(self) -> str:
        """Return the API key the rest of the app should use for LLM calls."""
        if self.primary_llm_api_key:
            return self.primary_llm_api_key
        raise ValueError("Missing LLM API key. Set PRIMARY_LLM_API_KEY.")

    @property
    def llm_model(self) -> str:
        """Expose the configured model name (defaults to the Grok-compatible one)."""
        return self.primary_llm_model

    @property
    def llm_temperature(self) -> float:
        """Expose the configured sampling temperature."""
        return self.primary_llm_temperature

    @property
    def llm_base_url(self) -> str | None:
        """Optional override so we can talk to non-OpenAI-compatible endpoints (e.g., xAI)."""
        return self.primary_llm_base_url


@lru_cache
def get_settings() -> Settings:
    return Settings()

