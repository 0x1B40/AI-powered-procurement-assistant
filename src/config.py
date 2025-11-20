from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_uri: str = Field(env="MONGODB_URI")
    mongodb_db: str = Field(env="MONGODB_DB", default="california_procurement")
    mongodb_collection: str = Field(env="MONGODB_COLLECTION", default="purchase_orders")

    grok_api_key: str = Field(env="GROK_API_KEY")
    grok_model: str = Field(env="GROK_MODEL", default="grok-4-1-fast-non-reasoning")
    grok_temperature: float = Field(env="GROK_TEMPERATURE", default=0.1)

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

