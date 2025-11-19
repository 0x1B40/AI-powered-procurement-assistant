from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_uri: str = Field(env="MONGODB_URI")
    mongodb_db: str = Field(env="MONGODB_DB", default="california_procurement")
    mongodb_collection: str = Field(env="MONGODB_COLLECTION", default="purchase_orders")

    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(env="OPENAI_MODEL", default="gpt-4o-mini")
    openai_temperature: float = Field(env="OPENAI_TEMPERATURE", default=0.1)

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

