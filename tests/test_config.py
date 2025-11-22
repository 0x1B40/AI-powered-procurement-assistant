from unittest.mock import patch
import pytest
from src.config import Settings, get_settings


class TestSettings:
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://test:27017",
        "MONGODB_DB": "test_db",
        "MONGODB_COLLECTION": "test_collection",
        "PRIMARY_LLM_API_KEY": "primary-key",
        "PRIMARY_LLM_MODEL": "my-primary-model",
        "PRIMARY_LLM_TEMPERATURE": "0.5",
    })
    def test_settings_from_env(self):
        settings = Settings()
        assert settings.mongodb_uri == "mongodb://test:27017"
        assert settings.mongodb_db == "test_db"
        assert settings.mongodb_collection == "test_collection"
        assert settings.primary_llm_api_key == "primary-key"
        assert settings.llm_api_key == "primary-key"
        assert settings.llm_model == "my-primary-model"
        assert settings.llm_temperature == 0.5

    def test_settings_defaults(self):
        with patch.dict("os.environ", {
            "MONGODB_URI": "mongodb://localhost:27017",
            "PRIMARY_LLM_API_KEY": "grok-key",
        }):
            settings = Settings()
            assert settings.mongodb_db == "california_procurement"
            assert settings.mongodb_collection == "purchase_orders"
            assert settings.llm_api_key == "grok-key"
            assert settings.llm_model == "grok-4-1-fast-non-reasoning"
            assert settings.llm_temperature == 0.1

    def test_get_settings_caching(self):
        with patch.dict("os.environ", {
            "MONGODB_URI": "mongodb://localhost:27017",
            "PRIMARY_LLM_API_KEY": "primary-key",
        }):
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2  # Should return the same cached instance
