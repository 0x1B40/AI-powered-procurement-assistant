from unittest.mock import patch
import pytest
from src.config import Settings, get_settings


class TestSettings:
    @patch.dict("os.environ", {
        "MONGODB_URI": "mongodb://test:27017",
        "MONGODB_DB": "test_db",
        "MONGODB_COLLECTION": "test_collection",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4",
        "OPENAI_TEMPERATURE": "0.5"
    })
    def test_settings_from_env(self):
        settings = Settings()
        assert settings.mongodb_uri == "mongodb://test:27017"
        assert settings.mongodb_db == "test_db"
        assert settings.mongodb_collection == "test_collection"
        assert settings.openai_api_key == "test-key"
        assert settings.openai_model == "gpt-4"
        assert settings.openai_temperature == 0.5

    def test_settings_defaults(self):
        with patch.dict("os.environ", {
            "MONGODB_URI": "mongodb://localhost:27017",
            "OPENAI_API_KEY": "test-key"
        }):
            settings = Settings()
            assert settings.mongodb_db == "california_procurement"
            assert settings.mongodb_collection == "purchase_orders"
            assert settings.openai_model == "gpt-4o-mini"
            assert settings.openai_temperature == 0.1

    def test_get_settings_caching(self):
        with patch.dict("os.environ", {
            "MONGODB_URI": "mongodb://localhost:27017",
            "OPENAI_API_KEY": "test-key"
        }):
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2  # Should return the same cached instance
