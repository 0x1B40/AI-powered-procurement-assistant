import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.utils.data_loader import snake_case, normalize_chunk, sanitize_currency_series


class TestDataLoader:
    def test_snake_case_basic(self):
        assert snake_case("Purchase Order") == "purchase_order"
        assert snake_case("PO Number") == "po_number"
        assert snake_case("Total Amount") == "total_amount"

    def test_snake_case_special_chars(self):
        assert snake_case("Item/Description") == "item_description"
        assert snake_case("Qtr-Year") == "qtr_year"
        assert snake_case("Multi  Space") == "multi_space"

    def test_snake_case_double_underscore(self):
        assert snake_case("Item__Description") == "item_description"

    def test_normalize_chunk(self):
        # Create test data
        test_data = {
            "Purchase Order": ["PO001", "PO002"],
            "Total Amount": [100.50, 200.75],
            "Item Description": ["Item A", "Item B"],
            "NaN Column": [None, float('nan')]
        }
        df = pd.DataFrame(test_data)

        result = list(normalize_chunk(df))

        expected = [
            {
                "purchase_order": "PO001",
                "total_amount": 100.50,
                "item_description": "Item A",
                "nan_column": None
            },
            {
                "purchase_order": "PO002",
                "total_amount": 200.75,
                "item_description": "Item B",
                "nan_column": None
            }
        ]

        # The function should convert all NaN/None values to None
        assert result[0]["nan_column"] is None  # First row has explicit None
        assert result[1]["nan_column"] is None  # Second row has NaN converted to None

        assert result == expected

    def test_normalize_chunk_currency(self):
        df = pd.DataFrame(
            {
                "Total Price": ["$1,200.00", "3,500", None, "invalid"],
                "Unit Price": ["$10.00", "$5", "1,234.56", None],
            }
        )

        result = list(normalize_chunk(df))

        assert result[0]["total_price"] == 1200.0
        assert result[1]["total_price"] == 3500.0
        assert result[2]["total_price"] is None
        assert result[3]["total_price"] is None

        assert result[0]["unit_price"] == 10.0
        assert result[1]["unit_price"] == 5.0
        assert result[2]["unit_price"] == 1234.56
        assert result[3]["unit_price"] is None

    def test_sanitize_currency_series(self):
        series = pd.Series(["$1,000.00", "  42 ", None, ""])
        cleaned = sanitize_currency_series(series)
        assert cleaned.tolist()[0] == 1000.0
        assert cleaned.tolist()[1] == 42.0
        assert pd.isna(cleaned.tolist()[2])
        assert pd.isna(cleaned.tolist()[3])

    @patch('src.data_loader.MongoClient')
    @patch('src.data_loader.get_settings')
    def test_load_csv_integration(self, mock_get_settings, mock_mongo_client):
        # This is a basic integration test structure
        # In a real scenario, you'd mock the entire MongoDB interaction
        mock_settings = MagicMock()
        mock_settings.mongodb_uri = "mongodb://test:27017"
        mock_settings.mongodb_db = "test_db"
        mock_settings.mongodb_collection = "test_collection"
        mock_get_settings.return_value = mock_settings

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_mongo_client.return_value = mock_client

        # Test that the function can be called without errors
        # (Full integration test would require actual CSV file and MongoDB)
        from src.utils.data_loader import load_csv
        from pathlib import Path

        # This would normally load a CSV, but we're just testing the setup
        # In practice, you'd create a temporary CSV file for testing
        pass
