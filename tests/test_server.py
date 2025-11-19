import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.server import app


class TestServer:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch('src.server.agent.chat')
    def test_chat_endpoint_success(self, mock_chat):
        mock_chat.return_value = "Query result: [{\"count\": 150}]"

        payload = {"question": "How many purchase orders were created in Q3 2014?"}
        response = self.client.post("/chat", json=payload)

        assert response.status_code == 200
        assert response.json() == {"answer": "Query result: [{\"count\": 150}]"}
        mock_chat.assert_called_once_with("How many purchase orders were created in Q3 2014?")

    def test_chat_endpoint_validation_error(self):
        # Test with empty question
        payload = {"question": ""}
        response = self.client.post("/chat", json=payload)
        assert response.status_code == 422  # Validation error

        # Test with question too short
        payload = {"question": "Hi"}
        response = self.client.post("/chat", json=payload)
        assert response.status_code == 422  # Validation error

    @patch('src.server.agent.chat')
    def test_chat_endpoint_agent_error(self, mock_chat):
        mock_chat.side_effect = Exception("Agent processing failed")

        payload = {"question": "Test question"}
        response = self.client.post("/chat", json=payload)

        assert response.status_code == 500
        assert "Agent processing failed" in response.json()["detail"]

    def test_openapi_docs_available(self):
        response = self.client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_json_available(self):
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert "info" in data
        assert data["info"]["title"] == "Procurement AI Assistant"
