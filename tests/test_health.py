"""
tests/test_health.py
--------------------
Basic test suite for the Phase 1 health endpoint.

Uses FastAPI's built-in TestClient (backed by httpx) to make real HTTP
requests against the application without spinning up a live server.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        """Health endpoint must return HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self):
        """Response body must contain status: healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_service_name(self):
        """Response body must identify the correct service."""
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "Autonomous Data Quality Investigation Agent"

    def test_health_agents_not_active(self):
        """agents_active must be False in Phase 1."""
        response = client.get("/health")
        data = response.json()
        assert data["agents_active"] is False

    def test_health_content_type_is_json(self):
        """Response Content-Type must be application/json."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]
