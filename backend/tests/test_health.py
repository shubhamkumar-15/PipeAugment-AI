import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import Settings, get_settings

client = TestClient(app)


def test_health_endpoint_success_and_payload():
    """Verify GET /api/v1/health returns HTTP 200 and expected default JSON payload."""
    # Deterministic settings override to isolate from external environment variables
    deterministic_settings = Settings(
        APP_NAME="PipeAugment AI",
        APP_VERSION="0.1.0",
    )
    app.dependency_overrides[get_settings] = lambda: deterministic_settings
    try:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "app_name": "PipeAugment AI",
            "version": "0.1.0",
        }
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_isolation_from_env():
    """Verify that dependency overrides ensure deterministic configuration."""
    custom_settings = Settings(
        APP_NAME="Isolated Test App",
        APP_VERSION="1.2.3",
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings
    try:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "Isolated Test App"
        assert data["version"] == "1.2.3"
    finally:
        app.dependency_overrides.clear()
