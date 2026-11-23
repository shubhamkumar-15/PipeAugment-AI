from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_cors_allowed_origin():
    """Verify that requests from allowed origin receive the access-control-allow-origin header."""
    headers = {"Origin": "http://localhost:3000"}
    response = client.get("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_disallowed_origin():
    """Verify that untrusted origins do not receive an allow-origin header."""
    headers = {"Origin": "http://malicious-site.com"}
    response = client.get("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_options():
    """Verify CORS preflight OPTIONS request returns correct methods and headers."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/api/v1/augment/traditional", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    allowed_methods = response.headers.get("access-control-allow-methods", "")
    assert "POST" in allowed_methods
