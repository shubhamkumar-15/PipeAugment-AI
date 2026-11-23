"""
Tests for Corrosion Analysis and Visual Layout Maps endpoints:
- POST /api/v1/images/{id}/analyze
- GET /api/v1/images/{id}/analysis
- GET /api/v1/images/{id}/maps/{map_type}
"""

import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4
from PIL import Image

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Sets up a test client with an isolated dataset directory."""
    settings = get_settings()
    monkeypatch.setattr(settings, "DATASET_ROOT", tmp_path / "datasets")
    (tmp_path / "datasets" / "original").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "metadata").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets" / "metadata" / "analysis").mkdir(parents=True, exist_ok=True)
    return TestClient(app)


def create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_analysis_not_found(client):
    """Verify 404 when requesting analysis for an image without precomputed analysis."""
    random_id = str(uuid4())
    res = client.get(f"/api/v1/images/{random_id}/analysis")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_invalid_map_type_rejected(client):
    """Verify 400 when requesting an unknown map type."""
    random_id = str(uuid4())
    res = client.get(f"/api/v1/images/{random_id}/maps/invalid_map_type")
    assert res.status_code == 400
    assert "Authorized" in res.json()["detail"]


def test_map_not_found(client):
    """Verify 404 when requesting an authorized map that has not yet been generated."""
    random_id = str(uuid4())
    res = client.get(f"/api/v1/images/{random_id}/maps/pipeline_canny")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_analyze_and_retrieve_roundtrip(client, tmp_path):
    """
    Uploads a test image, mocks worker execution, verifies POST /analyze,
    GET /analysis, and GET /maps/{map_type} responses.
    """
    settings = get_settings()
    # 1. Upload test image
    img_bytes = create_test_image_bytes()
    upload_res = client.post(
        "/api/v1/images/upload",
        files={"file": ("pipe_sample.jpg", img_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 201
    image_id = upload_res.json()["image_id"]

    analysis_dir = settings.DATASET_ROOT / "metadata" / "analysis" / str(image_id)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy map files
    for map_name in ["pipeline_canny.png", "corrosion_layout.png", "corrosion_overlay.png"]:
        dummy_png = Image.new("RGB", (100, 100), color=(10, 10, 10))
        dummy_png.save(analysis_dir / map_name, format="PNG")

    mock_analysis_payload = {
        "image_id": str(image_id),
        "predicted_class": "corroded",
        "confidence": 0.942,
        "corrosion_probability": 0.942,
        "non_corrosion_probability": 0.058,
        "severity_level": "Moderate",
        "rust_coverage_percent": 7.45,
        "model_checkpoint": "best_model.pth",
        "disclaimer": "Diagnostic CV screening assistance only.",
        "map_urls": {
            "pipeline_canny": f"/api/v1/images/{image_id}/maps/pipeline_canny",
            "corrosion_layout": f"/api/v1/images/{image_id}/maps/corrosion_layout",
            "corrosion_overlay": f"/api/v1/images/{image_id}/maps/corrosion_overlay",
        },
    }

    # Save mock analysis json to the location analyze_image.py would write
    with open(settings.DATASET_ROOT / "metadata" / "analysis" / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(mock_analysis_payload, f)

    # 2. Mock subprocess.run to simulate worker success
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(mock_analysis_payload)
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc):
        # Trigger POST /analyze
        analyze_res = client.post(f"/api/v1/images/{image_id}/analyze")
        assert analyze_res.status_code == 200
        data = analyze_res.json()
        assert data["image_id"] == str(image_id)
        assert data["predicted_class"] == "corroded"
        assert data["severity_level"] == "Moderate"
        assert data["rust_coverage_percent"] == 7.45
        assert "map_urls" in data

    # 3. Test GET /analysis retrieval
    get_res = client.get(f"/api/v1/images/{image_id}/analysis")
    assert get_res.status_code == 200
    assert get_res.json()["predicted_class"] == "corroded"

    # 4. Test GET /maps/{map_type} serving
    for map_type in ["pipeline_canny", "corrosion_layout", "corrosion_overlay"]:
        map_res = client.get(f"/api/v1/images/{image_id}/maps/{map_type}")
        assert map_res.status_code == 200
        assert map_res.headers["content-type"] == "image/png"
        assert len(map_res.content) > 0
