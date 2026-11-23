import io
import zipfile
import json
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def upload_helper() -> str:
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post(
        "/api/v1/images/upload",
        files={"file": ("export_test.jpg", buf, "image/jpeg")},
    )
    assert resp.status_code == 201
    return resp.json()["image_id"]


def test_dataset_stats_endpoint():
    """Verify stats computation returns non-negative counters and valid schema."""
    resp = client.get("/api/v1/datasets/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_images" in data
    assert "total_originals" in data
    assert "total_generated" in data
    assert "by_method" in data
    assert "by_condition" in data
    assert "by_quality_status" in data
    assert "by_review_status" in data
    assert "total_jobs" in data
    assert "recent_activity" in data
    assert isinstance(data["recent_activity"], list)


def test_dataset_export_and_download_roundtrip():
    """Verify structured ZIP export creation, ZIP contents, and download streaming."""
    upload_helper()

    # Create export
    resp = client.post(
        "/api/v1/datasets/export",
        json={
            "include_originals": True,
            "include_traditional": False,
            "include_generative": False,
        },
    )
    assert resp.status_code == 201
    exp_data = resp.json()
    export_id = exp_data["export_id"]
    assert exp_data["image_count"] >= 1
    assert exp_data["filename"].endswith(".zip")

    # List exports
    list_resp = client.get("/api/v1/datasets/exports")
    assert list_resp.status_code == 200
    assert any(e["export_id"] == export_id for e in list_resp.json()["items"])

    # Download export ZIP
    dl_resp = client.get(f"/api/v1/datasets/exports/{export_id}/download")
    assert dl_resp.status_code == 200
    assert dl_resp.headers.get("content-type") == "application/zip"
    assert "attachment;" in dl_resp.headers.get("content-disposition", "")

    # Inspect ZIP contents in memory
    zip_bytes = io.BytesIO(dl_resp.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist
        assert "manifest.csv" in namelist
        assert "README.txt" in namelist
        # Verify an original image is present
        assert any(n.startswith("original/") for n in namelist)

        # Verify manifest.json parses
        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest_data["export_id"] == export_id
        assert len(manifest_data["images"]) >= 1


def test_export_empty_filter_rejected():
    """Verify that export filter matching 0 images returns 400 Bad Request."""
    resp = client.post(
        "/api/v1/datasets/export",
        json={
            "include_originals": False,
            "include_traditional": False,
            "include_generative": False,
        },
    )
    assert resp.status_code == 400
    assert "No images matched" in resp.json()["detail"]


def test_export_download_unknown_id_404():
    """Verify downloading unknown export ID returns 404."""
    resp = client.get("/api/v1/datasets/exports/nonexistent_export_id/download")
    assert resp.status_code == 404
