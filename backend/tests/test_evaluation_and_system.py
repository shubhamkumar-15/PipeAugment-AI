import io
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def upload_helper() -> str:
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(80, 120, 160))
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = client.post(
        "/api/v1/images/upload",
        files={"file": ("review_test.png", buf, "image/png")},
    )
    assert resp.status_code == 201
    return resp.json()["image_id"]


def test_benchmark_results_endpoint():
    """Verify evaluation benchmark endpoint returns genuine Phase 7 results."""
    resp = client.get("/api/v1/evaluation/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "metadata" in data
    assert "experiments" in data
    assert "clean_test_summary" in data
    assert "overall_robustness_summary" in data
    assert "robustness_by_condition" in data
    assert data["metadata"]["test_sample_count"] == 60


def test_evaluation_chart_serving_and_allowlist():
    """Verify chart serving and rejection of non-allowlisted / traversal names."""
    # Allowlisted chart
    resp_chart = client.get("/api/v1/evaluation/charts/abc_clean_vs_stress_f1.png")
    assert resp_chart.status_code == 200
    assert resp_chart.headers.get("content-type") == "image/png"
    assert len(resp_chart.content) > 100

    # Non-allowlisted chart
    resp_bad = client.get("/api/v1/evaluation/charts/random_fake_chart.png")
    assert resp_bad.status_code == 404

    # Traversal attempt
    resp_trav = client.get("/api/v1/evaluation/charts/..%2f..%2fapp%2fmain.py")
    assert resp_trav.status_code == 404


def test_system_info_endpoint():
    """Verify system info endpoint returns real hardware, worker status, and capability descriptors."""
    resp = client.get("/api/v1/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app_name"] == "PipeAugment AI"
    assert "hardware" in data
    assert "gpu" in data["hardware"]
    assert "ai_worker" in data
    assert "capabilities" in data
    assert "augmentation_methods" in data["capabilities"]
    assert "traditional" in data["capabilities"]["augmentation_methods"]
    assert "generative" in data["capabilities"]["augmentation_methods"]
    assert "upload_limits" in data


def test_review_status_patch_and_persistence():
    """Verify human review status update PATCH endpoint."""
    img_id = upload_helper()

    # Update review status
    patch_resp = client.patch(
        f"/api/v1/images/{img_id}/review",
        json={
            "review_status": "approved",
            "reviewer": "Lead Pipeline Inspector",
            "review_notes": "Corrosion defect pattern and cylindrical seam fully intact.",
        },
    )
    assert patch_resp.status_code == 200
    meta = patch_resp.json()
    assert meta["review_status"] == "approved"
    assert meta["reviewer"] == "Lead Pipeline Inspector"
    assert "cylindrical seam" in meta["review_notes"]

    # Verify persistence via GET
    get_resp = client.get(f"/api/v1/images/{img_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["review_status"] == "approved"
