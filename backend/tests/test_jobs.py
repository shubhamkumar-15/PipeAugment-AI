import io
import time
from uuid import uuid4
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.services.job_manager import JobManager, reset_job_manager_for_testing

client = TestClient(app)


def make_tiny_jpeg() -> io.BytesIO:
    """Helper creating a minimal valid JPEG image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(180, 100, 50))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def upload_sample_image() -> str:
    """Uploads a test image and returns its UUID string."""
    buf = make_tiny_jpeg()
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("test_parent.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 201
    return response.json()["image_id"]


def test_unknown_parent_returns_404():
    """Verify that submitting a job with non-existent parent UUID returns 404."""
    unknown_id = str(uuid4())
    # Traditional
    resp_trad = client.post(
        "/api/v1/augment/traditional",
        json={"parent_image_ids": [unknown_id]},
    )
    assert resp_trad.status_code == 404
    assert f"'{unknown_id}' not found" in resp_trad.json()["detail"]

    # Generative
    resp_gen = client.post(
        "/api/v1/augment/generative",
        json={"parent_image_ids": [unknown_id]},
    )
    assert resp_gen.status_code == 404


def test_job_submission_and_lifecycle():
    """Verify traditional and generative job submission, queueing, and status retrieval."""
    parent_id = upload_sample_image()

    # Traditional job submission
    resp_trad = client.post(
        "/api/v1/augment/traditional",
        json={
            "parent_image_ids": [parent_id],
            "conditions": ["brightness"],
            "intensity": 0.5,
            "variations_per_image": 1,
            "seed": 42,
        },
    )
    assert resp_trad.status_code == 202
    job_data = resp_trad.json()
    job_id = job_data["job_id"]
    assert job_data["job_type"] == "traditional"
    assert job_data["total_items"] == 1
    assert job_data["status"] in ("queued", "running", "completed")

    # Get job status
    get_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["job_id"] == job_id

    # List jobs
    list_resp = client.get("/api/v1/jobs?limit=10")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] >= 1
    assert any(j["job_id"] == job_id for j in list_resp.json()["items"])


def test_mock_worker_execution_and_retry():
    """Verify mock runner execution, atomic status completion, and failed item retry."""
    settings = get_settings()
    parent_id = upload_sample_image()

    def mock_runner(job_spec, spec_file):
        """Simulates worker writing completed status with 1 failed item."""
        status_file = spec_file.parent / f"{job_spec['job_id']}.json"
        status_data = {
            "job_id": job_spec["job_id"],
            "job_type": job_spec["job_type"],
            "status": "partial_failed",
            "total_items": 1,
            "completed_items": 0,
            "failed_items": 1,
            "parent_image_ids": job_spec["parent_image_ids"],
            "config": job_spec["config"],
            "created_at": "2026-11-22T20:00:00Z",
            "started_at": "2026-11-22T20:00:01Z",
            "finished_at": "2026-11-22T20:00:02Z",
            "generated_image_ids": [],
            "item_results": [
                {
                    "parent_image_id": parent_id,
                    "status": "failed",
                    "generated_image_ids": [],
                    "error": "Simulated hardware timeout",
                }
            ],
            "error_message": "1 item failed",
        }
        import json
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f)

    # Inject mock manager
    test_manager = JobManager(settings=settings, runner_fn=mock_runner)
    reset_job_manager_for_testing(test_manager)

    try:
        # Submit job
        resp = client.post(
            "/api/v1/augment/traditional",
            json={"parent_image_ids": [parent_id]},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Wait briefly for mock runner execution
        time.sleep(0.3)

        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "partial_failed"
        assert status_resp.json()["failed_items"] == 1

        # Test retry endpoint
        retry_resp = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert retry_resp.status_code == 202
        new_job_id = retry_resp.json()["job_id"]
        assert new_job_id != job_id
        assert retry_resp.json()["total_items"] == 1

    finally:
        # Reset back to default manager
        reset_job_manager_for_testing(None)


def test_generative_modes_submission_and_canonicalization():
    """Verify that both controlnet_canny and img2img (and aliases) are accepted and canonicalized."""
    parent_id = upload_sample_image()

    # 1. Plain img2img
    resp1 = client.post(
        "/api/v1/augment/generative",
        json={
            "parent_image_ids": [parent_id],
            "mode": "img2img",
            "conditions": ["low_light_inspection"],
            "strength": 0.35,
        },
    )
    assert resp1.status_code == 202
    assert resp1.json()["config"]["mode"] == "img2img"

    # 2. plain_img2img alias canonicalizes to img2img
    resp2 = client.post(
        "/api/v1/augment/generative",
        json={
            "parent_image_ids": [parent_id],
            "mode": "plain_img2img",
            "conditions": ["bright_daylight"],
        },
    )
    assert resp2.status_code == 202
    assert resp2.json()["config"]["mode"] == "img2img"

    # 3. ControlNet Canny
    resp3 = client.post(
        "/api/v1/augment/generative",
        json={
            "parent_image_ids": [parent_id],
            "mode": "controlnet_canny",
            "conditions": ["directional_industrial_lighting"],
            "control_scale": 0.8,
        },
    )
    assert resp3.status_code == 202
    assert resp3.json()["config"]["mode"] == "controlnet_canny"

    # 4. controlnet_img2img alias canonicalizes to controlnet_canny
    resp4 = client.post(
        "/api/v1/augment/generative",
        json={
            "parent_image_ids": [parent_id],
            "mode": "controlnet_img2img",
        },
    )
    assert resp4.status_code == 202
    assert resp4.json()["config"]["mode"] == "controlnet_canny"

    # 5. Invalid mode rejected with 422
    resp_invalid = client.post(
        "/api/v1/augment/generative",
        json={
            "parent_image_ids": [parent_id],
            "mode": "unsupported_xyz",
        },
    )
    assert resp_invalid.status_code == 422
