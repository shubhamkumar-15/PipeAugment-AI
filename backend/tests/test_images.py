import hashlib
import io
import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings, get_settings
from app.main import app

client = TestClient(app)


def make_image_bytes(img_format: str = "JPEG", size: tuple[int, int] = (16, 16), color: tuple[int, int, int] = (200, 100, 50)) -> bytes:
    """Helper to generate valid in-memory image bytes."""
    buf = io.BytesIO()
    # JPEG does not support RGBA, so use RGB
    mode = "RGBA" if img_format in ("PNG", "WEBP") else "RGB"
    img = Image.new(mode, size, color)
    img.save(buf, format=img_format)
    return buf.getvalue()


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up a temporary isolated dataset directory for tests."""
    dataset_root = tmp_path / "datasets"
    test_settings = Settings(
        DATASET_ROOT=dataset_root,
        MAX_UPLOAD_BYTES=10 * 1024 * 1024,
        MAX_IMAGE_PIXELS=25_000_000,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.clear()


# 1. Health endpoint still works
def test_health_endpoint_still_works():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "PipeAugment AI"
    assert data["version"] == "0.1.0"


# 2. Valid JPEG upload
def test_valid_jpeg_upload(test_env):
    jpeg_bytes = make_image_bytes("JPEG", size=(20, 30))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["image_format"] == "JPEG"
    assert data["content_type"] == "image/jpeg"
    assert data["width"] == 20
    assert data["height"] == 30
    assert data["size_bytes"] == len(jpeg_bytes)
    assert data["sha256"] == hashlib.sha256(jpeg_bytes).hexdigest()
    assert data["original_filename"] == "sample.jpg"
    assert data["stored_filename"].endswith(".jpg")
    assert data["relative_path"] == f"original/{data['stored_filename']}"


# 3. Valid PNG upload
def test_valid_png_upload(test_env):
    png_bytes = make_image_bytes("PNG", size=(15, 25))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("sample.png", png_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["image_format"] == "PNG"
    assert data["content_type"] == "image/png"
    assert data["width"] == 15
    assert data["height"] == 25
    assert data["stored_filename"].endswith(".png")


# 4. Valid WEBP upload
def test_valid_webp_upload(test_env):
    webp_bytes = make_image_bytes("WEBP", size=(40, 40))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("sample.webp", webp_bytes, "image/webp")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["image_format"] == "WEBP"
    assert data["content_type"] == "image/webp"
    assert data["width"] == 40
    assert data["height"] == 40
    assert data["stored_filename"].endswith(".webp")


# 5. Stored original bytes exactly match uploaded bytes
def test_stored_original_bytes_exact_match(test_env):
    raw_bytes = make_image_bytes("JPEG", size=(50, 50))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("match.jpg", raw_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    stored_filename = response.json()["stored_filename"]
    stored_path = test_env.DATASET_ROOT / "original" / stored_filename
    assert stored_path.is_file()
    assert stored_path.read_bytes() == raw_bytes


# 6. Metadata dimensions, SHA-256, and byte count are correct
def test_metadata_fields_correct(test_env):
    w, h = 64, 48
    raw_bytes = make_image_bytes("PNG", size=(w, h))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("meta_test.png", raw_bytes, "image/png")},
    )
    assert response.status_code == 201
    meta = response.json()
    assert meta["width"] == w
    assert meta["height"] == h
    assert meta["size_bytes"] == len(raw_bytes)
    assert meta["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert meta["source_type"] == "user_upload"
    assert meta["annotation_status"] == "unannotated"
    assert meta["split"] == "unassigned"


# 7. Empty upload rejected
def test_empty_upload_rejected(test_env):
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


# 8. Text renamed to .jpg rejected
def test_text_renamed_to_jpg_rejected(test_env):
    text_content = b"This is plain text pretending to be a JPG image file."
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("fake.jpg", text_content, "image/jpeg")},
    )
    assert response.status_code == 422


# 9. Unsupported actual format rejected
def test_unsupported_actual_format_rejected(test_env):
    buf = io.BytesIO()
    bmp_img = Image.new("RGB", (20, 20), (0, 255, 0))
    bmp_img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    # Uploaded with .bmp extension
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("test.bmp", bmp_bytes, "image/bmp")},
    )
    assert response.status_code == 415

    # Uploaded with .jpg extension but BMP content
    response2 = client.post(
        "/api/v1/images/upload",
        files={"file": ("test.jpg", bmp_bytes, "image/jpeg")},
    )
    assert response2.status_code == 415


# 10. Filename-format mismatch rejected
def test_filename_format_mismatch_rejected(test_env):
    # Valid PNG bytes with .jpg extension
    png_bytes = make_image_bytes("PNG", size=(20, 20))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("mismatch.jpg", png_bytes, "image/jpeg")},
    )
    assert response.status_code == 415
    assert "extension" in response.json()["detail"].lower()


# 11. Byte-size limit enforced using small overridden test limit
def test_byte_size_limit_enforced(tmp_path: Path):
    small_limit_settings = Settings(
        DATASET_ROOT=tmp_path / "datasets",
        MAX_UPLOAD_BYTES=100,  # small limit: 100 bytes
        MAX_IMAGE_PIXELS=25_000_000,
    )
    app.dependency_overrides[get_settings] = lambda: small_limit_settings
    try:
        large_bytes = make_image_bytes("PNG", size=(60, 60))
        assert len(large_bytes) > 100
        response = client.post(
            "/api/v1/images/upload",
            files={"file": ("oversize.png", large_bytes, "image/png")},
        )
        assert response.status_code == 413
    finally:
        app.dependency_overrides.clear()


# 12. Pixel limit enforced using small overridden test limit
def test_pixel_limit_enforced(tmp_path: Path):
    small_pixel_settings = Settings(
        DATASET_ROOT=tmp_path / "datasets",
        MAX_UPLOAD_BYTES=10 * 1024 * 1024,
        MAX_IMAGE_PIXELS=100,  # limit to 100 pixels
    )
    app.dependency_overrides[get_settings] = lambda: small_pixel_settings
    try:
        # 16x16 = 256 pixels > 100
        img_bytes = make_image_bytes("PNG", size=(16, 16))
        response = client.post(
            "/api/v1/images/upload",
            files={"file": ("large_pixels.png", img_bytes, "image/png")},
        )
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "pixel" in detail or "decompression" in detail
    finally:
        app.dependency_overrides.clear()


# 13. Animated image rejected
def test_animated_image_rejected(test_env):
    # Create animated WEBP with 2 frames
    buf = io.BytesIO()
    frame1 = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
    frame2 = Image.new("RGBA", (20, 20), (0, 255, 0, 255))
    frame1.save(buf, format="WEBP", save_all=True, append_images=[frame2], duration=100)
    animated_bytes = buf.getvalue()

    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("animated.webp", animated_bytes, "image/webp")},
    )
    assert response.status_code == 422
    assert "animated" in response.json()["detail"].lower()


# 14. Listing and pagination
def test_listing_and_pagination(test_env):
    # Upload 3 distinct images
    ids = []
    for i in range(3):
        raw = make_image_bytes("PNG", size=(10 + i, 10 + i))
        res = client.post(
            "/api/v1/images/upload",
            files={"file": (f"img_{i}.png", raw, "image/png")},
        )
        assert res.status_code == 201
        ids.append(res.json()["image_id"])

    # Page 1 (limit 2, offset 0)
    res_page1 = client.get("/api/v1/images?limit=2&offset=0")
    assert res_page1.status_code == 200
    data1 = res_page1.json()
    assert data1["total"] == 3
    assert len(data1["items"]) == 2
    assert data1["limit"] == 2
    assert data1["offset"] == 0

    # Page 2 (limit 2, offset 2)
    res_page2 = client.get("/api/v1/images?limit=2&offset=2")
    assert res_page2.status_code == 200
    data2 = res_page2.json()
    assert data2["total"] == 3
    assert len(data2["items"]) == 1

    # Invalid pagination
    assert client.get("/api/v1/images?limit=0").status_code == 422
    assert client.get("/api/v1/images?limit=101").status_code == 422
    assert client.get("/api/v1/images?offset=-1").status_code == 422


# 15. Metadata retrieval
def test_metadata_retrieval(test_env):
    raw = make_image_bytes("JPEG", size=(32, 32))
    res = client.post(
        "/api/v1/images/upload",
        files={"file": ("meta.jpg", raw, "image/jpeg")},
    )
    assert res.status_code == 201
    img_id = res.json()["image_id"]

    res_get = client.get(f"/api/v1/images/{img_id}")
    assert res_get.status_code == 200
    assert res_get.json()["image_id"] == img_id
    assert res_get.json()["width"] == 32
    assert res_get.json()["height"] == 32


# 16. Original file retrieval returns exact uploaded bytes
def test_original_file_retrieval(test_env):
    raw = make_image_bytes("PNG", size=(25, 25))
    res = client.post(
        "/api/v1/images/upload",
        files={"file": ("retrieve.png", raw, "image/png")},
    )
    img_id = res.json()["image_id"]

    res_file = client.get(f"/api/v1/images/{img_id}/file")
    assert res_file.status_code == 200
    assert res_file.content == raw
    assert res_file.headers["content-type"] == "image/png"


# 17. Unknown UUID returns 404
def test_unknown_uuid_returns_404(test_env):
    unknown_id = uuid4()
    assert client.get(f"/api/v1/images/{unknown_id}").status_code == 404
    assert client.get(f"/api/v1/images/{unknown_id}/file").status_code == 404


# 18. Missing multipart file rejected
def test_missing_multipart_file_rejected(test_env):
    response = client.post("/api/v1/images/upload", data={})
    assert response.status_code == 422


# 19. Filename path traversal cannot escape storage directories
def test_path_traversal_sanitized(test_env):
    raw = make_image_bytes("JPEG", size=(16, 16))
    response = client.post(
        "/api/v1/images/upload",
        files={"file": ("../../../../etc/passwd.jpg", raw, "image/jpeg")},
    )
    assert response.status_code == 201
    meta = response.json()
    assert meta["original_filename"] == "passwd.jpg"
    assert ".." not in meta["stored_filename"]
    assert ".." not in meta["relative_path"]

    stored_file = test_env.DATASET_ROOT / "original" / meta["stored_filename"]
    assert stored_file.is_file()
    assert stored_file.resolve().is_relative_to(test_env.DATASET_ROOT.resolve())


# 20. Simulated metadata-write failure leaves no orphaned image file
def test_simulated_metadata_write_failure_cleans_up(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "datasets"
    test_settings = Settings(
        DATASET_ROOT=dataset_root,
        MAX_UPLOAD_BYTES=10 * 1024 * 1024,
        MAX_IMAGE_PIXELS=25_000_000,
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    # Monkeypatch ImageMetadata.model_dump_json to raise an exception simulating metadata failure
    from app.schemas.image import ImageMetadata
    original_dump = ImageMetadata.model_dump_json

    def failing_dump(*args, **kwargs):
        raise IOError("Simulated metadata write disk failure")

    monkeypatch.setattr(ImageMetadata, "model_dump_json", failing_dump)

    try:
        raw = make_image_bytes("JPEG", size=(20, 20))
        response = client.post(
            "/api/v1/images/upload",
            files={"file": ("fail_test.jpg", raw, "image/jpeg")},
        )
        assert response.status_code == 500

        # Verify no image files or temp files are left in original/
        original_dir = dataset_root / "original"
        if original_dir.exists():
            files_in_original = list(original_dir.glob("*"))
            assert len(files_in_original) == 0, f"Found orphaned files: {files_in_original}"

        metadata_dir = dataset_root / "metadata"
        if metadata_dir.exists():
            files_in_metadata = list(metadata_dir.glob("*"))
            assert len(files_in_metadata) == 0, f"Found orphaned metadata: {files_in_metadata}"
    finally:
        app.dependency_overrides.clear()
