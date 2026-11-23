# PipeAugment AI — Backend REST API Reference

The PipeAugment AI backend is a production-grade FastAPI service providing immutable image storage, background asynchronous AI/CV augmentation job management, dataset ZIP export, multi-seed evaluation benchmark metrics, and hardware telemetry.

- **Base URL**: `http://localhost:8000/api/v1`
- **Documentation**: Swagger UI at `http://localhost:8000/docs`, ReDoc at `http://localhost:8000/redoc`
- **CORS Allowed Origins**: `http://localhost:3000`, `http://127.0.0.1:3000` (Next.js frontend)

---

## 1. System & Health Endpoints

### `GET /api/v1/health`
Health check endpoint reporting API status, version, and database/storage readiness.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "app_name": "PipeAugment AI"
}
```

### `GET /api/v1/system/info`
Returns genuine hardware status (queried via `nvidia-smi`), AI worker virtual environment readiness, augmentation capabilities, and upload constraints.
- **Response `200 OK`**:
```json
{
  "app_name": "PipeAugment AI",
  "app_version": "0.1.0",
  "hardware": {
    "gpu": {
      "gpu_detected": true,
      "model_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
      "total_memory_mb": 8188,
      "free_memory_mb": 7654,
      "driver_version": "576.02"
    }
  },
  "ai_worker": {
    "environment_ready": true,
    "python_executable": "D:\\vscode\\Gen AI Project\\pipeaugment-ai\\.venv-ai\\Scripts\\python.exe"
  },
  "capabilities": {
    "augmentation_methods": ["traditional", "generative"],
    "traditional_transforms": [
      "brightness", "contrast", "gamma", "uneven_illumination",
      "soft_shadow", "gaussian_blur", "gaussian_noise",
      "jpeg_compression", "downsample_restore"
    ],
    "generative_conditions": [
      "low_light_inspection", "directional_industrial_lighting",
      "bright_daylight", "harsh_shadows", "wet_surface", "dusty_surface"
    ],
    "frozen_controlnet_config": {
      "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
      "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
      "control_scale": 0.80,
      "strength": 0.35,
      "guidance_scale": 5.0,
      "num_inference_steps": 30,
      "canny_thresholds": [100, 200]
    }
  },
  "upload_limits": {
    "max_upload_bytes": 10485760,
    "max_upload_mb": 10,
    "max_image_pixels": 25000000,
    "supported_formats": ["JPEG", "PNG", "WEBP"]
  }
}
```

---

## 2. Image Management & Gallery Endpoints

### `POST /api/v1/images/upload`
Uploads and validates an original pipeline inspection image (multipart/form-data).
- **Validation**: Strict Pillow format check (`JPEG`, `PNG`, `WEBP`), dimension check ($\le 25\text{M}$ pixels), byte size check ($\le 10\text{MB}$). Animated frames are rejected.
- **Storage**: Immutable UUID v4 storage in `datasets/original/{image_id}.{ext}` with paired metadata in `datasets/metadata/{image_id}.json`.
- **Response `201 Created`**: Returns `ImageMetadata`.

### `GET /api/v1/images`
Retrieves paginated image metadata sorted newest first, supporting multi-attribute filtering without loading pixel data.
- **Query Parameters**:
  - `limit` (int, default=20, max=100)
  - `offset` (int, default=0)
  - `source_type` (string, optional: `user_upload`, `traditional`, `generative`)
  - `parent_image_id` (UUID, optional: fetch variations of a specific original)
  - `condition` (string, optional: filter by condition/transform)
  - `quality_status` (string, optional: `pass`, `flag_for_review`, `fail`)
  - `review_status` (string, optional: `pending`, `approved`, `rejected`, `needs_review`)
  - `search` (string, optional: filename substring match)
- **Response `200 OK`**:
```json
{
  "items": [
    {
      "image_id": "c8889ff9-79a8-4c80-be34-e406456f917d",
      "parent_image_id": null,
      "original_filename": "inspection_pipe_01.jpg",
      "stored_filename": "c8889ff9-79a8-4c80-be34-e406456f917d.jpg",
      "relative_path": "original/c8889ff9-79a8-4c80-be34-e406456f917d.jpg",
      "image_format": "JPEG",
      "content_type": "image/jpeg",
      "width": 1920,
      "height": 1080,
      "size_bytes": 482104,
      "sha256": "4a5c...78e1",
      "created_at": "2026-11-22T20:10:00Z",
      "source_type": "user_upload",
      "generation_method": null,
      "label": null,
      "annotation_status": "unannotated",
      "split": "unassigned"
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### `GET /api/v1/images/{image_id}`
Returns metadata JSON for a single image by UUID.
- **Response `200 OK`**: `ImageMetadata`
- **Response `404 Not Found`**: If UUID does not exist.

### `GET /api/v1/images/{image_id}/file`
Serves raw image bytes with correct MIME `Content-Type` for both original uploads (`datasets/original/`) and generated variations (`datasets/augmented/app_generated/`). Strictly protected against directory traversal attacks.
- **Response `200 OK`**: Binary image stream.

### `PATCH /api/v1/images/{image_id}/review`
Atomically updates the human review decision in the image's metadata JSON.
- **Request Body**:
```json
{
  "review_status": "approved",
  "review_notes": "Corrosion defect pattern and cylindrical seam fully intact.",
  "reviewer": "Senior Pipeline Inspector"
}
```
- **Response `200 OK`**: Updated `ImageMetadata`.

### `DELETE /api/v1/images/{image_id}`
Safely deletes an image file and its metadata JSON. Raw K-Pipelines benchmark files are protected and cannot be deleted.
- **Query Parameter**: `delete_children` (bool, default=false). If true and the image is an original, all generated variations are also deleted.
- **Response `200 OK`**:
```json
{
  "image_id": "c8889ff9-79a8-4c80-be34-e406456f917d",
  "deleted": true,
  "deleted_files": ["original/c8889ff9.jpg", "metadata/c8889ff9.json"],
  "deleted_children_count": 0
}
```

### `POST /api/v1/images/{image_id}/analyze`
Executes real-time inference using the trained Experiment C ResNet-18 classifier (`.venv-ai`), extracts `layer4` Grad-CAM attention heatmaps, computes color-space rust coverage, determines diagnostic severity levels, and synthesizes 3 visual inspection layout maps saved under `datasets/metadata/analysis/{image_id}/`.
- **Response `200 OK`**:
```json
{
  "image_id": "c1a2b3c4-...",
  "predicted_class": "corroded",
  "confidence": 0.942,
  "corrosion_probability": 0.942,
  "non_corrosion_probability": 0.058,
  "severity_level": "Moderate",
  "rust_coverage_percent": 7.45,
  "model_checkpoint": "best_model.pth",
  "disclaimer": "Diagnostic CV screening assistance only. Not certified NDT, ultrasonic thickness gauging, or metallographic inspection.",
  "map_urls": {
    "pipeline_canny": "/api/v1/images/c1a2b3c4-.../maps/pipeline_canny",
    "corrosion_layout": "/api/v1/images/c1a2b3c4-.../maps/corrosion_layout",
    "corrosion_overlay": "/api/v1/images/c1a2b3c4-.../maps/corrosion_overlay"
  }
}
```

### `GET /api/v1/images/{image_id}/analysis`
Retrieves precomputed or cached corrosion analysis results for a given image.
- **Response `200 OK`**: `CorrosionAnalysisResponse`
- **Response `404 Not Found`**: If analysis has not yet been executed.

### `GET /api/v1/images/{image_id}/maps/{map_type}`
Streams one of the 3 visual inspection layout maps with `image/png` media type:
- `pipeline_canny`: Pipeline Structural Layout (Canny edge map on obsidian canvas)
- `corrosion_layout`: Corrosion Layout Map (Isolated candidate rust mask + cyber-saffron contours)
- `corrosion_overlay`: AI Attention Heatmap (ResNet-18 layer4 Grad-CAM overlay)
- **Response `200 OK`**: PNG image stream.
- **Response `400 Bad Request`**: If `map_type` is not in the authorized allowlist.
- **Response `404 Not Found`**: If map has not yet been generated.

---

## 3. Augmentation & Background Job Endpoints

All augmentation requests run asynchronously through a single-worker FIFO background queue, preventing concurrent GPU VRAM exhaustion.

### `POST /api/v1/augment/traditional`
Enqueues a traditional computer vision augmentation job.
- **Request Body**:
```json
{
  "parent_image_ids": ["c8889ff9-79a8-4c80-be34-e406456f917d"],
  "conditions": ["brightness", "contrast", "uneven_illumination"],
  "intensity": 0.5,
  "variations_per_image": 2,
  "seed": 42
}
```
- **Response `202 Accepted`**:
```json
{
  "job_id": "7b0fa563-3561-4fa3-9f82-a0cf66e51147",
  "job_type": "traditional",
  "status": "queued",
  "total_items": 1,
  "completed_items": 0,
  "failed_items": 0,
  "parent_image_ids": ["c8889ff9-79a8-4c80-be34-e406456f917d"],
  "config": {
    "conditions": ["brightness", "contrast", "uneven_illumination"],
    "intensity": 0.5,
    "variations_per_image": 2,
    "seed": 42
  },
  "created_at": "2026-11-22T21:40:00Z",
  "started_at": null,
  "finished_at": null,
  "generated_image_ids": [],
  "item_results": [],
  "error_message": null
}
```

### `POST /api/v1/augment/generative`
Enqueues a generative diffusion augmentation job using frozen ControlNet Canny or plain img2img.
- **Request Body**:
```json
{
  "parent_image_ids": ["c8889ff9-79a8-4c80-be34-e406456f917d"],
  "mode": "controlnet_canny",
  "conditions": ["low_light_inspection", "directional_industrial_lighting"],
  "strength": 0.35,
  "control_scale": 0.80,
  "variations_per_image": 1,
  "seed": 42
}
```
- **Response `202 Accepted`**: Returns initial `JobStatus`.

### `GET /api/v1/jobs`
Lists background jobs with pagination and optional filtering.
- **Query Parameters**: `limit`, `offset`, `status` (`queued`, `running`, `completed`, `partial_failed`, `failed`), `type` (`traditional`, `generative`).
- **Response `200 OK`**: `JobListResponse`

### `GET /api/v1/jobs/{job_id}`
Returns real-time job progress, completed items count, and generated image UUIDs.
- **Response `200 OK`**: `JobStatus`

### `POST /api/v1/jobs/{job_id}/retry`
Extracts failed items from a failed or partial_failed job and re-queues them as a new job.
- **Response `202 Accepted`**: New `JobStatus`.

---

## 4. Dataset Management & ZIP Export Endpoints

### `GET /api/v1/datasets/stats`
Computes real-time dataset statistics directly from storage.
- **Response `200 OK`**:
```json
{
  "total_images": 120,
  "total_originals": 40,
  "total_generated": 80,
  "by_method": {
    "traditional": 48,
    "generative": 32
  },
  "by_condition": {
    "brightness": 24,
    "low_light_inspection": 16,
    "directional_industrial_lighting": 16
  },
  "by_quality_status": {
    "pass": 72,
    "flag_for_review": 8,
    "fail": 0,
    "unvalidated": 40
  },
  "by_review_status": {
    "approved": 52,
    "rejected": 4,
    "needs_review": 6,
    "pending": 18
  },
  "total_jobs": 14,
  "recent_activity": [...]
}
```

### `POST /api/v1/datasets/export`
Builds a self-contained ZIP archive on disk in `datasets/exports/`.
- **Request Body**:
```json
{
  "include_originals": true,
  "include_traditional": true,
  "include_generative": true,
  "only_quality_passed": true,
  "only_human_approved": false,
  "conditions": null
}
```
- **Response `201 Created`**:
```json
{
  "export_id": "c5cb38be",
  "filename": "export_20261122_161321_c5cb38be.zip",
  "image_count": 84,
  "size_bytes": 45120892,
  "download_url": "/api/v1/datasets/exports/c5cb38be/download",
  "created_at": "2026-11-22T21:45:00Z",
  "filters_applied": {...}
}
```

### `GET /api/v1/datasets/exports`
Lists available generated ZIP exports.
- **Response `200 OK`**: `ExportListResponse`

### `GET /api/v1/datasets/exports/{export_id}/download`
Streams the ZIP file from disk with `Content-Type: application/zip` and `Content-Disposition: attachment; filename="..."`.
- **Response `200 OK`**: Binary ZIP stream.

---

## 5. Model Evaluation Endpoints

### `GET /api/v1/evaluation/benchmark`
Reads and serves the multi-seed evaluation benchmark results from `experiments/final_evaluation/final_benchmark_results.json`.
- **Response `200 OK`**: Master JSON with metadata, clean test metrics (Mean ± Std), confusion matrices, and robustness by condition.

### `GET /api/v1/evaluation/charts/{chart_name}`
Serves publication-grade evaluation charts from `experiments/final_evaluation/charts/`:
- `abc_clean_vs_stress_f1.png`
- `abc_corrosion_recall_robustness.png`
- `abc_confusion_matrices_clean.png`
- `stress_suite_preview.png`
- **Response `200 OK`**: `image/png` stream.

---

## 6. How to Run the Backend Server

Start the FastAPI service using `uvicorn` inside `backend/.venv`:

```powershell
# From project root:
cd "d:\vscode\Gen AI Project\pipeaugment-ai"
& ".\backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
API docs will be available at `http://127.0.0.1:8000/docs`.
