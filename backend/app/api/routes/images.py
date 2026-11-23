import json
import logging
import subprocess
from pathlib import Path
from uuid import UUID
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.schemas.image import (
    ImageMetadata,
    ImageListResponse,
    ReviewUpdateRequest,
    DeleteImageResponse,
    CorrosionAnalysisResponse,
)
from app.services.image_storage import ImageStorageService
from app.services.image_validation import validate_image_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])

AUTHORIZED_MAP_TYPES = {"pipeline_canny", "corrosion_layout", "corrosion_overlay"}


def get_image_storage(settings: Settings = Depends(get_settings)) -> ImageStorageService:
    return ImageStorageService(dataset_root=settings.DATASET_ROOT)


@router.post(
    "/upload",
    response_model=ImageMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an original pipeline inspection image",
)
async def upload_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> ImageMetadata:
    """
    Upload and validate an original pipeline inspection image.
    Supports JPEG, PNG, and WEBP formats up to MAX_UPLOAD_BYTES.
    Validates image bytes, dimensions, and pixel limits using Pillow.
    Persists original image immutably and saves metadata in datasets/metadata/.
    """
    validated = await validate_image_upload(
        file=file,
        max_upload_bytes=settings.MAX_UPLOAD_BYTES,
        max_image_pixels=settings.MAX_IMAGE_PIXELS,
    )
    return storage.save_image(validated)


@router.get(
    "",
    response_model=ImageListResponse,
    summary="List stored images with pagination and filtering",
)
def list_images(
    limit: int = Query(default=20, ge=1, le=100, description="Page limit (1-100)"),
    offset: int = Query(default=0, ge=0, description="Offset starting point"),
    source_type: Optional[str] = Query(default=None, description="Filter by source_type ('user_upload' | 'traditional' | 'generative')"),
    parent_image_id: Optional[UUID] = Query(default=None, description="Filter by parent_image_id"),
    condition: Optional[str] = Query(default=None, description="Filter by augmentation condition"),
    quality_status: Optional[str] = Query(default=None, description="Filter by quality status ('pass' | 'flag_for_review' | 'fail')"),
    review_status: Optional[str] = Query(default=None, description="Filter by review status ('pending' | 'approved' | 'rejected' | 'needs_review')"),
    search: Optional[str] = Query(default=None, description="Search by original filename substring"),
    storage: ImageStorageService = Depends(get_image_storage),
) -> ImageListResponse:
    """Retrieve paginated metadata for images, sorted newest first with optional multi-attribute filters."""
    return storage.list_images(
        limit=limit,
        offset=offset,
        source_type=source_type,
        parent_image_id=parent_image_id,
        condition=condition,
        quality_status=quality_status,
        review_status=review_status,
        search=search,
    )


@router.get(
    "/{image_id}",
    response_model=ImageMetadata,
    summary="Get image metadata by ID",
)
def get_image_metadata(
    image_id: UUID,
    storage: ImageStorageService = Depends(get_image_storage),
) -> ImageMetadata:
    """Retrieve metadata for a specific image by UUID."""
    return storage.get_metadata(image_id)


@router.get(
    "/{image_id}/file",
    summary="Retrieve image file by ID",
)
def get_image_file(
    image_id: UUID,
    storage: ImageStorageService = Depends(get_image_storage),
) -> FileResponse:
    """Serve the immutable original or generated image bytes with appropriate Content-Type."""
    file_path, content_type, filename = storage.get_image_file_path_and_type(image_id)
    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=filename,
    )


@router.patch(
    "/{image_id}/review",
    response_model=ImageMetadata,
    summary="Update human review decision for an image",
)
def update_image_review(
    image_id: UUID,
    review_update: ReviewUpdateRequest,
    storage: ImageStorageService = Depends(get_image_storage),
) -> ImageMetadata:
    """Update human review status (approved, rejected, needs_review), notes, and reviewer name."""
    return storage.update_review(image_id, review_update)


@router.delete(
    "/{image_id}",
    response_model=DeleteImageResponse,
    summary="Delete an image and its metadata",
)
def delete_image(
    image_id: UUID,
    delete_children: bool = Query(default=False, description="If deleting an original, also delete all its generated variations"),
    storage: ImageStorageService = Depends(get_image_storage),
) -> DeleteImageResponse:
    """Safely delete an image file and metadata while protecting raw benchmark files."""
    return storage.delete_image(image_id, delete_children=delete_children)


@router.post(
    "/{image_id}/analyze",
    response_model=CorrosionAnalysisResponse,
    summary="Run live ResNet-18 corrosion analysis and generate layout maps",
)
def analyze_image(
    image_id: UUID,
    storage: ImageStorageService = Depends(get_image_storage),
    settings: Settings = Depends(get_settings),
) -> CorrosionAnalysisResponse:
    """
    Executes live ResNet-18 classifier inference on an image via .venv-ai worker.
    Computes corrosion verdict, confidence, severity level, rust coverage %,
    and generates 3 visual layout maps:
    (1) Pipeline Structural Layout (Canny Edge Map)
    (2) Corrosion Layout Map (Isolated Mask + Contours)
    (3) AI Attention Heatmap (Grad-CAM Overlay)
    """
    file_path, _, _ = storage.get_image_file_path_and_type(image_id)
    analysis_base_dir = settings.DATASET_ROOT / "metadata" / "analysis"
    analysis_base_dir.mkdir(parents=True, exist_ok=True)

    worker_script = settings.PROJECT_ROOT / "ai" / "scripts" / "analyze_image.py"
    if not worker_script.exists():
        raise HTTPException(status_code=500, detail="Analysis worker script not found.")
    if not settings.AI_PYTHON_PATH.exists():
        raise HTTPException(status_code=500, detail="AI Python environment not configured.")

    cmd = [
        str(settings.AI_PYTHON_PATH),
        str(worker_script),
        "--image-path", str(file_path),
        "--image-id", str(image_id),
        "--output-dir", str(analysis_base_dir),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            logger.error("Analysis worker failed: %s", proc.stderr)
            raise RuntimeError(proc.stderr.strip() or "Analysis worker failed")

        result_json_path = analysis_base_dir / f"{image_id}.json"
        if result_json_path.exists():
            with open(result_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(proc.stdout)

        return CorrosionAnalysisResponse.model_validate(data)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Image analysis timed out.")
    except Exception as exc:
        logger.error("Analysis error for %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.get(
    "/{image_id}/analysis",
    response_model=CorrosionAnalysisResponse,
    summary="Get cached corrosion analysis for an image",
)
def get_image_analysis(
    image_id: UUID,
    settings: Settings = Depends(get_settings),
) -> CorrosionAnalysisResponse:
    """Retrieve pre-computed or cached corrosion analysis for an image."""
    analysis_file = settings.DATASET_ROOT / "metadata" / "analysis" / f"{image_id}.json"
    if not analysis_file.is_file():
        alt_file = settings.DATASET_ROOT / "metadata" / "analysis" / str(image_id) / "analysis.json"
        if alt_file.is_file():
            analysis_file = alt_file
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis for image '{image_id}' not found. Run POST /images/{image_id}/analyze first.",
            )

    try:
        with open(analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CorrosionAnalysisResponse.model_validate(data)
    except Exception as exc:
        logger.error("Failed to read analysis for %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail="Stored analysis data is invalid or corrupt.")


@router.get(
    "/{image_id}/maps/{map_type}",
    summary="Retrieve visual layout map PNG for an image",
)
def get_image_layout_map(
    image_id: UUID,
    map_type: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """
    Serve one of three generated visual inspection layout maps:
    - pipeline_canny: Pipeline Structural Layout (Canny Edge Map)
    - corrosion_layout: Corrosion Layout Map (Isolated Mask + Contours)
    - corrosion_overlay: AI Attention Heatmap (Grad-CAM Overlay)
    """
    if map_type not in AUTHORIZED_MAP_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid map_type '{map_type}'. Authorized: {', '.join(sorted(AUTHORIZED_MAP_TYPES))}",
        )

    candidate_1 = settings.DATASET_ROOT / "metadata" / "analysis" / str(image_id) / f"{map_type}.png"
    candidate_2 = settings.DATASET_ROOT / "metadata" / "analysis" / f"{image_id}_{map_type}.png"

    resolved_path: Optional[Path] = None
    if candidate_1.is_file():
        resolved_path = candidate_1
    elif candidate_2.is_file():
        resolved_path = candidate_2

    if resolved_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Map '{map_type}' for image '{image_id}' not found. Run POST /images/{image_id}/analyze first.",
        )

    resolved_path = resolved_path.resolve()
    if not resolved_path.is_relative_to(settings.DATASET_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid map file path.")

    return FileResponse(
        path=str(resolved_path),
        media_type="image/png",
        filename=f"{image_id}_{map_type}.png",
    )
