from uuid import UUID
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import Settings, get_settings
from app.schemas.job import (
    TraditionalAugmentRequest,
    GenerativeAugmentRequest,
    JobStatus,
    JobListResponse,
)
from app.services.image_storage import ImageStorageService
from app.services.job_manager import JobManager, get_job_manager

router = APIRouter(prefix="", tags=["augmentation-jobs"])


def get_image_storage(settings: Settings = Depends(get_settings)) -> ImageStorageService:
    return ImageStorageService(dataset_root=settings.DATASET_ROOT)


def resolve_parent_image_info(storage: ImageStorageService, parent_id: UUID) -> Dict[str, Any]:
    """Helper to resolve parent image metadata and disk path."""
    meta = storage.get_metadata(parent_id)
    file_path, _, _ = storage.get_image_file_path_and_type(parent_id)
    return {
        "image_id": str(meta.image_id),
        "file_path": str(file_path),
        "relative_path": meta.relative_path,
        "sha256": meta.sha256,
        "label": meta.label,
        "original_filename": meta.original_filename,
    }


@router.post(
    "/augment/traditional",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a traditional computer vision augmentation job",
)
def create_traditional_job(
    request: TraditionalAugmentRequest,
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> JobStatus:
    """
    Enqueue a traditional CV augmentation job across one or more parent images.
    Applies condition-controlled transformations, evaluates Phase 4 quality metrics,
    and saves lossless PNGs with paired metadata.
    """
    # 1. Validate that all parent images exist
    parent_images = []
    for pid in request.parent_image_ids:
        parent_images.append(resolve_parent_image_info(storage, pid))

    # 2. Enqueue job
    manager = get_job_manager(settings)
    job_config = {
        "conditions": request.conditions,
        "intensity": request.intensity,
        "variations_per_image": request.variations_per_image,
        "seed": request.seed or 42,
    }
    return manager.submit_job(
        job_type="traditional",
        parent_image_ids=request.parent_image_ids,
        parent_images=parent_images,
        config=job_config,
    )


@router.post(
    "/augment/generative",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a generative diffusion augmentation job",
)
def create_generative_job(
    request: GenerativeAugmentRequest,
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> JobStatus:
    """
    Enqueue a generative diffusion augmentation job using the frozen ControlNet Canny
    or plain img2img pipeline. Evaluates Phase 4 quality metrics and saves PNGs with paired metadata.
    """
    # 1. Validate that all parent images exist
    parent_images = []
    for pid in request.parent_image_ids:
        parent_images.append(resolve_parent_image_info(storage, pid))

    # 2. Enqueue job
    manager = get_job_manager(settings)
    job_config = {
        "mode": request.mode,
        "conditions": request.conditions,
        "strength": request.strength,
        "control_scale": request.control_scale,
        "variations_per_image": request.variations_per_image,
        "seed": request.seed or 42,
    }
    return manager.submit_job(
        job_type="generative",
        parent_image_ids=request.parent_image_ids,
        parent_images=parent_images,
        config=job_config,
    )


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List background augmentation jobs",
)
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Page limit (1-100)"),
    offset: int = Query(default=0, ge=0, description="Offset starting point"),
    status: Optional[str] = Query(default=None, description="Filter by status ('queued' | 'running' | 'completed' | 'partial_failed' | 'failed')"),
    type: Optional[str] = Query(default=None, description="Filter by type ('traditional' | 'generative')"),
    settings: Settings = Depends(get_settings),
) -> JobListResponse:
    """Retrieve paginated list of augmentation jobs, sorted newest first."""
    manager = get_job_manager(settings)
    return manager.list_jobs(limit=limit, offset=offset, status_filter=status, type_filter=type)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Get background job status by ID",
)
def get_job(
    job_id: UUID,
    settings: Settings = Depends(get_settings),
) -> JobStatus:
    """Retrieve live status, progress, and generated image IDs for a background job."""
    manager = get_job_manager(settings)
    return manager.get_job(job_id)


@router.post(
    "/jobs/{job_id}/retry",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry failed items from a failed or partial_failed job",
)
def retry_job(
    job_id: UUID,
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> JobStatus:
    """Re-queues any failed parent images from a previous job as a new background job."""
    manager = get_job_manager(settings)
    return manager.retry_job(
        job_id=job_id,
        parent_resolver=lambda pid: resolve_parent_image_info(storage, pid),
    )
