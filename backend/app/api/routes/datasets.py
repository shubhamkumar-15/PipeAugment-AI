import csv
import io
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings
from app.schemas.dataset import (
    DatasetStatsResponse,
    DatasetExportRequest,
    DatasetExportResponse,
    ExportListResponse,
)
from app.services.image_storage import ImageStorageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["dataset-management"])


def get_image_storage(settings: Settings = Depends(get_settings)) -> ImageStorageService:
    return ImageStorageService(dataset_root=settings.DATASET_ROOT)


@router.get(
    "/stats",
    response_model=DatasetStatsResponse,
    summary="Get real-time dataset inventory and quality statistics",
)
def get_dataset_stats(
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> DatasetStatsResponse:
    """Computes live dataset statistics and recent activity directly from metadata storage."""
    metadata_dir = settings.DATASET_ROOT / "metadata"
    jobs_dir = settings.JOBS_DIR

    total_originals = 0
    total_generated = 0
    by_method: Dict[str, int] = {}
    by_condition: Dict[str, int] = {}
    by_quality: Dict[str, int] = {"pass": 0, "flag_for_review": 0, "fail": 0, "unvalidated": 0}
    by_review: Dict[str, int] = {"pending": 0, "approved": 0, "rejected": 0, "needs_review": 0}
    recent_activity: List[Dict[str, Any]] = []

    if metadata_dir.exists():
        for meta_file in metadata_dir.glob("*.json"):
            if meta_file.name.startswith((".", "jobs")):
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                source_type = meta.get("source_type", "user_upload")
                if source_type == "user_upload":
                    total_originals += 1
                else:
                    total_generated += 1

                # By method
                method = meta.get("generation_method") or source_type
                by_method[method] = by_method.get(method, 0) + 1

                # By condition
                cond = meta.get("condition")
                if cond:
                    by_condition[cond] = by_condition.get(cond, 0) + 1

                # Quality status
                q_status = meta.get("quality_status")
                if q_status in by_quality:
                    by_quality[q_status] += 1
                else:
                    by_quality["unvalidated"] += 1

                # Review status
                r_status = meta.get("review_status")
                if r_status in by_review:
                    by_review[r_status] += 1

                recent_activity.append({
                    "type": "image",
                    "id": meta.get("image_id"),
                    "filename": meta.get("original_filename"),
                    "source_type": source_type,
                    "created_at": meta.get("created_at"),
                })
            except Exception as exc:
                logger.error("Error reading %s for stats: %s", meta_file.name, exc)

    # Job count
    total_jobs = 0
    if jobs_dir.exists():
        for jf in jobs_dir.glob("*.json"):
            if not jf.name.startswith((".", "spec_")):
                total_jobs += 1
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        j_data = json.load(f)
                    recent_activity.append({
                        "type": "job",
                        "id": j_data.get("job_id"),
                        "job_type": j_data.get("job_type"),
                        "status": j_data.get("status"),
                        "created_at": j_data.get("created_at"),
                    })
                except Exception:
                    pass

    # Sort recent activity newest first, limit to 10
    recent_activity.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    recent_activity = recent_activity[:10]

    return DatasetStatsResponse(
        total_images=total_originals + total_generated,
        total_originals=total_originals,
        total_generated=total_generated,
        by_method=by_method,
        by_condition=by_condition,
        by_quality_status=by_quality,
        by_review_status=by_review,
        total_jobs=total_jobs,
        recent_activity=recent_activity,
    )


@router.post(
    "/export",
    response_model=DatasetExportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Export a filtered pipeline dataset as a structured ZIP archive",
)
def export_dataset(
    request: DatasetExportRequest,
    settings: Settings = Depends(get_settings),
    storage: ImageStorageService = Depends(get_image_storage),
) -> DatasetExportResponse:
    """
    Builds a structured, self-contained dataset export archive on disk.
    Includes original images, augmented variations, paired JSON manifest, tabular CSV,
    and a README containing pipeline lineage and annotation validity caveats.
    """
    exports_dir = settings.EXPORTS_DIR
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Filter images from storage
    all_images = storage.list_images(limit=10000, offset=0).items
    filtered: List[Any] = []

    for img in all_images:
        # Source type checks
        if img.source_type == "user_upload" and not request.include_originals:
            continue
        if img.source_type == "traditional" and not request.include_traditional:
            continue
        if img.source_type == "generative" and not request.include_generative:
            continue

        # Quality pass filter
        if request.only_quality_passed and img.quality_status != "pass":
            continue

        # Human approved filter
        if request.only_human_approved and img.review_status != "approved":
            continue

        # Condition filter
        if request.conditions and img.condition not in request.conditions:
            continue

        filtered.append(img)

    if not filtered:
        raise HTTPException(
            status_code=400,
            detail="No images matched the specified export filter criteria.",
        )

    short_id = uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_filename = f"export_{timestamp}_{short_id}.zip"
    final_zip_path = exports_dir / zip_filename
    temp_zip_path = exports_dir / f".tmp_{zip_filename}"

    manifest_records = []
    csv_rows = []

    with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in filtered:
            try:
                file_path, _, _ = storage.get_image_file_path_and_type(item.image_id)
            except HTTPException:
                continue

            # Determine archive folder path
            if item.source_type == "user_upload":
                arc_path = f"original/{item.stored_filename}"
            elif item.source_type == "traditional":
                arc_path = f"augmented/traditional/{item.stored_filename}"
            else:
                arc_path = f"augmented/generative/{item.stored_filename}"

            zf.write(file_path, arcname=arc_path)

            item_dict = item.model_dump(mode="json")
            item_dict["export_archive_path"] = arc_path
            manifest_records.append(item_dict)

            csv_rows.append({
                "image_id": str(item.image_id),
                "parent_image_id": str(item.parent_image_id) if item.parent_image_id else "",
                "archive_path": arc_path,
                "source_type": item.source_type,
                "generation_method": item.generation_method or "",
                "label": item.label or "unlabeled",
                "condition": item.condition or "",
                "quality_status": item.quality_status or "unvalidated",
                "review_status": item.review_status or "unreviewed",
                "sha256": item.sha256,
                "width": item.width,
                "height": item.height,
            })

        # Add manifest.json
        manifest_json_bytes = json.dumps({
            "export_id": short_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(manifest_records),
            "filters_applied": request.model_dump(),
            "images": manifest_records,
        }, indent=2).encode("utf-8")
        zf.writestr("manifest.json", manifest_json_bytes)

        # Add manifest.csv
        csv_buffer = io.StringIO()
        if csv_rows:
            writer = csv.DictWriter(csv_buffer, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        zf.writestr("manifest.csv", csv_buffer.getvalue().encode("utf-8"))

        # Add README.txt
        readme_text = f"""PipeAugment AI — Dataset Export Archive
===================================================
Export ID: {short_id}
Generated: {datetime.now(timezone.utc).isoformat()}
Image Count: {len(manifest_records)}

Directory Layout:
  original/                 Original uploaded parent pipeline inspection images
  augmented/traditional/    Deterministic traditional computer vision augmentations
  augmented/generative/     Quality-screened ControlNet Canny generative augmentations
  manifest.json             Full metadata lineage, parameters, quality scores, and hashes
  manifest.csv              Tabular summary suitable for standard ML dataset loaders
  README.txt                This information file

Dataset Integrity & Annotation Validity Notice:
- Generative augmentations in this archive were generated using ControlNet Canny (c=0.80, s=0.35)
  and evaluated through the PipeAugment AI Phase 4 QualityValidator.
- High-level classification labels are inherited from parent images.
- If using for object detection or segmentation, verify that defect bounding boxes or masks
  remain valid under spatial and lighting transformations.
"""
        zf.writestr("README.txt", readme_text.encode("utf-8"))

    # Atomic rename
    os.replace(temp_zip_path, final_zip_path)
    size_bytes = final_zip_path.stat().st_size

    # Save export metadata JSON
    export_meta = {
        "export_id": short_id,
        "filename": zip_filename,
        "image_count": len(manifest_records),
        "size_bytes": size_bytes,
        "download_url": f"/api/v1/datasets/exports/{short_id}/download",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filters_applied": request.model_dump(),
    }
    with open(exports_dir / f"{short_id}.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return DatasetExportResponse.model_validate(export_meta)


@router.get(
    "/exports",
    response_model=ExportListResponse,
    summary="List generated dataset ZIP export archives",
)
def list_exports(settings: Settings = Depends(get_settings)) -> ExportListResponse:
    """Retrieve list of available generated ZIP exports."""
    exports_dir = settings.EXPORTS_DIR
    if not exports_dir.exists():
        return ExportListResponse(items=[], total=0)

    items = []
    for meta_file in exports_dir.glob("*.json"):
        if meta_file.name.startswith("."):
            continue
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify zip actually exists
            zip_p = exports_dir / data["filename"]
            if zip_p.exists():
                items.append(DatasetExportResponse.model_validate(data))
        except Exception:
            pass

    items.sort(key=lambda x: x.created_at, reverse=True)
    return ExportListResponse(items=items, total=len(items))


@router.get(
    "/exports/{export_id}/download",
    summary="Download a structured dataset ZIP export",
)
def download_export(
    export_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream a dataset ZIP archive from disk with Content-Disposition headers."""
    exports_dir = settings.EXPORTS_DIR
    meta_path = exports_dir / f"{export_id}.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail=f"Export '{export_id}' not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    zip_path = (exports_dir / meta["filename"]).resolve()
    # Path containment check
    try:
        if not zip_path.is_relative_to(exports_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid export path.")
    except AttributeError:
        if exports_dir.resolve() not in zip_path.parents:
            raise HTTPException(status_code=400, detail="Invalid export path.")

    if not zip_path.is_file():
        raise HTTPException(status_code=404, detail="Export ZIP file missing on disk.")

    file_size = zip_path.stat().st_size
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=meta["filename"],
        headers={
            "Content-Disposition": f'attachment; filename="{meta["filename"]}"',
            "Content-Length": str(file_size),
        },
    )
