import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from app.schemas.image import (
    ImageMetadata,
    ImageListResponse,
    ReviewUpdateRequest,
    DeleteImageResponse,
)
from app.services.image_validation import ValidatedImage

logger = logging.getLogger(__name__)


class ImageStorageService:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.resolve()
        self.original_dir = (dataset_root / "original").resolve()
        self.augmented_dir = (dataset_root / "augmented" / "app_generated").resolve()
        self.metadata_dir = (dataset_root / "metadata").resolve()

    def _ensure_directories(self) -> None:
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        validated: ValidatedImage,
        parent_image_id: UUID | None = None,
    ) -> ImageMetadata:
        """
        Atomically persist the validated original image bytes into datasets/original/
        and its metadata JSON record into datasets/metadata/.
        """
        self._ensure_directories()

        image_id = uuid4()
        stored_filename = f"{image_id}.{validated.canonical_ext}"
        relative_path = f"original/{stored_filename}"
        final_image_path = self.original_dir / stored_filename
        final_metadata_path = self.metadata_dir / f"{image_id}.json"

        temp_image_path = self.original_dir / f".tmp_{image_id}.{validated.canonical_ext}"
        temp_metadata_path = self.metadata_dir / f".tmp_{image_id}.json"

        image_committed = False
        try:
            # 1. Write original image bytes to a temp file, then rename atomically
            with open(temp_image_path, "wb") as f:
                f.write(validated.raw_bytes)
            os.replace(temp_image_path, final_image_path)
            image_committed = True

            # 2. Build metadata record
            metadata = ImageMetadata(
                image_id=image_id,
                parent_image_id=parent_image_id,
                original_filename=validated.sanitized_filename,
                stored_filename=stored_filename,
                relative_path=relative_path,
                image_format=validated.detected_format,
                content_type=validated.content_type,
                width=validated.width,
                height=validated.height,
                size_bytes=validated.size_bytes,
                sha256=validated.sha256,
                created_at=datetime.now(timezone.utc),
                source_type="user_upload",
                generation_method=None,
                label=None,
                annotation_status="unannotated",
                split="unassigned",
            )

            # 3. Write metadata JSON to temp file, then rename atomically
            metadata_json = metadata.model_dump_json(indent=2)
            with open(temp_metadata_path, "w", encoding="utf-8") as f:
                f.write(metadata_json)
            os.replace(temp_metadata_path, final_metadata_path)

            return metadata

        except Exception as exc:
            logger.error("Failed to store image or metadata for %s: %s", image_id, str(exc))
            # Clean up temporary and partially written files
            if temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except OSError:
                    pass
            if temp_metadata_path.exists():
                try:
                    temp_metadata_path.unlink()
                except OSError:
                    pass
            if image_committed and final_image_path.exists():
                try:
                    final_image_path.unlink()
                except OSError:
                    pass
            if final_metadata_path.exists():
                try:
                    final_metadata_path.unlink()
                except OSError:
                    pass
            raise HTTPException(
                status_code=500,
                detail="Failed to store uploaded image.",
            ) from exc

    def get_metadata(self, image_id: UUID) -> ImageMetadata:
        """Retrieve metadata for a given image UUID."""
        metadata_path = self.metadata_dir / f"{image_id}.json"
        if not metadata_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Image with id '{image_id}' not found.",
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ImageMetadata.model_validate_json(content)
        except Exception as exc:
            logger.error("Corrupt metadata encountered for image id %s: %s", image_id, str(exc))
            raise HTTPException(
                status_code=500,
                detail="Stored image metadata is invalid or corrupt.",
            ) from exc

    def list_images(
        self,
        limit: int = 20,
        offset: int = 0,
        source_type: Optional[str] = None,
        parent_image_id: Optional[UUID] = None,
        condition: Optional[str] = None,
        quality_status: Optional[str] = None,
        review_status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> ImageListResponse:
        """
        List image metadata with pagination and optional filtering, sorted newest first.
        Does not load image pixel data into memory.
        """
        if not self.metadata_dir.exists():
            return ImageListResponse(items=[], total=0, limit=limit, offset=offset)

        all_metadata: List[ImageMetadata] = []
        for file_path in self.metadata_dir.glob("*.json"):
            if file_path.name.startswith("."):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                meta = ImageMetadata.model_validate_json(content)

                # Verify image file exists on disk to prevent orphaned broken cards
                candidate_img_path = (self.dataset_root / meta.relative_path).resolve()
                if not candidate_img_path.is_file():
                    logger.warning("Excluding orphaned metadata %s: %s missing on disk", file_path.name, candidate_img_path)
                    continue

                # Apply query filters
                if source_type and meta.source_type != source_type:
                    continue
                if parent_image_id and meta.parent_image_id != parent_image_id:
                    continue
                if condition and meta.condition != condition:
                    continue
                if quality_status and meta.quality_status != quality_status:
                    continue
                if review_status and meta.review_status != review_status:
                    continue
                if search and search.lower() not in meta.original_filename.lower():
                    continue

                all_metadata.append(meta)
            except Exception as exc:
                logger.error("Skipping corrupt metadata file %s: %s", file_path.name, str(exc))

        # Sort newest first (descending created_at, then image_id)
        all_metadata.sort(key=lambda m: (m.created_at, str(m.image_id)), reverse=True)

        total = len(all_metadata)
        paged_items = all_metadata[offset : offset + limit]

        return ImageListResponse(
            items=paged_items,
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_image_file_path_and_type(self, image_id: UUID) -> tuple[Path, str, str]:
        """
        Resolve the original or generated image file path through its metadata.
        Guarantees that the path remains strictly inside datasets/ (DATASET_ROOT).
        Returns (resolved_file_path, content_type, original_filename).
        """
        metadata = self.get_metadata(image_id)
        # Resolve via relative_path
        candidate_path = (self.dataset_root / metadata.relative_path).resolve()

        # Strict path containment check inside DATASET_ROOT
        try:
            if not candidate_path.is_relative_to(self.dataset_root):
                logger.error("Path traversal detected for image %s: %s", image_id, candidate_path)
                raise HTTPException(status_code=400, detail="Invalid file path.")
        except AttributeError:
            if self.dataset_root not in candidate_path.parents:
                raise HTTPException(status_code=400, detail="Invalid file path.")

        if not candidate_path.is_file():
            logger.error("File missing on disk for image %s: %s", image_id, candidate_path)
            raise HTTPException(
                status_code=404,
                detail="Image file not found on disk.",
            )

        return candidate_path, metadata.content_type, metadata.original_filename

    def update_review(
        self,
        image_id: UUID,
        review_update: ReviewUpdateRequest,
    ) -> ImageMetadata:
        """Atomically update human review decision, notes, and reviewer."""
        metadata = self.get_metadata(image_id)
        metadata.review_status = review_update.review_status
        if review_update.review_notes is not None:
            metadata.review_notes = review_update.review_notes
        if review_update.reviewer is not None:
            metadata.reviewer = review_update.reviewer

        final_metadata_path = self.metadata_dir / f"{image_id}.json"
        temp_metadata_path = self.metadata_dir / f".tmp_{image_id}.json"

        with open(temp_metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        os.replace(temp_metadata_path, final_metadata_path)

        return metadata

    def delete_image(self, image_id: UUID, delete_children: bool = False) -> DeleteImageResponse:
        """
        Safely delete an image and its metadata JSON.
        Guarantees that raw K-Pipelines benchmark files are protected.
        If delete_children is True and image is an original, also removes all generated variations.
        """
        metadata = self.get_metadata(image_id)
        if "raw" in metadata.relative_path:
            raise HTTPException(status_code=400, detail="Protected dataset files cannot be deleted.")

        deleted_files: List[str] = []
        deleted_children_count = 0

        # If original and delete_children requested, delete children first
        if metadata.source_type == "user_upload" and delete_children:
            all_children = self.list_images(limit=1000, parent_image_id=image_id).items
            for child in all_children:
                child_file = (self.dataset_root / child.relative_path).resolve()
                child_meta = self.metadata_dir / f"{child.image_id}.json"
                if child_file.exists():
                    try:
                        child_file.unlink()
                        deleted_files.append(str(child.relative_path))
                    except OSError:
                        pass
                if child_meta.exists():
                    try:
                        child_meta.unlink()
                        deleted_files.append(f"metadata/{child.image_id}.json")
                    except OSError:
                        pass
                deleted_children_count += 1

        # Delete target image file
        target_file = (self.dataset_root / metadata.relative_path).resolve()
        if target_file.exists():
            target_file.unlink()
            deleted_files.append(str(metadata.relative_path))

        # Delete target metadata file
        meta_file = self.metadata_dir / f"{image_id}.json"
        if meta_file.exists():
            meta_file.unlink()
            deleted_files.append(f"metadata/{image_id}.json")

        return DeleteImageResponse(
            image_id=image_id,
            deleted=True,
            deleted_files=deleted_files,
            deleted_children_count=deleted_children_count,
        )
