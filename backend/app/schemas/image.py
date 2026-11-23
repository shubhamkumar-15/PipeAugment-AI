from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class ImageMetadata(BaseModel):
    image_id: UUID
    parent_image_id: Optional[UUID] = None
    original_filename: str
    stored_filename: str
    relative_path: str
    image_format: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    sha256: str
    created_at: datetime
    source_type: Literal["user_upload", "traditional", "generative"] = "user_upload"
    generation_method: Optional[str] = None
    label: Optional[str] = None
    annotation_status: str = "unannotated"
    split: str = "unassigned"

    # Augmented / quality validation / review fields
    condition: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    quality_status: Optional[Literal["pass", "flag_for_review", "fail"]] = None
    quality_flags: Optional[List[str]] = None
    review_status: Optional[Literal["pending", "approved", "rejected", "needs_review"]] = None
    review_notes: Optional[str] = None
    reviewer: Optional[str] = None
    label_preserved: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)


class CorrosionMapUrls(BaseModel):
    pipeline_canny: str
    corrosion_layout: str
    corrosion_overlay: str


class CorrosionAnalysisResponse(BaseModel):
    image_id: str
    predicted_class: Literal["corroded", "non_corroded"]
    confidence: float
    corrosion_probability: float
    non_corrosion_probability: float
    severity_level: Literal["None", "Low", "Moderate", "Severe"]
    rust_coverage_percent: float
    model_checkpoint: str
    disclaimer: str
    map_urls: CorrosionMapUrls


class ImageListResponse(BaseModel):
    items: List[ImageMetadata]
    total: int
    limit: int
    offset: int


class ReviewUpdateRequest(BaseModel):
    review_status: Literal["approved", "rejected", "needs_review", "pending"]
    review_notes: Optional[str] = None
    reviewer: Optional[str] = None


class DeleteImageResponse(BaseModel):
    image_id: UUID
    deleted: bool
    deleted_files: List[str]
    deleted_children_count: int = 0
