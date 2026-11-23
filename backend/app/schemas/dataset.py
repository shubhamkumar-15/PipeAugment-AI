from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DatasetStatsResponse(BaseModel):
    total_images: int
    total_originals: int
    total_generated: int
    by_method: Dict[str, int]
    by_condition: Dict[str, int]
    by_quality_status: Dict[str, int]
    by_review_status: Dict[str, int]
    total_jobs: int
    recent_activity: List[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class DatasetExportRequest(BaseModel):
    include_originals: bool = Field(default=True, description="Include original parent images")
    include_traditional: bool = Field(default=True, description="Include traditional CV variations")
    include_generative: bool = Field(default=True, description="Include generative diffusion variations")
    only_quality_passed: bool = Field(default=False, description="Export only images that passed Phase 4 QualityValidator")
    only_human_approved: bool = Field(default=False, description="Export only images with review_status='approved'")
    conditions: Optional[List[str]] = Field(default=None, description="Optional condition filters")


class DatasetExportResponse(BaseModel):
    export_id: str
    filename: str
    image_count: int
    size_bytes: int
    download_url: str
    created_at: datetime
    filters_applied: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class ExportListResponse(BaseModel):
    items: List[DatasetExportResponse]
    total: int
