from datetime import datetime
from typing import Literal, Any, List, Optional, Dict
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator


class TraditionalAugmentRequest(BaseModel):
    parent_image_ids: List[UUID] = Field(..., min_length=1, description="List of parent image UUIDs to augment")
    conditions: Optional[List[str]] = Field(
        default=None,
        description="List of transforms: brightness, contrast, gamma, uneven_illumination, soft_shadow, gaussian_blur, gaussian_noise, jpeg_compression, downsample_restore",
    )
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Augmentation intensity from 0.0 to 1.0")
    variations_per_image: int = Field(default=1, ge=1, le=4, description="Variations to generate per parent (1-4)")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducibility")


class GenerativeAugmentRequest(BaseModel):
    parent_image_ids: List[UUID] = Field(..., min_length=1, description="List of parent image UUIDs to augment")
    mode: Literal["controlnet_canny", "img2img"] = Field(
        default="controlnet_canny",
        description="Generative mode: controlnet_canny (recommended) or img2img",
    )
    conditions: Optional[List[str]] = Field(
        default=None,
        description="Lighting conditions: low_light_inspection, directional_industrial_lighting, bright_daylight, harsh_shadows, wet_surface, dusty_surface",
    )
    strength: float = Field(default=0.35, ge=0.15, le=0.55, description="Diffusion img2img strength (0.15-0.55)")
    control_scale: float = Field(default=0.80, ge=0.30, le=1.0, description="ControlNet conditioning scale (0.30-1.0)")
    variations_per_image: int = Field(default=1, ge=1, le=4, description="Variations to generate per parent (1-4)")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducibility")

    @field_validator("mode", mode="before")
    @classmethod
    def canonicalize_mode(cls, v: Any) -> str:
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("plain_img2img", "plain-img2img", "img2img", "plain"):
                return "img2img"
            if normalized in ("controlnet_canny", "controlnet-canny", "controlnet_img2img", "controlnet"):
                return "controlnet_canny"
        return v


class JobStatus(BaseModel):
    job_id: UUID
    job_type: Literal["traditional", "generative"]
    status: Literal["queued", "running", "completed", "partial_failed", "failed"]
    stage: Optional[str] = Field(default=None, description="Execution stage: queued, loading_model, synthesizing, scoring_quality, completed")
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    parent_image_ids: List[UUID]
    config: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generated_image_ids: List[UUID] = Field(default_factory=list)
    item_results: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class JobListResponse(BaseModel):
    items: List[JobStatus]
    total: int
    limit: int
    offset: int
