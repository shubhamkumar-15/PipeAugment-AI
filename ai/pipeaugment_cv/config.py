"""Configuration and range validation for PipeAugment CV traditional augmentation engine."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any


@dataclass(frozen=True)
class ParameterRange:
    min_val: float
    max_val: float
    param_type: type = float

    def validate(self, val: Any, name: str = "") -> None:
        if not isinstance(val, (int, float)):
            raise TypeError(f"Parameter '{name}' must be numeric, got {type(val).__name__}: {val}")
        if not (self.min_val <= float(val) <= self.max_val):
            raise ValueError(
                f"Parameter '{name}'={val} outside allowed range [{self.min_val}, {self.max_val}]"
            )


@dataclass
class TraditionalConfig:
    version: str = "traditional_v1"
    base_seed: int = 42

    # Lighting parameter ranges
    brightness_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.80, 1.20, float))
    contrast_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.85, 1.15, float))
    gamma_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.85, 1.20, float))
    uneven_illum_intensity_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.85, 1.15, float))
    soft_shadow_intensity_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.75, 0.90, float))

    # Image-quality parameter ranges
    gaussian_blur_sigma_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.40, 0.90, float))
    gaussian_noise_std_range: ParameterRange = field(default_factory=lambda: ParameterRange(2.0, 6.0, float))
    jpeg_quality_range: ParameterRange = field(default_factory=lambda: ParameterRange(75, 90, int))
    downsample_scale_range: ParameterRange = field(default_factory=lambda: ParameterRange(0.65, 0.90, float))

    lighting_transforms: Tuple[str, ...] = (
        "brightness",
        "contrast",
        "gamma",
        "uneven_illumination",
        "soft_shadow",
    )

    quality_transforms: Tuple[str, ...] = (
        "gaussian_blur",
        "gaussian_noise",
        "jpeg_compression",
        "downsample_restore",
    )
