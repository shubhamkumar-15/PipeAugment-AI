"""
PipeAugment AI — Phase 4 Quality Validation Configuration.
Defines threshold dataclasses and validation profiles for screening
augmented images (traditional and generative) prior to training admission.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class QualityThresholdProfile:
    """Threshold configuration for a specific augmentation category."""
    min_ssim: float = 0.50
    max_lpips: float = 0.35
    min_edge_iou: float = 0.15
    max_rust_area_delta: float = 0.10
    max_clipping_low_rate: float = 0.05
    max_clipping_high_rate: float = 0.05
    min_pixel_abs_diff: float = 1.0

    def validate(self) -> None:
        if not (0.0 <= self.min_ssim <= 1.0):
            raise ValueError(f"min_ssim must be in [0, 1], got {self.min_ssim}")
        if not (0.0 <= self.max_lpips <= 2.0):
            raise ValueError(f"max_lpips must be in [0, 2], got {self.max_lpips}")
        if not (0.0 <= self.min_edge_iou <= 1.0):
            raise ValueError(f"min_edge_iou must be in [0, 1], got {self.min_edge_iou}")
        if not (0.0 <= self.max_rust_area_delta <= 1.0):
            raise ValueError(f"max_rust_area_delta must be in [0, 1], got {self.max_rust_area_delta}")
        if not (0.0 <= self.max_clipping_low_rate <= 1.0):
            raise ValueError(f"max_clipping_low_rate must be in [0, 1], got {self.max_clipping_low_rate}")
        if not (0.0 <= self.max_clipping_high_rate <= 1.0):
            raise ValueError(f"max_clipping_high_rate must be in [0, 1], got {self.max_clipping_high_rate}")
        if self.min_pixel_abs_diff < 0.0:
            raise ValueError(f"min_pixel_abs_diff must be >= 0, got {self.min_pixel_abs_diff}")


@dataclass
class QualityValidationConfig:
    """Overall quality validation configuration containing profile policies."""
    profiles: Dict[str, QualityThresholdProfile] = field(default_factory=lambda: {
        "generative": QualityThresholdProfile(
            min_ssim=0.50,
            max_lpips=0.35,
            min_edge_iou=0.15,
            max_rust_area_delta=0.10,
            max_clipping_low_rate=0.05,
            max_clipping_high_rate=0.05,
            min_pixel_abs_diff=1.0,
        ),
        "traditional": QualityThresholdProfile(
            min_ssim=0.65,
            max_lpips=0.25,
            min_edge_iou=0.20,
            max_rust_area_delta=0.08,
            max_clipping_low_rate=0.05,
            max_clipping_high_rate=0.05,
            min_pixel_abs_diff=1.0,
        ),
    })
    default_profile: str = "generative"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    lpips_net: str = "alex"
    use_gpu: bool = True

    def validate(self) -> None:
        if not self.profiles:
            raise ValueError("profiles dict cannot be empty")
        if self.default_profile not in self.profiles:
            raise ValueError(f"default_profile '{self.default_profile}' not in profiles: {list(self.profiles.keys())}")
        for name, prof in self.profiles.items():
            prof.validate()
        if not (0 <= self.canny_low_threshold < self.canny_high_threshold <= 255):
            raise ValueError(f"Invalid Canny thresholds: [{self.canny_low_threshold}, {self.canny_high_threshold}]")

    def get_profile(self, profile_name: Optional[str] = None) -> QualityThresholdProfile:
        name = profile_name or self.default_profile
        if name not in self.profiles:
            raise KeyError(f"Unknown profile '{name}'. Available: {list(self.profiles.keys())}")
        return self.profiles[name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_profile": self.default_profile,
            "canny_low_threshold": self.canny_low_threshold,
            "canny_high_threshold": self.canny_high_threshold,
            "lpips_net": self.lpips_net,
            "use_gpu": self.use_gpu,
            "profiles": {k: asdict(v) for k, v in self.profiles.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityValidationConfig":
        profiles = {}
        for k, v in data.get("profiles", {}).items():
            profiles[k] = QualityThresholdProfile(**v)
        return cls(
            profiles=profiles,
            default_profile=data.get("default_profile", "generative"),
            canny_low_threshold=data.get("canny_low_threshold", 100),
            canny_high_threshold=data.get("canny_high_threshold", 200),
            lpips_net=data.get("lpips_net", "alex"),
            use_gpu=data.get("use_gpu", True),
        )

    @classmethod
    def from_json(cls, path: Path) -> "QualityValidationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    load_json = from_json

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
