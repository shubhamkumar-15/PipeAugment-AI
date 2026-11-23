"""
PipeAugment AI — Phase 4 Quality Validation Engine.
Evaluates structural (SSIM), perceptual (LPIPS), edge retention (Canny),
and diagnostic rust-region stability (HSV/Lab heuristic).
"""

from .config import QualityValidationConfig, QualityThresholdProfile
from .metrics import evaluate_image_validity, compute_ssim_rgb, compute_edge_retention, LPIPSEvaluator
from .corrosion_diag import compute_rust_candidate_mask, evaluate_rust_stability
from .validator import QualityValidator

__all__ = [
    "QualityValidationConfig",
    "QualityThresholdProfile",
    "evaluate_image_validity",
    "compute_ssim_rgb",
    "compute_edge_retention",
    "LPIPSEvaluator",
    "compute_rust_candidate_mask",
    "evaluate_rust_stability",
    "QualityValidator",
]
