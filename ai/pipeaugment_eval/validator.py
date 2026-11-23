"""
PipeAugment AI — QualityValidator.
Combines technical image validity, multi-channel SSIM, AlexNet LPIPS perceptual distance,
Canny edge retention, and HSV/Lab diagnostic rust-candidate stability into a unified
three-state decision gate: 'pass' | 'flag_for_review' | 'fail'.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .config import QualityValidationConfig, QualityThresholdProfile
from .metrics import evaluate_image_validity, compute_ssim_rgb, compute_edge_retention, LPIPSEvaluator
from .corrosion_diag import evaluate_rust_stability


class QualityValidator:
    """
    Quality validator applying configurable threshold profiles across traditional
    and generative pipeline augmentations.
    """

    def __init__(
        self,
        config: Optional[QualityValidationConfig] = None,
        lpips_evaluator: Optional[LPIPSEvaluator] = None,
    ):
        self.config = config or QualityValidationConfig()
        self.config.validate()
        self._lpips = lpips_evaluator

    @property
    def lpips(self) -> LPIPSEvaluator:
        if self._lpips is None:
            self._lpips = LPIPSEvaluator(
                net=self.config.lpips_net,
                use_gpu=self.config.use_gpu,
            )
        return self._lpips

    def validate_pair(
        self,
        source_rgb: np.ndarray,
        generated_rgb: np.ndarray,
        category: str = "generative",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single (source, generated) image pair against the profile for `category`.

        Returns:
            Dict containing quality_status ('pass' | 'flag_for_review' | 'fail'),
            list of warning flags, detailed quantitative metrics, and metadata.
        """
        profile = self.config.get_profile(category)
        flags: List[str] = []

        # 1. Technical Image Validity Checks
        validity = evaluate_image_validity(generated_rgb, source_rgb)

        if not validity["dimension_preserved"]:
            flags.append("invalid_dimensions")
        if not validity["finite_pixels"]:
            flags.append("non_finite_pixels")
        if validity["is_constant"]:
            flags.append("constant_image")
        if validity["is_all_black"]:
            flags.append("all_black")
        if validity["is_all_white"]:
            flags.append("all_white")
        if validity["is_identical_pixels"]:
            flags.append("identical_copy")

        # If fundamentally invalid, immediately fail
        if not validity["valid"]:
            return {
                "quality_status": "fail",
                "flags": flags,
                "category": category,
                "validity": validity,
                "metrics": {
                    "ssim_rgb": 0.0,
                    "lpips": 1.0,
                    "mean_pixel_abs_diff": validity["mean_pixel_abs_diff"],
                    "clipping_low_rate": validity["clipping_low_rate"],
                    "clipping_high_rate": validity["clipping_high_rate"],
                },
                "corrosion_diagnostic": None,
                "metadata": metadata or {},
            }

        # Check clipping & identity thresholds
        if validity["clipping_low_rate"] > profile.max_clipping_low_rate:
            flags.append("excessive_clipping_low")
        if validity["clipping_high_rate"] > profile.max_clipping_high_rate:
            flags.append("excessive_clipping_high")
        if validity["mean_pixel_abs_diff"] < profile.min_pixel_abs_diff:
            flags.append("near_identical_copy")

        # 2. Structural & Perceptual Similarity
        ssim_val = compute_ssim_rgb(source_rgb, generated_rgb)
        if ssim_val < profile.min_ssim:
            flags.append("low_ssim")

        lpips_val = self.lpips.evaluate(source_rgb, generated_rgb)
        if lpips_val > profile.max_lpips:
            flags.append("high_lpips")

        # 3. Edge Retention Heuristic
        edge_diag = compute_edge_retention(
            source_rgb,
            generated_rgb,
            low_thresh=self.config.canny_low_threshold,
            high_thresh=self.config.canny_high_threshold,
        )
        if edge_diag["edge_overlap_iou"] < profile.min_edge_iou:
            flags.append("low_edge_iou")

        # 4. Diagnostic Rust-Region Stability (HSV/Lab Heuristic)
        rust_diag = evaluate_rust_stability(source_rgb, generated_rgb)
        if abs(rust_diag["rust_area_delta"]) > profile.max_rust_area_delta:
            flags.append("rust_area_shift_high")

        # 5. Three-state Decision Gate
        quality_status = "flag_for_review" if flags else "pass"

        return {
            "quality_status": quality_status,
            "flags": flags,
            "category": category,
            "validity": validity,
            "metrics": {
                "ssim_rgb": ssim_val,
                "lpips": lpips_val,
                "edge_overlap_iou": edge_diag["edge_overlap_iou"],
                "edge_overlap_dice": edge_diag["edge_overlap_dice"],
                "source_edge_density": edge_diag["source_edge_density"],
                "output_edge_density": edge_diag["output_edge_density"],
                "mean_pixel_abs_diff": validity["mean_pixel_abs_diff"],
                "clipping_low_rate": validity["clipping_low_rate"],
                "clipping_high_rate": validity["clipping_high_rate"],
                "source_rust_ratio": rust_diag["source_rust_ratio"],
                "output_rust_ratio": rust_diag["output_rust_ratio"],
                "rust_area_delta": rust_diag["rust_area_delta"],
                "rust_mask_iou": rust_diag["rust_mask_iou"],
            },
            "corrosion_diagnostic": rust_diag,
            "metadata": metadata or {},
        }
