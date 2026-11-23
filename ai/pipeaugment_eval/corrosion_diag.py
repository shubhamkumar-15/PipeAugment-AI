"""
PipeAugment AI — Diagnostic Rust-Region Stability (HSV/Lab Heuristic).

CRITICAL SCIENTIFIC NOTICE:
This module provides an approximate color-threshold candidate diagnostic.
It is NOT ground-truth metallurgical corrosion segmentation or ultrasonic testing.
It serves solely as an exploratory screening heuristic to track whether
surface color distributions shift drastically during augmentation.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


# Documented color thresholds for heuristic rust-candidate segmentation
HSV_RUST_HUE_MIN = 6       # Orange-brown lower hue bound (OpenCV scale: 0-180)
HSV_RUST_HUE_MAX = 26      # Yellow-brown upper hue bound
HSV_RUST_SAT_MIN = 45      # Minimum saturation (filters neutral steel gray)
HSV_RUST_SAT_MAX = 255
HSV_RUST_VAL_MIN = 35      # Minimum brightness (filters deep shadow)
HSV_RUST_VAL_MAX = 210     # Maximum brightness (filters blown-out specular highlights)

LAB_RUST_A_MIN = 130       # Red-green axis lower bound (OpenCV uint8 scale: 0-255)
LAB_RUST_B_MIN = 135       # Yellow-blue axis lower bound


def compute_rust_candidate_mask(
    rgb_image: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes an approximate binary mask of candidate rust pixels in an RGB image
    using joint HSV and CIE-Lab color space thresholding.

    Returns:
        (binary_mask, rust_ratio)
        where binary_mask is a 2D boolean ndarray and rust_ratio is in [0.0, 1.0].
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != np.uint8:
        raise ValueError("rgb_image must be a 3-channel uint8 ndarray of shape (H, W, 3)")

    # 1. HSV candidate mask
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_mask = (
        (hsv[:, :, 0] >= HSV_RUST_HUE_MIN)
        & (hsv[:, :, 0] <= HSV_RUST_HUE_MAX)
        & (hsv[:, :, 1] >= HSV_RUST_SAT_MIN)
        & (hsv[:, :, 1] <= HSV_RUST_SAT_MAX)
        & (hsv[:, :, 2] >= HSV_RUST_VAL_MIN)
        & (hsv[:, :, 2] <= HSV_RUST_VAL_MAX)
    )

    # 2. Lab candidate refinement (elevated red and yellow chromaticity)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    lab_mask = (
        (lab[:, :, 1] >= LAB_RUST_A_MIN)
        & (lab[:, :, 2] >= LAB_RUST_B_MIN)
    )

    # Joint candidate mask
    rust_mask = np.logical_and(hsv_mask, lab_mask)
    ratio = float(np.mean(rust_mask))

    return rust_mask, round(ratio, 5)


def evaluate_rust_stability(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates approximate rust-candidate region stability between source and generated image.

    Measures:
      - source_rust_ratio: fraction of source pixels in rust-color candidate mask
      - output_rust_ratio: fraction of generated pixels in rust-color candidate mask
      - rust_area_delta: (output_rust_ratio - source_rust_ratio)
      - rust_mask_iou: Intersection over Union of candidate masks.
        * If both masks are empty, returns None (not applicable), NOT 0.0 or 1.0.

    CRITICAL SCIENTIFIC LABELING:
      Tagged explicitly as 'approximate_color_diagnostic'.
      Never claim high IoU proves ground-truth corrosion preservation.
    """
    src_mask, src_ratio = compute_rust_candidate_mask(source_rgb)
    gen_mask, gen_ratio = compute_rust_candidate_mask(generated_rgb)

    inter = int(np.logical_and(src_mask, gen_mask).sum())
    union = int(np.logical_or(src_mask, gen_mask).sum())

    if union == 0:
        # Both masks are completely empty (e.g. pristine non-corroded pipe)
        mask_iou = None
    else:
        mask_iou = round(float(inter / union), 5)

    area_delta = round(gen_ratio - src_ratio, 5)

    return {
        "diagnostic_type": "approximate_color_diagnostic",
        "scientific_caveat": "Heuristic color-candidate mask only; NOT ground-truth metallurgical corrosion segmentation.",
        "source_rust_ratio": src_ratio,
        "output_rust_ratio": gen_ratio,
        "rust_area_delta": area_delta,
        "rust_mask_iou": mask_iou,
    }
