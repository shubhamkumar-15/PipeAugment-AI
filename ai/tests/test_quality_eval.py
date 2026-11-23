import sys
import json
import pytest
import numpy as np
import cv2
import torch
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup paths
AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_eval.config import QualityThresholdProfile, QualityValidationConfig
from pipeaugment_eval.metrics import (
    evaluate_image_validity,
    compute_ssim_rgb,
    compute_edge_retention,
    LPIPSEvaluator,
)
from pipeaugment_eval.corrosion_diag import (
    compute_rust_candidate_mask,
    evaluate_rust_stability,
)
from pipeaugment_eval.validator import QualityValidator


# =====================================================================
# 1. Configuration & Threshold Validation Tests
# =====================================================================

def test_quality_profile_validation():
    """Verify validation of bounds on QualityThresholdProfile."""
    valid_profile = QualityThresholdProfile(
        min_ssim=0.50,
        max_lpips=0.35,
        min_edge_iou=0.15,
        max_rust_area_delta=0.10,
    )
    valid_profile.validate()

    with pytest.raises(ValueError, match="min_ssim"):
        QualityThresholdProfile(min_ssim=1.5).validate()

    with pytest.raises(ValueError, match="min_ssim"):
        QualityThresholdProfile(min_ssim=-0.1).validate()

    with pytest.raises(ValueError, match="max_lpips"):
        QualityThresholdProfile(max_lpips=2.5).validate()

    with pytest.raises(ValueError, match="min_edge_iou"):
        QualityThresholdProfile(min_edge_iou=-0.01).validate()

    with pytest.raises(ValueError, match="max_rust_area_delta"):
        QualityThresholdProfile(max_rust_area_delta=1.5).validate()


def test_quality_validation_config_serialization(tmp_path):
    """Verify serialization to JSON and round-trip reloading."""
    config = QualityValidationConfig()
    config_file = tmp_path / "test_config.json"
    config.save_json(config_file)

    assert config_file.exists()
    loaded = QualityValidationConfig.load_json(config_file)
    assert loaded.lpips_net == config.lpips_net
    assert "generative" in loaded.profiles
    assert "traditional" in loaded.profiles
    assert loaded.profiles["generative"].min_ssim == 0.50
    assert loaded.profiles["traditional"].min_ssim == 0.65


# =====================================================================
# 2. Image Technical Validity Tests
# =====================================================================

def test_image_validity_valid_pair():
    """Verify a valid image pair passes validity checks."""
    np.random.seed(42)
    parent = np.random.randint(50, 200, size=(128, 128, 3), dtype=np.uint8)
    generated = parent.copy()
    generated[10:30, 10:30, :] = 100  # Slight perturbation

    res = evaluate_image_validity(generated, parent)
    assert res["valid"] is True
    assert res["dimension_preserved"] is True
    assert res["dtype_uint8"] is True
    assert res["finite_pixels"] is True
    assert res["is_constant"] is False
    assert res["is_all_black"] is False
    assert res["is_all_white"] is False
    assert res["is_identical_pixels"] is False
    assert res["mean_pixel_abs_diff"] > 0.0


def test_image_validity_shape_mismatch():
    """Verify dimension mismatch is flagged."""
    parent = np.full((128, 128, 3), 100, dtype=np.uint8)
    generated = np.full((64, 64, 3), 100, dtype=np.uint8)

    res = evaluate_image_validity(generated, parent)
    assert res["valid"] is False
    assert res["dimension_preserved"] is False


def test_image_validity_identical_copy():
    """Verify identical bit-for-bit image is flagged."""
    parent = np.full((128, 128, 3), 100, dtype=np.uint8)
    generated = parent.copy()

    res = evaluate_image_validity(generated, parent)
    assert res["valid"] is False
    assert res["is_identical_pixels"] is True
    assert res["mean_pixel_abs_diff"] == 0.0


def test_image_validity_all_black_and_white():
    """Verify all-black or all-white degenerate images are flagged."""
    parent = np.random.randint(50, 200, size=(64, 64, 3), dtype=np.uint8)

    black = np.zeros((64, 64, 3), dtype=np.uint8)
    res_b = evaluate_image_validity(black, parent)
    assert res_b["valid"] is False
    assert res_b["is_all_black"] is True
    assert res_b["is_constant"] is True

    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    res_w = evaluate_image_validity(white, parent)
    assert res_w["valid"] is False
    assert res_w["is_all_white"] is True
    assert res_w["is_constant"] is True


# =====================================================================
# 3. SSIM Tests
# =====================================================================

def test_compute_ssim_rgb_identical_and_perturbed():
    """Verify SSIM gives 1.0 for identical images and drops under perturbation."""
    np.random.seed(42)
    img1 = np.random.randint(30, 220, size=(128, 128, 3), dtype=np.uint8)
    ssim_self = compute_ssim_rgb(img1, img1)
    assert pytest.approx(ssim_self, rel=1e-3) == 1.0

    # Add Gaussian noise
    noise = np.random.normal(0, 25, img1.shape).astype(np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ssim_noisy = compute_ssim_rgb(img1, img2)
    assert 0.0 < ssim_noisy < 1.0
    assert ssim_noisy < 0.95


# =====================================================================
# 4. Edge Retention Tests
# =====================================================================

def test_compute_edge_retention():
    """Verify edge retention IoU and Dice heuristics."""
    # Synthetic image with a bright vertical edge in the center
    img1 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img1[:, 64:, :] = 220  # High contrast edge at column 64

    # Identical image -> IoU = 1.0
    ret_self = compute_edge_retention(img1, img1, low_thresh=50, high_thresh=150)
    assert ret_self["edge_overlap_iou"] == 1.0
    assert ret_self["edge_overlap_dice"] == 1.0
    assert ret_self["source_edge_density"] > 0.0

    # Completely flat image with no edges on both sides -> handles empty union
    flat1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    flat2 = np.full((128, 128, 3), 100, dtype=np.uint8)
    ret_flat = compute_edge_retention(flat1, flat2)
    assert ret_flat["edge_overlap_iou"] == 1.0
    assert ret_flat["source_edge_density"] == 0.0

    # Disjoint edges
    img2 = np.full((128, 128, 3), 50, dtype=np.uint8)
    img2[64:, :, :] = 220  # Horizontal edge
    ret_disjoint = compute_edge_retention(img1, img2, low_thresh=50, high_thresh=150)
    assert ret_disjoint["edge_overlap_iou"] < 0.2


# =====================================================================
# 5. Diagnostic Rust-Region Stability Tests
# =====================================================================

def test_rust_candidate_mask_and_stability():
    """Verify HSV/Lab diagnostic rust-candidate segmentation."""
    # 1. Plain neutral steel gray -> 0 rust ratio
    gray_pipe = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, gray_ratio = compute_rust_candidate_mask(gray_pipe)
    assert gray_ratio == 0.0

    # 2. Add an orange-brown rust color patch:
    # Rust in RGB is approximately (180, 75, 25)
    corroded_pipe = gray_pipe.copy()
    corroded_pipe[20:60, 20:60, :] = [180, 75, 25]
    _, corroded_ratio = compute_rust_candidate_mask(corroded_pipe)
    assert corroded_ratio > 0.10

    # 3. Both clean pipes -> rust_mask_iou is None (not 0.0 or 1.0)
    stability_clean = evaluate_rust_stability(gray_pipe, gray_pipe)
    assert stability_clean["source_rust_ratio"] == 0.0
    assert stability_clean["output_rust_ratio"] == 0.0
    assert stability_clean["rust_mask_iou"] is None
    assert stability_clean["rust_area_delta"] == 0.0
    assert stability_clean["diagnostic_type"] == "approximate_color_diagnostic"

    # 4. Detected shift when rust is expanded
    expanded_pipe = corroded_pipe.copy()
    expanded_pipe[10:80, 10:80, :] = [180, 75, 25]
    stability_expanded = evaluate_rust_stability(corroded_pipe, expanded_pipe)
    assert stability_expanded["rust_area_delta"] > 0.0
    assert stability_expanded["rust_mask_iou"] is not None


# =====================================================================
# 6. LPIPSEvaluator Mocked Unit Tests
# =====================================================================

def test_lpips_evaluator_mocked():
    """Verify LPIPSEvaluator normalizes tensors correctly without downloading weights."""
    mock_lpips_instance = MagicMock()
    mock_lpips_instance.to.return_value = mock_lpips_instance
    mock_lpips_instance.return_value = torch.tensor([[[[0.1234]]]])

    with patch("lpips.LPIPS", return_value=mock_lpips_instance):
        evaluator = LPIPSEvaluator(net="alex", use_gpu=False)

        img1 = np.full((64, 64, 3), 50, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 60, dtype=np.uint8)

        score = evaluator.evaluate(img1, img2)
        assert pytest.approx(score, rel=1e-3) == 0.1234

        # Check call arguments to mock
        assert mock_lpips_instance.called
        args, _ = mock_lpips_instance.call_args
        t1, t2 = args
        # Verify shape (1, 3, 64, 64)
        assert t1.shape == (1, 3, 64, 64)
        assert t2.shape == (1, 3, 64, 64)
        # Verify normalization range in [-1.0, 1.0]
        assert float(t1.min()) >= -1.0
        assert float(t1.max()) <= 1.0


# =====================================================================
# 7. QualityValidator Decision Gate Tests
# =====================================================================

def test_quality_validator_decision_gate():
    """Verify pass, flag_for_review, and fail decisions from QualityValidator."""
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = 0.10

    config = QualityValidationConfig()
    validator = QualityValidator(config=config, lpips_evaluator=mock_evaluator)

    # Synthetic pipe: dark background (30), steel cylinder in center (120)
    source = np.full((128, 128, 3), 30, dtype=np.uint8)
    source[:, 30:98, :] = 120

    # 1. High-quality perturbation (slight brightness shift) -> PASS
    gen_good = source.copy()
    gen_good[:, 30:98, :] = 125
    res_pass = validator.validate_pair(source, gen_good, category="generative")
    assert res_pass["quality_status"] == "pass"
    assert len(res_pass["flags"]) == 0

    # 2. Exceed LPIPS threshold -> FLAG_FOR_REVIEW
    mock_evaluator.evaluate.return_value = 0.45  # Exceeds max_lpips 0.35
    res_flag = validator.validate_pair(source, gen_good, category="generative")
    assert res_flag["quality_status"] == "flag_for_review"
    assert "high_lpips" in res_flag["flags"]

    # 3. Fatal invalidity (all black) -> FAIL
    black = np.zeros((128, 128, 3), dtype=np.uint8)
    res_fail = validator.validate_pair(source, black, category="generative")
    assert res_fail["quality_status"] == "fail"
    assert "all_black" in res_fail["flags"]
