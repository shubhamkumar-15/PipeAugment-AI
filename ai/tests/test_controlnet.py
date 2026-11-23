import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_gen.config import GenerativePilotConfig
from pipeaugment_gen.metadata import (
    derive_parent_seed,
    make_controlnet_output_id,
    compute_canny_edge_map,
    compute_edge_overlap_diagnostics,
    evaluate_technical_checks,
    GenerativeOutputRecord,
)
from pipeaugment_gen.engine import GenerativeDiffusionEngine


def test_controlnet_config_validation():
    """Verify GenerativePilotConfig validates ControlNet parameters."""
    # Valid ControlNet config
    cfg = GenerativePilotConfig(
        mode="controlnet_img2img",
        controlnet_id="lllyasviel/control_v11p_sd15_canny",
        controlnet_scales=[0.5, 0.8],
        canny_low_threshold=100,
        canny_high_threshold=200,
    )
    cfg.validate()

    # Invalid mode
    with pytest.raises(ValueError, match="Unsupported mode"):
        GenerativePilotConfig(mode="invalid_mode").validate()

    # Empty controlnet_id in controlnet mode
    with pytest.raises(ValueError, match="controlnet_id must not be empty"):
        GenerativePilotConfig(mode="controlnet_img2img", controlnet_id="").validate()

    # Invalid controlnet_scales
    with pytest.raises(ValueError, match="ControlNet scale"):
        GenerativePilotConfig(mode="controlnet_img2img", controlnet_scales=[-0.5]).validate()

    # Invalid canny thresholds
    with pytest.raises(ValueError, match="Invalid Canny thresholds"):
        GenerativePilotConfig(
            mode="controlnet_img2img",
            canny_low_threshold=200,
            canny_high_threshold=100,
        ).validate()


def test_controlnet_output_naming():
    """Verify deterministic ControlNet output IDs."""
    parent_id = "27a30c35-ace6-401d-86ba-e75895d9f016_png_jpg.rf.dc5d5a2a4e1e6f177b67cd9d5fad7a57"
    out_id_05 = make_controlnet_output_id(parent_id, "low_light_inspection", 0.35, 0.50)
    assert out_id_05 == "gen_cnet_canny_27a30c35-ace6-401d-8_low_light_inspection_s35_c50"

    out_id_08 = make_controlnet_output_id(parent_id, "directional_industrial_lighting", 0.35, 0.80)
    assert out_id_08 == "gen_cnet_canny_27a30c35-ace6-401d-8_directional_industrial_lighting_s35_c80"


def test_canny_edge_map_computation():
    """Verify Canny edge map outputs 2D uint8 with correct density."""
    # Synthetic RGB image with a sharp edge
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:, 256:] = 255  # Vertical boundary

    canny_map, density = compute_canny_edge_map(img, low_thresh=100, high_thresh=200)
    assert canny_map.shape == (512, 512)
    assert canny_map.dtype == np.uint8
    assert 0.0 < density < 0.05  # Single vertical line has small positive density
    assert (canny_map == 255).sum() > 0


def test_edge_overlap_diagnostics_edge_cases():
    """Verify edge overlap IoU and Dice heuristic computations under all edge conditions."""
    # Case 1: Identical edge masks
    canny = np.zeros((512, 512), dtype=np.uint8)
    canny[100:150, 100:150] = 255
    # Construct an image that produces exactly the same edge mask
    gen_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    gen_rgb[100:150, 100:150] = 255

    # Test with identical source and generated canny
    with patch("pipeaugment_gen.metadata.compute_canny_edge_map", return_value=(canny, 0.01)):
        diag = compute_edge_overlap_diagnostics(canny, gen_rgb)
        assert diag["edge_overlap_iou_heuristic"] == 1.0
        assert diag["edge_overlap_dice_heuristic"] == 1.0

    # Case 2: Completely disjoint edge masks
    other_canny = np.zeros((512, 512), dtype=np.uint8)
    other_canny[300:350, 300:350] = 255
    with patch("pipeaugment_gen.metadata.compute_canny_edge_map", return_value=(other_canny, 0.01)):
        diag = compute_edge_overlap_diagnostics(canny, gen_rgb)
        assert diag["edge_overlap_iou_heuristic"] == 0.0
        assert diag["edge_overlap_dice_heuristic"] == 0.0

    # Case 3: Empty edge masks (no edges detected in either)
    empty_canny = np.zeros((512, 512), dtype=np.uint8)
    with patch("pipeaugment_gen.metadata.compute_canny_edge_map", return_value=(empty_canny, 0.0)):
        diag = compute_edge_overlap_diagnostics(empty_canny, gen_rgb)
        assert diag["edge_overlap_iou_heuristic"] == 1.0
        assert diag["edge_overlap_dice_heuristic"] == 1.0


def test_mocked_controlnet_engine_generation():
    """Test ControlNet engine generation with mocked pipeline (zero download)."""
    cfg = GenerativePilotConfig(
        mode="controlnet_img2img",
        controlnet_id="lllyasviel/control_v11p_sd15_canny",
        controlnet_scales=[0.5, 0.8],
        strengths=[0.35],
    )
    engine = GenerativeDiffusionEngine(cfg)

    # Mock pipeline and inference
    mock_pipe = MagicMock()
    mock_result = MagicMock()
    synthetic_out = Image.new("RGB", (512, 512), color=(110, 130, 150))
    mock_result.images = [synthetic_out]
    mock_result.has_nsfw_concept = [False]
    mock_pipe.return_value = mock_result
    mock_pipe.__class__.__name__ = "MockControlNetImg2ImgPipeline"
    mock_pipe.scheduler.__class__.__name__ = "PNDMScheduler"
    mock_pipe.scheduler.config = {"beta_start": 0.00085}
    mock_pipe.safety_checker = MagicMock()

    engine._pipe = mock_pipe
    engine._loaded_controlnet_revision = "mock_cnet_rev_123"
    engine._resolved_model_info = {
        "pipeline_class": "MockControlNetImg2ImgPipeline",
        "scheduler_class": "PNDMScheduler",
        "scheduler_config": {"beta_start": 0.00085},
    }

    parent_img = np.full((512, 512, 3), 120, dtype=np.uint8)
    source_canny = np.zeros((512, 512), dtype=np.uint8)
    source_canny[200:250, 200:250] = 255

    out_arr, record, tech_stats = engine.generate(
        source_image=parent_img,
        parent_id="parent_uuid_cnet_test",
        parent_sha256="dummy_sha256",
        parent_split="train",
        source_label="corroded",
        condition="low_light_inspection",
        strength=0.35,
        seed=999,
        control_scale=0.5,
        source_canny=source_canny,
    )

    # Check mock called with correct ControlNet arguments
    mock_pipe.assert_called_once()
    call_kwargs = mock_pipe.call_args[1]
    assert "image" in call_kwargs
    assert "control_image" in call_kwargs
    assert "controlnet_conditioning_scale" in call_kwargs
    assert call_kwargs["controlnet_conditioning_scale"] == 0.5
    assert call_kwargs["strength"] == 0.35
    assert call_kwargs["num_inference_steps"] == 30

    # Verify metadata completeness
    assert record.generation_mode == "controlnet_img2img"
    assert record.controlnet_id == "lllyasviel/control_v11p_sd15_canny"
    assert record.control_scale == 0.5
    assert record.training_eligibility is False
    assert record.human_review_status == "pending"
    assert record.output_id == "gen_cnet_canny_parent_uuid_cnet_tes_low_light_inspection_s35_c50"
    assert record.effective_denoising_steps == int(30 * 0.35)


def test_sources_manifest_parity_and_train_isolation():
    """Verify sources.csv is strictly from train.csv and matches diffusion_v1."""
    diff_sources_path = PROJECT_ROOT / "experiments" / "pilots" / "diffusion_v1" / "sources.csv"
    train_manifest_path = PROJECT_ROOT / "datasets" / "manifests" / "train.csv"
    val_manifest_path = PROJECT_ROOT / "datasets" / "manifests" / "val.csv"
    test_manifest_path = PROJECT_ROOT / "datasets" / "manifests" / "test.csv"

    assert diff_sources_path.exists(), "diffusion_v1/sources.csv must exist"
    sources_df = pd.read_csv(diff_sources_path)
    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    # Exactly 6 parents (3 non-corroded, 3 corroded)
    assert len(sources_df) == 6
    assert (sources_df["label"] == "non_corroded").sum() == 3
    assert (sources_df["label"] == "corroded").sum() == 3

    # All parents strictly in train.csv
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    for p_id in sources_df["image_id"]:
        assert p_id in train_ids, f"Parent {p_id} not in train split!"
        assert p_id not in val_ids, f"Parent {p_id} leaked from val split!"
        assert p_id not in test_ids, f"Parent {p_id} leaked from test split!"
