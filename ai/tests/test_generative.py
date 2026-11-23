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
from pipeaugment_gen.prompts import get_prompt_for_condition, get_negative_prompt, CONDITIONS, BASE_INTENT
from pipeaugment_gen.metadata import (
    derive_parent_seed,
    make_output_id,
    evaluate_technical_checks,
    compute_ssim_rgb,
    GenerativeOutputRecord,
)
from pipeaugment_gen.engine import GenerativeDiffusionEngine
from scripts.run_diffusion_pilot import select_pilot_parents


def test_config_validation():
    """Verify GenerativePilotConfig validates parameters and catches invalid settings."""
    cfg = GenerativePilotConfig()
    cfg.validate()  # Default should pass

    # Verify both img2img and controlnet modes validate cleanly
    cfg_plain = GenerativePilotConfig(mode="plain_img2img")
    cfg_plain.validate()
    assert not cfg_plain.is_controlnet
    assert cfg_plain.canonical_mode == "img2img"

    cfg_img2img = GenerativePilotConfig(mode="img2img")
    cfg_img2img.validate()
    assert not cfg_img2img.is_controlnet
    assert cfg_img2img.canonical_mode == "img2img"

    cfg_cnet_canny = GenerativePilotConfig(mode="controlnet_canny")
    cfg_cnet_canny.validate()
    assert cfg_cnet_canny.is_controlnet
    assert cfg_cnet_canny.canonical_mode == "controlnet_canny"

    cfg_cnet_img2img = GenerativePilotConfig(mode="controlnet_img2img")
    cfg_cnet_img2img.validate()
    assert cfg_cnet_img2img.is_controlnet
    assert cfg_cnet_img2img.canonical_mode == "controlnet_canny"

    with pytest.raises(ValueError, match="Unsupported mode"):
        GenerativePilotConfig(mode="unsupported_xyz").validate()

    with pytest.raises(ValueError, match="num_inference_steps"):
        GenerativePilotConfig(num_inference_steps=2).validate()

    with pytest.raises(ValueError, match="guidance_scale"):
        GenerativePilotConfig(guidance_scale=0.5).validate()

    with pytest.raises(ValueError, match="Strength"):
        GenerativePilotConfig(strengths=[0.01]).validate()

    with pytest.raises(ValueError, match="Target size"):
        GenerativePilotConfig(target_size=(256, 256)).validate()


def test_prompt_construction():
    """Verify prompt builder neutral intent and condition appending without class leaks."""
    for cond_name in ["low_light_inspection", "directional_industrial_lighting"]:
        prompt = get_prompt_for_condition(cond_name)
        assert BASE_INTENT in prompt
        assert CONDITIONS[cond_name]["prompt_extension"] in prompt
        # Confirm no class-specific leaks
        assert "add rust" not in prompt
        assert "remove rust" not in prompt

    with pytest.raises(ValueError, match="Unknown condition"):
        get_prompt_for_condition("nonexistent_condition")

    neg_prompt = get_negative_prompt()
    assert "changed pipe geometry" in neg_prompt
    assert "invented surface defects" in neg_prompt


def test_stable_seed_derivation():
    """Verify deterministic, platform-independent seed derivation."""
    parent_id = "test_parent_uuid_12345"
    seed1 = derive_parent_seed(parent_id, 42)
    seed2 = derive_parent_seed(parent_id, 42)
    assert seed1 == seed2
    assert isinstance(seed1, int)
    assert 0 <= seed1 <= (2**31 - 1)

    # Different parent gets different seed
    seed3 = derive_parent_seed("other_parent_uuid", 42)
    assert seed1 != seed3


def test_output_naming():
    """Verify structured output naming conventions."""
    out_id = make_output_id("005f6893-350c-4325-b357-7406681884b9", "low_light_inspection", 0.20)
    assert out_id == "gen_diff_005f6893-350c-4325-b_low_light_inspection_s20"

    out_id2 = make_output_id("005f6893-350c-4325-b357-7406681884b9", "directional_industrial_lighting", 0.35)
    assert out_id2 == "gen_diff_005f6893-350c-4325-b_directional_industrial_lighting_s35"


def test_training_only_parent_selection():
    """Verify pilot parents are deterministically selected strictly from train.csv."""
    train_manifest = PROJECT_ROOT / "datasets" / "manifests" / "train.csv"
    val_manifest = PROJECT_ROOT / "datasets" / "manifests" / "val.csv"
    test_manifest = PROJECT_ROOT / "datasets" / "manifests" / "test.csv"

    parents = select_pilot_parents(train_manifest, num_per_class=3, selection_seed=42)
    assert len(parents) == 6
    assert (parents["label"] == "non_corroded").sum() == 3
    assert (parents["label"] == "corroded").sum() == 3

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    test_df = pd.read_csv(test_manifest)

    parent_ids = set(parents["image_id"])
    assert parent_ids.issubset(set(train_df["image_id"]))
    assert parent_ids.isdisjoint(set(val_df["image_id"]))
    assert parent_ids.isdisjoint(set(test_df["image_id"]))


def test_ssim_rgb_computation():
    """Verify multi-channel SSIM calculation on synthetic arrays."""
    img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    ssim_identical = compute_ssim_rgb(img, img)
    assert abs(ssim_identical - 1.0) < 1e-4

    # Shifted image has high but < 1.0 SSIM
    img_shifted = np.clip(img.astype(int) + 15, 0, 255).astype(np.uint8)
    ssim_shifted = compute_ssim_rgb(img, img_shifted)
    assert 0.8 < ssim_shifted < 1.0


def test_technical_sanity_checks():
    """Verify technical checks detect valid generations and flag degenerate outputs."""
    parent = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    valid_gen = np.clip(parent.astype(int) + 10, 0, 255).astype(np.uint8)

    stats = evaluate_technical_checks(valid_gen, parent)
    assert stats["status"] == "passed"
    assert stats["dimension_preserved"] is True
    assert stats["dtype_uint8"] is True
    assert stats["is_constant"] is False

    # Degenerate: identical pixels
    stats_identical = evaluate_technical_checks(parent, parent)
    assert stats_identical["status"] == "failed"
    assert stats_identical["is_identical_pixels"] is True

    # Degenerate: all black
    black_img = np.zeros((512, 512, 3), dtype=np.uint8)
    stats_black = evaluate_technical_checks(black_img, parent)
    assert stats_black["status"] == "failed"
    assert stats_black["is_all_black"] is True

    # Degenerate: safety violation
    stats_safety = evaluate_technical_checks(valid_gen, parent, safety_flag=True)
    assert stats_safety["status"] == "failed"
    assert stats_safety["safety_violation"] is True


def test_mocked_engine_generation_and_metadata_completeness():
    """Test engine generation with a mocked Diffusers pipeline (no model download)."""
    cfg = GenerativePilotConfig()
    engine = GenerativeDiffusionEngine(cfg)

    # Mock the pipeline
    mock_pipe = MagicMock()
    mock_result = MagicMock()

    # Synthetic output image: slight variation of input
    synthetic_out = Image.new("RGB", (512, 512), color=(120, 140, 160))
    mock_result.images = [synthetic_out]
    mock_result.has_nsfw_concept = [False]
    mock_pipe.return_value = mock_result
    mock_pipe.__class__.__name__ = "MockStableDiffusionImg2ImgPipeline"
    mock_pipe.scheduler.__class__.__name__ = "PNDMScheduler"
    mock_pipe.scheduler.config = {"beta_start": 0.00085}
    mock_pipe.safety_checker = MagicMock()

    engine._pipe = mock_pipe
    engine._resolved_model_info = {
        "pipeline_class": "MockStableDiffusionImg2ImgPipeline",
        "scheduler_class": "PNDMScheduler",
        "scheduler_config": {"beta_start": 0.00085},
    }

    parent_img = np.full((512, 512, 3), 100, dtype=np.uint8)
    out_arr, record, tech_stats = engine.generate(
        source_image=parent_img,
        parent_id="test_parent_1",
        parent_sha256="dummy_sha256",
        parent_split="train",
        source_label="corroded",
        condition="low_light_inspection",
        strength=0.20,
        seed=12345
    )

    # Check mock called with correct parameters
    mock_pipe.assert_called_once()
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["strength"] == 0.20
    assert call_kwargs["num_inference_steps"] == 30
    assert call_kwargs["guidance_scale"] == 5.0

    # Verify metadata record completeness
    assert record.output_id == "gen_diff_test_parent_1_low_light_inspection_s20"
    assert record.parent_id == "test_parent_1"
    assert record.condition == "low_light_inspection"
    assert record.strength == 0.20
    assert record.effective_denoising_steps == 6  # 30 * 0.20
    assert record.human_review_status == "pending"
    assert record.training_eligibility is False
    assert record.technical_validation_status == "passed"
    assert out_arr.shape == (512, 512, 3)


def test_pending_outputs_excluded_from_training():
    """Verify that pilot outputs cannot be treated as valid training data."""
    record = GenerativeOutputRecord(
        output_id="out1",
        parent_id="p1",
        parent_sha256="sha",
        parent_split="train",
        source_label="corroded",
        model_id="sd15",
        model_revision="main",
        pipeline_class="pipe",
        scheduler_config={},
        positive_prompt="p",
        negative_prompt="n",
        condition="low_light",
        strength=0.20,
        num_inference_steps=30,
        effective_denoising_steps=6,
        guidance_scale=5.0,
        seed=42,
        input_dimensions="512x512",
        output_dimensions="512x512",
        preprocessing="rgb",
        precision="fp16",
        device="cuda",
        offload_strategy="none",
        runtime_seconds=1.0,
        peak_gpu_allocated_mb=100.0,
        timestamp_utc="2026-11-22T00:00:00Z",
        technical_validation_status="passed",
        safety_flag=False,
        human_review_status="pending",
        training_eligibility=False
    )
    rec_dict = record.to_dict()
    assert rec_dict["training_eligibility"] is False
    assert rec_dict["human_review_status"] == "pending"


def test_raw_source_immutability(tmp_path):
    """Verify load_source_rgb_image does not alter original image file or array."""
    raw_path = tmp_path / "test_raw.png"
    img = Image.new("RGB", (512, 512), color=(50, 100, 150))
    img.save(raw_path)

    with open(raw_path, "rb") as f:
        bytes_before = f.read()

    from scripts.run_diffusion_pilot import load_source_rgb_image
    loaded = load_source_rgb_image(raw_path)

    with open(raw_path, "rb") as f:
        bytes_after = f.read()

    assert bytes_before == bytes_after
    assert loaded.shape == (512, 512, 3)
