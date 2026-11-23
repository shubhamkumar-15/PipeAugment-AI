import sys
import pytest
import numpy as np
import cv2
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_cv.config import TraditionalConfig
from pipeaugment_cv.engine import TraditionalAugmentationEngine, derive_output_seed
from pipeaugment_cv.transforms import (
    validate_rgb_image,
    apply_brightness,
    apply_contrast,
    apply_gamma,
    apply_uneven_illumination,
    apply_soft_shadow,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_downsample_restore,
)


@pytest.fixture
def synthetic_rgb_image():
    """Deterministic 64x64 non-saturated RGB synthetic image."""
    rng = np.random.default_rng(12345)
    # Range [60, 190] avoids floor/ceiling saturation
    return rng.integers(60, 190, size=(64, 64, 3), dtype=np.uint8)


def test_output_dimensions_preserved(synthetic_rgb_image):
    engine = TraditionalAugmentationEngine()
    for t_name in engine.config.lighting_transforms + engine.config.quality_transforms:
        out, params, status = engine.execute(synthetic_rgb_image, t_name, "test_parent", 1)
        assert out.shape == synthetic_rgb_image.shape, f"Shape mismatch for {t_name}: {out.shape}"
        assert status["dimension_preserved"] is True


def test_output_dtype_uint8(synthetic_rgb_image):
    engine = TraditionalAugmentationEngine()
    for t_name in engine.config.lighting_transforms + engine.config.quality_transforms:
        out, _, status = engine.execute(synthetic_rgb_image, t_name, "test_parent", 1)
        assert out.dtype == np.uint8, f"Dtype mismatch for {t_name}: {out.dtype}"
        assert status["dtype_uint8"] is True


def test_input_array_not_modified(synthetic_rgb_image):
    copy_orig = synthetic_rgb_image.copy()
    engine = TraditionalAugmentationEngine()
    engine.execute(synthetic_rgb_image, "brightness", "test_parent", 1, explicit_params={"factor": 1.2})
    assert np.array_equal(synthetic_rgb_image, copy_orig), "Input array was modified in-place!"


def test_rgb_channel_order_retained():
    """Verify red and blue channels are not swapped (BGR inversion check)."""
    # Create asymmetric color image: Red=200, Green=100, Blue=50
    asym_img = np.zeros((32, 32, 3), dtype=np.uint8)
    asym_img[:, :, 0] = 200  # R
    asym_img[:, :, 1] = 100  # G
    asym_img[:, :, 2] = 50   # B

    compressed = apply_jpeg_compression(asym_img, quality=90)
    # R channel must remain distinctly higher than B channel
    assert np.mean(compressed[:, :, 0]) > 170
    assert np.mean(compressed[:, :, 2]) < 80


def test_same_seed_produces_identical_output(synthetic_rgb_image):
    engine = TraditionalAugmentationEngine()
    out1, p1, _ = engine.execute(synthetic_rgb_image, "gaussian_noise", "sample_001", 1, base_seed=42)
    out2, p2, _ = engine.execute(synthetic_rgb_image, "gaussian_noise", "sample_001", 1, base_seed=42)
    assert p1 == p2
    assert np.array_equal(out1, out2), "Identical seed produced non-identical outputs!"


def test_different_parameters_produce_expected_change(synthetic_rgb_image):
    brighter = apply_brightness(synthetic_rgb_image, factor=1.2)
    darker = apply_brightness(synthetic_rgb_image, factor=0.8)
    assert np.mean(brighter) > np.mean(synthetic_rgb_image)
    assert np.mean(darker) < np.mean(synthetic_rgb_image)


def test_invalid_parameters_rejected(synthetic_rgb_image):
    with pytest.raises(ValueError):
        apply_brightness(synthetic_rgb_image, factor=-0.5)
    with pytest.raises(ValueError):
        apply_contrast(synthetic_rgb_image, factor=5.0)
    with pytest.raises(ValueError):
        apply_gamma(synthetic_rgb_image, gamma=-1.0)
    with pytest.raises(ValueError):
        apply_gaussian_blur(synthetic_rgb_image, sigma=0.0)
    with pytest.raises(ValueError):
        apply_jpeg_compression(synthetic_rgb_image, quality=120)
    with pytest.raises(ValueError):
        apply_downsample_restore(synthetic_rgb_image, scale=1.5)


def test_noise_generation_uses_local_rng(synthetic_rgb_image):
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    rng3 = np.random.default_rng(888)

    noisy1 = apply_gaussian_noise(synthetic_rgb_image, std=4.0, rng=rng1)
    noisy2 = apply_gaussian_noise(synthetic_rgb_image, std=4.0, rng=rng2)
    noisy3 = apply_gaussian_noise(synthetic_rgb_image, std=4.0, rng=rng3)

    assert np.array_equal(noisy1, noisy2), "Same local RNG seed must produce identical noise."
    assert not np.array_equal(noisy1, noisy3), "Different local RNG seeds must produce different noise."


def test_gamma_identity_behaves_correctly(synthetic_rgb_image):
    gamma_identity = apply_gamma(synthetic_rgb_image, gamma=1.0)
    assert np.array_equal(gamma_identity, synthetic_rgb_image), "Gamma 1.0 must be exact identity!"


def test_brightness_direction_on_non_saturated_image(synthetic_rgb_image):
    # Mean of synthetic_rgb_image is around 125
    orig_mean = np.mean(synthetic_rgb_image)
    out_110 = apply_brightness(synthetic_rgb_image, factor=1.10)
    out_090 = apply_brightness(synthetic_rgb_image, factor=0.90)

    assert np.mean(out_110) > orig_mean + 5.0
    assert np.mean(out_090) < orig_mean - 5.0


def test_jpeg_round_trip_returns_valid_image(synthetic_rgb_image):
    jpeg_out = apply_jpeg_compression(synthetic_rgb_image, quality=85)
    validate_rgb_image(jpeg_out)
    assert jpeg_out.shape == synthetic_rgb_image.shape
    assert jpeg_out.dtype == np.uint8


def test_downsample_restore_retains_dimensions(synthetic_rgb_image):
    restored = apply_downsample_restore(synthetic_rgb_image, scale=0.75)
    assert restored.shape == synthetic_rgb_image.shape
    assert restored.dtype == np.uint8


def test_corrupt_source_image_fails_clearly():
    # 2D grayscale
    with pytest.raises(ValueError):
        validate_rgb_image(np.zeros((64, 64), dtype=np.uint8))
    # Float array
    with pytest.raises(ValueError):
        validate_rgb_image(np.zeros((64, 64, 3), dtype=np.float32))
    # 4-channel RGBA
    with pytest.raises(ValueError):
        validate_rgb_image(np.zeros((64, 64, 4), dtype=np.uint8))
