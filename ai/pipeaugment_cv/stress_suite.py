"""
Deterministic Robustness Stress Suite for PipeAugment AI Final Evaluation.
Applies controlled, deterministic synthetic perturbations strictly at evaluation time.
Input and Output: 3-channel RGB uint8 NumPy arrays of identical dimensions.
"""

from typing import Dict, Any, List
import hashlib
import numpy as np

from .transforms import (
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


CONDITIONS: Dict[str, Dict[str, Any]] = {
    "clean_test": {
        "name": "clean_test",
        "display_name": "Clean Held-Out Test Split",
        "description": "Unmodified 60 held-out test images from K-Pipelines test manifest.",
        "is_stress": False,
    },
    "severe_low_light": {
        "name": "severe_low_light",
        "display_name": "Severe Low-Light",
        "description": "Controlled Synthetic Perturbation: Out-of-distribution dimming (0.55x) and non-linear contrast compression (gamma 1.45).",
        "is_stress": True,
    },
    "harsh_shadows_uneven": {
        "name": "harsh_shadows_uneven",
        "display_name": "Harsh Shadows & Uneven Light",
        "description": "Controlled Synthetic Perturbation: Strong directional gradient + deep spatial shadow (0.50x falloff).",
        "is_stress": True,
    },
    "overexposed_bright": {
        "name": "overexposed_bright",
        "display_name": "Overexposed Glare",
        "description": "Controlled Synthetic Perturbation: High inspection glare (1.40x brightness, 1.20x contrast).",
        "is_stress": True,
    },
    "sensor_blur_noise": {
        "name": "sensor_blur_noise",
        "display_name": "Sensor Blur & Noise",
        "description": "Controlled Synthetic Perturbation: Defocus optical blur (sigma 1.4) + Gaussian sensor noise (std 12.0).",
        "is_stress": True,
    },
    "heavy_compression": {
        "name": "heavy_compression",
        "display_name": "Heavy Low-Bitrate Compression",
        "description": "Controlled Synthetic Perturbation: Downscale/upscale (0.50x) + aggressive lossy JPEG compression (Q35).",
        "is_stress": True,
    },
    "combined_harsh_field": {
        "name": "combined_harsh_field",
        "display_name": "Combined Harsh Field Conditions",
        "description": "Controlled Synthetic Perturbation: Combined low-light (0.65x) + shadow (0.60x) + blur (sigma 1.0) + noise (std 8.0) + JPEG (Q50).",
        "is_stress": True,
    },
}

CONDITION_KEYS: List[str] = list(CONDITIONS.keys())


def _derive_seed(condition_name: str, image_id: str) -> int:
    """Derives a deterministic 32-bit unsigned integer seed from condition and image_id."""
    key = f"{condition_name}_{image_id}".encode("utf-8")
    hex_digest = hashlib.sha256(key).hexdigest()[:8]
    return int(hex_digest, 16)


def apply_stress_condition(img_rgb: np.ndarray, condition_name: str, image_id: str) -> np.ndarray:
    """
    Applies a deterministic evaluation perturbation to an RGB image.
    Never modifies input array in-place. Returns uint8 (H, W, 3).
    """
    validate_rgb_image(img_rgb)
    if condition_name not in CONDITIONS:
        raise ValueError(f"Unknown condition: '{condition_name}'. Authorized: {CONDITION_KEYS}")

    if condition_name == "clean_test":
        return img_rgb.copy()

    seed = _derive_seed(condition_name, image_id)
    rng = np.random.default_rng(seed)

    if condition_name == "severe_low_light":
        # Out-of-distribution dimming + non-linear gamma curve
        t1 = apply_brightness(img_rgb, factor=0.55)
        return apply_gamma(t1, gamma=1.45)

    elif condition_name == "harsh_shadows_uneven":
        # Directional gradient + deep spatial shadow
        angle = float(rng.uniform(0.0, 360.0))
        h, w = img_rgb.shape[:2]
        cx = float(w * rng.uniform(0.3, 0.7))
        cy = float(h * rng.uniform(0.3, 0.7))
        radius = float(min(h, w) * rng.uniform(0.4, 0.6))
        t1 = apply_uneven_illumination(img_rgb, angle_deg=angle, intensity=0.50)
        return apply_soft_shadow(t1, center_x=cx, center_y=cy, radius=radius, shadow_intensity=0.50)

    elif condition_name == "overexposed_bright":
        # Inspection glare and washed out highlights
        t1 = apply_brightness(img_rgb, factor=1.40)
        return apply_contrast(t1, factor=1.20)

    elif condition_name == "sensor_blur_noise":
        # Optical blur + sensor noise
        t1 = apply_gaussian_blur(img_rgb, sigma=1.4)
        return apply_gaussian_noise(t1, std=12.0, rng=rng)

    elif condition_name == "heavy_compression":
        # Transmission loss: downsample/restore + Q35 JPEG
        t1 = apply_downsample_restore(img_rgb, scale=0.50)
        return apply_jpeg_compression(t1, quality=35)

    elif condition_name == "combined_harsh_field":
        # Compound industrial perturbation
        h, w = img_rgb.shape[:2]
        cx = float(w * rng.uniform(0.3, 0.7))
        cy = float(h * rng.uniform(0.3, 0.7))
        radius = float(min(h, w) * 0.5)
        t1 = apply_brightness(img_rgb, factor=0.65)
        t2 = apply_soft_shadow(t1, center_x=cx, center_y=cy, radius=radius, shadow_intensity=0.60)
        t3 = apply_gaussian_blur(t2, sigma=1.0)
        t4 = apply_gaussian_noise(t3, std=8.0, rng=rng)
        return apply_jpeg_compression(t4, quality=50)

    raise RuntimeError(f"Unhandled condition: {condition_name}")
