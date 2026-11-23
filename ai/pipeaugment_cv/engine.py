"""
Traditional Augmentation Engine for PipeAugment AI.
Orchestrates condition-controlled transformations, stable seed derivation,
parameter recording, and quantitative validation.
"""

import hashlib
from typing import Dict, Any, Tuple, Optional
import numpy as np

from .config import TraditionalConfig
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


def derive_output_seed(parent_id: str, variant_idx: int, base_seed: int) -> int:
    """
    Derives an integer seed deterministically from parent_id, variant index, and base_seed
    using SHA-256. Never relies on Python's randomized hash().
    """
    key = f"{parent_id}_var{variant_idx}_seed{base_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Take first 8 hex characters (32-bit unsigned integer)
    return int(digest[:8], 16)


class TraditionalAugmentationEngine:
    def __init__(self, config: Optional[TraditionalConfig] = None):
        self.config = config or TraditionalConfig()

    def sample_parameters(
        self,
        transform_name: str,
        rng: np.random.Generator,
        img_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Samples transform parameters within config ranges using a local RNG."""
        h, w = img_shape[:2]
        if transform_name == "brightness":
            factor = float(rng.uniform(self.config.brightness_range.min_val, self.config.brightness_range.max_val))
            while abs(factor - 1.0) < 0.02:
                factor = float(rng.uniform(self.config.brightness_range.min_val, self.config.brightness_range.max_val))
            return {"factor": round(factor, 4)}

        elif transform_name == "contrast":
            factor = float(rng.uniform(self.config.contrast_range.min_val, self.config.contrast_range.max_val))
            while abs(factor - 1.0) < 0.02:
                factor = float(rng.uniform(self.config.contrast_range.min_val, self.config.contrast_range.max_val))
            return {"factor": round(factor, 4)}

        elif transform_name == "gamma":
            gamma = float(rng.uniform(self.config.gamma_range.min_val, self.config.gamma_range.max_val))
            while abs(gamma - 1.0) < 0.02:
                gamma = float(rng.uniform(self.config.gamma_range.min_val, self.config.gamma_range.max_val))
            return {"gamma": round(gamma, 4)}

        elif transform_name == "uneven_illumination":
            angle = float(rng.uniform(0.0, 360.0))
            intensity = float(rng.uniform(self.config.uneven_illum_intensity_range.min_val, self.config.uneven_illum_intensity_range.max_val))
            return {"angle_deg": round(angle, 2), "intensity": round(intensity, 4)}

        elif transform_name == "soft_shadow":
            cx = float(rng.uniform(0.15 * w, 0.85 * w))
            cy = float(rng.uniform(0.15 * h, 0.85 * h))
            radius = float(rng.uniform(0.25 * min(h, w), 0.65 * min(h, w)))
            intensity = float(rng.uniform(self.config.soft_shadow_intensity_range.min_val, self.config.soft_shadow_intensity_range.max_val))
            return {
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
                "radius": round(radius, 1),
                "shadow_intensity": round(intensity, 4),
            }

        elif transform_name == "gaussian_blur":
            sigma = float(rng.uniform(self.config.gaussian_blur_sigma_range.min_val, self.config.gaussian_blur_sigma_range.max_val))
            return {"sigma": round(sigma, 4)}

        elif transform_name == "gaussian_noise":
            std = float(rng.uniform(self.config.gaussian_noise_std_range.min_val, self.config.gaussian_noise_std_range.max_val))
            return {"std": round(std, 3)}

        elif transform_name == "jpeg_compression":
            q = int(rng.integers(self.config.jpeg_quality_range.min_val, self.config.jpeg_quality_range.max_val + 1))
            return {"quality": q}

        elif transform_name == "downsample_restore":
            scale = float(rng.uniform(self.config.downsample_scale_range.min_val, self.config.downsample_scale_range.max_val))
            return {"scale": round(scale, 4)}

        else:
            raise ValueError(f"Unknown transform name: {transform_name}")

    def apply_transform(
        self,
        img: np.ndarray,
        transform_name: str,
        params: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Applies a specified transform using provided parameters."""
        validate_rgb_image(img)
        
        if transform_name == "brightness":
            return apply_brightness(img, params["factor"])
        elif transform_name == "contrast":
            return apply_contrast(img, params["factor"])
        elif transform_name == "gamma":
            return apply_gamma(img, params["gamma"])
        elif transform_name == "uneven_illumination":
            return apply_uneven_illumination(img, params["angle_deg"], params["intensity"])
        elif transform_name == "soft_shadow":
            return apply_soft_shadow(
                img,
                params["center_x"],
                params["center_y"],
                params["radius"],
                params["shadow_intensity"]
            )
        elif transform_name == "gaussian_blur":
            return apply_gaussian_blur(img, params["sigma"])
        elif transform_name == "gaussian_noise":
            local_rng = rng if rng is not None else np.random.default_rng(42)
            return apply_gaussian_noise(img, params["std"], local_rng)
        elif transform_name == "jpeg_compression":
            return apply_jpeg_compression(img, params["quality"])
        elif transform_name == "downsample_restore":
            return apply_downsample_restore(img, params["scale"])
        else:
            raise ValueError(f"Unsupported transform: {transform_name}")

    def execute(
        self,
        img: np.ndarray,
        transform_name: str,
        parent_id: str,
        variant_idx: int,
        base_seed: Optional[int] = None,
        explicit_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """
        Executes a single transform pipeline:
        - Validates input
        - Derives deterministic seed
        - Samples or validates parameters
        - Executes transformation
        - Computes quantitative technical validation
        """
        validate_rgb_image(img)
        seed = derive_output_seed(parent_id, variant_idx, base_seed or self.config.base_seed)
        local_rng = np.random.default_rng(seed)

        if explicit_params is not None:
            params = dict(explicit_params)
        else:
            params = self.sample_parameters(transform_name, local_rng, img.shape)

        out_img = self.apply_transform(img, transform_name, params, rng=local_rng)

        # Technical validation metrics
        dim_match = bool(out_img.shape == img.shape)
        dtype_match = bool(out_img.dtype == np.uint8)
        finite_check = bool(np.isfinite(out_img).all())
        clipping_low = float(np.mean(out_img == 0))
        clipping_high = float(np.mean(out_img == 255))
        pixel_abs_diff = float(np.mean(np.abs(out_img.astype(float) - img.astype(float))))
        is_constant = bool(np.all(out_img == out_img[0, 0, 0]))

        validation_status = {
            "status": "passed" if (dim_match and dtype_match and finite_check and not is_constant and pixel_abs_diff > 0.0) else "failed",
            "dimension_preserved": dim_match,
            "dtype_uint8": dtype_match,
            "finite_pixels": finite_check,
            "is_constant": is_constant,
            "pixel_difference_detected": bool(pixel_abs_diff > 0.0),
            "clipping_low_rate": round(clipping_low, 5),
            "clipping_high_rate": round(clipping_high, 5),
            "mean_pixel_abs_diff": round(pixel_abs_diff, 4),
            "output_mean": round(float(np.mean(out_img)), 3),
            "output_std": round(float(np.std(out_img)), 3),
        }

        return out_img, params, validation_status
