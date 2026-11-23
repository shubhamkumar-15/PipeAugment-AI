"""
Deterministic Traditional Image Transformations for PipeAugment AI.
Input and Output: 3-channel RGB uint8 NumPy arrays of identical dimensions.
All operations validate input integrity, use local RNGs, and do not mutate inputs in-place.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def validate_rgb_image(img: np.ndarray, name: str = "input_image") -> None:
    """Validates that image is a non-empty, finite, 3-channel RGB uint8 ndarray."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(img).__name__}")
    if img.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got shape {img.shape}")
    if img.shape[0] < 8 or img.shape[1] < 8:
        raise ValueError(f"{name} dimensions too small: {img.shape}")


# 1. Brightness Multiplier
def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjusts image brightness multiplicatively.
    Formula: I_out = clip(I_in * factor, 0, 255)
    """
    validate_rgb_image(img)
    if not isinstance(factor, (int, float)) or not (0.1 <= factor <= 3.0):
        raise ValueError(f"Invalid brightness factor: {factor}")
    out = np.clip(img.astype(np.float32) * float(factor), 0.0, 255.0)
    return out.astype(np.uint8)


# 2. Contrast Adjustment
def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjusts image contrast relative to midpoint 128.0.
    Formula: I_out = clip(128.0 + factor * (I_in - 128.0), 0, 255)
    """
    validate_rgb_image(img)
    if not isinstance(factor, (int, float)) or not (0.1 <= factor <= 3.0):
        raise ValueError(f"Invalid contrast factor: {factor}")
    out = np.clip(128.0 + float(factor) * (img.astype(np.float32) - 128.0), 0.0, 255.0)
    return out.astype(np.uint8)


# 3. Gamma Correction
def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies non-linear gamma correction via 256-element LUT.
    Formula: I_out = 255.0 * (I_in / 255.0) ** gamma
    Note: gamma = 1.0 yields the exact identity.
    """
    validate_rgb_image(img)
    if not isinstance(gamma, (int, float)) or not (0.1 <= gamma <= 3.0):
        raise ValueError(f"Invalid gamma exponent: {gamma}")
    inv_gamma = float(gamma)
    lut = np.array([np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0.0, 255.0) for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)


# 4. Smooth Multiplicative Uneven Illumination
def apply_uneven_illumination(img: np.ndarray, angle_deg: float, intensity: float) -> np.ndarray:
    """
    Simulates smooth, low-amplitude directional illumination falloff across the pipeline.
    Formula: I_out = clip(I_in * Mask, 0, 255), where Mask is a smoothed 2D linear gradient.
    """
    validate_rgb_image(img)
    h, w = img.shape[:2]
    theta = np.deg2rad(angle_deg)
    
    # 2D coordinate grid normalized to [-1, 1]
    y_coords = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Linear gradient along angle theta
    gradient = xx * np.cos(theta) + yy * np.sin(theta)
    # Normalize gradient to [-1, 1]
    max_g = np.sqrt(2.0)
    gradient = gradient / max_g
    
    # Scale amplitude by (intensity - 1.0)
    delta = abs(float(intensity) - 1.0)
    sign = 1.0 if float(intensity) >= 1.0 else -1.0
    illum_mask = 1.0 + sign * delta * gradient
    
    # Smooth with Gaussian blur to prevent any sharp transitions
    blur_ksize = int(min(h, w) // 4) | 1  # ensure odd
    illum_mask = cv2.GaussianBlur(illum_mask, (blur_ksize, blur_ksize), sigmaX=min(h, w) / 6.0)
    illum_mask_3d = np.expand_dims(illum_mask, axis=2)
    
    out = np.clip(img.astype(np.float32) * illum_mask_3d, 0.0, 255.0)
    return out.astype(np.uint8)


# 5. Soft Spatial Shadow
def apply_soft_shadow(
    img: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    shadow_intensity: float
) -> np.ndarray:
    """
    Simulates a soft, low-amplitude spatial shadow with Gaussian falloff.
    Avoids hard cutout shapes.
    Formula: Mask(x, y) = 1.0 - (1.0 - shadow_intensity) * exp(-d^2 / (2 * radius^2))
             I_out = clip(I_in * Mask, 0, 255)
    """
    validate_rgb_image(img)
    h, w = img.shape[:2]
    cx = float(center_x)
    cy = float(center_y)
    r = max(10.0, float(radius))
    intensity = float(shadow_intensity)
    
    y_indices = np.arange(h, dtype=np.float32)
    x_indices = np.arange(w, dtype=np.float32)
    xx, yy = np.meshgrid(x_indices, y_indices)
    
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    falloff = np.exp(-dist_sq / (2.0 * (r ** 2)))
    attenuation = (1.0 - intensity) * falloff
    shadow_mask = 1.0 - attenuation
    
    shadow_mask_3d = np.expand_dims(shadow_mask, axis=2)
    out = np.clip(img.astype(np.float32) * shadow_mask_3d, 0.0, 255.0)
    return out.astype(np.uint8)


# 6. Gaussian Blur
def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies Gaussian blurring for optical defocus simulation.
    Formula: Convolves image with 2D Gaussian kernel of standard deviation sigma.
    """
    validate_rgb_image(img)
    if not isinstance(sigma, (int, float)) or sigma <= 0.0:
        raise ValueError(f"Invalid Gaussian blur sigma: {sigma}")
    ksize = int(np.ceil(3.0 * float(sigma))) * 2 + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


# 7. Gaussian Noise
def apply_gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Injects zero-mean additive Gaussian sensor noise using a local RNG generator.
    Formula: I_out = clip(I_in + N(0, std^2), 0, 255)
    """
    validate_rgb_image(img)
    if not isinstance(std, (int, float)) or std < 0.0:
        raise ValueError(f"Invalid Gaussian noise std: {std}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")
    noise = rng.normal(loc=0.0, scale=float(std), size=img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0.0, 255.0)
    return out.astype(np.uint8)


# 8. JPEG Compression Artifacts
def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """
    Simulates lossy transmission compression via JPEG encode and decode round trip.
    """
    validate_rgb_image(img)
    q = int(quality)
    if not (1 <= q <= 100):
        raise ValueError(f"Invalid JPEG quality factor: {q}")
    # OpenCV imencode expects BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not success:
        raise RuntimeError(f"cv2.imencode failed for JPEG quality={q}")
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


# 9. Downsampling and Restoration
def apply_downsample_restore(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Simulates optical resolution loss by downsampling then bicubic upsampling back to original size.
    Original dimensions (H, W) are strictly preserved.
    """
    validate_rgb_image(img)
    s = float(scale)
    if not (0.1 <= s <= 1.0):
        raise ValueError(f"Invalid downsample scale: {s}")
    h, w = img.shape[:2]
    down_w = max(4, int(round(w * s)))
    down_h = max(4, int(round(h * s)))
    
    down = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored
