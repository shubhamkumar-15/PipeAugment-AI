import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import cv2


def derive_parent_seed(parent_id: str, base_seed: int = 42) -> int:
    """
    Derives a stable, platform-independent 32-bit positive integer seed
    for a given parent image and base seed using SHA-256.
    """
    hash_bytes = hashlib.sha256(f"diffusion_parent:{parent_id}:{base_seed}".encode("utf-8")).digest()
    seed = int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)
    return seed


def make_output_id(parent_id: str, condition: str, strength: float) -> str:
    """Creates a deterministic collision-free output ID for plain img2img."""
    str_strength = f"{int(strength * 100):02d}"
    prefix = parent_id[:20]
    return f"gen_diff_{prefix}_{condition}_s{str_strength}"


def make_controlnet_output_id(parent_id: str, condition: str, strength: float, control_scale: float) -> str:
    """Creates a deterministic collision-free output ID for ControlNet img2img."""
    str_strength = f"{int(strength * 100):02d}"
    str_cscale = f"{int(control_scale * 100):02d}"
    prefix = parent_id[:20]
    return f"gen_cnet_canny_{prefix}_{condition}_s{str_strength}_c{str_cscale}"


def compute_canny_edge_map(
    rgb_image: np.ndarray,
    low_thresh: int = 100,
    high_thresh: int = 200
) -> Tuple[np.ndarray, float]:
    """
    Computes a 2D single-channel Canny edge map (uint8, 0 or 255)
    and returns (canny_map, edge_density).
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {rgb_image.shape}")
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, low_thresh, high_thresh)
    density = float(np.mean(canny > 0))
    return canny, round(density, 6)


def compute_edge_overlap_diagnostics(
    source_canny: np.ndarray,
    gen_rgb: np.ndarray,
    low_thresh: int = 100,
    high_thresh: int = 200
) -> Dict[str, float]:
    """
    Computes heuristic edge overlap diagnostics between source Canny edge map
    and the Canny edge map computed on the generated image.
    Handles all edge cases (empty masks).
    NOTE: This is a descriptive heuristic, not proof of structural or corrosion preservation.
    """
    gen_canny, gen_density = compute_canny_edge_map(gen_rgb, low_thresh, high_thresh)
    src_mask = source_canny > 0
    gen_mask = gen_canny > 0

    inter = int(np.logical_and(src_mask, gen_mask).sum())
    union = int(np.logical_or(src_mask, gen_mask).sum())

    iou = 1.0 if union == 0 else float(inter / union)
    denom = int(src_mask.sum() + gen_mask.sum())
    dice = 1.0 if denom == 0 else float((2.0 * inter) / denom)

    return {
        "source_edge_density": float(np.mean(src_mask)),
        "output_edge_density": float(np.mean(gen_mask)),
        "edge_overlap_iou_heuristic": round(iou, 5),
        "edge_overlap_dice_heuristic": round(dice, 5),
    }


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def compute_ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes multi-channel Mean Structural Similarity Index (SSIM)
    averaged across RGB channels with data_range=255 and standard 11x11 Gaussian window.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch for SSIM: {img1.shape} vs {img2.shape}")

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    kernel = (11, 11)
    sigma = 1.5

    channel_ssims = []
    for c in range(img1.shape[2]):
        c1 = I1[:, :, c]
        c2 = I2[:, :, c]
        mu1 = cv2.GaussianBlur(c1, kernel, sigma)
        mu2 = cv2.GaussianBlur(c2, kernel, sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(c1 ** 2, kernel, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(c2 ** 2, kernel, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(c1 * c2, kernel, sigma) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        channel_ssims.append(float(np.mean(ssim_map)))

    return float(np.mean(channel_ssims))


def evaluate_technical_checks(
    generated_img: np.ndarray,
    parent_img: np.ndarray,
    safety_flag: bool = False
) -> Dict[str, Any]:
    """
    Evaluates technical sanity of the generated output against parent.
    Verifies dimensions, finite uint8 pixels, nonconstant content,
    and non-identical generation.
    """
    dim_match = bool(generated_img.shape == parent_img.shape)
    dtype_match = bool(generated_img.dtype == np.uint8)
    finite_check = bool(np.isfinite(generated_img).all())
    is_constant = bool(np.all(generated_img == generated_img[0, 0, 0]))

    diff = np.abs(generated_img.astype(float) - parent_img.astype(float))
    mean_abs_diff = float(np.mean(diff))
    is_identical_pixels = bool(mean_abs_diff == 0.0)

    clipping_low = float(np.mean(generated_img == 0))
    clipping_high = float(np.mean(generated_img == 255))
    is_all_black = bool(clipping_low >= 0.999)
    is_all_white = bool(clipping_high >= 0.999)

    ssim_val = compute_ssim_rgb(generated_img, parent_img)

    passed = (
        dim_match
        and dtype_match
        and finite_check
        and not is_constant
        and not is_all_black
        and not is_all_white
        and not is_identical_pixels
        and not safety_flag
    )

    return {
        "status": "passed" if passed else "failed",
        "dimension_preserved": dim_match,
        "dtype_uint8": dtype_match,
        "finite_pixels": finite_check,
        "is_constant": is_constant,
        "is_all_black": is_all_black,
        "is_all_white": is_all_white,
        "is_identical_pixels": is_identical_pixels,
        "safety_violation": safety_flag,
        "mean_pixel_abs_diff": round(mean_abs_diff, 4),
        "clipping_low_rate": round(clipping_low, 5),
        "clipping_high_rate": round(clipping_high, 5),
        "ssim_rgb": round(ssim_val, 5),
    }


@dataclass
class GenerativeOutputRecord:
    """Full metadata schema for a generated diffusion output."""
    output_id: str
    parent_id: str
    parent_sha256: str
    parent_split: str
    source_label: str  # provisional
    model_id: str
    model_revision: str
    pipeline_class: str
    scheduler_config: Dict[str, Any]
    positive_prompt: str
    negative_prompt: str
    condition: str
    strength: float
    num_inference_steps: int
    effective_denoising_steps: int
    guidance_scale: float
    seed: int
    input_dimensions: str
    output_dimensions: str
    preprocessing: str
    precision: str
    device: str
    offload_strategy: str
    runtime_seconds: float
    peak_gpu_allocated_mb: float
    timestamp_utc: str
    technical_validation_status: str
    safety_flag: bool
    human_review_status: str = "pending"
    training_eligibility: bool = False
    mean_pixel_abs_diff: float = 0.0
    ssim_rgb: float = 0.0
    clipping_low_rate: float = 0.0
    clipping_high_rate: float = 0.0
    output_sha256: str = ""
    relative_output_path: str = ""
    # ControlNet & execution provenance
    generation_mode: str = "plain_img2img"
    controlnet_id: Optional[str] = None
    controlnet_revision: Optional[str] = None
    control_scale: Optional[float] = None
    canny_low_threshold: Optional[int] = None
    canny_high_threshold: Optional[int] = None
    canny_edge_density: Optional[float] = None
    peak_gpu_reserved_mb: float = 0.0
    device_memory_free_mb: float = 0.0
    edge_overlap_iou_heuristic: Optional[float] = None
    edge_overlap_dice_heuristic: Optional[float] = None
    vae_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
