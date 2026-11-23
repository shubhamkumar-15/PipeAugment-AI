"""
PipeAugment AI — Phase 4 Core Evaluation Metrics.
Provides implementations for image validity checks, multi-channel SSIM (data_range=255),
LPIPS perceptual distance (AlexNet trunk), and Canny edge retention IoU/Dice.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch


def evaluate_image_validity(
    generated_img: np.ndarray,
    parent_img: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates technical validity of a generated or augmented image against its parent.
    Checks dimensions, dtype, finite numbers, non-constant values, clipping, and copy status.
    """
    dim_match = bool(generated_img.shape == parent_img.shape)
    dtype_match = bool(generated_img.dtype == np.uint8)
    finite_check = bool(np.isfinite(generated_img).all())
    is_constant = bool(np.all(generated_img == generated_img[0, 0, 0]))

    if dim_match:
        diff = np.abs(generated_img.astype(float) - parent_img.astype(float))
        mean_abs_diff = float(np.mean(diff))
        is_identical_pixels = bool(mean_abs_diff == 0.0)
    else:
        mean_abs_diff = 999.0
        is_identical_pixels = False

    clipping_low = float(np.mean(generated_img == 0))
    clipping_high = float(np.mean(generated_img == 255))
    is_all_black = bool(clipping_low >= 0.999)
    is_all_white = bool(clipping_high >= 0.999)

    valid = (
        dim_match
        and dtype_match
        and finite_check
        and not is_constant
        and not is_all_black
        and not is_all_white
        and not is_identical_pixels
    )

    return {
        "valid": valid,
        "dimension_preserved": dim_match,
        "dtype_uint8": dtype_match,
        "finite_pixels": finite_check,
        "is_constant": is_constant,
        "is_all_black": is_all_black,
        "is_all_white": is_all_white,
        "is_identical_pixels": is_identical_pixels,
        "mean_pixel_abs_diff": round(mean_abs_diff, 4),
        "clipping_low_rate": round(clipping_low, 5),
        "clipping_high_rate": round(clipping_high, 5),
    }


def compute_ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes multi-channel Mean Structural Similarity Index (SSIM)
    averaged across RGB channels with data_range=255 and standard 11x11 Gaussian window (sigma=1.5).
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

    return round(float(np.mean(channel_ssims)), 5)


def compute_edge_retention(
    source_img: np.ndarray,
    gen_img: np.ndarray,
    low_thresh: int = 100,
    high_thresh: int = 200,
) -> Dict[str, float]:
    """
    Computes heuristic Canny edge retention IoU and Dice between source and generated image.
    Handles edge-empty masks safely without division by zero.
    NOTE: Descriptive heuristic, not proof of defect retention.
    """
    if source_img.ndim != 3 or gen_img.ndim != 3:
        raise ValueError("Images must be 3-channel RGB ndarrays")

    src_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)

    src_canny = cv2.Canny(src_gray, low_thresh, high_thresh)
    gen_canny = cv2.Canny(gen_gray, low_thresh, high_thresh)

    src_mask = src_canny > 0
    gen_mask = gen_canny > 0

    inter = int(np.logical_and(src_mask, gen_mask).sum())
    union = int(np.logical_or(src_mask, gen_mask).sum())

    iou = 1.0 if union == 0 else float(inter / union)
    denom = int(src_mask.sum() + gen_mask.sum())
    dice = 1.0 if denom == 0 else float((2.0 * inter) / denom)

    return {
        "source_edge_density": round(float(np.mean(src_mask)), 5),
        "output_edge_density": round(float(np.mean(gen_mask)), 5),
        "edge_overlap_iou": round(iou, 5),
        "edge_overlap_dice": round(dice, 5),
    }


class LPIPSEvaluator:
    """
    Wraps Learned Perceptual Image Patch Similarity (LPIPS) using AlexNet trunk.
    Evaluates in inference mode with proper [-1.0, 1.0] float32 normalization.
    """

    def __init__(self, net: str = "alex", use_gpu: bool = True, model_override: Optional[Any] = None):
        self.net = net
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        if model_override is not None:
            self._loss_fn = model_override
        else:
            import lpips
            self._loss_fn = lpips.LPIPS(net=self.net).to(self.device)
            self._loss_fn.eval()

    def evaluate(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
        """
        Calculates LPIPS distance between two (H, W, 3) uint8 RGB images.
        Normalizes uint8 [0, 255] to float32 [-1.0, 1.0] tensor of shape (1, 3, H, W).
        """
        if img1_rgb.shape != img2_rgb.shape or img1_rgb.ndim != 3:
            raise ValueError(f"Shape mismatch for LPIPS: {img1_rgb.shape} vs {img2_rgb.shape}")

        # Normalize to [-1.0, 1.0]
        t1 = torch.from_numpy(img1_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1.0

        t1 = t1.to(self.device)
        t2 = t2.to(self.device)

        with torch.inference_mode():
            dist = self._loss_fn(t1, t2)
            val = float(dist.squeeze().item())

        return round(val, 5)
