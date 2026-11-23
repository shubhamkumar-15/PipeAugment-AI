"""
PipeAugment AI — Real-Time ResNet-18 Corrosion Inspector & Visual Layout Maps.
Executes trained Experiment C ResNet-18 model inference, computes class probabilities,
extracts layer4 Grad-CAM activation heatmaps, determines corrosion severity levels,
and synthesizes 3 diagnostic visual inspection layout maps.

DISCLAIMER:
All corrosion classifications, candidate masks, and localization heatmaps produced
by this module are diagnostic computer vision assistance tools for research and
inspection screening. They do NOT constitute certified non-destructive testing (NDT),
ultrasonic thickness measurement, or metallographic metallurgical evaluation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

try:
    from pipeaugment_eval.corrosion_diag import compute_rust_candidate_mask
except ImportError:
    from ai.pipeaugment_eval.corrosion_diag import compute_rust_candidate_mask


CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent

DEFAULT_CHECKPOINT_PATTERN = "experiments/runs/pipeaugment_resnet18_seed42_*/best_model.pth"
DISCLAIMER_TEXT = (
    "Diagnostic CV screening assistance only. "
    "Not certified NDT, ultrasonic thickness gauging, or metallographic inspection."
)


def find_default_checkpoint() -> Optional[Path]:
    """Finds the best available Experiment C ResNet-18 model checkpoint."""
    matches = list(PROJECT_ROOT.glob(DEFAULT_CHECKPOINT_PATTERN))
    if matches:
        return matches[0]
    # Fallback to any pipeaugment_resnet18 checkpoint
    alt_matches = list(PROJECT_ROOT.glob("experiments/runs/pipeaugment_resnet18_*/best_model.pth"))
    if alt_matches:
        return alt_matches[0]
    # Fallback to baseline checkpoint
    base_matches = list(PROJECT_ROOT.glob("experiments/runs/baseline_resnet18_*/best_model.pth"))
    if base_matches:
        return base_matches[0]
    return None


class PipeAugmentInspector:
    """
    Inference-ready Corrosion Classifier, Grad-CAM Attention Generator,
    and Visual Inspection Layout Map Synthesizer.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if checkpoint_path is None:
            resolved_ckpt = find_default_checkpoint()
            if resolved_ckpt is None:
                raise FileNotFoundError("No trained ResNet-18 checkpoint found in experiments/runs/.")
            self.checkpoint_path = Path(resolved_ckpt)
        else:
            self.checkpoint_path = Path(checkpoint_path)
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.class_names = ["non_corroded", "corroded"]
        self._load_model()

        # Input preprocessing pipeline matching training evaluation
        self.transforms = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self) -> None:
        """Instantiates ResNet-18 and registers hooks for Grad-CAM."""
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.class_names))

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        self.model = model

        # Setup Grad-CAM storage and hooks on layer4
        self._activations: List[torch.Tensor] = []
        self._gradients: List[torch.Tensor] = []

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self._activations.append(out)

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            self._gradients.append(grad_out[0])

        self.model.layer4.register_forward_hook(forward_hook)
        self.model.layer4.register_full_backward_hook(backward_hook)

    def predict_and_cam(self, image_rgb: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Runs model inference and extracts Grad-CAM heatmap for the corrosion class (index 1).
        Returns (prediction_info, cam_heatmap_2d_float).
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(f"image_rgb must be uint8 (H, W, 3), got {image_rgb.shape}, {image_rgb.dtype}")

        h, w = image_rgb.shape[:2]
        pil_img = Image.fromarray(image_rgb, mode="RGB")
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)

        self._activations.clear()
        self._gradients.clear()

        # Forward pass
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Backward pass for corrosion class (index 1) to extract attention
        corrosion_score = logits[0, 1]
        self.model.zero_grad()
        corrosion_score.backward()

        # Grad-CAM computation on layer4
        cam_map = np.zeros((h, w), dtype=np.float32)
        if self._activations and self._gradients:
            act = self._activations[0][0].detach().cpu().numpy()  # (512, 7, 7)
            grad = self._gradients[0][0].detach().cpu().numpy()    # (512, 7, 7)
            weights = np.mean(grad, axis=(1, 2))                   # (512,)

            cam_7x7 = np.zeros(act.shape[1:], dtype=np.float32)
            for i, w_val in enumerate(weights):
                cam_7x7 += w_val * act[i]

            cam_7x7 = np.maximum(cam_7x7, 0)  # ReLU
            max_val = np.max(cam_7x7)
            if max_val > 1e-8:
                cam_7x7 = cam_7x7 / max_val
                # Upsample back to original image dimensions
                cam_map = cv2.resize(cam_7x7, (w, h), interpolation=cv2.INTER_LINEAR)
                cam_map = np.clip(cam_map, 0.0, 1.0)

        pred_info = {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "corrosion_probability": round(float(probs[1]), 4),
            "non_corrosion_probability": round(float(probs[0]), 4),
        }

        return pred_info, cam_map

    def analyze(self, image_rgb: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Full diagnostic analysis: classification, Grad-CAM heatmap, and color candidate mask.
        Returns (analysis_dict, cam_map, rust_mask).
        """
        pred_info, cam_map = self.predict_and_cam(image_rgb)
        rust_mask, rust_ratio = compute_rust_candidate_mask(image_rgb)
        rust_pct = round(rust_ratio * 100.0, 2)

        # Severity level heuristic combining neural verdict and rust coverage %
        pred_class = pred_info["predicted_class"]
        if pred_class == "non_corroded" and rust_pct < 1.0:
            severity = "None"
        elif rust_pct < 5.0:
            severity = "Low"
        elif rust_pct < 20.0:
            severity = "Moderate"
        else:
            severity = "Severe"

        analysis = {
            "predicted_class": pred_class,
            "confidence": pred_info["confidence"],
            "corrosion_probability": pred_info["corrosion_probability"],
            "non_corrosion_probability": pred_info["non_corrosion_probability"],
            "severity_level": severity,
            "rust_coverage_percent": rust_pct,
            "model_checkpoint": self.checkpoint_path.name,
            "disclaimer": DISCLAIMER_TEXT,
        }

        return analysis, cam_map, rust_mask

    def generate_layout_maps(
        self,
        image_rgb: np.ndarray,
        output_dir: Union[str, Path],
        prefix: str = "",
        cam_map: Optional[np.ndarray] = None,
        rust_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """
        Synthesizes and writes 3 visual inspection layout maps to disk:
        1. pipeline_canny: Pipeline Structural Layout (Canny Edge Map)
        2. corrosion_layout: Corrosion Layout Map (Isolated Corrosion Mask + Contours Overlay)
        3. corrosion_overlay: AI Attention Heatmap (ResNet-18 Grad-CAM Overlay)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        h, w = image_rgb.shape[:2]

        if cam_map is None or rust_mask is None:
            _, cam_map, rust_mask = self.analyze(image_rgb)

        # -------------------------------------------------------------
        # Map 1: Pipeline Structural Layout (Canny Edges on Obsidian)
        # -------------------------------------------------------------
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray, 100, 200)
        canny_canvas = np.full((h, w, 3), (13, 14, 21), dtype=np.uint8)  # Obsidian #0D0E15
        canny_canvas[canny_edges > 0] = (255, 255, 255)                  # Crisp white structural edges

        canny_file = out_path / f"{prefix}pipeline_canny.png"
        Image.fromarray(canny_canvas, mode="RGB").save(canny_file, format="PNG")

        # -------------------------------------------------------------
        # Map 2: Corrosion Layout Map (Isolated Rust Regions on Obsidian)
        # -------------------------------------------------------------
        corr_canvas = np.full((h, w, 3), (15, 17, 23), dtype=np.uint8)   # Dark Obsidian #0F1117
        corr_canvas[rust_mask] = (245, 158, 11)                           # Cyber Saffron fill
        contours, _ = cv2.findContours(
            rust_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Outline contours with bright cyber saffron
        cv2.drawContours(corr_canvas, contours, -1, (251, 191, 36), 2)

        layout_file = out_path / f"{prefix}corrosion_layout.png"
        Image.fromarray(corr_canvas, mode="RGB").save(layout_file, format="PNG")

        # -------------------------------------------------------------
        # Map 3: AI Attention Heatmap (Grad-CAM Overlay + Contours)
        # -------------------------------------------------------------
        cam_uint8 = (cam_map * 255.0).astype(np.uint8)
        # Apply Inferno colormap (Black -> Saffron -> Yellow)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend 65% image + 35% heatmap
        overlay = cv2.addWeighted(image_rgb, 0.65, heatmap_rgb, 0.35, 0)
        # Overlay subtle saffron contours for detected corrosion regions
        if contours:
            cv2.drawContours(overlay, contours, -1, (251, 191, 36), 1)

        overlay_file = out_path / f"{prefix}corrosion_overlay.png"
        Image.fromarray(overlay, mode="RGB").save(overlay_file, format="PNG")

        return {
            "pipeline_canny": canny_file,
            "corrosion_layout": layout_file,
            "corrosion_overlay": overlay_file,
        }
