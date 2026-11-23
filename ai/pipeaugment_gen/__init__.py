"""
PipeAugment AI - Generative Diffusion Module (Pretrained SD 1.5 Image-to-Image Pilot).
Condition-controlled, corrosion-preserving pipeline image augmentation.
"""

from .config import GenerativePilotConfig
from .prompts import get_prompt_for_condition, NEGATIVE_PROMPT, CONDITIONS
from .engine import GenerativeDiffusionEngine
from .metadata import (
    compute_canny_edge_map,
    compute_edge_overlap_diagnostics,
    make_controlnet_output_id,
    make_output_id,
    derive_parent_seed,
    GenerativeOutputRecord,
)

__all__ = [
    "GenerativePilotConfig",
    "get_prompt_for_condition",
    "NEGATIVE_PROMPT",
    "CONDITIONS",
    "GenerativeDiffusionEngine",
    "compute_canny_edge_map",
    "compute_edge_overlap_diagnostics",
    "make_controlnet_output_id",
    "make_output_id",
    "derive_parent_seed",
    "GenerativeOutputRecord",
]
