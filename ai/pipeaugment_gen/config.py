from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class GenerativePilotConfig:
    """
    Configuration parameters for pretrained Stable Diffusion v1.5 Image-to-Image Pilot.
    """
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_revision: str = "main"
    torch_dtype: str = "float16"
    variant: str = "fp16"
    use_safetensors: bool = True
    device: str = "cuda"
    offload_strategy: str = "none"  # "none" (direct pipeline.to("cuda")) or "model_cpu_offload"

    # Generation parameters
    mode: str = "plain_img2img"  # "plain_img2img" or "controlnet_img2img"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    use_dpm_solver: bool = True
    controlnet_id: str = "lllyasviel/control_v11p_sd15_canny"
    controlnet_revision: str = "main"
    controlnet_scales: List[float] = field(default_factory=lambda: [0.5, 0.8])
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200

    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    strengths: List[float] = field(default_factory=lambda: [0.20, 0.35])
    conditions: List[str] = field(default_factory=lambda: [
        "low_light_inspection",
        "directional_industrial_lighting"
    ])
    base_seed: int = 42
    target_size: Tuple[int, int] = (512, 512)
    num_parents_per_class: int = 3

    # Provenance
    version: str = "diffusion_pilot_v1"

    @property
    def is_controlnet(self) -> bool:
        """Returns True if the configured mode utilizes ControlNet guidance."""
        return self.mode in ("controlnet_img2img", "controlnet_canny")

    @property
    def canonical_mode(self) -> str:
        """Returns canonical public mode identifier: 'controlnet_canny' or 'img2img'."""
        return "controlnet_canny" if self.is_controlnet else "img2img"

    def validate(self) -> None:
        """Validates configuration parameters."""
        if not self.model_id:
            raise ValueError("model_id must not be empty.")
        if self.mode not in ("plain_img2img", "img2img", "controlnet_img2img", "controlnet_canny"):
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.is_controlnet:
            if not self.controlnet_id:
                raise ValueError("controlnet_id must not be empty in controlnet_img2img mode.")
            for sc in self.controlnet_scales:
                if not (0.0 <= sc <= 2.0):
                    raise ValueError(f"ControlNet scale {sc} out of valid range [0.0, 2.0]")
            if not (0 <= self.canny_low_threshold < self.canny_high_threshold <= 255):
                raise ValueError(f"Invalid Canny thresholds: low={self.canny_low_threshold}, high={self.canny_high_threshold}")
        if self.torch_dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype}")
        if self.num_inference_steps < 5 or self.num_inference_steps > 150:
            raise ValueError(f"num_inference_steps must be in [5, 150], got {self.num_inference_steps}")
        if self.guidance_scale < 1.0 or self.guidance_scale > 20.0:
            raise ValueError(f"guidance_scale must be in [1.0, 20.0], got {self.guidance_scale}")
        for s in self.strengths:
            if not (0.05 <= s <= 0.95):
                raise ValueError(f"Strength {s} out of valid range [0.05, 0.95]")
        if self.target_size != (512, 512):
            raise ValueError(f"Target size must be (512, 512) for SD 1.5, got {self.target_size}")
        if self.num_parents_per_class < 1:
            raise ValueError("num_parents_per_class must be at least 1")
        if not self.conditions:
            raise ValueError("conditions list cannot be empty")
