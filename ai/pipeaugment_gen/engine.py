"""
Reusable Generative Diffusion Engine for PipeAugment AI.
Supports both pretrained Stable Diffusion v1.5 Image-to-Image and
Canny-conditioned ControlNet Image-to-Image in float16 precision.
Manages GPU placement/offloading, performs single-image condition-controlled
generative augmentation, and computes technical validation and diagnostic metrics.
"""

import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from .config import GenerativePilotConfig
from .prompts import get_prompt_for_condition, get_negative_prompt
from .metadata import (
    derive_parent_seed,
    make_output_id,
    make_controlnet_output_id,
    compute_canny_edge_map,
    compute_edge_overlap_diagnostics,
    evaluate_technical_checks,
    GenerativeOutputRecord,
    compute_sha256_bytes,
)


class GenerativeDiffusionEngine:
    """
    Inference-only generative augmentation engine wrapping Hugging Face Diffusers
    Stable Diffusion v1.5 img2img and ControlNet img2img pipelines.
    """

    def __init__(self, config: Optional[GenerativePilotConfig] = None):
        self.config = config or GenerativePilotConfig()
        self.config.validate()
        self._pipe = None
        self._loaded_revision: Optional[str] = None
        self._loaded_controlnet_revision: Optional[str] = None
        self._resolved_model_info: Dict[str, Any] = {}
        self.active_offload_strategy: str = self.config.offload_strategy

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load_pipeline(self) -> None:
        """
        Loads the diffusion pipeline once into memory with float16 precision.
        Selects between AutoPipelineForImage2Image (plain) and
        StableDiffusionControlNetImg2ImgPipeline (controlnet) based on self.config.mode.
        """
        if self._pipe is not None:
            return

        torch_dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        custom_vae = None
        if getattr(self.config, "vae_id", None):
            from diffusers import AutoencoderKL
            print(f"Loading custom VAE: {self.config.vae_id} (dtype={self.config.torch_dtype})...")
            custom_vae = AutoencoderKL.from_pretrained(
                self.config.vae_id,
                torch_dtype=torch_dtype,
            )

        if self.config.is_controlnet:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

            print(f"Loading ControlNet model: {self.config.controlnet_id} (dtype={self.config.torch_dtype})...")
            controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_id,
                revision=self.config.controlnet_revision,
                torch_dtype=torch_dtype,
                variant=self.config.variant,
                use_safetensors=self.config.use_safetensors,
            )
            print(f"Loading Diffusers ControlNet pipeline: {self.config.model_id} (dtype={self.config.torch_dtype})...")
            cnet_kwargs = {
                "controlnet": controlnet,
                "revision": self.config.model_revision,
                "torch_dtype": torch_dtype,
                "variant": self.config.variant,
                "use_safetensors": self.config.use_safetensors,
            }
            if custom_vae is not None:
                cnet_kwargs["vae"] = custom_vae
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config.model_id,
                **cnet_kwargs,
            )
            self._loaded_controlnet_revision = getattr(controlnet, "_internal_dict", {}).get("_commit_hash", self.config.controlnet_revision)
        else:
            from diffusers import AutoPipelineForImage2Image

            print(f"Loading Diffusers pipeline: {self.config.model_id} (dtype={self.config.torch_dtype})...")
            pipe_kwargs = {
                "revision": self.config.model_revision,
                "torch_dtype": torch_dtype,
                "variant": self.config.variant,
                "use_safetensors": self.config.use_safetensors,
            }
            if custom_vae is not None:
                pipe_kwargs["vae"] = custom_vae
            pipe = AutoPipelineForImage2Image.from_pretrained(
                self.config.model_id,
                **pipe_kwargs,
            )

        # Scheduler configuration
        if getattr(self.config, "use_dpm_solver", False):
            from diffusers import DPMSolverMultistepScheduler
            print("Configuring DPMSolverMultistepScheduler (DPM++ 2M Karras)...")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )

        # Device placement
        if self.config.device == "cuda" and torch.cuda.is_available():
            if self.config.offload_strategy == "model_cpu_offload":
                pipe.enable_model_cpu_offload()
                self.active_offload_strategy = "model_cpu_offload"
            else:
                pipe = pipe.to("cuda")
                self.active_offload_strategy = "none"
        else:
            pipe = pipe.to("cpu")
            self.active_offload_strategy = "cpu"

        self._pipe = pipe
        self._loaded_revision = getattr(pipe, "_internal_dict", {}).get("_commit_hash", self.config.model_revision)
        self._resolved_model_info = {
            "model_id": self.config.model_id,
            "vae_id": getattr(self.config, "vae_id", None),
            "pipeline_class": pipe.__class__.__name__,
            "scheduler_class": pipe.scheduler.__class__.__name__,
            "scheduler_config": dict(pipe.scheduler.config),
            "safety_checker_enabled": (pipe.safety_checker is not None),
        }
        if self.config.is_controlnet:
            self._resolved_model_info["controlnet_id"] = self.config.controlnet_id
            self._resolved_model_info["controlnet_revision"] = self._loaded_controlnet_revision

        print(f"Pipeline {pipe.__class__.__name__} successfully initialized (offload={self.active_offload_strategy}).")

    def generate(
        self,
        source_image: np.ndarray,
        parent_id: str,
        parent_sha256: str,
        parent_split: str,
        source_label: str,
        condition: str,
        strength: float,
        seed: Optional[int] = None,
        control_scale: Optional[float] = None,
        source_canny: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, GenerativeOutputRecord, Dict[str, Any]]:
        """
        Executes single-image condition-controlled generative augmentation.
        Supports both plain img2img and Canny ControlNet img2img.
        """
        if not self.is_loaded:
            self.load_pipeline()

        # Validate inputs
        if source_image.ndim != 3 or source_image.shape[2] != 3 or source_image.dtype != np.uint8:
            raise ValueError(f"source_image must be uint8 RGB (H, W, 3), got {source_image.shape}, {source_image.dtype}")

        h, w = source_image.shape[:2]
        if (w, h) != self.config.target_size:
            raise ValueError(f"source_image dimension must match target_size {self.config.target_size}, got {(w, h)}")

        is_controlnet = self.config.is_controlnet
        if is_controlnet and control_scale is None:
            raise ValueError("control_scale must be provided in controlnet_img2img mode")

        resolved_seed = seed if seed is not None else derive_parent_seed(parent_id, self.config.base_seed)

        if is_controlnet:
            output_id = make_controlnet_output_id(parent_id, condition, strength, float(control_scale))
        else:
            output_id = make_output_id(parent_id, condition, strength)

        pos_prompt = get_prompt_for_condition(condition)
        neg_prompt = get_negative_prompt()

        effective_steps = int(self.config.num_inference_steps * strength)
        if effective_steps < 1:
            effective_steps = 1

        pil_input = Image.fromarray(source_image, mode="RGB")

        # Prepare ControlNet condition if required
        pil_canny = None
        canny_density = None
        if is_controlnet:
            if source_canny is None:
                source_canny, canny_density = compute_canny_edge_map(
                    source_image,
                    low_thresh=self.config.canny_low_threshold,
                    high_thresh=self.config.canny_high_threshold
                )
            else:
                canny_density = float(np.mean(source_canny > 0))
            # Diffusers ControlNet accepts 3-channel RGB PIL image for control_image
            canny_rgb = np.stack([source_canny, source_canny, source_canny], axis=-1)
            pil_canny = Image.fromarray(canny_rgb, mode="RGB")

        # Generator setup
        gen_device = "cuda" if (self.config.device == "cuda" and self.active_offload_strategy == "none") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(resolved_seed)

        call_kwargs = {
            "prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "image": pil_input,
            "strength": strength,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "generator": generator,
            "output_type": "pil",
        }
        if is_controlnet:
            call_kwargs["control_image"] = pil_canny
            call_kwargs["controlnet_conditioning_scale"] = float(control_scale)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        try:
            with torch.inference_mode():
                result = self._pipe(**call_kwargs)
        except torch.cuda.OutOfMemoryError as oom:
            if self.active_offload_strategy == "none" and torch.cuda.is_available():
                print("CUDA OOM encountered. Flushing cache and retrying with model_cpu_offload()...")
                torch.cuda.empty_cache()
                self._pipe.enable_model_cpu_offload()
                self.active_offload_strategy = "model_cpu_offload"
                gen_device = "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(resolved_seed)
                call_kwargs["generator"] = generator
                with torch.inference_mode():
                    result = self._pipe(**call_kwargs)
            else:
                raise oom

        runtime = time.time() - t0
        peak_gpu_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        peak_reserved_mb = (torch.cuda.max_memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        free_gpu_mb = 0.0
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gpu_mb = free_bytes / (1024 * 1024)

        out_pil = result.images[0]
        out_arr = np.array(out_pil, dtype=np.uint8)

        # Check safety checker flag
        has_nsfw = False
        if hasattr(result, "has_nsfw_concept") and result.has_nsfw_concept is not None:
            has_nsfw = bool(result.has_nsfw_concept[0])

        tech_stats = evaluate_technical_checks(out_arr, source_image, safety_flag=has_nsfw)

        # Edge overlap heuristic diagnostics (descriptive only)
        edge_diag = None
        if is_controlnet and source_canny is not None:
            edge_diag = compute_edge_overlap_diagnostics(
                source_canny,
                out_arr,
                low_thresh=self.config.canny_low_threshold,
                high_thresh=self.config.canny_high_threshold
            )

        record = GenerativeOutputRecord(
            output_id=output_id,
            parent_id=parent_id,
            parent_sha256=parent_sha256,
            parent_split=parent_split,
            source_label=source_label,
            model_id=self.config.model_id,
            model_revision=self._loaded_revision or self.config.model_revision,
            pipeline_class=self._resolved_model_info.get("pipeline_class", self._pipe.__class__.__name__),
            scheduler_config=self._resolved_model_info.get("scheduler_config", {}),
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            condition=condition,
            strength=strength,
            num_inference_steps=self.config.num_inference_steps,
            effective_denoising_steps=effective_steps,
            guidance_scale=self.config.guidance_scale,
            seed=resolved_seed,
            input_dimensions=f"{w}x{h}",
            output_dimensions=f"{out_arr.shape[1]}x{out_arr.shape[0]}",
            preprocessing="exif_transpose_rgb_512x512",
            precision=self.config.torch_dtype,
            device=self.config.device,
            offload_strategy=self.active_offload_strategy,
            runtime_seconds=round(runtime, 3),
            peak_gpu_allocated_mb=round(peak_gpu_mb, 2),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            technical_validation_status=tech_stats["status"],
            safety_flag=has_nsfw,
            human_review_status="pending",
            training_eligibility=False,
            mean_pixel_abs_diff=tech_stats["mean_pixel_abs_diff"],
            ssim_rgb=tech_stats["ssim_rgb"],
            clipping_low_rate=tech_stats["clipping_low_rate"],
            clipping_high_rate=tech_stats["clipping_high_rate"],
            # ControlNet & execution provenance
            generation_mode=self.config.mode,
            controlnet_id=self.config.controlnet_id if is_controlnet else None,
            controlnet_revision=self._loaded_controlnet_revision if is_controlnet else None,
            control_scale=float(control_scale) if is_controlnet else None,
            canny_low_threshold=self.config.canny_low_threshold if is_controlnet else None,
            canny_high_threshold=self.config.canny_high_threshold if is_controlnet else None,
            canny_edge_density=round(canny_density, 6) if canny_density is not None else None,
            peak_gpu_reserved_mb=round(peak_reserved_mb, 2),
            device_memory_free_mb=round(free_gpu_mb, 2),
            edge_overlap_iou_heuristic=edge_diag["edge_overlap_iou_heuristic"] if edge_diag else None,
            edge_overlap_dice_heuristic=edge_diag["edge_overlap_dice_heuristic"] if edge_diag else None,
            vae_id=getattr(self.config, "vae_id", None),
        )

        return out_arr, record, tech_stats
