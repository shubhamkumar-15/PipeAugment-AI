"""
PipeAugment AI — Persistent Warm GPU AI Worker Service.
Executes in .venv-ai on http://127.0.0.1:8001.

Keeps PyTorch, SD 1.5, ControlNet Canny, sd-vae-ft-mse VAE,
QualityValidator (LPIPS), and ResNet-18 (Grad-CAM) warm in VRAM.
Reduces Studio generation stall from ~35s cold subprocess start to ~2-4s warm inference.
Provides HTTP endpoints:
  GET  /health
  POST /run_job
  POST /shutdown
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

# Ensure ai directory is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [AI-WORKER] %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_worker")

# Import AI modules
try:
    import torch
except ImportError:
    torch = None

from pipeaugment_cv.engine import TraditionalAugmentationEngine, derive_output_seed
from pipeaugment_cv.config import TraditionalConfig
from pipeaugment_eval.validator import QualityValidator
from pipeaugment_eval.config import QualityValidationConfig
from pipeaugment_gen.engine import GenerativeDiffusionEngine
from pipeaugment_gen.config import GenerativePilotConfig
from pipeaugment_gen.prompts import CONDITIONS as GEN_CONDITIONS
from scripts.run_api_job import sample_calibrated_traditional_params, update_job_status_atomically


class WarmModelManager:
    """Manages pre-warmed models kept persistently in VRAM/RAM."""

    def __init__(self, device: str = "cuda"):
        self.device = device if (torch is not None and torch.cuda.is_available() and device == "cuda") else "cpu"
        self.lock = threading.Lock()
        logger.info("Initializing WarmModelManager on device: %s", self.device)

        # 1. Warm Traditional Engine
        self.cv_engine = TraditionalAugmentationEngine(TraditionalConfig(base_seed=42))
        logger.info("[✓] Traditional Augmentation Engine warm.")

        # 2. Warm Quality Validator (loads LPIPS if available)
        val_cfg_path = AI_DIR / "configs" / "quality_validation_v1.json"
        val_config = QualityValidationConfig.from_json(val_cfg_path) if val_cfg_path.exists() else QualityValidationConfig()
        self.validator = QualityValidator(config=val_config)
        logger.info("[✓] Quality Validator (LPIPS/SSIM/Edge) warm.")

        # 3. Warm ResNet-18 Inspector
        self.inspector = None
        try:
            from pipeaugment_ml.inspector import PipeAugmentInspector
            self.inspector = PipeAugmentInspector()
            logger.info("[✓] ResNet-18 Corrosion Inspector (Grad-CAM) warm.")
        except Exception as exc:
            logger.warning("ResNet-18 Inspector unavailable: %s", exc)

        # 4. Warm Generative Pipeline (Default: ControlNet Canny)
        self.gen_engine: Optional[GenerativeDiffusionEngine] = None
        self.current_gen_mode: Optional[str] = None
        self._preload_generative_pipeline("controlnet_img2img")

    def _preload_generative_pipeline(self, engine_mode: str) -> None:
        """Loads and pre-warms the generative diffusion pipeline into VRAM."""
        frozen_cfg_path = AI_DIR / "configs" / "generative_expansion_v1.json"
        frozen_cfg = {}
        if frozen_cfg_path.exists():
            try:
                with open(frozen_cfg_path, "r", encoding="utf-8") as f:
                    frozen_cfg = json.load(f)
            except Exception as e:
                logger.warning("Could not read frozen config: %s", e)

        gen_config = GenerativePilotConfig(
            model_id=frozen_cfg.get("model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
            controlnet_id=frozen_cfg.get("controlnet_id", "lllyasviel/control_v11p_sd15_canny"),
            mode=engine_mode,
            vae_id="stabilityai/sd-vae-ft-mse",
            use_dpm_solver=True,
            device=self.device,
            offload_strategy="none",
            base_seed=42,
            guidance_scale=7.5,
            num_inference_steps=30,
            canny_low_threshold=int(frozen_cfg.get("canny_low_threshold", 100)),
            canny_high_threshold=int(frozen_cfg.get("canny_high_threshold", 200)),
        )

        try:
            logger.info("Pre-warming GenerativeDiffusionEngine (mode=%s, VAE=sd-vae-ft-mse)...", engine_mode)
            engine = GenerativeDiffusionEngine(config=gen_config)
            engine.load_pipeline()
            self.gen_engine = engine
            self.current_gen_mode = engine_mode
            vram_mb = 0.0
            if torch is not None and torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info("[✓] Generative Pipeline warm in VRAM (Allocated: %.1f MB).", vram_mb)
        except Exception as exc:
            logger.error("Failed to pre-warm Generative Pipeline: %s", exc)
            self.gen_engine = None

    def get_generative_engine(self, required_engine_mode: str, num_steps: int, guidance_scale: float) -> GenerativeDiffusionEngine:
        """Returns warm engine, reloading only if the underlying architecture mode changed."""
        if self.gen_engine is None or self.current_gen_mode != required_engine_mode:
            if self.gen_engine is not None and torch is not None and torch.cuda.is_available():
                del self.gen_engine
                torch.cuda.empty_cache()
            self._preload_generative_pipeline(required_engine_mode)

        if self.gen_engine is None:
            raise RuntimeError(f"Generative pipeline could not be loaded for mode: {required_engine_mode}")

        self.gen_engine.config.num_inference_steps = num_steps
        self.gen_engine.config.guidance_scale = guidance_scale
        return self.gen_engine

    def get_vram_allocated_mb(self) -> float:
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        return 0.0


# Global singleton instance
WARM_MANAGER: Optional[WarmModelManager] = None


def execute_job_warm(job_spec: Dict[str, Any], status_file: Path) -> Dict[str, Any]:
    """Executes a job using warm models with sequential locking and atomic stage updates."""
    global WARM_MANAGER
    if WARM_MANAGER is None:
        WARM_MANAGER = WarmModelManager()

    job_type = job_spec.get("job_type", "traditional")
    job_id = job_spec.get("job_id")

    # Load initial status file
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)
    else:
        status_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "running",
            "stage": "loading_model",
            "total_items": len(job_spec.get("parent_images", [])),
            "completed_items": 0,
            "failed_items": 0,
            "parent_image_ids": [p["image_id"] for p in job_spec.get("parent_images", [])],
            "config": job_spec.get("config", {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "generated_image_ids": [],
            "item_results": [],
            "error_message": None,
        }

    status_data["status"] = "running"
    status_data["stage"] = "loading_model"
    status_data["started_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)

    start_time = time.time()

    with WARM_MANAGER.lock:
        if job_type == "traditional":
            _run_traditional_warm(job_spec, status_file, status_data)
        elif job_type == "generative":
            _run_generative_warm(job_spec, status_file, status_data)
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    elapsed = round(time.time() - start_time, 2)
    logger.info("Job %s completed in %.2fs (status=%s)", job_id, elapsed, status_data.get("status"))
    return status_data


def _run_traditional_warm(job_spec: Dict[str, Any], status_file: Path, status_data: Dict[str, Any]) -> None:
    config_dict = job_spec.get("config", {})
    conditions = config_dict.get("conditions") or ["brightness", "contrast", "uneven_illumination"]
    intensity = float(config_dict.get("intensity", 0.55))
    intensity = max(0.10, min(1.0, intensity))
    variations_per_image = int(config_dict.get("variations_per_image", 1))
    variations_per_image = max(1, min(4, variations_per_image))
    base_seed = int(config_dict.get("seed", 42))

    augmented_dir = Path(job_spec["output_dirs"]["augmented_dir"]) / "traditional"
    metadata_dir = Path(job_spec["output_dirs"]["metadata_dir"])
    augmented_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cv_engine = WARM_MANAGER.cv_engine
    validator = WARM_MANAGER.validator
    inspector = WARM_MANAGER.inspector

    parent_images = job_spec.get("parent_images", [])
    completed_items = 0
    failed_items = 0

    for item_idx, parent_info in enumerate(parent_images):
        parent_id = str(parent_info["image_id"])
        file_path = Path(parent_info["file_path"])
        parent_label = parent_info.get("label")

        item_result = {
            "parent_image_id": parent_id,
            "status": "running",
            "generated_image_ids": [],
            "quality_statuses": [],
            "error": None,
        }

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Parent image missing: {file_path}")

            pil_img = Image.open(file_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            src_rgb = np.array(pil_img)

            parent_pred_class = parent_label
            if inspector is not None:
                try:
                    p_info, _ = inspector.predict_and_cam(src_rgb)
                    parent_pred_class = p_info["predicted_class"]
                except Exception as e_p:
                    logger.warning("Inspector failed on parent: %s", e_p)

            for var_idx in range(variations_per_image):
                cond = conditions[var_idx % len(conditions)]
                out_seed = derive_output_seed(parent_id, var_idx, base_seed)
                local_rng = np.random.default_rng(out_seed)

                status_data["stage"] = "synthesizing"
                update_job_status_atomically(status_file, status_data)

                params = sample_calibrated_traditional_params(cv_engine, cond, local_rng, src_rgb.shape, intensity)
                transformed_rgb = cv_engine.apply_transform(src_rgb, cond, params, local_rng)

                status_data["stage"] = "scoring_quality"
                update_job_status_atomically(status_file, status_data)

                q_res = validator.validate_pair(src_rgb, transformed_rgb, category="traditional")
                quality_status = q_res["quality_status"]
                quality_flags = q_res["flags"]

                label_preserved = True
                var_verdict = parent_pred_class
                var_conf = 1.0
                if inspector is not None:
                    try:
                        v_info, _ = inspector.predict_and_cam(transformed_rgb)
                        var_verdict = v_info["predicted_class"]
                        var_conf = float(v_info["confidence"])
                        if parent_pred_class is not None:
                            label_preserved = bool(var_verdict == parent_pred_class)
                    except Exception as e_v:
                        logger.warning("Inspector failed on variation: %s", e_v)

                quality_metrics = {
                    **q_res["metrics"],
                    "ssim": q_res["metrics"].get("ssim_rgb", 0.0),
                    "corrosion_diagnostic": q_res.get("corrosion_diagnostic"),
                    "label_preserved": label_preserved,
                    "parent_verdict": parent_pred_class,
                    "variation_verdict": var_verdict,
                    "variation_confidence": var_conf,
                }

                gen_id = str(uuid4())
                stored_filename = f"{gen_id}.png"
                out_png_path = augmented_dir / stored_filename
                Image.fromarray(transformed_rgb, mode="RGB").save(out_png_path, format="PNG")

                with open(out_png_path, "rb") as f:
                    file_bytes = f.read()
                gen_sha256 = hashlib.sha256(file_bytes).hexdigest()
                size_bytes = len(file_bytes)
                h, w = transformed_rgb.shape[:2]

                rel_path = f"augmented/app_generated/traditional/{stored_filename}"

                meta_json = {
                    "image_id": gen_id,
                    "parent_image_id": parent_id,
                    "original_filename": f"{parent_info.get('original_filename', parent_id)}_trad_{cond}.png",
                    "stored_filename": stored_filename,
                    "relative_path": rel_path,
                    "image_format": "PNG",
                    "content_type": "image/png",
                    "width": w,
                    "height": h,
                    "size_bytes": size_bytes,
                    "sha256": gen_sha256,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_type": "traditional",
                    "generation_method": cond,
                    "label": parent_label,
                    "annotation_status": "unannotated",
                    "split": "unassigned",
                    "condition": cond,
                    "parameters": params,
                    "seed": out_seed,
                    "label_preserved": label_preserved,
                    "quality_metrics": quality_metrics,
                    "quality_status": quality_status,
                    "quality_flags": quality_flags,
                    "review_status": "pending",
                    "review_notes": None,
                    "reviewer": None,
                }
                meta_file = metadata_dir / f"{gen_id}.json"
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_json, f, indent=2)

                item_result["generated_image_ids"].append(gen_id)
                item_result["quality_statuses"].append(quality_status)
                status_data["generated_image_ids"].append(gen_id)

            item_result["status"] = "success"
            completed_items += 1

        except Exception as exc:
            item_result["status"] = "failed"
            item_result["error"] = str(exc)
            failed_items += 1

        status_data["item_results"].append(item_result)
        status_data["completed_items"] = completed_items
        status_data["failed_items"] = failed_items
        update_job_status_atomically(status_file, status_data)

    if failed_items == 0:
        status_data["status"] = "completed"
        status_data["stage"] = "completed"
    elif completed_items > 0:
        status_data["status"] = "partial_failed"
        status_data["stage"] = "completed"
    else:
        status_data["status"] = "failed"
        status_data["stage"] = "completed"

    status_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)


def _run_generative_warm(job_spec: Dict[str, Any], status_file: Path, status_data: Dict[str, Any]) -> None:
    config_dict = job_spec.get("config", {})
    raw_mode = str(config_dict.get("mode", "controlnet_canny")).strip().lower()
    if raw_mode in ("controlnet_canny", "controlnet_img2img", "controlnet"):
        mode = "controlnet_canny"
        engine_mode = "controlnet_img2img"
    else:
        mode = "img2img"
        engine_mode = "plain_img2img"

    strength = float(config_dict.get("strength", 0.45))
    strength = max(0.15, min(0.75, strength))
    control_scale = float(config_dict.get("control_scale", 0.75))
    control_scale = max(0.30, min(1.0, control_scale))

    # Calibrate DPM++ 2M Karras in Studio to 18 effective steps:
    num_steps = min(45, max(24, int(round(18 / max(strength, 0.25)))))
    guidance_scale = float(config_dict.get("guidance_scale", 7.5))
    guidance_scale = max(1.0, min(15.0, guidance_scale))

    conditions = config_dict.get("conditions") or ["low_light_inspection", "directional_industrial_lighting"]
    valid_conditions = [c for c in conditions if c in GEN_CONDITIONS]
    if not valid_conditions:
        valid_conditions = ["low_light_inspection"]

    variations_per_image = int(config_dict.get("variations_per_image", 1))
    variations_per_image = max(1, min(4, variations_per_image))
    base_seed = int(config_dict.get("seed", 42))

    augmented_dir = Path(job_spec["output_dirs"]["augmented_dir"]) / "generative"
    metadata_dir = Path(job_spec["output_dirs"]["metadata_dir"])
    augmented_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    status_data["stage"] = "loading_model"
    update_job_status_atomically(status_file, status_data)

    engine = WARM_MANAGER.get_generative_engine(engine_mode, num_steps, guidance_scale)
    validator = WARM_MANAGER.validator
    inspector = WARM_MANAGER.inspector

    parent_images = job_spec.get("parent_images", [])
    completed_items = 0
    failed_items = 0

    for item_idx, parent_info in enumerate(parent_images):
        parent_id = str(parent_info["image_id"])
        file_path = Path(parent_info["file_path"])
        parent_label = parent_info.get("label")

        item_result = {
            "parent_image_id": parent_id,
            "status": "running",
            "generated_image_ids": [],
            "quality_statuses": [],
            "error": None,
        }

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Parent image missing: {file_path}")

            pil_img = Image.open(file_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            src_rgb = np.array(pil_img)
            orig_h, orig_w = src_rgb.shape[:2]

            target_size = engine.config.target_size
            needs_resize = (orig_w, orig_h) != target_size
            if needs_resize:
                pil_resized = pil_img.resize(target_size, Image.Resampling.BICUBIC)
                input_rgb = np.array(pil_resized)
            else:
                input_rgb = src_rgb

            parent_pred_class = parent_label
            if inspector is not None:
                try:
                    p_info, _ = inspector.predict_and_cam(src_rgb)
                    parent_pred_class = p_info["predicted_class"]
                except Exception as e_p:
                    logger.warning("Inspector failed on parent: %s", e_p)

            for var_idx in range(variations_per_image):
                cond = valid_conditions[var_idx % len(valid_conditions)]
                var_seed = base_seed + var_idx * 1000 + item_idx

                status_data["stage"] = "synthesizing"
                update_job_status_atomically(status_file, status_data)

                out_rgb, record, tech_stats = engine.generate(
                    source_image=input_rgb,
                    parent_id=parent_id,
                    parent_sha256=parent_info.get("sha256", "none"),
                    parent_split="unassigned",
                    source_label=parent_label or "corroded",
                    condition=cond,
                    strength=strength,
                    seed=var_seed,
                    control_scale=control_scale if engine.config.is_controlnet else None,
                )

                if record.safety_flag:
                    raise ValueError("Diffusion safety checker flagged the output image.")

                gen_pil = Image.fromarray(out_rgb, mode="RGB")
                if needs_resize:
                    gen_pil = gen_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
                gen_rgb = np.array(gen_pil)

                status_data["stage"] = "scoring_quality"
                update_job_status_atomically(status_file, status_data)

                q_res = validator.validate_pair(src_rgb, gen_rgb, category="generative")
                quality_status = q_res["quality_status"]
                quality_flags = q_res["flags"]

                label_preserved = True
                var_verdict = parent_pred_class
                var_conf = 1.0
                if inspector is not None:
                    try:
                        v_info, _ = inspector.predict_and_cam(gen_rgb)
                        var_verdict = v_info["predicted_class"]
                        var_conf = float(v_info["confidence"])
                        if parent_pred_class is not None:
                            label_preserved = bool(var_verdict == parent_pred_class)
                    except Exception as e_v:
                        logger.warning("Inspector failed on variation: %s", e_v)

                quality_metrics = {
                    **q_res["metrics"],
                    "ssim": q_res["metrics"].get("ssim_rgb", 0.0),
                    "runtime_seconds": record.runtime_seconds,
                    "corrosion_diagnostic": q_res.get("corrosion_diagnostic"),
                    "label_preserved": label_preserved,
                    "parent_verdict": parent_pred_class,
                    "variation_verdict": var_verdict,
                    "variation_confidence": var_conf,
                }

                gen_id = str(uuid4())
                stored_filename = f"{gen_id}.png"
                out_png_path = augmented_dir / stored_filename
                gen_pil.save(out_png_path, format="PNG")

                with open(out_png_path, "rb") as f:
                    file_bytes = f.read()
                gen_sha256 = hashlib.sha256(file_bytes).hexdigest()
                size_bytes = len(file_bytes)
                h, w = gen_rgb.shape[:2]

                rel_path = f"augmented/app_generated/generative/{stored_filename}"

                meta_json = {
                    "image_id": gen_id,
                    "parent_image_id": parent_id,
                    "original_filename": f"{parent_info.get('original_filename', parent_id)}_gen_{cond}.png",
                    "stored_filename": stored_filename,
                    "relative_path": rel_path,
                    "image_format": "PNG",
                    "content_type": "image/png",
                    "width": w,
                    "height": h,
                    "size_bytes": size_bytes,
                    "sha256": gen_sha256,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_type": "generative",
                    "generation_method": mode,
                    "label": parent_label,
                    "annotation_status": "unannotated",
                    "split": "unassigned",
                    "condition": cond,
                    "parameters": {
                        "mode": mode,
                        "model_name": record.pipeline_class,
                        "model_id": record.model_id,
                        "vae_id": engine.config.vae_id,
                        "scheduler": "DPMSolverMultistepScheduler",
                        "runtime_seconds": record.runtime_seconds,
                        "peak_gpu_mb": record.peak_gpu_allocated_mb,
                        "strength": strength,
                        "control_scale": control_scale if engine.config.is_controlnet else None,
                        "guidance_scale": engine.config.guidance_scale,
                        "inference_steps": engine.config.num_inference_steps,
                        "effective_denoising_steps": record.effective_denoising_steps,
                    },
                    "seed": var_seed,
                    "label_preserved": label_preserved,
                    "quality_metrics": quality_metrics,
                    "quality_status": quality_status,
                    "quality_flags": quality_flags,
                    "review_status": "pending",
                    "review_notes": None,
                    "reviewer": None,
                }
                meta_file = metadata_dir / f"{gen_id}.json"
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_json, f, indent=2)

                item_result["generated_image_ids"].append(gen_id)
                item_result["quality_statuses"].append(quality_status)
                status_data["generated_image_ids"].append(gen_id)

            item_result["status"] = "success"
            completed_items += 1

        except Exception as exc:
            logger.error("Item failed in generative job: %s", exc, exc_info=True)
            item_result["status"] = "failed"
            item_result["error"] = str(exc)
            failed_items += 1

        status_data["item_results"].append(item_result)
        status_data["completed_items"] = completed_items
        status_data["failed_items"] = failed_items
        update_job_status_atomically(status_file, status_data)

    if failed_items == 0:
        status_data["status"] = "completed"
        status_data["stage"] = "completed"
    elif completed_items > 0:
        status_data["status"] = "partial_failed"
        status_data["stage"] = "completed"
    else:
        status_data["status"] = "failed"
        status_data["stage"] = "completed"

    status_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)


class WarmWorkerHTTPHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for persistent AI worker."""

    def log_message(self, format, *args):
        logger.debug("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            global WARM_MANAGER
            gpu_name = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "CPU"
            vram_mb = WARM_MANAGER.get_vram_allocated_mb() if WARM_MANAGER else 0.0
            self._send_json(200, {
                "status": "healthy",
                "worker": "pipeaugment-warm-ai-worker",
                "cuda_available": torch.cuda.is_available() if torch else False,
                "device_name": gpu_name,
                "vram_allocated_mb": vram_mb,
                "models_warm": [
                    "controlnet_canny",
                    "traditional",
                    "quality_validator",
                    "inspector",
                ],
            })
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path == "/run_job":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            spec_file_str = payload.get("spec_file")
            if spec_file_str:
                spec_path = Path(spec_file_str)
                if not spec_path.exists():
                    self._send_json(404, {"error": f"Spec file not found: {spec_file_str}"})
                    return
                with open(spec_path, "r", encoding="utf-8") as f:
                    job_spec = json.load(f)
                status_file = spec_path.parent / f"{job_spec['job_id']}.json"
            else:
                job_spec = payload.get("job_spec")
                status_file_str = payload.get("status_file")
                if not job_spec or not status_file_str:
                    self._send_json(400, {"error": "Either 'spec_file' or ('job_spec' and 'status_file') required."})
                    return
                status_file = Path(status_file_str)

            try:
                res = execute_job_warm(job_spec, status_file)
                self._send_json(200, res)
            except Exception as exc:
                logger.error("Job execution error: %s", exc, exc_info=True)
                try:
                    self._send_json(500, {"error": str(exc)})
                except Exception:
                    pass

        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send_json(404, {"error": "Not Found"})


def main():
    parser = argparse.ArgumentParser(description="PipeAugment AI Warm Worker Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default 8001)")
    args = parser.parse_args()

    global WARM_MANAGER
    logger.info("Starting PipeAugment AI Warm Worker...")
    WARM_MANAGER = WarmModelManager()

    server = HTTPServer((args.host, args.port), WarmWorkerHTTPHandler)
    logger.info("=================================================================")
    logger.info("  WARM GPU AI WORKER LISTENING ON http://%s:%d", args.host, args.port)
    logger.info("  PyTorch + ControlNet + VAE + Validator ready for warm inference")
    logger.info("=================================================================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Worker received KeyboardInterrupt, shutting down...")
    finally:
        server.server_close()
        logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
