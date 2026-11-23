"""
PipeAugment AI — Isolated AI Worker for Background API Jobs.
Executes in .venv-ai.
Supports 'traditional' and 'generative' augmentation jobs.
Updates job status JSON atomically after every item.
"""

import sys
import os
import json
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

# Ensure ai directory is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_cv.engine import TraditionalAugmentationEngine, derive_output_seed
from pipeaugment_cv.config import TraditionalConfig
from pipeaugment_eval.validator import QualityValidator
from pipeaugment_eval.config import QualityValidationConfig


def sample_calibrated_traditional_params(
    cv_engine: TraditionalAugmentationEngine,
    cond: str,
    rng: np.random.Generator,
    img_shape: Tuple[int, int, int],
    intensity: float
) -> Dict[str, Any]:
    """
    Calibrates interactive Studio Traditional CV parameters according to user intensity [0.10, 1.00] (default 0.55)
    so lighting/shadow/blur/noise variations show clear, unmistakable visual differences.
    """
    h, w = img_shape[:2]
    intensity = max(0.10, min(1.0, intensity))

    if cond == "brightness":
        sign = float(rng.choice([-1.0, 1.0]))
        delta = float(rng.uniform(0.08 + 0.15 * intensity, 0.12 + 0.32 * intensity))
        factor = round(max(0.40, min(1.70, 1.0 + sign * delta)), 4)
        return {"factor": factor}

    elif cond == "contrast":
        sign = float(rng.choice([-1.0, 1.0]))
        delta = float(rng.uniform(0.10 + 0.18 * intensity, 0.14 + 0.38 * intensity))
        factor = round(max(0.40, min(1.70, 1.0 + sign * delta)), 4)
        return {"factor": factor}

    elif cond == "gamma":
        sign = float(rng.choice([-1.0, 1.0]))
        delta = float(rng.uniform(0.12 + 0.20 * intensity, 0.18 + 0.42 * intensity))
        gamma = round(max(0.45, min(1.80, 1.0 + sign * delta)), 4)
        return {"gamma": gamma}

    elif cond == "uneven_illumination":
        angle = round(float(rng.uniform(0.0, 360.0)), 2)
        sign = float(rng.choice([-1.0, 1.0]))
        delta = float(rng.uniform(0.16 + 0.22 * intensity, 0.20 + 0.32 * intensity))
        illum_intensity = round(1.0 + sign * delta, 4)
        return {"angle_deg": angle, "intensity": illum_intensity}

    elif cond == "soft_shadow":
        cx = round(float(rng.uniform(0.20 * w, 0.80 * w)), 1)
        cy = round(float(rng.uniform(0.20 * h, 0.80 * h)), 1)
        radius = round(float(rng.uniform(0.25 * min(h, w), 0.55 * min(h, w))), 1)
        shadow_intensity = round(float(rng.uniform(0.18 + 0.25 * intensity, 0.25 + 0.45 * intensity)), 4)
        return {"center_x": cx, "center_y": cy, "radius": radius, "shadow_intensity": shadow_intensity}

    elif cond == "gaussian_blur":
        sigma = round(float(rng.uniform(0.8 + 1.2 * intensity, 1.4 + 2.8 * intensity)), 3)
        return {"sigma": sigma}

    elif cond == "gaussian_noise":
        std = round(float(rng.uniform(5.0 + 8.0 * intensity, 8.0 + 16.0 * intensity)), 2)
        return {"std": std}

    elif cond == "jpeg_compression":
        quality = int(round(rng.uniform(75.0 - 30.0 * intensity, 85.0 - 25.0 * intensity)))
        return {"quality": max(25, min(90, quality))}

    elif cond == "downsample_restore":
        scale = round(float(rng.uniform(0.60 - 0.25 * intensity, 0.75 - 0.20 * intensity)), 3)
        return {"scale": max(0.25, min(0.80, scale))}

    else:
        return cv_engine.sample_parameters(cond, rng, img_shape)


def get_inspector():
    """Lazily loads PipeAugmentInspector if available."""
    try:
        from pipeaugment_ml.inspector import PipeAugmentInspector
        return PipeAugmentInspector()
    except Exception as exc:
        print(f"Warning: PipeAugmentInspector unavailable ({exc})")
        return None


def update_job_status_atomically(status_file: Path, status_data: Dict[str, Any]) -> None:
    """Writes updated job status JSON using atomic temp file replacement."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = status_file.parent / f".tmp_{status_file.name}_{os.getpid()}"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)
    os.replace(temp_file, status_file)


def run_traditional_job(job_spec: Dict[str, Any], status_file: Path, status_data: Dict[str, Any]) -> None:
    """Executes a traditional augmentation job."""
    config_dict = job_spec.get("config", {})
    conditions = config_dict.get("conditions") or ["brightness", "contrast", "uneven_illumination"]
    intensity = float(config_dict.get("intensity", 0.5))
    intensity = max(0.0, min(1.0, intensity))
    variations_per_image = int(config_dict.get("variations_per_image", 1))
    variations_per_image = max(1, min(4, variations_per_image))
    base_seed = int(config_dict.get("seed", 42))

    augmented_dir = Path(job_spec["output_dirs"]["augmented_dir"]) / "traditional"
    metadata_dir = Path(job_spec["output_dirs"]["metadata_dir"])
    augmented_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    cv_engine = TraditionalAugmentationEngine(TraditionalConfig(base_seed=base_seed))
    # Load validator with traditional profile
    val_cfg_path = AI_DIR / "configs" / "quality_validation_v1.json"
    val_config = QualityValidationConfig.from_json(val_cfg_path) if val_cfg_path.exists() else QualityValidationConfig()
    validator = QualityValidator(config=val_config)
    inspector = get_inspector()

    parent_images = job_spec.get("parent_images", [])
    status_data["status"] = "running"
    status_data["started_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)

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
                raise FileNotFoundError(f"Parent image file missing: {file_path}")

            pil_img = Image.open(file_path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            src_rgb = np.array(pil_img)

            # Predict parent with ResNet-18 inspector if available
            parent_pred_class = parent_label
            if inspector is not None:
                try:
                    p_info, _ = inspector.predict_and_cam(src_rgb)
                    parent_pred_class = p_info["predicted_class"]
                except Exception as e_p:
                    print(f"Warning: Inspector failed on parent {parent_id}: {e_p}")

            # Generate variations
            for var_idx in range(variations_per_image):
                cond = conditions[var_idx % len(conditions)]
                out_seed = derive_output_seed(parent_id, var_idx, base_seed)
                local_rng = np.random.default_rng(out_seed)

                # Sample parameters with calibrated visible intensity scaling
                params = sample_calibrated_traditional_params(cv_engine, cond, local_rng, src_rgb.shape, intensity)
                # Apply transform
                transformed_rgb = cv_engine.apply_transform(src_rgb, cond, params, local_rng)

                # Validate with QualityValidator
                q_res = validator.validate_pair(src_rgb, transformed_rgb, category="traditional")
                quality_status = q_res["quality_status"]
                quality_flags = q_res["flags"]

                # Run ResNet-18 label preservation check
                label_preserved = True
                var_verdict = parent_pred_class
                var_conf = 1.0
                if inspector is not None:
                    try:
                        v_info, _ = inspector.predict_and_cam(transformed_rgb)
                        var_verdict = v_info["predicted_class"]
                        var_conf = v_info["confidence"]
                        if parent_pred_class:
                            label_preserved = bool(var_verdict == parent_pred_class)
                    except Exception as e_v:
                        print(f"Warning: Inspector failed on variation: {e_v}")

                quality_metrics = {
                    **q_res["metrics"],
                    "ssim": q_res["metrics"].get("ssim_rgb", 0.0),
                    "corrosion_diagnostic": q_res.get("corrosion_diagnostic"),
                    "label_preserved": label_preserved,
                    "parent_verdict": parent_pred_class,
                    "variation_verdict": var_verdict,
                    "variation_confidence": var_conf,
                }

                # Save lossless PNG
                gen_id = str(uuid4())
                stored_filename = f"{gen_id}.png"
                out_png_path = augmented_dir / stored_filename
                Image.fromarray(transformed_rgb, mode="RGB").save(out_png_path, format="PNG")

                # Compute hash and size
                with open(out_png_path, "rb") as f:
                    file_bytes = f.read()
                gen_sha256 = hashlib.sha256(file_bytes).hexdigest()
                size_bytes = len(file_bytes)
                h, w = transformed_rgb.shape[:2]

                rel_path = f"augmented/app_generated/traditional/{stored_filename}"

                # Write paired metadata JSON
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

    # Final status determination
    if failed_items == 0:
        status_data["status"] = "completed"
    elif completed_items > 0:
        status_data["status"] = "partial_failed"
    else:
        status_data["status"] = "failed"

    status_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)


def run_generative_job(job_spec: Dict[str, Any], status_file: Path, status_data: Dict[str, Any]) -> None:
    """Executes a generative diffusion augmentation job (ControlNet Canny / img2img)."""
    import torch
    from pipeaugment_gen.engine import GenerativeDiffusionEngine
    from pipeaugment_gen.config import GenerativePilotConfig
    from pipeaugment_gen.prompts import CONDITIONS as GEN_CONDITIONS

    config_dict = job_spec.get("config", {})
    raw_mode = str(config_dict.get("mode", "controlnet_canny")).strip().lower()
    if raw_mode in ("controlnet_canny", "controlnet_img2img", "controlnet"):
        mode = "controlnet_canny"
        engine_mode = "controlnet_img2img"
    else:
        mode = "img2img"
        engine_mode = "plain_img2img"

    # Load frozen defaults
    frozen_cfg_path = AI_DIR / "configs" / "generative_expansion_v1.json"
    frozen_cfg = {}
    if frozen_cfg_path.exists():
        with open(frozen_cfg_path, "r", encoding="utf-8") as f:
            frozen_cfg = json.load(f)

    # Parameters with user overrides clamped to authorized ranges
    strength = float(config_dict.get("strength", 0.45))
    strength = max(0.15, min(0.75, strength))

    control_scale = float(config_dict.get("control_scale", 0.75))
    control_scale = max(0.30, min(1.0, control_scale))

    # Calibrate DPM++ 2M Karras in Studio to 18 effective steps
    num_steps = min(45, max(24, int(round(18 / max(strength, 0.25)))))

    guidance_scale = float(config_dict.get("guidance_scale", 7.5))
    guidance_scale = max(1.0, min(15.0, guidance_scale))

    conditions = config_dict.get("conditions") or ["low_light_inspection", "directional_industrial_lighting"]
    # Verify all conditions exist in prompt templates
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

    # Initialize Generative Pilot Config with VAE upgrade and DPM++ solver
    gen_config = GenerativePilotConfig(
        model_id=frozen_cfg.get("model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        controlnet_id=frozen_cfg.get("controlnet_id", "lllyasviel/control_v11p_sd15_canny"),
        mode=engine_mode,
        vae_id="stabilityai/sd-vae-ft-mse",
        use_dpm_solver=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        offload_strategy="none",
        base_seed=base_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        canny_low_threshold=int(frozen_cfg.get("canny_low_threshold", 100)),
        canny_high_threshold=int(frozen_cfg.get("canny_high_threshold", 200)),
    )

    # Load QualityValidator with generative profile
    val_cfg_path = AI_DIR / "configs" / "quality_validation_v1.json"
    val_config = QualityValidationConfig.from_json(val_cfg_path) if val_cfg_path.exists() else QualityValidationConfig()
    validator = QualityValidator(config=val_config)
    inspector = get_inspector()

    parent_images = job_spec.get("parent_images", [])
    status_data["status"] = "running"
    status_data["started_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)

    engine = GenerativeDiffusionEngine(config=gen_config)
    print(f"Loading generative diffusion pipeline (mode={engine_mode})...")
    engine.load_pipeline()

    completed_items = 0
    failed_items = 0

    try:
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
                    raise FileNotFoundError(f"Parent image file missing: {file_path}")

                pil_img = Image.open(file_path)
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                src_rgb = np.array(pil_img)
                orig_h, orig_w = src_rgb.shape[:2]

                needs_resize = (orig_w, orig_h) != gen_config.target_size
                if needs_resize:
                    pil_resized = pil_img.resize(gen_config.target_size, Image.Resampling.BICUBIC)
                    input_rgb = np.array(pil_resized)
                else:
                    input_rgb = src_rgb

                # Predict parent with ResNet-18 inspector if available
                parent_pred_class = parent_label
                if inspector is not None:
                    try:
                        p_info, _ = inspector.predict_and_cam(src_rgb)
                        parent_pred_class = p_info["predicted_class"]
                    except Exception as e_p:
                        print(f"Warning: Inspector failed on parent {parent_id}: {e_p}")

                for var_idx in range(variations_per_image):
                    cond = valid_conditions[var_idx % len(valid_conditions)]
                    var_seed = base_seed + var_idx * 1000 + item_idx

                    # Generate variation
                    out_rgb, record, tech_stats = engine.generate(
                        source_image=input_rgb,
                        parent_id=parent_id,
                        parent_sha256=parent_info.get("sha256", "none"),
                        parent_split="unassigned",
                        source_label=parent_label or "corroded",
                        condition=cond,
                        strength=strength,
                        seed=var_seed,
                        control_scale=control_scale if gen_config.is_controlnet else None,
                    )

                    if record.safety_flag:
                        raise ValueError("Diffusion safety checker flagged the output image.")

                    gen_pil = Image.fromarray(out_rgb, mode="RGB")
                    if needs_resize:
                        gen_pil = gen_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
                    gen_rgb = np.array(gen_pil)

                    # Run Quality Validation
                    q_res = validator.validate_pair(src_rgb, gen_rgb, category="generative")
                    quality_status = q_res["quality_status"]
                    quality_flags = q_res["flags"]

                    # Run ResNet-18 label preservation check
                    label_preserved = True
                    var_verdict = parent_pred_class
                    var_conf = 1.0
                    if inspector is not None:
                        try:
                            v_info, _ = inspector.predict_and_cam(gen_rgb)
                            var_verdict = v_info["predicted_class"]
                            var_conf = v_info["confidence"]
                            if parent_pred_class:
                                label_preserved = bool(var_verdict == parent_pred_class)
                        except Exception as e_v:
                            print(f"Warning: Inspector failed on variation: {e_v}")

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

                    # Save lossless PNG
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

                    # Write paired metadata JSON
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
                            "vae_id": gen_config.vae_id,
                            "scheduler": "DPMSolverMultistepScheduler",
                            "runtime_seconds": record.runtime_seconds,
                            "peak_gpu_mb": record.peak_gpu_allocated_mb,
                            "strength": strength,
                            "control_scale": control_scale if gen_config.is_controlnet else None,
                            "guidance_scale": gen_config.guidance_scale,
                            "inference_steps": gen_config.num_inference_steps,
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
                item_result["status"] = "failed"
                item_result["error"] = str(exc)
                failed_items += 1

            status_data["item_results"].append(item_result)
            status_data["completed_items"] = completed_items
            status_data["failed_items"] = failed_items
            update_job_status_atomically(status_file, status_data)

    finally:
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failed_items == 0:
        status_data["status"] = "completed"
    elif completed_items > 0:
        status_data["status"] = "partial_failed"
    else:
        status_data["status"] = "failed"

    status_data["finished_at"] = datetime.now(timezone.utc).isoformat()
    update_job_status_atomically(status_file, status_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="PipeAugment AI Isolated Background Job Worker")
    parser.add_argument("--job-spec", type=str, required=True, help="Path to job spec JSON file")
    args = parser.parse_args()

    spec_path = Path(args.job_spec)
    if not spec_path.exists():
        sys.stderr.write(f"Job spec file not found: {spec_path}\n")
        sys.exit(1)

    with open(spec_path, "r", encoding="utf-8") as f:
        job_spec = json.load(f)

    status_file = Path(job_spec["status_file"])
    with open(status_file, "r", encoding="utf-8") as f:
        status_data = json.load(f)

    job_type = job_spec.get("job_type")
    try:
        if job_type == "traditional":
            run_traditional_job(job_spec, status_file, status_data)
        elif job_type == "generative":
            run_generative_job(job_spec, status_file, status_data)
        else:
            raise ValueError(f"Unknown job type: '{job_type}'. Authorized: traditional, generative")
    except Exception as exc:
        status_data["status"] = "failed"
        status_data["error_message"] = str(exc)
        status_data["finished_at"] = datetime.now(timezone.utc).isoformat()
        update_job_status_atomically(status_file, status_data)
        sys.stderr.write(f"Fatal worker exception: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
