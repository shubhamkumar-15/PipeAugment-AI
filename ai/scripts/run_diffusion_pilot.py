#!/usr/bin/env python3
"""
Pretrained Diffusion Image-to-Image Pilot Execution Script for PipeAugment AI.
Generates condition-controlled variations from 6 deterministically selected training parents
(3 non-corroded, 3 corroded) across 2 environmental lighting conditions and 2 strengths (0.20, 0.35).
Outputs are stored under experiments/pilots/diffusion_v1/ with complete metadata, technical validation,
and contact sheets for human review.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from PIL import Image, ImageOps

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_gen.config import GenerativePilotConfig
from pipeaugment_gen.engine import GenerativeDiffusionEngine
from pipeaugment_gen.metadata import derive_parent_seed, make_output_id
from pipeaugment_gen.prompts import get_prompt_for_condition
from scripts.build_diffusion_contact_sheets import build_parent_contact_sheet


def load_source_rgb_image(image_path: Path) -> np.ndarray:
    """Loads image with EXIF orientation correction and converts to 512x512 RGB uint8 ndarray."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_pilot_parents(
    train_manifest_path: Path,
    num_per_class: int = 3,
    selection_seed: int = 42
) -> pd.DataFrame:
    """
    Deterministically selects training parents for the pilot.
    Strictly restricted to train.csv; never validation or test.
    """
    df = pd.read_csv(train_manifest_path)
    # Sort deterministically by image_id
    df_sorted = df.sort_values("image_id").reset_index(drop=True)

    non_corroded = df_sorted[df_sorted["label"] == "non_corroded"].sample(
        n=num_per_class, random_state=selection_seed
    )
    corroded = df_sorted[df_sorted["label"] == "corroded"].sample(
        n=num_per_class, random_state=selection_seed
    )

    parents_df = pd.concat([non_corroded, corroded], ignore_index=True)
    return parents_df


def run_diffusion_pilot(
    config_path: Path,
    smoke_test: bool = False,
    resume: bool = True
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Resolve paths
    train_manifest = PROJECT_ROOT / cfg_dict["dataset"]["train_manifest"]
    raw_root = PROJECT_ROOT / cfg_dict["dataset"]["raw_root"]
    output_dir = PROJECT_ROOT / cfg_dict["storage"]["output_dir"]

    # Subdirectories
    outputs_dir = output_dir / "outputs"
    metadata_dir = output_dir / "metadata"
    contact_sheets_dir = output_dir / "contact_sheets"
    for d in [outputs_dir, metadata_dir, contact_sheets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "generation.log"
    def log_print(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")

    log_print(f"=== Starting PipeAugment AI Pretrained Diffusion Pilot ===")
    log_print(f"Mode: {'Smoke Test (1 output)' if smoke_test else 'Full Pilot (24 outputs)'}")
    log_print(f"Output Directory: {output_dir}")

    # Build config & engine
    engine_cfg = GenerativePilotConfig(
        model_id=cfg_dict["model"]["model_id"],
        model_revision=cfg_dict["model"]["model_revision"],
        torch_dtype=cfg_dict["model"]["torch_dtype"],
        variant=cfg_dict["model"]["variant"],
        use_safetensors=cfg_dict["model"]["use_safetensors"],
        device=cfg_dict["model"]["device"],
        offload_strategy=cfg_dict["model"].get("offload_strategy", "none"),
        num_inference_steps=cfg_dict["generation"]["num_inference_steps"],
        guidance_scale=cfg_dict["generation"]["guidance_scale"],
        strengths=cfg_dict["generation"]["strengths"],
        conditions=cfg_dict["generation"]["conditions"],
        base_seed=cfg_dict["dataset"]["selection_seed"],
        num_parents_per_class=cfg_dict["dataset"]["num_parents_per_class"],
    )
    engine = GenerativeDiffusionEngine(engine_cfg)

    # Select parents
    parents_df = select_pilot_parents(
        train_manifest,
        num_per_class=engine_cfg.num_parents_per_class,
        selection_seed=engine_cfg.base_seed
    )
    sources_csv = output_dir / "sources.csv"
    parents_df.to_csv(sources_csv, index=False)
    log_print(f"Selected {len(parents_df)} training parents (3 non_corroded, 3 corroded). Manifest: {sources_csv.name}")

    # Initialize Engine (Model load)
    engine.load_pipeline()

    attempted_count = 0
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    all_records: List[Dict[str, Any]] = []
    parent_variant_map: Dict[str, List[Dict[str, Any]]] = {row["image_id"]: [] for _, row in parents_df.iterrows()}

    conditions = engine_cfg.conditions
    strengths = engine_cfg.strengths

    # If smoke test, limit to 1 parent, 1 condition, 1 strength
    if smoke_test:
        eval_parents = parents_df.head(1)
        conditions = conditions[:1]
        strengths = strengths[:1]
    else:
        eval_parents = parents_df

    t_start = datetime.now(timezone.utc)

    for p_idx, (_, parent_row) in enumerate(eval_parents.iterrows(), 1):
        parent_id = str(parent_row["image_id"])
        parent_rel_path = str(parent_row["relative_path"])
        parent_label = str(parent_row["label"])
        parent_sha256 = str(parent_row["sha256"])
        parent_split = str(parent_row["split"])

        parent_full_path = raw_root / parent_rel_path
        parent_img = load_source_rgb_image(parent_full_path)

        # One fixed seed per parent, reused across all its configurations
        parent_seed = derive_parent_seed(parent_id, engine_cfg.base_seed)

        log_print(f"[{p_idx}/{len(eval_parents)}] Processing Parent {parent_id[:20]} ({parent_label}) | Seed: {parent_seed}")

        for cond in conditions:
            for str_val in strengths:
                out_id = make_output_id(parent_id, cond, str_val)
                out_png_path = outputs_dir / f"{out_id}.png"
                out_json_path = metadata_dir / f"{out_id}.json"

                # Check resume
                if resume and out_png_path.exists() and out_json_path.exists():
                    log_print(f"  [Skipped - Exists] {out_id}")
                    with open(out_json_path, "r", encoding="utf-8") as f:
                        rec_dict = json.load(f)
                    all_records.append(rec_dict)
                    skipped_count += 1
                    parent_variant_map[parent_id].append({
                        "condition": cond,
                        "strength": str_val,
                        "image_path": str(out_png_path),
                        "seed": parent_seed,
                        "ssim": rec_dict.get("ssim_rgb", 0.0),
                        "effective_steps": rec_dict.get("effective_denoising_steps", 0),
                    })
                    continue

                attempted_count += 1
                try:
                    out_arr, record, tech_stats = engine.generate(
                        source_image=parent_img,
                        parent_id=parent_id,
                        parent_sha256=parent_sha256,
                        parent_split=parent_split,
                        source_label=parent_label,
                        condition=cond,
                        strength=str_val,
                        seed=parent_seed,
                    )

                    # Save PNG losslessly
                    Image.fromarray(out_arr).save(out_png_path, format="PNG")
                    out_sha256 = compute_file_sha256(out_png_path)
                    record.output_sha256 = out_sha256
                    record.relative_output_path = str(out_png_path.relative_to(PROJECT_ROOT).as_posix())

                    # Save JSON metadata
                    rec_dict = record.to_dict()
                    with open(out_json_path, "w", encoding="utf-8") as jf:
                        json.dump(rec_dict, jf, indent=2)

                    all_records.append(rec_dict)

                    if tech_stats["status"] == "passed":
                        successful_count += 1
                        log_print(
                            f"  + Generated {out_id} | Runtime: {record.runtime_seconds}s | "
                            f"VRAM: {record.peak_gpu_allocated_mb}MB | SSIM: {record.ssim_rgb} | Diff: {record.mean_pixel_abs_diff}"
                        )
                    else:
                        failed_count += 1
                        log_print(f"  ! FAILED technical checks: {out_id} | Details: {tech_stats}")

                    parent_variant_map[parent_id].append({
                        "condition": cond,
                        "strength": str_val,
                        "image_path": str(out_png_path),
                        "seed": parent_seed,
                        "ssim": record.ssim_rgb,
                        "effective_steps": record.effective_denoising_steps,
                    })

                except Exception as e:
                    failed_count += 1
                    log_print(f"  X ERROR generating {out_id}: {str(e)}")

    # Build Contact Sheets if not smoke test or if full set generated
    contact_sheet_paths = []
    if not smoke_test:
        log_print("Generating high-resolution parent contact sheets...")
        for p_idx, (_, parent_row) in enumerate(parents_df.iterrows(), 1):
            parent_id = str(parent_row["image_id"])
            parent_rel_path = str(parent_row["relative_path"])
            parent_label = str(parent_row["label"])
            parent_full_path = raw_root / parent_rel_path

            sheet_path = contact_sheets_dir / f"contact_sheet_parent_{p_idx:02d}_{parent_id[:16]}.png"
            built_path = build_parent_contact_sheet(
                parent_id=parent_id,
                parent_label=parent_label,
                parent_img_path=parent_full_path,
                variants=parent_variant_map[parent_id],
                output_path=sheet_path
            )
            contact_sheet_paths.append(str(built_path.as_posix()))
            log_print(f"  Saved sheet: {built_path.name}")

    # Generate review.csv for human review
    review_rows = []
    for rec in all_records:
        review_rows.append({
            "output_id": rec["output_id"],
            "parent_id": rec["parent_id"],
            "condition": rec["condition"],
            "strength": rec["strength"],
            "geometry_preserved": "unreviewed",
            "corrosion_preserved": "unreviewed",
            "condition_visible": "unreviewed",
            "realism_acceptable": "unreviewed",
            "decision": "pending",
            "reviewer": "",
            "review_notes": "",
        })
    review_csv = output_dir / "review.csv"
    pd.DataFrame(review_rows).to_csv(review_csv, index=False)

    # Save summary.json
    total_time = (datetime.now(timezone.utc) - t_start).total_seconds()
    runtimes = [r["runtime_seconds"] for r in all_records if "runtime_seconds" in r]
    peak_vrams = [r["peak_gpu_allocated_mb"] for r in all_records if "peak_gpu_allocated_mb" in r]
    ssim_vals = [r["ssim_rgb"] for r in all_records if "ssim_rgb" in r]

    summary = {
        "pilot_name": cfg_dict["pilot_name"],
        "model_id": engine_cfg.model_id,
        "model_revision": engine._loaded_revision or engine_cfg.model_revision,
        "license": "CreativeML OpenRAIL-M",
        "precision": engine_cfg.torch_dtype,
        "device": engine_cfg.device,
        "offload_strategy": engine.active_offload_strategy,
        "parent_count": len(eval_parents),
        "total_outputs_targeted": len(eval_parents) * len(conditions) * len(strengths),
        "attempted_count": attempted_count,
        "successful_count": successful_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "total_valid_outputs": len(all_records),
        "human_review_status": "pending",
        "training_eligibility": False,
        "runtime_metrics": {
            "total_elapsed_seconds": round(total_time, 2),
            "avg_runtime_per_image_seconds": round(float(np.mean(runtimes)), 2) if runtimes else 0.0,
            "max_runtime_seconds": round(float(np.max(runtimes)), 2) if runtimes else 0.0,
        },
        "memory_metrics": {
            "max_peak_gpu_allocated_mb": round(float(np.max(peak_vrams)), 2) if peak_vrams else 0.0,
            "avg_peak_gpu_allocated_mb": round(float(np.mean(peak_vrams)), 2) if peak_vrams else 0.0,
        },
        "ssim_diagnostics": {
            "mean_ssim_rgb": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
            "min_ssim_rgb": round(float(np.min(ssim_vals)), 4) if ssim_vals else 0.0,
            "max_ssim_rgb": round(float(np.max(ssim_vals)), 4) if ssim_vals else 0.0,
        },
        "contact_sheets": contact_sheet_paths,
        "review_csv": str(review_csv.as_posix()),
        "generation_start_utc": t_start.isoformat(),
        "generation_end_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    log_print(f"Pilot execution finished. Summary saved to {summary_json.name}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pretrained diffusion image-to-image pilot.")
    parser.add_argument("--config", type=str, default=str(AI_DIR / "configs" / "diffusion_pilot_v1.json"), help="Config JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 output smoke test")
    parser.add_argument("--pilot", action="store_true", help="Run full 24-output pilot")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg_file = Path(args.config)
    if not cfg_file.exists():
        print(f"Error: Config file not found: {cfg_file}")
        sys.exit(1)

    res = run_diffusion_pilot(
        config_path=cfg_file,
        smoke_test=args.smoke_test,
        resume=(not args.no_resume)
    )
    print("Execution Result:")
    print(json.dumps(res, indent=2))
