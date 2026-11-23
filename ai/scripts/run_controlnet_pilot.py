#!/usr/bin/env python3
"""
PipeAugment AI — Canny-ControlNet Comparison Pilot Execution Script.
Evaluates whether structural conditioning via ControlNet Canny improves upon
the existing pretrained Stable Diffusion v1.5 Image-to-Image pilot.

Conditioning:
- Model: lllyasviel/control_v11p_sd15_canny + stable-diffusion-v1-5/stable-diffusion-v1-5
- Strength: 0.35 (fixed, matched to diffusion_v1)
- Condition scales: [0.50, 0.80]
- Parents: 6 deterministic training parents from diffusion_v1/sources.csv
- Seeds: Identical fixed seed per parent matching diffusion_v1
- Conditions: low_light_inspection, directional_industrial_lighting
- Total outputs: 6 parents x 2 conditions x 2 scales = 24 outputs.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import cv2

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_gen.config import GenerativePilotConfig
from pipeaugment_gen.engine import GenerativeDiffusionEngine
from pipeaugment_gen.metadata import (
    derive_parent_seed,
    make_output_id,
    make_controlnet_output_id,
    compute_canny_edge_map,
    compute_edge_overlap_diagnostics,
)
from pipeaugment_gen.prompts import get_prompt_for_condition
from scripts.build_controlnet_comparison_sheets import build_comparison_sheet


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


def run_controlnet_pilot(
    config_path: Path,
    smoke_test: bool = False,
    resume: bool = True
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Resolve paths
    sources_manifest_path = PROJECT_ROOT / cfg_dict["dataset"]["sources_manifest"]
    raw_root = PROJECT_ROOT / cfg_dict["dataset"]["raw_root"]
    output_dir = PROJECT_ROOT / cfg_dict["storage"]["output_dir"]
    ref_diff_dir = PROJECT_ROOT / cfg_dict["storage"].get("reference_diffusion_dir", "experiments/pilots/diffusion_v1")

    # Subdirectories
    outputs_dir = output_dir / "outputs"
    metadata_dir = output_dir / "metadata"
    control_maps_dir = output_dir / "control_maps"
    contact_sheets_dir = output_dir / "contact_sheets"
    for d in [outputs_dir, metadata_dir, control_maps_dir, contact_sheets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "generation.log"
    def log_print(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")

    log_print("=== Starting PipeAugment AI Canny-ControlNet Pilot ===")
    log_print(f"Mode: {'Smoke Test (1 output)' if smoke_test else 'Full Pilot (24 outputs)'}")
    log_print(f"Output Directory: {output_dir}")

    # Load sources manifest
    if not sources_manifest_path.exists():
        raise FileNotFoundError(f"Sources manifest not found: {sources_manifest_path}")

    sources_df = pd.read_csv(sources_manifest_path)
    log_print(f"Loaded {len(sources_df)} parents from {sources_manifest_path.name}")

    # Copy sources manifest for provenance
    local_sources_csv = output_dir / "sources.csv"
    sources_df.to_csv(local_sources_csv, index=False)

    # Build config & engine
    engine_cfg = GenerativePilotConfig(
        model_id=cfg_dict["model"]["model_id"],
        model_revision=cfg_dict["model"]["model_revision"],
        controlnet_id=cfg_dict["model"]["controlnet_id"],
        controlnet_revision=cfg_dict["model"]["controlnet_revision"],
        mode=cfg_dict["generation"]["mode"],
        torch_dtype=cfg_dict["model"]["torch_dtype"],
        variant=cfg_dict["model"]["variant"],
        use_safetensors=cfg_dict["model"]["use_safetensors"],
        device=cfg_dict["model"]["device"],
        offload_strategy=cfg_dict["model"].get("offload_strategy", "none"),
        canny_low_threshold=cfg_dict["controlnet"]["canny_low_threshold"],
        canny_high_threshold=cfg_dict["controlnet"]["canny_high_threshold"],
        controlnet_scales=cfg_dict["controlnet"]["control_scales"],
        num_inference_steps=cfg_dict["generation"]["num_inference_steps"],
        guidance_scale=cfg_dict["generation"]["guidance_scale"],
        strengths=cfg_dict["generation"]["strengths"],
        conditions=cfg_dict["generation"]["conditions"],
        base_seed=cfg_dict["dataset"]["selection_seed"],
        num_parents_per_class=len(sources_df) // 2,
    )
    engine_cfg.validate()
    engine = GenerativeDiffusionEngine(engine_cfg)

    # Initialize Engine (Model load)
    engine.load_pipeline()

    attempted_count = 0
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    all_records: List[Dict[str, Any]] = []

    conditions = engine_cfg.conditions
    strengths = engine_cfg.strengths  # [0.35]
    control_scales = engine_cfg.controlnet_scales  # [0.5, 0.8]

    # Pre-generate and cache Canny maps for all parents
    log_print("Generating and saving Canny edge control maps...")
    canny_maps: Dict[str, np.ndarray] = {}
    canny_map_paths: Dict[str, Path] = {}
    for _, p_row in sources_df.iterrows():
        p_id = str(p_row["image_id"])
        p_img_path = raw_root / str(p_row["relative_path"])
        p_img = load_source_rgb_image(p_img_path)
        c_map, density = compute_canny_edge_map(
            p_img,
            low_thresh=engine_cfg.canny_low_threshold,
            high_thresh=engine_cfg.canny_high_threshold
        )
        canny_maps[p_id] = c_map
        c_path = control_maps_dir / f"{p_id[:20]}_canny.png"
        Image.fromarray(c_map).save(c_path, format="PNG")
        canny_map_paths[p_id] = c_path
        log_print(f"  Canny map saved: {c_path.name} | Edge density: {density:.4f}")

    # Set evaluation targets
    if smoke_test:
        eval_parents = sources_df.head(1)
        eval_conditions = conditions[:1]
        eval_scales = control_scales[:1]
    else:
        eval_parents = sources_df
        eval_conditions = conditions
        eval_scales = control_scales

    t_start = datetime.now(timezone.utc)

    for p_idx, (_, parent_row) in enumerate(eval_parents.iterrows(), 1):
        parent_id = str(parent_row["image_id"])
        parent_rel_path = str(parent_row["relative_path"])
        parent_label = str(parent_row["label"])
        parent_sha256 = str(parent_row["sha256"])
        parent_split = str(parent_row["split"])

        parent_full_path = raw_root / parent_rel_path
        parent_img = load_source_rgb_image(parent_full_path)
        source_canny = canny_maps[parent_id]

        # Fixed deterministic seed per parent
        parent_seed = derive_parent_seed(parent_id, engine_cfg.base_seed)

        log_print(f"[{p_idx}/{len(eval_parents)}] Processing Parent {parent_id[:20]} ({parent_label}) | Seed: {parent_seed}")

        for cond in eval_conditions:
            for str_val in strengths:
                for c_scale in eval_scales:
                    out_id = make_controlnet_output_id(parent_id, cond, str_val, c_scale)
                    out_png_path = outputs_dir / f"{out_id}.png"
                    out_json_path = metadata_dir / f"{out_id}.json"

                    # Check resume
                    if resume and out_png_path.exists() and out_json_path.exists():
                        log_print(f"  [Skipped - Exists] {out_id}")
                        with open(out_json_path, "r", encoding="utf-8") as f:
                            rec_dict = json.load(f)
                        all_records.append(rec_dict)
                        skipped_count += 1
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
                            control_scale=c_scale,
                            source_canny=source_canny,
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
                                f"Alloc: {record.peak_gpu_allocated_mb}MB | Reserv: {record.peak_gpu_reserved_mb}MB | "
                                f"SSIM: {record.ssim_rgb} | Edge IoU: {record.edge_overlap_iou_heuristic}"
                            )
                        else:
                            failed_count += 1
                            log_print(f"  ! FAILED technical checks: {out_id} | Details: {tech_stats}")

                    except Exception as e:
                        failed_count += 1
                        log_print(f"  X ERROR generating {out_id}: {str(e)}")

    # Build Comparison Sheets if not smoke test or if full set generated
    comparison_sheet_paths = []
    if not smoke_test:
        log_print("Generating 12 side-by-side comparison sheets...")
        for p_idx, (_, parent_row) in enumerate(sources_df.iterrows(), 1):
            parent_id = str(parent_row["image_id"])
            parent_label = str(parent_row["label"])
            parent_rel_path = str(parent_row["relative_path"])
            parent_full_path = raw_root / parent_rel_path
            canny_map_path = canny_map_paths[parent_id]

            for cond in conditions:
                # 1. Plain img2img s35
                plain_id = make_output_id(parent_id, cond, 0.35)
                plain_img_path = ref_diff_dir / "outputs" / f"{plain_id}.png"
                plain_json_path = ref_diff_dir / "metadata" / f"{plain_id}.json"
                plain_meta = None
                if plain_json_path.exists():
                    with open(plain_json_path, "r", encoding="utf-8") as pf:
                        plain_meta = json.load(pf)

                # 2. ControlNet c=0.50
                c05_id = make_controlnet_output_id(parent_id, cond, 0.35, 0.50)
                c05_img_path = outputs_dir / f"{c05_id}.png"
                c05_json_path = metadata_dir / f"{c05_id}.json"
                c05_meta = None
                if c05_json_path.exists():
                    with open(c05_json_path, "r", encoding="utf-8") as c05f:
                        c05_meta = json.load(c05f)

                # 3. ControlNet c=0.80
                c08_id = make_controlnet_output_id(parent_id, cond, 0.35, 0.80)
                c08_img_path = outputs_dir / f"{c08_id}.png"
                c08_json_path = metadata_dir / f"{c08_id}.json"
                c08_meta = None
                if c08_json_path.exists():
                    with open(c08_json_path, "r", encoding="utf-8") as c08f:
                        c08_meta = json.load(c08f)

                sheet_name = f"sheet_parent_{p_idx:02d}_{parent_id[:16]}_{cond}.png"
                sheet_dest = contact_sheets_dir / sheet_name

                built_path = build_comparison_sheet(
                    parent_id=parent_id,
                    parent_label=parent_label,
                    condition=cond,
                    parent_img_path=parent_full_path,
                    canny_map_path=canny_map_path,
                    plain_img_path=plain_img_path if plain_img_path.exists() else None,
                    plain_meta=plain_meta,
                    cnet_05_path=c05_img_path if c05_img_path.exists() else None,
                    cnet_05_meta=c05_meta,
                    cnet_08_path=c08_img_path if c08_img_path.exists() else None,
                    cnet_08_meta=c08_meta,
                    output_path=sheet_dest,
                )
                comparison_sheet_paths.append(str(built_path.as_posix()))
                log_print(f"  Saved comparison sheet: {built_path.name}")

    # Generate review.csv for human domain review
    review_rows = []
    for rec in all_records:
        review_rows.append({
            "output_id": rec.get("output_id", ""),
            "parent_id": rec.get("parent_id", ""),
            "condition": rec.get("condition", ""),
            "method": "controlnet_canny",
            "control_scale": rec.get("control_scale", 0.0),
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
    alloc_vrams = [r["peak_gpu_allocated_mb"] for r in all_records if "peak_gpu_allocated_mb" in r]
    res_vrams = [r["peak_gpu_reserved_mb"] for r in all_records if "peak_gpu_reserved_mb" in r]
    free_vrams = [r["device_memory_free_mb"] for r in all_records if "device_memory_free_mb" in r]
    ssim_vals = [r["ssim_rgb"] for r in all_records if "ssim_rgb" in r]
    edge_ious = [r["edge_overlap_iou_heuristic"] for r in all_records if r.get("edge_overlap_iou_heuristic") is not None]
    edge_dices = [r["edge_overlap_dice_heuristic"] for r in all_records if r.get("edge_overlap_dice_heuristic") is not None]

    summary = {
        "pilot_name": cfg_dict["pilot_name"],
        "model_id": engine_cfg.model_id,
        "model_revision": engine._loaded_revision or engine_cfg.model_revision,
        "controlnet_id": engine_cfg.controlnet_id,
        "controlnet_revision": engine._loaded_controlnet_revision or engine_cfg.controlnet_revision,
        "license": "CreativeML OpenRAIL-M",
        "precision": engine_cfg.torch_dtype,
        "device": engine_cfg.device,
        "offload_strategy": engine.active_offload_strategy,
        "parent_count": len(eval_parents),
        "total_outputs_targeted": len(eval_parents) * len(eval_conditions) * len(eval_scales),
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
            "max_peak_gpu_allocated_mb": round(float(np.max(alloc_vrams)), 2) if alloc_vrams else 0.0,
            "avg_peak_gpu_allocated_mb": round(float(np.mean(alloc_vrams)), 2) if alloc_vrams else 0.0,
            "max_peak_gpu_reserved_mb": round(float(np.max(res_vrams)), 2) if res_vrams else 0.0,
            "avg_peak_gpu_reserved_mb": round(float(np.mean(res_vrams)), 2) if res_vrams else 0.0,
            "min_device_memory_free_mb": round(float(np.min(free_vrams)), 2) if free_vrams else 0.0,
        },
        "ssim_diagnostics": {
            "mean_ssim_rgb": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
            "min_ssim_rgb": round(float(np.min(ssim_vals)), 4) if ssim_vals else 0.0,
            "max_ssim_rgb": round(float(np.max(ssim_vals)), 4) if ssim_vals else 0.0,
        },
        "edge_overlap_diagnostics": {
            "mean_edge_overlap_iou": round(float(np.mean(edge_ious)), 4) if edge_ious else 0.0,
            "min_edge_overlap_iou": round(float(np.min(edge_ious)), 4) if edge_ious else 0.0,
            "max_edge_overlap_iou": round(float(np.max(edge_ious)), 4) if edge_ious else 0.0,
            "mean_edge_overlap_dice": round(float(np.mean(edge_dices)), 4) if edge_dices else 0.0,
        },
        "contact_sheets": comparison_sheet_paths,
        "review_csv": str(review_csv.as_posix()),
        "matched_plain_baseline_dir": str(ref_diff_dir.as_posix()),
        "generation_start_utc": t_start.isoformat(),
        "generation_end_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    log_print(f"ControlNet pilot execution finished. Summary saved to {summary_json.name}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PipeAugment AI Canny-ControlNet pilot.")
    parser.add_argument("--config", type=str, default=str(AI_DIR / "configs" / "controlnet_pilot_v1.json"), help="Config JSON path")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 output smoke test")
    parser.add_argument("--pilot", action="store_true", help="Run full 24-output pilot")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg_file = Path(args.config)
    if not cfg_file.exists():
        print(f"Error: Config file not found: {cfg_file}")
        sys.exit(1)

    res = run_controlnet_pilot(
        config_path=cfg_file,
        smoke_test=args.smoke_test,
        resume=(not args.no_resume)
    )
    print("Execution Result:")
    print(json.dumps(res, indent=2))
