#!/usr/bin/env python3
"""
PipeAugment AI — Balanced Training-Only Generative Expansion & Quality Gating.
Generates 200 condition-controlled synthetic variations across 100 training parents
using the frozen winning ControlNet Canny configuration (scale=0.80, strength=0.35).

Every output is automatically screened through Phase 4 QualityValidator.
Only outputs with quality_status == 'pass' and safety_flag == False are admitted
to Experiment C training.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch

# Ensure ai directory is on python path
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_gen.config import GenerativePilotConfig
from pipeaugment_gen.engine import GenerativeDiffusionEngine
from pipeaugment_eval.config import QualityValidationConfig
from pipeaugment_eval.validator import QualityValidator


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_source_rgb(image_path: Path, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.uint8)


def sample_expansion_parents(
    train_manifest_path: Path,
    val_manifest_path: Path,
    test_manifest_path: Path,
    seed: int = 42,
    num_per_class: int = 50,
) -> pd.DataFrame:
    """
    Deterministically samples 50 non_corroded and 50 corroded parents strictly from train.csv.
    Verifies zero overlap with val.csv or test.csv.
    """
    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    # Disjointness sanity check
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])
    assert len(train_ids & val_ids) == 0, "Train and Val manifests share image IDs!"
    assert len(train_ids & test_ids) == 0, "Train and Test manifests share image IDs!"

    non_corr = train_df[train_df["label"] == "non_corroded"].sample(n=num_per_class, random_state=seed)
    corr = train_df[train_df["label"] == "corroded"].sample(n=num_per_class, random_state=seed)

    sampled = pd.concat([non_corr, corr], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Verify zero leakage
    sampled_ids = set(sampled["image_id"])
    assert len(sampled_ids & val_ids) == 0, "Sampled parents leak into val set!"
    assert len(sampled_ids & test_ids) == 0, "Sampled parents leak into test set!"

    return sampled


def make_expansion_output_id(parent_id: str, condition: str) -> str:
    p_prefix = parent_id[:20]
    return f"aug_gen_{p_prefix}_{condition}"


def build_spot_check_sheet(
    records: List[Dict[str, Any]],
    output_image_path: Path,
    raw_root: Path,
    augmented_dir: Path,
    num_admitted_samples: int = 16,
    num_flagged_samples: int = 8,
    seed: int = 42,
) -> None:
    """
    Builds a high-resolution contact sheet for visual spot checking.
    Selects 8 corroded + 8 non-corroded admitted outputs, plus up to 8 flagged outputs.
    """
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    admitted_corr = [r for r in records if r["training_eligibility"] and r["source_class"] == "corroded"]
    admitted_non = [r for r in records if r["training_eligibility"] and r["source_class"] == "non_corroded"]
    flagged = [r for r in records if not r["training_eligibility"]]

    selected_admitted = []
    if len(admitted_non) >= 8:
        selected_admitted.extend(rng.choice(admitted_non, size=8, replace=False).tolist())
    else:
        selected_admitted.extend(admitted_non)

    if len(admitted_corr) >= 8:
        selected_admitted.extend(rng.choice(admitted_corr, size=8, replace=False).tolist())
    else:
        selected_admitted.extend(admitted_corr)

    selected_flagged = []
    if flagged:
        k_flagged = min(len(flagged), num_flagged_samples)
        selected_flagged.extend(rng.choice(flagged, size=k_flagged, replace=False).tolist())

    display_items = selected_admitted + selected_flagged
    if not display_items:
        return

    # Canvas dimensions: 4 columns (Parent, Canny, Generated, Metrics) per row
    thumb_w, thumb_h = 256, 256
    row_h = thumb_h + 30
    canvas_w = thumb_w * 3 + 360
    header_h = 70
    canvas_h = header_h + len(display_items) * row_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(14, 19, 28))
    draw = ImageDraw.Draw(canvas)

    # Title Header
    draw.text((20, 15), "PipeAugment AI — Generative Expansion Spot-Check Audit Sheet", fill=(255, 90, 0))
    subtitle = f"Sampled: {len(selected_admitted)} Admitted ({len(admitted_non)} non, {len(admitted_corr)} corr available) | {len(selected_flagged)} Flagged | Frozen ControlNet scale=0.80, s=0.35"
    draw.text((20, 38), subtitle, fill=(160, 175, 195))

    for idx, item in enumerate(display_items):
        y_top = header_h + idx * row_h

        # 1. Parent
        p_path = raw_root / item["parent_relative_path"]
        p_thumb = Image.open(p_path).convert("RGB").resize((thumb_w, thumb_h))
        canvas.paste(p_thumb, (20, y_top))

        # 2. Canny Edge Map
        c_path = augmented_dir / "control_maps" / f"{item['parent_image_id'][:20]}_canny.png"
        if c_path.exists():
            c_thumb = Image.open(c_path).convert("RGB").resize((thumb_w, thumb_h))
        else:
            c_thumb = Image.new("RGB", (thumb_w, thumb_h), color=(0, 0, 0))
        canvas.paste(c_thumb, (30 + thumb_w, y_top))

        # 3. Generated Output
        g_path = augmented_dir / f"{item['generated_image_id']}.png"
        if g_path.exists():
            g_thumb = Image.open(g_path).convert("RGB").resize((thumb_w, thumb_h))
        else:
            g_thumb = Image.new("RGB", (thumb_w, thumb_h), color=(50, 0, 0))
        canvas.paste(g_thumb, (40 + thumb_w * 2, y_top))

        # 4. Text Info & Quality Badge
        tx = 50 + thumb_w * 3
        status_color = (16, 185, 129) if item["training_eligibility"] else (239, 68, 68)
        status_text = "ADMITTED (Pass)" if item["training_eligibility"] else f"EXCLUDED ({item['quality_status']})"

        draw.text((tx, y_top + 10), f"ID: {item['generated_image_id']}", fill=(255, 255, 255))
        draw.text((tx, y_top + 32), f"Class: {item['source_class']} | Condition: {item['condition']}", fill=(200, 215, 230))
        draw.text((tx, y_top + 54), f"SSIM: {item['ssim_rgb']:.4f} | LPIPS: {item['lpips']:.4f} | Edge IoU: {item['edge_overlap_iou']:.4f}", fill=(245, 158, 11))
        draw.text((tx, y_top + 76), f"Rust Area Delta: {item['rust_area_delta']} | Flags: {item['flags'] or 'None'}", fill=(160, 175, 195))
        draw.text((tx, y_top + 105), f"Status: {status_text}", fill=status_color)
        draw.line([(20, y_top + row_h - 2), (canvas_w - 20, y_top + row_h - 2)], fill=(36, 48, 68))

    canvas.save(output_image_path, format="PNG")
    print(f"Spot-check sheet saved: {output_image_path} ({canvas_w}x{canvas_h})")


def main():
    parser = argparse.ArgumentParser(description="Generate balanced training-only generative expansion for PipeAugment AI.")
    parser.add_argument("--config", type=str, default="ai/configs/generative_expansion_v1.json", help="Path to frozen expansion config")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-output smoke test on 1 parent")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume and skip completed outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f)

    smoke_test = args.smoke_test
    resume = args.resume

    # Paths
    manifests_dir = PROJECT_ROOT / "datasets" / "manifests"
    raw_root = PROJECT_ROOT / "datasets" / "raw" / "k-pipelines"
    augmented_dir = PROJECT_ROOT / "datasets" / "augmented" / "generative_v1"
    metadata_dir = augmented_dir / "metadata"
    control_maps_dir = augmented_dir / "control_maps"

    for d in [augmented_dir, metadata_dir, control_maps_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print("PIPEAUGMENT AI — BALANCED TRAINING-ONLY GENERATIVE EXPANSION")
    print("====================================================================")
    print(f"Config: {config_path}")
    print(f"Device: {args.device} | Smoke Test: {smoke_test} | Resume: {resume}")

    # 1. Deterministic Parent Selection
    train_csv = manifests_dir / "train.csv"
    val_csv = manifests_dir / "val.csv"
    test_csv = manifests_dir / "test.csv"

    num_per_class = 1 if smoke_test else exp_cfg["expansion_targets"]["class_balance"]["non_corroded"]
    parents_df = sample_expansion_parents(
        train_csv, val_csv, test_csv,
        seed=42, num_per_class=num_per_class
    )
    print(f"Sampled {len(parents_df)} distinct training parents ({len(parents_df[parents_df['label']=='non_corroded'])} non_corroded, {len(parents_df[parents_df['label']=='corroded'])} corroded).")
    print(f"Zero overlap with val.csv ({len(pd.read_csv(val_csv))} rows) and test.csv ({len(pd.read_csv(test_csv))} rows) verified.")

    # 2. Setup QualityValidator
    q_config_path = PROJECT_ROOT / "ai" / "configs" / "quality_validation_v1.json"
    q_config = QualityValidationConfig.from_json(q_config_path)
    validator = QualityValidator(config=q_config)
    print("Phase 4 QualityValidator initialized with generative screening profile.")

    # 3. Setup Generative Diffusion Engine with Frozen ControlNet Canny Config
    conditions = list(exp_cfg["conditions"].keys())
    if smoke_test:
        conditions = conditions[:1]

    engine_cfg = GenerativePilotConfig(
        model_id=exp_cfg["model_id"],
        model_revision=exp_cfg["model_revision"],
        controlnet_id=exp_cfg["controlnet_id"],
        controlnet_revision=exp_cfg["controlnet_revision"],
        mode="controlnet_img2img",
        torch_dtype=exp_cfg["precision"],
        device=args.device,
        offload_strategy=exp_cfg.get("offload_strategy", "none"),
        canny_low_threshold=exp_cfg["canny_low_threshold"],
        canny_high_threshold=exp_cfg["canny_high_threshold"],
        controlnet_scales=[exp_cfg["control_scale"]],
        num_inference_steps=exp_cfg["num_inference_steps"],
        guidance_scale=exp_cfg["guidance_scale"],
        strengths=[exp_cfg["strength"]],
        conditions=conditions,
        base_seed=42,
    )
    engine_cfg.validate()

    engine = GenerativeDiffusionEngine(engine_cfg)
    print("Loading Stable Diffusion v1.5 + Canny ControlNet pipeline into GPU...")
    engine.load_pipeline()

    # 4. Generate & Screen Loop
    target_count = len(parents_df) * len(conditions)
    print(f"Targeting {target_count} total generated variations ({len(conditions)} per parent)...")

    attempted_count = 0
    passed_count = 0
    flagged_count = 0
    failed_count = 0
    skipped_count = 0

    records: List[Dict[str, Any]] = []
    admitted_records: List[Dict[str, Any]] = []

    t_start = time.time()

    for p_idx, (_, parent_row) in enumerate(parents_df.iterrows(), 1):
        parent_id = str(parent_row["image_id"])
        parent_prefix = parent_id[:20]
        parent_rel_path = str(parent_row["relative_path"])
        parent_label = str(parent_row["label"])
        parent_sha256 = str(parent_row["sha256"])
        parent_group = str(parent_row["group_id"])
        parent_full_path = raw_root / parent_rel_path

        parent_rgb = load_source_rgb(parent_full_path)

        # Precompute Canny edge map
        c_map_path = control_maps_dir / f"{parent_prefix}_canny.png"
        if c_map_path.exists():
            c_map = cv2.imread(str(c_map_path), cv2.IMREAD_GRAYSCALE)
        else:
            gray = cv2.cvtColor(parent_rgb, cv2.COLOR_RGB2GRAY)
            c_map = cv2.Canny(gray, exp_cfg["canny_low_threshold"], exp_cfg["canny_high_threshold"])
            cv2.imwrite(str(c_map_path), c_map)

        for cond in conditions:
            out_id = make_expansion_output_id(parent_id, cond)
            out_png = augmented_dir / f"{out_id}.png"
            out_json = metadata_dir / f"{out_id}.json"

            # Deterministic seed per (parent, condition)
            seed_hash = hashlib.sha256(f"{parent_id}_{cond}_seed42".encode()).hexdigest()
            out_seed = int(seed_hash[:8], 16)

            # Check resume
            if resume and out_png.exists() and out_json.exists():
                try:
                    with open(out_json, "r", encoding="utf-8") as jf:
                        meta_rec = json.load(jf)
                    records.append(meta_rec)
                    if meta_rec.get("training_eligibility", False):
                        passed_count += 1
                        admitted_records.append(meta_rec)
                    else:
                        flagged_count += 1
                    skipped_count += 1
                    print(f"[{p_idx}/{len(parents_df)}] [Resumed] {out_id} | Status: {meta_rec.get('quality_status')}")
                    continue
                except Exception:
                    pass

            attempted_count += 1
            t_gen_0 = time.time()

            try:
                # 1. Generate image using engine
                out_arr, pilot_rec, tech_stats = engine.generate(
                    source_image=parent_rgb,
                    parent_id=parent_id,
                    parent_sha256=parent_sha256,
                    parent_split="train",
                    source_label=parent_label,
                    condition=cond,
                    strength=exp_cfg["strength"],
                    seed=out_seed,
                    control_scale=exp_cfg["control_scale"],
                    source_canny=c_map,
                )
                gen_runtime = time.time() - t_gen_0

                # 2. Run Phase 4 Automated QualityValidator
                val_res = validator.validate_pair(
                    source_rgb=parent_rgb,
                    generated_rgb=out_arr,
                    category="generative",
                    metadata={"parent_id": parent_id, "condition": cond}
                )

                q_status = val_res["quality_status"]  # 'pass' | 'flag_for_review' | 'fail'
                q_flags = val_res["flags"]
                q_metrics = val_res["metrics"]
                safety_flagged = bool(tech_stats.get("safety_flag", False))

                # Admission rule: strictly pass AND not safety_flagged
                is_admitted = (q_status == "pass") and (not safety_flagged)

                # Save output image
                Image.fromarray(out_arr).save(out_png, format="PNG")
                out_sha256 = compute_file_sha256(out_png)

                meta_entry = {
                    "generated_image_id": out_id,
                    "parent_image_id": parent_id,
                    "parent_relative_path": parent_rel_path,
                    "parent_sha256": parent_sha256,
                    "parent_group_id": parent_group,
                    "source_class": parent_label,
                    "class_index": 0 if parent_label == "non_corroded" else 1,
                    "condition": cond,
                    "seed": out_seed,
                    "strength": exp_cfg["strength"],
                    "control_scale": exp_cfg["control_scale"],
                    "output_sha256": out_sha256,
                    "relative_output_path": f"datasets/augmented/generative_v1/{out_id}.png",
                    "dimensions": "512x512",
                    "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "generation_runtime_seconds": round(gen_runtime, 3),
                    "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
                    "ssim_rgb": q_metrics.get("ssim_rgb"),
                    "lpips": q_metrics.get("lpips"),
                    "edge_overlap_iou": q_metrics.get("edge_overlap_iou"),
                    "edge_overlap_dice": q_metrics.get("edge_overlap_dice"),
                    "rust_area_delta": q_metrics.get("rust_area_delta"),
                    "rust_mask_iou": q_metrics.get("rust_mask_iou"),
                    "quality_status": q_status,
                    "flags": ";".join(q_flags) if q_flags else "",
                    "safety_flagged": safety_flagged,
                    "training_eligibility": is_admitted,
                    "label_provenance": "inherited_under_quality_screened_generative_policy",
                    "human_review_status": "spot_check_pending",
                }

                # Save JSON metadata
                with open(out_json, "w", encoding="utf-8") as jf:
                    json.dump(meta_entry, jf, indent=2)

                records.append(meta_entry)
                if is_admitted:
                    passed_count += 1
                    admitted_records.append(meta_entry)
                elif q_status == "flag_for_review":
                    flagged_count += 1
                else:
                    failed_count += 1

                print(
                    f"[{p_idx:03d}/{len(parents_df)}] {out_id} | "
                    f"{parent_label[:4]} | {cond[:9]} | "
                    f"SSIM: {q_metrics.get('ssim_rgb'):.4f} | "
                    f"LPIPS: {q_metrics.get('lpips'):.4f} | "
                    f"Edge: {q_metrics.get('edge_overlap_iou'):.4f} | "
                    f"Status: {q_status} | Admitted: {is_admitted} | {gen_runtime:.2f}s"
                )

            except Exception as e:
                failed_count += 1
                print(f"[{p_idx:03d}/{len(parents_df)}] ERROR generating {out_id}: {e}")

    total_time = time.time() - t_start

    print("\n--------------------------------------------------------------------")
    print(f"GENERATION & SCREENING COMPLETE: {len(records)} outputs processed in {total_time:.1f}s")
    print(f"Admitted (Pass): {len(admitted_records)} | Flagged: {flagged_count} | Failed: {failed_count} | Resumed: {skipped_count}")
    print("--------------------------------------------------------------------")

    # 5. Save Summary Ledgers
    gen_ledger_csv = manifests_dir / "generative_v1_generated.csv"
    pd.DataFrame(records).to_csv(gen_ledger_csv, index=False)
    print(f"Saved complete generation ledger: {gen_ledger_csv} ({len(records)} rows)")

    admitted_csv = manifests_dir / "generative_v1_admitted.csv"
    pd.DataFrame(admitted_records).to_csv(admitted_csv, index=False)
    print(f"Saved admitted candidate manifest: {admitted_csv} ({len(admitted_records)} rows)")

    summary_json = manifests_dir / "generative_v1_generation_summary.json"
    summary_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_targets": target_count,
        "total_generated": len(records),
        "total_admitted": len(admitted_records),
        "total_flagged": sum(1 for r in records if r["quality_status"] == "flag_for_review"),
        "total_failed": sum(1 for r in records if r["quality_status"] == "fail"),
        "admission_rate": round(len(admitted_records) / max(1, len(records)), 4),
        "admitted_by_class": {
            "non_corroded": sum(1 for r in admitted_records if r["source_class"] == "non_corroded"),
            "corroded": sum(1 for r in admitted_records if r["source_class"] == "corroded"),
        },
        "admitted_by_condition": {
            "low_light_inspection": sum(1 for r in admitted_records if r["condition"] == "low_light_inspection"),
            "directional_industrial_lighting": sum(1 for r in admitted_records if r["condition"] == "directional_industrial_lighting"),
        },
        "mean_admitted_ssim": round(float(np.mean([r["ssim_rgb"] for r in admitted_records])), 4) if admitted_records else 0.0,
        "mean_admitted_lpips": round(float(np.mean([r["lpips"] for r in admitted_records])), 4) if admitted_records else 0.0,
        "mean_admitted_edge_iou": round(float(np.mean([r["edge_overlap_iou"] for r in admitted_records])), 4) if admitted_records else 0.0,
        "total_runtime_seconds": round(total_time, 2),
        "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved generation summary: {summary_json}")

    # 6. Build Spot-Check Sheet
    spot_check_png = PROJECT_ROOT / "experiments" / "evaluations" / "generative_expansion_v1" / "spot_check_sheet.png"
    build_spot_check_sheet(records, spot_check_png, raw_root, augmented_dir)

    # 7. Build Combined Experiment C Training Manifest
    print("\nBuilding combined Experiment C training manifest (train_pipeaugment_v1.csv)...")
    train_orig_df = pd.read_csv(train_csv)
    trad_csv = manifests_dir / "traditional_v1_generated.csv"
    trad_df = pd.read_csv(trad_csv)

    combined_rows = []

    # A. 480 Originals
    for _, row in train_orig_df.iterrows():
        combined_rows.append({
            "image_id": str(row["image_id"]),
            "dataset": "k-pipelines-standard",
            "relative_path": str(row["relative_path"]),
            "label": str(row["label"]),
            "class_index": int(row["class_index"]),
            "split": "train",
            "group_id": str(row["group_id"]),
            "sha256": str(row["sha256"]),
            "source_type": "original",
        })

    # B. 960 Traditional variations
    parent_group_lookup = dict(zip(train_orig_df["image_id"], train_orig_df["group_id"]))
    for _, row in trad_df.iterrows():
        p_id = str(row["parent_image_id"])
        combined_rows.append({
            "image_id": str(row["generated_image_id"]),
            "dataset": "k-pipelines-traditional-v1",
            "relative_path": f"datasets/augmented/{row['relative_output_path']}",
            "label": str(row["source_class"]),
            "class_index": int(row["class_index"]),
            "split": "train",
            "group_id": parent_group_lookup.get(p_id, "unknown"),
            "sha256": str(row["output_sha256"]),
            "source_type": "traditional",
        })

    # C. Admitted Generative variations
    for row in admitted_records:
        p_id = str(row["parent_image_id"])
        combined_rows.append({
            "image_id": str(row["generated_image_id"]),
            "dataset": "k-pipelines-generative-v1",
            "relative_path": str(row["relative_output_path"]),
            "label": str(row["source_class"]),
            "class_index": int(row["class_index"]),
            "split": "train",
            "group_id": parent_group_lookup.get(p_id, "unknown"),
            "sha256": str(row["output_sha256"]),
            "source_type": "generative",
        })

    combined_df = pd.DataFrame(combined_rows)
    combined_csv = manifests_dir / "train_pipeaugment_v1.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Created combined manifest: {combined_csv}")
    print(f"Total training pool: {len(combined_df)} samples ({len(train_orig_df)} originals + {len(trad_df)} traditional + {len(admitted_records)} generative)")
    print(f"Class breakdown: {len(combined_df[combined_df['class_index']==0])} non_corroded, {len(combined_df[combined_df['class_index']==1])} corroded")


if __name__ == "__main__":
    main()
