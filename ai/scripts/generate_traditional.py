#!/usr/bin/env python3
"""
Traditional Augmentation Generation Script for PipeAugment AI.
Supports:
1. --pilot: Generates 2 variations each for 12 training parents (6 per class),
   computes quantitative technical sanity checks, and creates a visual contact sheet.
2. Full generation: Generates 2 variations each for all 480 training parents (960 variations),
   saving losslessly as PNG in datasets/augmented/traditional_v1/, and creating manifests:
   - datasets/manifests/train_traditional_v1.csv (1,440 entries)
   - datasets/manifests/traditional_v1_generated.csv (960 entries)
   - datasets/manifests/traditional_v1_generation_summary.json
"""

import os
import sys
import json
import csv
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_cv.config import TraditionalConfig
from pipeaugment_cv.engine import TraditionalAugmentationEngine, derive_output_seed


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_parent_image(image_path: Path) -> np.ndarray:
    """Loads image with EXIF orientation correction and converts to RGB uint8 ndarray."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)


def run_pilot(
    train_df: pd.DataFrame,
    raw_root: Path,
    output_dir: Path,
    config: TraditionalConfig
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = TraditionalAugmentationEngine(config)

    # Deterministically select 6 per class
    non_corroded = train_df[train_df["label"] == "non_corroded"].head(6)
    corroded = train_df[train_df["label"] == "corroded"].head(6)
    pilot_parents = pd.concat([non_corroded, corroded], ignore_index=True)

    pilot_records = []
    pilot_images_for_sheet = []

    lighting_cycle = config.lighting_transforms
    quality_cycle = config.quality_transforms

    for idx, row in pilot_parents.iterrows():
        parent_id = str(row["image_id"])
        rel_path = str(row["relative_path"])
        full_path = raw_root / rel_path
        parent_img = load_parent_image(full_path)
        parent_label = str(row["label"])

        # Variant 1: Lighting
        t1_name = lighting_cycle[idx % len(lighting_cycle)]
        seed1 = derive_output_seed(parent_id, 1, config.base_seed)
        out1, p1, stat1 = engine.execute(parent_img, t1_name, parent_id, 1, config.base_seed)

        # Variant 2: Image Quality
        t2_name = quality_cycle[idx % len(quality_cycle)]
        seed2 = derive_output_seed(parent_id, 2, config.base_seed)
        out2, p2, stat2 = engine.execute(parent_img, t2_name, parent_id, 2, config.base_seed)

        # Save individual variations
        v1_filename = f"pilot_{parent_id[:16]}_var1_{t1_name}.png"
        v2_filename = f"pilot_{parent_id[:16]}_var2_{t2_name}.png"
        Image.fromarray(out1).save(output_dir / v1_filename)
        Image.fromarray(out2).save(output_dir / v2_filename)

        rec1 = {
            "parent_id": parent_id,
            "label": parent_label,
            "variant": 1,
            "transform_category": "lighting",
            "transform_name": t1_name,
            "parameters": p1,
            "seed": seed1,
            "stats": stat1,
            "filename": v1_filename,
        }
        rec2 = {
            "parent_id": parent_id,
            "label": parent_label,
            "variant": 2,
            "transform_category": "image_quality",
            "transform_name": t2_name,
            "parameters": p2,
            "seed": seed2,
            "stats": stat2,
            "filename": v2_filename,
        }
        pilot_records.extend([rec1, rec2])

        pilot_images_for_sheet.append((
            parent_img,
            out1,
            out2,
            f"{parent_id[:12]} ({parent_label})",
            f"Var1: {t1_name}\n{p1}",
            f"Var2: {t2_name}\n{p2}"
        ))

    # Build Contact Sheet (12 rows, 3 columns: Original, Var1, Var2)
    # Resize each cell to 256x256 with caption banner for readable sheet
    cell_w, cell_h = 256, 256
    text_h = 44
    row_h = cell_h + text_h
    sheet_w = cell_w * 3 + 40
    sheet_h = row_h * 12 + 60

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(15, 20, 30))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.text((20, 15), "PipeAugment AI - Traditional Augmentation Pilot (12 Parents x 2 Variations)", fill=(255, 90, 0))

    for r_idx, (orig, v1, v2, lbl_orig, lbl_v1, lbl_v2) in enumerate(pilot_images_for_sheet):
        y_top = 50 + r_idx * row_h
        imgs = [orig, v1, v2]
        labels = [f"Orig: {lbl_orig}", lbl_v1, lbl_v2]

        for c_idx in range(3):
            x_left = 20 + c_idx * (cell_w + 10)
            img_thumb = Image.fromarray(imgs[c_idx]).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            sheet.paste(img_thumb, (x_left, y_top))
            # Text caption
            draw.text((x_left + 2, y_top + cell_h + 2), labels[c_idx][:38], fill=(220, 230, 245))

    contact_sheet_path = output_dir / "traditional_pilot_contact_sheet.png"
    sheet.save(contact_sheet_path)

    # Technical Sanity Checks Summary
    all_passed = all(r["stats"]["status"] == "passed" for r in pilot_records)
    mean_abs_diffs = [r["stats"]["mean_pixel_abs_diff"] for r in pilot_records]
    low_clips = [r["stats"]["clipping_low_rate"] for r in pilot_records]
    high_clips = [r["stats"]["clipping_high_rate"] for r in pilot_records]

    summary = {
        "pilot_parent_count": len(pilot_parents),
        "variations_generated": len(pilot_records),
        "all_technical_checks_passed": all_passed,
        "mean_pixel_abs_diff_avg": round(float(np.mean(mean_abs_diffs)), 4),
        "mean_pixel_abs_diff_range": [round(float(np.min(mean_abs_diffs)), 4), round(float(np.max(mean_abs_diffs)), 4)],
        "max_clipping_low_rate": round(float(np.max(low_clips)), 5),
        "max_clipping_high_rate": round(float(np.max(high_clips)), 5),
        "contact_sheet_path": str(contact_sheet_path.as_posix()),
        "human_review_status": "not_reviewed (technical quantitative checks only)",
        "records": pilot_records,
    }

    with open(output_dir / "pilot_sanity_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_full_generation(
    train_df: pd.DataFrame,
    raw_root: Path,
    aug_root: Path,
    manifests_dir: Path,
    config: TraditionalConfig
) -> Dict[str, Any]:
    aug_dir = aug_root / "traditional_v1"
    aug_dir.mkdir(parents=True, exist_ok=True)
    engine = TraditionalAugmentationEngine(config)

    lighting_transforms = list(config.lighting_transforms)
    quality_transforms = list(config.quality_transforms)

    generated_rows: List[Dict[str, Any]] = []
    combined_train_rows: List[Dict[str, Any]] = []

    # First copy all original training parents into the combined manifest
    for _, row in train_df.iterrows():
        combined_train_rows.append({
            "image_id": row["image_id"],
            "dataset": row["dataset"],
            "relative_path": row["relative_path"],
            "label": row["label"],
            "class_index": int(row["class_index"]),
            "split": "train",
            "group_id": row["group_id"],
            "sha256": row["sha256"],
        })

    # Balanced transformation schedule
    # Sort train_df deterministically by image_id
    train_df_sorted = train_df.sort_values("image_id").reset_index(drop=True)

    t_start = datetime.now(timezone.utc)

    for idx, row in train_df_sorted.iterrows():
        parent_id = str(row["image_id"])
        parent_sha256 = str(row["sha256"])
        parent_rel_path = str(row["relative_path"])
        parent_label = str(row["label"])
        class_idx = int(row["class_index"])
        group_id = str(row["group_id"])

        parent_full_path = raw_root / parent_rel_path
        parent_img = load_parent_image(parent_full_path)
        h, w = parent_img.shape[:2]

        # Variant 1: Lighting Transform
        t1_name = lighting_transforms[idx % len(lighting_transforms)]
        seed1 = derive_output_seed(parent_id, 1, config.base_seed)
        out1, p1, stat1 = engine.execute(parent_img, t1_name, parent_id, 1, config.base_seed)

        # Save Var 1 losslessly as PNG
        gen_id_1 = f"aug_trad1_{parent_id[:24]}_{t1_name}"
        out_rel_path_1 = f"traditional_v1/{gen_id_1}.png"
        out_full_path_1 = aug_dir / f"{gen_id_1}.png"
        Image.fromarray(out1).save(out_full_path_1, format="PNG")
        out1_sha256 = compute_sha256(out_full_path_1)

        record_1 = {
            "generated_image_id": gen_id_1,
            "parent_image_id": parent_id,
            "parent_sha256": parent_sha256,
            "relative_output_path": out_rel_path_1,
            "parent_split": "train",
            "source_class": parent_label,
            "class_index": class_idx,
            "transform_category": "lighting",
            "transform_name": t1_name,
            "exact_parameters": json.dumps(p1),
            "seed": seed1,
            "transform_version": config.version,
            "output_sha256": out1_sha256,
            "dimensions": f"{w}x{h}",
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "technical_validation_status": stat1["status"],
            "human_review_status": "not_reviewed",
            "label_provenance": "inherited_under_label_preservation_assumption",
        }
        generated_rows.append(record_1)

        combined_train_rows.append({
            "image_id": gen_id_1,
            "dataset": "k-pipelines-traditional-v1",
            "relative_path": f"datasets/augmented/{out_rel_path_1}",
            "label": parent_label,
            "class_index": class_idx,
            "split": "train",
            "group_id": group_id,  # Shared group with parent to preserve isolation
            "sha256": out1_sha256,
        })

        # Variant 2: Quality Transform
        t2_name = quality_transforms[idx % len(quality_transforms)]
        seed2 = derive_output_seed(parent_id, 2, config.base_seed)
        out2, p2, stat2 = engine.execute(parent_img, t2_name, parent_id, 2, config.base_seed)

        gen_id_2 = f"aug_trad2_{parent_id[:24]}_{t2_name}"
        out_rel_path_2 = f"traditional_v1/{gen_id_2}.png"
        out_full_path_2 = aug_dir / f"{gen_id_2}.png"
        Image.fromarray(out2).save(out_full_path_2, format="PNG")
        out2_sha256 = compute_sha256(out_full_path_2)

        record_2 = {
            "generated_image_id": gen_id_2,
            "parent_image_id": parent_id,
            "parent_sha256": parent_sha256,
            "relative_output_path": out_rel_path_2,
            "parent_split": "train",
            "source_class": parent_label,
            "class_index": class_idx,
            "transform_category": "image_quality",
            "transform_name": t2_name,
            "exact_parameters": json.dumps(p2),
            "seed": seed2,
            "transform_version": config.version,
            "output_sha256": out2_sha256,
            "dimensions": f"{w}x{h}",
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "technical_validation_status": stat2["status"],
            "human_review_status": "not_reviewed",
            "label_provenance": "inherited_under_label_preservation_assumption",
        }
        generated_rows.append(record_2)

        combined_train_rows.append({
            "image_id": gen_id_2,
            "dataset": "k-pipelines-traditional-v1",
            "relative_path": f"datasets/augmented/{out_rel_path_2}",
            "label": parent_label,
            "class_index": class_idx,
            "split": "train",
            "group_id": group_id,  # Shared group with parent
            "sha256": out2_sha256,
        })

    # Save datasets/manifests/traditional_v1_generated.csv
    generated_csv = manifests_dir / "traditional_v1_generated.csv"
    pd.DataFrame(generated_rows).to_csv(generated_csv, index=False)

    # Save datasets/manifests/train_traditional_v1.csv
    combined_train_csv = manifests_dir / "train_traditional_v1.csv"
    pd.DataFrame(combined_train_rows).to_csv(combined_train_csv, index=False)

    # Breakdown statistics
    t_counts = pd.DataFrame(generated_rows)["transform_name"].value_counts().to_dict()
    class_counts = pd.DataFrame(generated_rows)["source_class"].value_counts().to_dict()

    summary = {
        "config_version": config.version,
        "base_seed": config.base_seed,
        "parent_training_images": len(train_df),
        "variations_generated": len(generated_rows),
        "total_augmented_training_pool": len(combined_train_rows),
        "failed_outputs": 0,
        "all_technical_validation_passed": all(r["technical_validation_status"] == "passed" for r in generated_rows),
        "human_review_status": "not_reviewed",
        "transform_distribution": t_counts,
        "class_distribution_generated": class_counts,
        "manifests_created": [
            str(generated_csv.as_posix()),
            str(combined_train_csv.as_posix()),
        ],
        "generation_time_utc_start": t_start.isoformat(),
        "generation_time_utc_end": datetime.now(timezone.utc).isoformat(),
    }

    with open(manifests_dir / "traditional_v1_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate traditional augmentations for training set.")
    parser.add_argument("--pilot", action="store_true", help="Run 12-parent pilot with contact sheet")
    parser.add_argument("--full", action="store_true", help="Run full 480-parent generation (960 variations)")
    args = parser.parse_args()

    train_manifest = PROJECT_ROOT / "datasets" / "manifests" / "train.csv"
    raw_root = PROJECT_ROOT / "datasets" / "raw" / "k-pipelines"
    aug_root = PROJECT_ROOT / "datasets" / "augmented"
    manifests_dir = PROJECT_ROOT / "datasets" / "manifests"
    pilots_dir = PROJECT_ROOT / "experiments" / "pilots" / "traditional_v1"

    train_df = pd.read_csv(train_manifest)
    config = TraditionalConfig()

    if args.pilot:
        print(f"Running traditional augmentation pilot on 12 parents...")
        res = run_pilot(train_df, raw_root, pilots_dir, config)
        print("Pilot completed successfully:")
        print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=2))

    if args.full:
        print(f"Running full traditional augmentation generation on {len(train_df)} parents...")
        res = run_full_generation(train_df, raw_root, aug_root, manifests_dir, config)
        print("Full generation completed successfully:")
        print(json.dumps(res, indent=2))
