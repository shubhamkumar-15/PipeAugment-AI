#!/usr/bin/env python3
"""
Dataset Audit Script for K-Pipelines Dataset.
Audits image integrity, dimensions, formats, exact SHA-256 duplicates,
cross-split duplicates, label conflicts, and perceptual hash near-duplicate candidates.
"""

import os
import sys
import hashlib
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

from PIL import Image
import imagehash


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
    docs_dir: Path,
    phash_threshold: int = 4
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    standard_root = dataset_root / "Standard"
    if not standard_root.exists():
        raise FileNotFoundError(f"Expected Standard dataset folder at {standard_root}")

    inventory: List[Dict[str, Any]] = []
    corrupt_files: List[Dict[str, Any]] = []
    sha_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    phash_list: List[Tuple[str, Any, Dict[str, Any]]] = []

    # Traverse Standard dataset splits
    splits_found = sorted([d.name for d in standard_root.iterdir() if d.is_dir()])
    
    for split in splits_found:
        split_dir = standard_root / split
        class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        for class_name in class_dirs:
            target_dir = split_dir / class_name
            for img_path in sorted(target_dir.iterdir()):
                if img_path.is_file() and not img_path.name.startswith("."):
                    rel_path = img_path.relative_to(dataset_root).as_posix()
                    file_size = img_path.stat().st_size
                    try:
                        sha256 = compute_sha256(img_path)
                        with Image.open(img_path) as img:
                            img.verify()
                        # Re-open for mode/size and phash computation
                        with Image.open(img_path) as img:
                            fmt = img.format
                            width, height = img.size
                            mode = img.mode
                            phash_val = str(imagehash.phash(img))
                            phash_obj = imagehash.hex_to_hash(phash_val)
                        
                        record = {
                            "image_id": img_path.stem,
                            "filename": img_path.name,
                            "relative_path": rel_path,
                            "source_split": split,
                            "source_class": class_name,
                            "format": fmt,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "file_size_bytes": file_size,
                            "sha256": sha256,
                            "phash": phash_val,
                        }
                        inventory.append(record)
                        sha_map[sha256].append(record)
                        phash_list.append((rel_path, phash_obj, record))
                    except Exception as exc:
                        corrupt_files.append({
                            "relative_path": rel_path,
                            "source_split": split,
                            "source_class": class_name,
                            "error": str(exc),
                        })

    # Exact duplicate analysis
    exact_duplicates_groups = [records for records in sha_map.values() if len(records) > 1]
    cross_split_duplicates: List[Dict[str, Any]] = []
    label_conflicts: List[Dict[str, Any]] = []

    duplicate_candidates_rows: List[Dict[str, Any]] = []
    dup_group_id = 1

    for records in exact_duplicates_groups:
        splits = {r["source_split"] for r in records}
        labels = {r["source_class"] for r in records}
        is_cross_split = len(splits) > 1
        is_label_conflict = len(labels) > 1

        if is_cross_split:
            cross_split_duplicates.append({
                "sha256": records[0]["sha256"],
                "records": records,
            })
        if is_label_conflict:
            label_conflicts.append({
                "sha256": records[0]["sha256"],
                "records": records,
            })

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                duplicate_candidates_rows.append({
                    "group_id": f"exact_{dup_group_id}",
                    "candidate_type": "exact_duplicate",
                    "image_1": records[i]["relative_path"],
                    "image_2": records[j]["relative_path"],
                    "sha256_1": records[i]["sha256"],
                    "sha256_2": records[j]["sha256"],
                    "phash_distance": 0,
                    "split_1": records[i]["source_split"],
                    "split_2": records[j]["source_split"],
                    "class_1": records[i]["source_class"],
                    "class_2": records[j]["source_class"],
                    "same_label": records[i]["source_class"] == records[j]["source_class"],
                })
        dup_group_id += 1

    # Near-duplicate analysis via perceptual hash
    near_dup_group_id = 1
    near_duplicate_count = 0
    n = len(phash_list)
    for i in range(n):
        path1, phash1, rec1 = phash_list[i]
        for j in range(i + 1, n):
            path2, phash2, rec2 = phash_list[j]
            # Skip if they are already exact duplicates
            if rec1["sha256"] == rec2["sha256"]:
                continue
            distance = phash1 - phash2
            if distance <= phash_threshold:
                near_duplicate_count += 1
                duplicate_candidates_rows.append({
                    "group_id": f"near_{near_dup_group_id}",
                    "candidate_type": "near_duplicate_candidate",
                    "image_1": path1,
                    "image_2": path2,
                    "sha256_1": rec1["sha256"],
                    "sha256_2": rec2["sha256"],
                    "phash_distance": int(distance),
                    "split_1": rec1["source_split"],
                    "split_2": rec2["source_split"],
                    "class_1": rec1["source_class"],
                    "class_2": rec2["source_class"],
                    "same_label": rec1["source_class"] == rec2["source_class"],
                })
                near_dup_group_id += 1

    # Write inventory CSV
    inventory_csv = output_dir / "k_pipelines_inventory.csv"
    with open(inventory_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "filename", "relative_path", "source_split",
            "source_class", "format", "width", "height", "mode",
            "file_size_bytes", "sha256", "phash"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    # Write duplicate candidates CSV
    candidates_csv = output_dir / "k_pipelines_duplicate_candidates.csv"
    with open(candidates_csv, "w", newline="", encoding="utf-8") as f:
        cand_fieldnames = [
            "group_id", "candidate_type", "image_1", "image_2",
            "sha256_1", "sha256_2", "phash_distance", "split_1",
            "split_2", "class_1", "class_2", "same_label"
        ]
        writer = csv.DictWriter(f, fieldnames=cand_fieldnames)
        writer.writeheader()
        writer.writerows(duplicate_candidates_rows)

    # Compute breakdown statistics
    split_counts: Dict[str, int] = defaultdict(int)
    class_counts: Dict[str, int] = defaultdict(int)
    split_class_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    formats: Dict[str, int] = defaultdict(int)
    dimensions: Dict[str, int] = defaultdict(int)

    for r in inventory:
        split_counts[r["source_split"]] += 1
        class_counts[r["source_class"]] += 1
        split_class_counts[r["source_split"]][r["source_class"]] += 1
        formats[r["format"]] += 1
        dim_str = f"{r['width']}x{r['height']}"
        dimensions[dim_str] += 1

    # Write Markdown audit report
    audit_report_md = docs_dir / "k-pipelines-dataset-audit.md"
    with open(audit_report_md, "w", encoding="utf-8") as f:
        f.write("# K-Pipelines Dataset Audit Report\n\n")
        f.write("## Overview\n")
        f.write("- **Dataset Name:** K-Pipelines (Standard Version)\n")
        f.write(f"- **Total Readable Images:** {len(inventory)}\n")
        f.write(f"- **Corrupt Images:** {len(corrupt_files)}\n")
        f.write(f"- **Unique SHA-256 Checksums:** {len(sha_map)}\n")
        f.write(f"- **Exact Duplicate Groups:** {len(exact_duplicates_groups)}\n")
        f.write(f"- **Cross-Split Exact Duplicates:** {len(cross_split_duplicates)}\n")
        f.write(f"- **Label Conflicts (Exact):** {len(label_conflicts)}\n")
        f.write(f"- **Near-Duplicate Candidates (pHash <= {phash_threshold}):** {near_duplicate_count}\n\n")

        f.write("## Split and Class Distribution\n\n")
        f.write("| Split | Class | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for s in sorted(split_class_counts.keys()):
            for c in sorted(split_class_counts[s].keys()):
                cnt = split_class_counts[s][c]
                pct = (cnt / len(inventory)) * 100 if inventory else 0
                f.write(f"| `{s}` | `{c}` | {cnt} | {pct:.1f}% |\n")
        f.write(f"| **Total** | -- | **{len(inventory)}** | **100%** |\n\n")

        f.write("## Formats and Dimensions\n\n")
        f.write("- **Image Formats:** " + ", ".join([f"{k}: {v}" for k, v in formats.items()]) + "\n")
        f.write("- **Dimensions:** " + ", ".join([f"{k}: {v}" for k, v in dimensions.items()]) + "\n\n")

        f.write("## Duplicate Analysis\n\n")
        if exact_duplicates_groups:
            f.write(f"### Exact Duplicate Groups ({len(exact_duplicates_groups)})\n")
            for i, group in enumerate(exact_duplicates_groups, 1):
                f.write(f"- Group {i} (SHA-256 `{group[0]['sha256'][:12]}`):\n")
                for item in group:
                    f.write(f"  - `{item['relative_path']}` (split: `{item['source_split']}`, class: `{item['source_class']}`)\n")
            f.write("\n")
        else:
            f.write("No exact duplicates (SHA-256 identical) found.\n\n")

        f.write(f"### Near-Duplicate Candidates ({near_duplicate_count} pairs found with pHash distance <= {phash_threshold})\n")
        f.write("Candidate pairs are recorded in `datasets/manifests/k_pipelines_duplicate_candidates.csv`.\n")
        f.write("*Note:* As mandated by audit policy, perceptual hash candidates are flagged as potential visual similarities rather than confirmed duplicates, and raw images remain strictly immutable.\n\n")

        f.write("## Task Suitability\n\n")
        f.write("- **Image Classification:** Supported out of the box. Image labels correspond to directory structure (`corrosion`, `no_corrosion`).\n")
        f.write("- **Bounding Box Object Detection:** Not directly annotated in the source repository (no YOLO / COCO / Pascal VOC XML bounding box files are provided).\n")
        f.write("- **Pixel-Level Segmentation:** Not annotated (no binary or polygon masks provided in Standard release).\n\n")

        f.write("## Provenance and Licensing\n\n")
        f.write("- **License:** GNU General Public License v3.0 (GPL-3.0)\n")
        f.write("- **Citation:** Ramos, Casas, and Rivas-Echeverría (2025), *Intelligent Systems with Applications*, Vol. 25, 200463.\n")
        f.write("- **Downstream Notice:** Dataset images were generated via Stable Diffusion for pipeline research. Redistribution or commercial deployment requires legal review under GPL v3.0.\n")

    summary = {
        "total_images": len(inventory),
        "corrupt_images": len(corrupt_files),
        "splits": split_counts,
        "classes": class_counts,
        "split_class_breakdown": split_class_counts,
        "formats": formats,
        "dimensions": dimensions,
        "exact_duplicate_groups": len(exact_duplicates_groups),
        "cross_split_duplicates": len(cross_split_duplicates),
        "label_conflicts": len(label_conflicts),
        "near_duplicate_candidates": near_duplicate_count,
    }

    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = project_root / "datasets" / "raw" / "k-pipelines"
    manifests_dir = project_root / "datasets" / "manifests"
    docs_dir = project_root / "docs"

    print(f"Auditing K-Pipelines dataset at {dataset_root}...")
    summary = audit_dataset(dataset_root, manifests_dir, docs_dir)
    print("Audit completed successfully.")
    print(json.dumps(summary, indent=2))
