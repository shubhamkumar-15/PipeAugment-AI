#!/usr/bin/env python3
"""
Manifest Preparation Script for K-Pipelines Dataset.
Generates reproducible, leakage-free train.csv, val.csv, and test.csv manifests
with class mappings and split metadata.
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Explicit label mapping from source directory names to standard project labels
LABEL_MAPPING = {
    "no_corrosion": "non_corroded",
    "corrosion": "corroded",
}

CLASS_TO_INDEX = {
    "non_corroded": 0,
    "corroded": 1,
}

INDEX_TO_CLASS = {
    0: "non_corroded",
    1: "corroded",
}


def prepare_manifests(
    inventory_csv_path: Path,
    manifests_dir: Path,
    seed: int = 42
) -> Dict[str, Any]:
    if not inventory_csv_path.exists():
        raise FileNotFoundError(f"Inventory file not found at {inventory_csv_path}. Run audit_dataset.py first.")

    manifests_dir.mkdir(parents=True, exist_ok=True)

    with open(inventory_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        raise ValueError("Inventory file is empty.")

    # Group by SHA-256 to assign group_ids
    sha_to_group: Dict[str, str] = {}
    group_counter = 1
    for r in records:
        sha = r["sha256"]
        if sha not in sha_to_group:
            sha_to_group[sha] = f"grp_{group_counter:04d}"
            group_counter += 1

    # Check for cross-split duplicates
    sha_splits: Dict[str, set] = defaultdict(set)
    for r in records:
        split_name = "val" if r["source_split"] == "valid" else r["source_split"]
        sha_splits[r["sha256"]].add(split_name)

    cross_split_shas = {sha: s for sha, s in sha_splits.items() if len(s) > 1}
    if cross_split_shas:
        raise RuntimeError(
            f"Detected {len(cross_split_shas)} cross-split duplicate SHA-256 hashes! "
            f"Published splits contain leakage and cannot be used without re-splitting."
        )

    # Build split manifests
    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": []
    }

    for r in records:
        source_split = r["source_split"]
        target_split = "val" if source_split == "valid" else source_split
        if target_split not in split_records:
            raise ValueError(f"Unknown source split '{source_split}' in record: {r}")

        source_class = r["source_class"]
        if source_class not in LABEL_MAPPING:
            raise ValueError(f"Unknown source class '{source_class}'. Expected one of {list(LABEL_MAPPING.keys())}")

        label = LABEL_MAPPING[source_class]
        class_idx = CLASS_TO_INDEX[label]
        group_id = sha_to_group[r["sha256"]]

        manifest_row = {
            "image_id": r["image_id"],
            "dataset": "k-pipelines-standard",
            "relative_path": r["relative_path"],
            "label": label,
            "class_index": class_idx,
            "split": target_split,
            "group_id": group_id,
            "sha256": r["sha256"],
        }
        split_records[target_split].append(manifest_row)

    manifest_fieldnames = [
        "image_id", "dataset", "relative_path", "label",
        "class_index", "split", "group_id", "sha256"
    ]

    # Save CSVs for each split
    split_counts: Dict[str, int] = {}
    split_class_breakdown: Dict[str, Dict[str, int]] = {}

    for split_name, rows in split_records.items():
        csv_path = manifests_dir / f"{split_name}.csv"
        # Sort deterministically by image_id
        rows = sorted(rows, key=lambda x: x["image_id"])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        split_counts[split_name] = len(rows)
        class_dist: Dict[str, int] = defaultdict(int)
        for row in rows:
            class_dist[row["label"]] += 1
        split_class_breakdown[split_name] = dict(class_dist)

    # Save class_mapping.json
    class_mapping_data = {
        "class_to_index": CLASS_TO_INDEX,
        "index_to_class": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        "source_mapping": LABEL_MAPPING,
    }
    with open(manifests_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping_data, f, indent=2)

    # Save split_metadata.json
    split_metadata = {
        "dataset_name": "k-pipelines-standard",
        "split_policy": "Preserved published train/valid/test splits (verified 0 cross-split duplicates / leakage-free)",
        "split_seed": seed,
        "total_images": len(records),
        "split_counts": split_counts,
        "class_distribution": split_class_breakdown,
        "classes": list(CLASS_TO_INDEX.keys()),
        "output_manifests": {
            "train": (manifests_dir / "train.csv").as_posix(),
            "val": (manifests_dir / "val.csv").as_posix(),
            "test": (manifests_dir / "test.csv").as_posix(),
        }
    }
    with open(manifests_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2)

    return split_metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    manifests_dir = project_root / "datasets" / "manifests"
    inventory_csv = manifests_dir / "k_pipelines_inventory.csv"

    print(f"Preparing reproducible manifests from {inventory_csv}...")
    metadata = prepare_manifests(inventory_csv, manifests_dir, seed=42)
    print("Manifest preparation successful:")
    print(json.dumps(metadata, indent=2))
