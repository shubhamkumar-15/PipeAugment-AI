#!/usr/bin/env python3
"""
PipeAugment AI — Import and Validate Human Generative Review.
Validates exported human review JSON/CSV files against review_inventory.json,
verifies file hashes, enforces strict approval criteria, and generates
a versioned, signed manifest (reviewed_manifest_<version>.json).

Never overwrites original pilot files or metadata.
"""

import os
import sys
import json
import csv
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent

ALLOWED_CRITERIA_VALUES = {"yes", "no", "uncertain", "unreviewed"}
ALLOWED_DECISION_VALUES = {"pending", "approved", "rejected", "needs_review"}


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_inventory(inventory_path: Path) -> Dict[str, Any]:
    if not inventory_path.exists():
        raise FileNotFoundError(f"Review inventory not found: {inventory_path}")
    with open(inventory_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_review_file(file_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parses JSON or CSV exported review file and returns list of records and header metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Review file not found: {file_path}")

    ext = file_path.suffix.lower()
    meta = {}
    records = []

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            meta = {
                "schema_version": data.get("schema_version", "1.0"),
                "exported_at_utc": data.get("exported_at_utc"),
                "inventory_created_at": data.get("inventory_created_at"),
                "reviewer_default": data.get("reviewer_default", ""),
            }
            records = data.get("reviews", [])
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unexpected JSON format in {file_path}")

    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        raise ValueError(f"Unsupported review file format '{ext}'. Expected .json or .csv")

    return records, meta


def validate_and_process_reviews(
    raw_records: List[Dict[str, Any]],
    inventory: Dict[str, Any],
    project_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validates review records against inventory:
    - All output IDs exist in inventory
    - Zero duplicate IDs
    - File hashes match both recorded SHA256 and actual file on disk
    - Criteria values in allowed set
    - Strict approval criteria: all 4 criteria == 'yes' and reviewer non-empty
    """
    outputs_lookup = inventory.get("outputs_lookup", {})
    if not outputs_lookup:
        raise ValueError("Invalid inventory: missing 'outputs_lookup'")

    seen_ids = set()
    validated_records = []
    issues = []

    for idx, rec in enumerate(raw_records, 1):
        out_id = str(rec.get("output_id", "")).strip()
        if not out_id:
            raise ValueError(f"Record #{idx} missing required 'output_id'")

        if out_id in seen_ids:
            raise ValueError(f"Duplicate output_id '{out_id}' found in review file at record #{idx}")
        seen_ids.add(out_id)

        if out_id not in outputs_lookup:
            raise ValueError(f"Unknown output_id '{out_id}' in record #{idx}. Not found in review_inventory.json")

        inv_item = outputs_lookup[out_id]

        # Verify parent ID
        parent_id = str(rec.get("parent_id", "")).strip()
        if parent_id and parent_id != inv_item["parent_id"]:
            raise ValueError(f"Parent ID mismatch for {out_id}: review={parent_id} vs inventory={inv_item['parent_id']}")

        # Verify on-disk image file hash
        rel_path = inv_item["relative_image_path"]
        actual_img_path = project_root / rel_path
        if not actual_img_path.exists():
            raise FileNotFoundError(f"Underlying output image missing on disk: {actual_img_path}")

        disk_sha256 = compute_file_sha256(actual_img_path)
        expected_sha256 = inv_item["output_sha256"]
        if disk_sha256 != expected_sha256:
            raise ValueError(
                f"Integrity check FAILED for {out_id}: Disk SHA-256 ({disk_sha256}) "
                f"does not match recorded inventory SHA-256 ({expected_sha256}). "
                f"Refusing to approve altered files."
            )

        # Also check exported sha if present
        rec_sha = str(rec.get("output_sha256", "")).strip()
        if rec_sha and rec_sha != expected_sha256:
            raise ValueError(
                f"Integrity check FAILED for {out_id}: Exported SHA-256 ({rec_sha}) "
                f"does not match inventory SHA-256 ({expected_sha256})."
            )

        # Validate review fields
        geom = str(rec.get("geometry_preserved", "unreviewed")).strip().lower()
        corr = str(rec.get("corrosion_preserved", "unreviewed")).strip().lower()
        cond = str(rec.get("condition_visible", "unreviewed")).strip().lower()
        real = str(rec.get("realism_acceptable", "unreviewed")).strip().lower()
        dec = str(rec.get("decision", "pending")).strip().lower()
        reviewer = str(rec.get("reviewer", "")).strip()
        notes = str(rec.get("review_notes", "")).strip()
        timestamp = rec.get("review_timestamp")

        for k, v in [("geometry_preserved", geom), ("corrosion_preserved", corr),
                     ("condition_visible", cond), ("realism_acceptable", real)]:
            if v not in ALLOWED_CRITERIA_VALUES:
                raise ValueError(f"Invalid value '{v}' for {k} in {out_id}. Must be one of {ALLOWED_CRITERIA_VALUES}")

        if dec not in ALLOWED_DECISION_VALUES:
            raise ValueError(f"Invalid decision '{dec}' for {out_id}. Must be one of {ALLOWED_DECISION_VALUES}")

        # Enforce strict approval rule
        if dec == "approved":
            missing_criteria = []
            if geom != "yes": missing_criteria.append(f"geometry_preserved='{geom}' (must be 'yes')")
            if corr != "yes": missing_criteria.append(f"corrosion_preserved='{corr}' (must be 'yes')")
            if cond != "yes": missing_criteria.append(f"condition_visible='{cond}' (must be 'yes')")
            if real != "yes": missing_criteria.append(f"realism_acceptable='{real}' (must be 'yes')")
            if not reviewer: missing_criteria.append("reviewer name is empty")

            if missing_criteria:
                raise ValueError(
                    f"Approval validation FAILED for {out_id}:\n"
                    f"  An 'approved' decision requires all 4 criteria to be 'yes' and reviewer name present.\n"
                    f"  Violations: {', '.join(missing_criteria)}"
                )

        validated_rec = {
            "output_id": out_id,
            "parent_id": inv_item["parent_id"],
            "condition": inv_item["condition"],
            "method": inv_item["method"],
            "strength": inv_item["strength"],
            "control_scale": inv_item["control_scale"],
            "relative_image_path": rel_path,
            "output_sha256": expected_sha256,
            "geometry_preserved": geom,
            "corrosion_preserved": corr,
            "condition_visible": cond,
            "realism_acceptable": real,
            "decision": dec,
            "reviewer": reviewer,
            "review_notes": notes,
            "review_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "ssim_rgb": inv_item.get("ssim_rgb"),
            "edge_overlap_iou_heuristic": inv_item.get("edge_overlap_iou_heuristic"),
        }
        validated_records.append(validated_rec)

    # Verify coverage
    all_inv_ids = set(outputs_lookup.keys())
    missing_from_review = all_inv_ids - seen_ids
    if missing_from_review:
        # Not all 48 were in the file: fill in remaining as pending unreviewed
        for m_id in sorted(missing_from_review):
            inv_item = outputs_lookup[m_id]
            validated_records.append({
                "output_id": m_id,
                "parent_id": inv_item["parent_id"],
                "condition": inv_item["condition"],
                "method": inv_item["method"],
                "strength": inv_item["strength"],
                "control_scale": inv_item["control_scale"],
                "relative_image_path": inv_item["relative_image_path"],
                "output_sha256": inv_item["output_sha256"],
                "geometry_preserved": "unreviewed",
                "corrosion_preserved": "unreviewed",
                "condition_visible": "unreviewed",
                "realism_acceptable": "unreviewed",
                "decision": "pending",
                "reviewer": "",
                "review_notes": "",
                "review_timestamp": None,
                "ssim_rgb": inv_item.get("ssim_rgb"),
                "edge_overlap_iou_heuristic": inv_item.get("edge_overlap_iou_heuristic"),
            })

    stats = {
        "total_records": len(validated_records),
        "approved": sum(1 for r in validated_records if r["decision"] == "approved"),
        "rejected": sum(1 for r in validated_records if r["decision"] == "rejected"),
        "needs_review": sum(1 for r in validated_records if r["decision"] == "needs_review"),
        "pending": sum(1 for r in validated_records if r["decision"] == "pending"),
    }

    return validated_records, stats


def generate_review_summary(records: List[Dict[str, Any]], inventory: Dict[str, Any]) -> str:
    """
    Constructs a thorough human-readable summary of the review status.
    Respects scientific limitations: reports pending if no reviews,
    does not declare winners based on SSIM, reports parent count.
    """
    total = len(records)
    approved = sum(1 for r in records if r["decision"] == "approved")
    rejected = sum(1 for r in records if r["decision"] == "rejected")
    needs_review = sum(1 for r in records if r["decision"] == "needs_review")
    pending = sum(1 for r in records if r["decision"] == "pending")

    lines = []
    lines.append("=" * 72)
    lines.append("PIPEAUGMENT AI — HUMAN GENERATIVE REVIEW SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Evaluation Scope: 48 generated candidate variations across 6 training parents")
    lines.append(f"Parent Distribution: 3 non-corroded parents, 3 corroded parents")
    lines.append(f"Total Evaluated Records: {total}")
    lines.append("-" * 72)
    lines.append(f"Overall Decision Status:")
    lines.append(f"  • Approved:     {approved:>2} / {total} ({approved/total*100:.1f}%)")
    lines.append(f"  • Rejected:     {rejected:>2} / {total} ({rejected/total*100:.1f}%)")
    lines.append(f"  • Needs Review: {needs_review:>2} / {total} ({needs_review/total*100:.1f}%)")
    lines.append(f"  • Pending:      {pending:>2} / {total} ({pending/total*100:.1f}%)")
    lines.append("-" * 72)

    if approved == 0 and rejected == 0 and needs_review == 0:
        lines.append("HUMAN REVIEW STATUS: PENDING")
        lines.append("All 48 candidate variations remain strictly unreviewed.")
        lines.append("No generative method or configuration is recommended for dataset expansion.")
        lines.append("Experiment C (classifier training) remains BLOCKED until human review is completed.")
        lines.append("=" * 72)
        return "\n".join(lines)

    # Breakdowns
    lines.append("Breakdown by Augmentation Method:")
    methods = [
        ("Plain SD 1.5 (strength 0.20)", lambda r: r["method"] == "plain_img2img" and r["strength"] == 0.20),
        ("Plain SD 1.5 (strength 0.35)", lambda r: r["method"] == "plain_img2img" and r["strength"] == 0.35),
        ("ControlNet Canny (scale 0.50, s=0.35)", lambda r: r["method"] == "controlnet_canny" and r["control_scale"] == 0.50),
        ("ControlNet Canny (scale 0.80, s=0.35)", lambda r: r["method"] == "controlnet_canny" and r["control_scale"] == 0.80),
    ]
    for label, fn in methods:
        sub = [r for r in records if fn(r)]
        app = sum(1 for r in sub if r["decision"] == "approved")
        rej = sum(1 for r in sub if r["decision"] == "rejected")
        n = len(sub)
        lines.append(f"  • {label:<38} Approved: {app:>2}/{n} | Rejected: {rej:>2}/{n}")

    lines.append("\nBreakdown by Environmental Lighting Condition:")
    for cond, label in [("low_light_inspection", "Low-Light Inspection"), ("directional_industrial_lighting", "Directional Lighting")]:
        sub = [r for r in records if r["condition"] == cond]
        app = sum(1 for r in sub if r["decision"] == "approved")
        rej = sum(1 for r in sub if r["decision"] == "rejected")
        lines.append(f"  • {label:<38} Approved: {app:>2}/{len(sub)} | Rejected: {rej:>2}/{len(sub)}")

    lines.append("\nBreakdown by Source Parent Class (Inherited Label):")
    # map parent classes
    sources_map = {g["parent_id"]: g["parent_label"] for g in inventory.get("groups", [])}
    for cls in ["non_corroded", "corroded"]:
        sub = [r for r in records if sources_map.get(r["parent_id"]) == cls]
        app = sum(1 for r in sub if r["decision"] == "approved")
        rej = sum(1 for r in sub if r["decision"] == "rejected")
        lines.append(f"  • Class: {cls:<31} Approved: {app:>2}/{len(sub)} | Rejected: {rej:>2}/{len(sub)}")

    # Rejection reasons
    rejections = [r for r in records if r["decision"] == "rejected"]
    if rejections:
        lines.append("\nStructured Failure Modes for Rejected Outputs:")
        f_geom = sum(1 for r in rejections if r.get("geometry_preserved") == "no")
        f_corr = sum(1 for r in rejections if r.get("corrosion_preserved") == "no")
        f_cond = sum(1 for r in rejections if r.get("condition_visible") == "no")
        f_real = sum(1 for r in rejections if r.get("realism_acceptable") == "no")
        lines.append(f"  • Pipe Geometry Distorted/Altered: {f_geom:>2} / {len(rejections)}")
        lines.append(f"  • Corrosion Hallucinated or Erased:  {f_corr:>2} / {len(rejections)}")
        lines.append(f"  • Condition Insufficiently Visible:  {f_cond:>2} / {len(rejections)}")
        lines.append(f"  • Visual Artifacts / Low Realism:   {f_real:>2} / {len(rejections)}")

    lines.append("-" * 72)
    lines.append("METHODOLOGICAL NOTICE:")
    lines.append("- Visual inspection was performed on a small sample of 6 training parents.")
    lines.append("- Higher approval in one method does NOT imply statistical significance.")
    lines.append("- Quantitative metrics (SSIM, Edge IoU) are descriptive heuristics, NOT proof of defect retention.")
    lines.append("=" * 72)
    return "\n".join(lines)


def import_review(
    input_file: Path,
    inventory_path: Path,
    output_dir: Path,
    version: str = "v1",
    print_summary: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for importing and certifying a human review file.
    """
    input_sha256 = compute_file_sha256(input_file)
    inventory = load_inventory(inventory_path)

    raw_records, file_meta = parse_review_file(input_file)
    print(f"Loaded {len(raw_records)} records from {input_file.name} (SHA-256: {input_sha256[:16]}...)")

    # Validate
    validated_records, stats = validate_and_process_reviews(raw_records, inventory, PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = output_dir / f"reviewed_manifest_{version}.json"
    out_csv_path = output_dir / f"reviewed_manifest_{version}.csv"

    approved_outputs = [r for r in validated_records if r["decision"] == "approved"]

    manifest = {
        "manifest_version": f"reviewed_{version}",
        "imported_from": str(input_file.relative_to(PROJECT_ROOT).as_posix()) if input_file.is_relative_to(PROJECT_ROOT) else str(input_file),
        "imported_file_sha256": input_sha256,
        "import_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_version": inventory.get("inventory_version", "1.0"),
        "total_records": len(validated_records),
        "stats": stats,
        "approved_count": len(approved_outputs),
        "approved_outputs": [r["output_id"] for r in approved_outputs],
        "training_eligibility": bool(len(approved_outputs) > 0),  # Only approved outputs can ever be training-eligible
        "records": validated_records,
    }

    # Write manifest JSON
    with open(out_json_path, "w", encoding="utf-8") as jf:
        json.dump(manifest, jf, indent=2)

    # Write manifest CSV
    fieldnames = [
        "output_id", "parent_id", "condition", "method", "strength", "control_scale",
        "relative_image_path", "output_sha256", "geometry_preserved", "corrosion_preserved",
        "condition_visible", "realism_acceptable", "decision", "reviewer", "review_notes",
        "review_timestamp", "ssim_rgb", "edge_overlap_iou_heuristic"
    ]
    with open(out_csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in validated_records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Validated {len(validated_records)} records: {stats['approved']} Approved, {stats['rejected']} Rejected, {stats['needs_review']} Needs Review, {stats['pending']} Pending.")
    print(f"Saved reviewed manifest: {out_json_path.name}")
    print(f"Saved reviewed CSV:      {out_csv_path.name}")

    summary_text = generate_review_summary(validated_records, inventory)
    if print_summary:
        print("\n" + summary_text)

    # Also save summary text alongside manifest
    summary_txt_path = output_dir / f"reviewed_summary_{version}.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as sf:
        sf.write(summary_text)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Import and validate human review file for generative pilot.")
    parser.add_argument("--input", type=str, required=True, help="Path to exported review JSON or CSV")
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1" / "review_inventory.json"),
        help="Path to review_inventory.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1"),
        help="Output directory for versioned reviewed manifests",
    )
    parser.add_argument("--version", type=str, default="v1", help="Version identifier (e.g. v1)")
    parser.add_argument("--summary", action="store_true", help="Print summary report")
    args = parser.parse_args()

    import_review(
        input_file=Path(args.input),
        inventory_path=Path(args.inventory),
        output_dir=Path(args.output_dir),
        version=args.version,
        print_summary=args.summary,
    )


if __name__ == "__main__":
    main()
