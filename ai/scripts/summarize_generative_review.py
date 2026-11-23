#!/usr/bin/env python3
"""
PipeAugment AI — Summarize Human Generative Review Status.
Inspects review inventory or imported review manifest and displays structured summary:
- Total outputs, approval / rejection / pending counts
- Approval rates by method, condition, and source class
- Rejection failure modes
- Honest scientific notices
"""

import sys
import json
import argparse
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from scripts.import_generative_review import generate_review_summary, load_inventory


def main():
    parser = argparse.ArgumentParser(description="Summarize human generative review status.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to reviewed_manifest_v1.json (if omitted, checks review_inventory.json)",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1" / "review_inventory.json"),
        help="Path to review_inventory.json",
    )
    args = parser.parse_args()

    inv_path = Path(args.inventory)
    inventory = load_inventory(inv_path)

    manifest_path = Path(args.manifest) if args.manifest else (inv_path.parent / "reviewed_manifest_v1.json")

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        records = manifest_data.get("records", [])
        print(f"[Loaded reviewed manifest: {manifest_path.name}]")
    else:
        # Generate initial pending status directly from inventory
        records = []
        for g in inventory.get("groups", []):
            for out in g.get("outputs", []):
                rec = dict(out)
                st = out.get("review_state", {})
                rec.update(st)
                records.append(rec)
        print(f"[No reviewed manifest found; reporting baseline inventory status]")

    summary_text = generate_review_summary(records, inventory)
    print(summary_text)


if __name__ == "__main__":
    main()
