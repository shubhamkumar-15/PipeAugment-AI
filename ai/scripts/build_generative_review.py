#!/usr/bin/env python3
"""
PipeAugment AI — Build Generative Offline Review Utility.
Constructs a self-contained offline HTML review interface and inventory JSON
for the completed plain-diffusion (diffusion_v1) and Canny-ControlNet (controlnet_canny_v1) pilots.

Enhanced for Phase 4:
- Ingests Phase 4 Quality Validation screening metrics (SSIM, LPIPS, Edge IoU, Rust Stability).
- Adds quick filter buttons (Method, Class, Status) for immediate candidate triage.
- Embeds automated screening badges on all candidate cards and high-res modal.
- Retains compatibility with prior review exports.

Output:
experiments/reviews/generative_pilot_v1/
  index.html
  review_inventory.json
  README.md
"""

import os
import sys
import json
import base64
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_as_data_uri(filepath: Path) -> str:
    """Reads image file and returns base64 data URI."""
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    ext = filepath.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_inventory(
    project_root: Path,
    pilots_dir: Path,
    embed_base64: bool = True
) -> Dict[str, Any]:
    """
    Builds the complete review inventory matching all 48 outputs across the 6 training parents.
    Ingests Phase 4 Quality Validation results and previous review exports if available.
    """
    diff_dir = pilots_dir / "diffusion_v1"
    cnet_dir = pilots_dir / "controlnet_canny_v1"

    sources_path = diff_dir / "sources.csv"
    if not sources_path.exists():
        raise FileNotFoundError(f"Sources manifest not found: {sources_path}")

    sources_df = pd.read_csv(sources_path)

    # Base model metadata
    diff_summary_path = diff_dir / "summary.json"
    cnet_summary_path = cnet_dir / "summary.json"
    diff_sum = json.load(open(diff_summary_path, encoding="utf-8")) if diff_summary_path.exists() else {}
    cnet_sum = json.load(open(cnet_summary_path, encoding="utf-8")) if cnet_summary_path.exists() else {}

    # Load Phase 4 quality validation metrics if available
    quality_csv_path = project_root / "experiments" / "evaluations" / "pilot_quality_v1" / "quality_results.csv"
    quality_lookup: Dict[str, Dict[str, Any]] = {}
    if quality_csv_path.exists():
        try:
            q_df = pd.read_csv(quality_csv_path)
            for _, q_row in q_df.iterrows():
                q_id = str(q_row["output_id"]).strip()
                quality_lookup[q_id] = {
                    "quality_status": str(q_row.get("quality_status", "unscreened")),
                    "flags": str(q_row.get("flags", "")) if pd.notna(q_row.get("flags")) else "",
                    "ssim_rgb": float(q_row["ssim_rgb"]) if pd.notna(q_row.get("ssim_rgb")) else None,
                    "lpips": float(q_row["lpips"]) if pd.notna(q_row.get("lpips")) else None,
                    "edge_overlap_iou": float(q_row["edge_overlap_iou"]) if pd.notna(q_row.get("edge_overlap_iou")) else None,
                    "edge_overlap_dice": float(q_row["edge_overlap_dice"]) if pd.notna(q_row.get("edge_overlap_dice")) else None,
                    "rust_area_delta": float(q_row["rust_area_delta"]) if pd.notna(q_row.get("rust_area_delta")) else None,
                }
            print(f"Loaded Phase 4 quality metrics for {len(quality_lookup)} outputs.")
        except Exception as e:
            print(f"Warning: Failed to load quality_results.csv: {e}")

    # Load existing review states from prior export or reviewed manifest v2
    reviews_dir = project_root / "experiments" / "reviews" / "generative_pilot_v1"
    existing_decisions: Dict[str, Dict[str, Any]] = {}
    default_reviewer = ""
    export_files = sorted(reviews_dir.glob("generative_review_export_*.json"))
    manifest_v2 = reviews_dir / "reviewed_manifest_v2.json"
    source_to_load = export_files[-1] if export_files else (manifest_v2 if manifest_v2.exists() else None)

    if source_to_load and source_to_load.exists():
        try:
            with open(source_to_load, "r", encoding="utf-8") as rf:
                r_data = json.load(rf)
            recs = r_data.get("reviews") or r_data.get("records") or []
            default_reviewer = r_data.get("reviewer_default") or r_data.get("reviewer") or ""
            for r in recs:
                if isinstance(r, dict) and "output_id" in r:
                    existing_decisions[r["output_id"]] = r
            print(f"Pre-loaded {len(existing_decisions)} prior reviews from {source_to_load.name} (Reviewer: '{default_reviewer}')")
        except Exception as e:
            print(f"Warning: Failed to pre-load reviews: {e}")

    groups = []
    outputs_lookup = {}
    group_idx = 1

    conditions = ["low_light_inspection", "directional_industrial_lighting"]

    for _, p_row in sources_df.iterrows():
        parent_id = str(p_row["image_id"])
        prefix = parent_id[:20]
        parent_label = str(p_row["label"])
        parent_rel_path = str(p_row["relative_path"])
        parent_full_path = project_root / "datasets" / "raw" / "k-pipelines" / parent_rel_path
        parent_sha256 = str(p_row["sha256"])

        # Canny edge map
        canny_rel_path = f"experiments/pilots/controlnet_canny_v1/control_maps/{prefix}_canny.png"
        canny_full_path = project_root / canny_rel_path
        canny_sha256 = compute_file_sha256(canny_full_path) if canny_full_path.exists() else ""

        # Base64 URIs for parent & canny
        parent_uri = load_image_as_data_uri(parent_full_path) if (embed_base64 and parent_full_path.exists()) else ""
        canny_uri = load_image_as_data_uri(canny_full_path) if (embed_base64 and canny_full_path.exists()) else ""

        for cond in conditions:
            cond_display = "Low-Light Inspection" if "low_light" in cond else "Directional Industrial Lighting"
            group_id = f"group_{group_idx:02d}_{prefix[:12]}_{cond}"

            # 4 candidates
            candidate_specs = [
                {
                    "method": "plain_img2img",
                    "strength": 0.20,
                    "control_scale": None,
                    "output_id": f"gen_diff_{prefix}_{cond}_s20",
                    "pilot_sub": "diffusion_v1",
                    "method_label": "Plain SD 1.5 (strength 0.20)",
                },
                {
                    "method": "plain_img2img",
                    "strength": 0.35,
                    "control_scale": None,
                    "output_id": f"gen_diff_{prefix}_{cond}_s35",
                    "pilot_sub": "diffusion_v1",
                    "method_label": "Plain SD 1.5 (strength 0.35)",
                },
                {
                    "method": "controlnet_canny",
                    "strength": 0.35,
                    "control_scale": 0.50,
                    "output_id": f"gen_cnet_canny_{prefix}_{cond}_s35_c50",
                    "pilot_sub": "controlnet_canny_v1",
                    "method_label": "ControlNet Canny (scale 0.50, strength 0.35)",
                },
                {
                    "method": "controlnet_canny",
                    "strength": 0.35,
                    "control_scale": 0.80,
                    "output_id": f"gen_cnet_canny_{prefix}_{cond}_s35_c80",
                    "pilot_sub": "controlnet_canny_v1",
                    "method_label": "ControlNet Canny (scale 0.80, strength 0.35)",
                },
            ]

            group_outputs = []
            for spec in candidate_specs:
                out_id = spec["output_id"]
                sub = spec["pilot_sub"]

                img_path = pilots_dir / sub / "outputs" / f"{out_id}.png"
                meta_path = pilots_dir / sub / "metadata" / f"{out_id}.json"

                if not img_path.exists():
                    raise FileNotFoundError(f"Missing output image: {img_path}")
                if not meta_path.exists():
                    raise FileNotFoundError(f"Missing output metadata: {meta_path}")

                with open(meta_path, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)

                img_sha256 = compute_file_sha256(img_path)
                data_uri = load_image_as_data_uri(img_path) if embed_base64 else ""

                rel_img_path = str(img_path.relative_to(project_root).as_posix())

                q_info = quality_lookup.get(out_id, {})
                prev_review = existing_decisions.get(out_id, {})

                initial_review_state = {
                    "geometry_preserved": prev_review.get("geometry_preserved", "unreviewed"),
                    "corrosion_preserved": prev_review.get("corrosion_preserved", "unreviewed"),
                    "condition_visible": prev_review.get("condition_visible", "unreviewed"),
                    "realism_acceptable": prev_review.get("realism_acceptable", "unreviewed"),
                    "decision": prev_review.get("decision", "pending"),
                    "reviewer": prev_review.get("reviewer", default_reviewer),
                    "review_notes": prev_review.get("review_notes", ""),
                    "review_timestamp": prev_review.get("review_timestamp", None),
                }

                record = {
                    "output_id": out_id,
                    "parent_id": parent_id,
                    "parent_prefix": prefix,
                    "parent_label": parent_label,
                    "condition": cond,
                    "method": spec["method"],
                    "method_label": spec["method_label"],
                    "strength": spec["strength"],
                    "control_scale": spec["control_scale"],
                    "effective_denoising_steps": meta.get("effective_denoising_steps", int(30 * spec["strength"])),
                    "seed": meta.get("seed", 0),
                    "relative_image_path": rel_img_path,
                    "output_sha256": img_sha256,
                    "ssim_rgb": q_info.get("ssim_rgb", meta.get("ssim_rgb", 0.0)),
                    "lpips": q_info.get("lpips"),
                    "edge_overlap_iou_heuristic": q_info.get("edge_overlap_iou", meta.get("edge_overlap_iou_heuristic")),
                    "edge_overlap_dice_heuristic": q_info.get("edge_overlap_dice", meta.get("edge_overlap_dice_heuristic")),
                    "rust_area_delta": q_info.get("rust_area_delta"),
                    "quality_status": q_info.get("quality_status", "unscreened"),
                    "quality_flags": q_info.get("flags", ""),
                    "mean_pixel_abs_diff": meta.get("mean_pixel_abs_diff", 0.0),
                    "runtime_seconds": meta.get("runtime_seconds", 0.0),
                    "data_uri": data_uri,
                    "review_state": initial_review_state,
                }

                group_outputs.append(record)
                outputs_lookup[out_id] = {
                    "output_id": out_id,
                    "parent_id": parent_id,
                    "condition": cond,
                    "method": spec["method"],
                    "strength": spec["strength"],
                    "control_scale": spec["control_scale"],
                    "output_sha256": img_sha256,
                    "relative_image_path": rel_img_path,
                    "ssim_rgb": record["ssim_rgb"],
                    "lpips": record["lpips"],
                    "edge_overlap_iou_heuristic": record["edge_overlap_iou_heuristic"],
                    "quality_status": record["quality_status"],
                    "quality_flags": record["quality_flags"],
                }

            group_data = {
                "group_id": group_id,
                "group_index": group_idx,
                "parent_id": parent_id,
                "parent_prefix": prefix,
                "parent_label": parent_label,
                "parent_label_display": f"Source label: {parent_label} (inherited, not independently verified)",
                "parent_relative_path": str(parent_full_path.relative_to(project_root).as_posix()),
                "parent_sha256": parent_sha256,
                "parent_data_uri": parent_uri,
                "condition": cond,
                "condition_display": cond_display,
                "canny_map_relative_path": canny_rel_path,
                "canny_map_sha256": canny_sha256,
                "canny_data_uri": canny_uri,
                "canny_edge_density": meta.get("canny_edge_density", 0.0),
                "outputs": group_outputs,
            }

            groups.append(group_data)
            group_idx += 1

    inventory = {
        "inventory_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_count": len(sources_df),
        "condition_count": len(conditions),
        "group_count": len(groups),
        "total_candidates": len(outputs_lookup),
        "reviewer_default": default_reviewer,
        "models": {
            "diffusion_backbone": {
                "model_id": diff_sum.get("model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
                "model_revision": diff_sum.get("model_revision", "main"),
                "license": "CreativeML OpenRAIL-M",
            },
            "controlnet": {
                "model_id": cnet_sum.get("controlnet_id", "lllyasviel/control_v11p_sd15_canny"),
                "model_revision": cnet_sum.get("controlnet_revision", "main"),
                "license": "CreativeML OpenRAIL-M",
            }
        },
        "evaluation_rule": "This is an exploratory visual inspection, not pixel-level proof or certified industrial metallurgical validation.",
        "approval_rule": "An approved decision strictly requires all 4 criteria to be Yes and a valid Reviewer Name.",
        "groups": groups,
        "outputs_lookup": outputs_lookup,
    }

    return inventory


def generate_html_content(inventory: Dict[str, Any], embed_base64: bool = True) -> str:
    """
    Renders the standalone offline review HTML page.
    Zero remote dependencies, zero external CDNs, fully compliant with Cyber-Saffron theme.
    """
    inv_json_str = json.dumps(inventory, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PipeAugment AI — Generative Pilot Review Utility</title>
<style>
  :root {{
    --bg-page: #0a0e17;
    --bg-card: #111827;
    --bg-card-alt: #162032;
    --bg-input: #0d131f;
    --border-color: #243044;
    --border-focus: #ff5a00;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --accent-primary: #ff5a00;
    --accent-hover: #ff7729;
    --accent-amber: #f59e0b;
    --accent-cyan: #38bdf8;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --pending: #64748b;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-page);
    color: var(--text-main);
    font-family: var(--font-sans);
    line-height: 1.5;
    padding-bottom: 60px;
  }}

  header.top-bar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(11, 16, 26, 0.95);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border-color);
    padding: 14px 28px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }}

  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .brand-badge {{
    background: linear-gradient(135deg, #ff5a00, #ff8c00);
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 8px;
    border-radius: 4px;
  }}
  .brand-title {{
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
  }}

  .header-actions {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}

  .reviewer-input-group {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: 6px;
  }}
  .reviewer-input-group label {{
    font-size: 12px;
    color: var(--accent-amber);
    font-weight: 600;
    white-space: nowrap;
  }}
  .reviewer-input-group input {{
    background: transparent;
    border: none;
    outline: none;
    color: #fff;
    font-size: 13px;
    width: 140px;
  }}

  button.btn {{
    background: var(--bg-card-alt);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    padding: 7px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  button.btn:hover {{
    border-color: var(--accent-primary);
    background: #1e2d42;
  }}
  button.btn-primary {{
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: #ffffff;
  }}
  button.btn-primary:hover {{
    background: var(--accent-hover);
    border-color: var(--accent-hover);
  }}

  /* Quick Filters Bar */
  .quick-filters-bar {{
    background: #0d131f;
    border-bottom: 1px solid var(--border-color);
    padding: 10px 28px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 20px;
  }}
  .filter-group-inline {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .filter-group-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--accent-amber);
    white-space: nowrap;
  }}
  .pill-group {{
    display: flex;
    gap: 4px;
    background: #111827;
    border: 1px solid var(--border-color);
    padding: 3px;
    border-radius: 6px;
  }}
  .btn-pill {{
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn-pill:hover {{
    color: #fff;
    background: #1a2538;
  }}
  .btn-pill.active {{
    background: var(--accent-primary);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
  }}

  /* Stats Bar */
  .stats-bar {{
    background: #0f1624;
    border-bottom: 1px solid var(--border-color);
    padding: 10px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .stats-items {{
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .stat-pill {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(255,255,255,0.04);
  }}
  .stat-pill .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }}
  .dot-approved {{ background: var(--success); }}
  .dot-rejected {{ background: var(--danger); }}
  .dot-needs {{ background: var(--warning); }}
  .dot-pending {{ background: var(--pending); }}

  .filter-controls {{
    display: flex;
    gap: 10px;
    align-items: center;
  }}
  .filter-controls select {{
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    outline: none;
  }}

  main.container {{
    max-width: 1720px;
    margin: 0 auto;
    padding: 24px;
  }}

  .banner-notice {{
    background: #141e2e;
    border-left: 4px solid var(--accent-primary);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 24px;
    font-size: 13px;
    color: #cbd5e1;
  }}
  .banner-notice strong {{ color: #ffffff; }}
  .banner-notice p {{ margin-top: 4px; }}

  .group-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    margin-bottom: 32px;
    overflow: hidden;
  }}
  .group-header {{
    background: var(--bg-card-alt);
    border-bottom: 1px solid var(--border-color);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .group-title-block {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .group-badge {{
    background: #243044;
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .group-title {{
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
  }}
  .group-subtitle {{
    font-size: 12px;
    color: var(--text-muted);
  }}

  .group-grid {{
    display: grid;
    grid-template-columns: 280px repeat(4, 1fr);
    gap: 16px;
    padding: 20px;
  }}
  @media (max-width: 1400px) {{
    .group-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media (max-width: 800px) {{
    .group-grid {{ grid-template-columns: 1fr; }}
  }}

  .candidate-box {{
    background: var(--bg-page);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .candidate-box.is-parent {{
    background: #0d1422;
    border-color: #2b3b55;
  }}

  .thumb-wrap {{
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
  }}
  .thumb-wrap img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    image-rendering: auto;
    display: block;
  }}
  .thumb-overlay-btn {{
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(0,0,0,0.75);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.2);
    font-size: 10px;
    padding: 3px 6px;
    border-radius: 4px;
    cursor: pointer;
    backdrop-filter: blur(4px);
  }}

  .meta-tag-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }}
  .meta-tag {{
    background: #1a2436;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: var(--font-mono);
  }}
  .meta-tag.meta-tag-pass {{
    background: #064e3b;
    border: 1px solid #059669;
    color: #a7f3d0;
    font-weight: 700;
  }}
  .meta-tag.meta-tag-flag {{
    background: #78350f;
    border: 1px solid #d97706;
    color: #fde68a;
    font-weight: 700;
  }}
  .meta-tag.meta-tag-fail {{
    background: #7f1d1d;
    border: 1px solid #dc2626;
    color: #fecaca;
    font-weight: 700;
  }}
  .automated-notice {{
    font-size: 10px;
    color: var(--text-dim);
    margin-top: -4px;
    font-style: italic;
  }}

  .control-group {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 11px;
  }}
  .control-label {{
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.5px;
  }}

  .criteria-radios {{
    display: flex;
    gap: 4px;
  }}
  .btn-check {{
    flex: 1;
    padding: 4px 6px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid var(--border-color);
    background: #111827;
    color: var(--text-muted);
    border-radius: 4px;
    cursor: pointer;
    text-align: center;
  }}
  .btn-check:hover {{
    background: #1a2538;
    color: #fff;
  }}
  .btn-check.selected-yes {{
    background: #065f46;
    border-color: var(--success);
    color: #a7f3d0;
  }}
  .btn-check.selected-no {{
    background: #7f1d1d;
    border-color: var(--danger);
    color: #fecaca;
  }}
  .btn-check.selected-uncertain {{
    background: #78350f;
    border-color: var(--warning);
    color: #fde68a;
  }}

  .decision-row {{
    display: flex;
    gap: 4px;
    margin-top: 4px;
  }}
  .btn-decision {{
    flex: 1;
    padding: 6px 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid var(--border-color);
    background: #111827;
    color: var(--text-dim);
    border-radius: 4px;
    cursor: pointer;
    text-align: center;
  }}
  .btn-decision:hover {{
    border-color: #475569;
    color: #cbd5e1;
  }}
  .btn-decision.active-approved {{
    background: var(--success);
    border-color: var(--success);
    color: #064e3b;
    font-weight: 800;
  }}
  .btn-decision.active-rejected {{
    background: var(--danger);
    border-color: var(--danger);
    color: #450a0a;
    font-weight: 800;
  }}
  .btn-decision.active-needs {{
    background: var(--warning);
    border-color: var(--warning);
    color: #451a03;
    font-weight: 800;
  }}

  .notes-input {{
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    font-size: 11px;
    font-family: inherit;
    padding: 6px;
    border-radius: 4px;
    resize: vertical;
    min-height: 40px;
  }}
  .notes-input:focus {{
    border-color: var(--border-focus);
    outline: none;
  }}

  /* Modal Inspection */
  .modal-backdrop {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(6px);
    z-index: 200;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }}
  .modal-backdrop.is-open {{
    display: flex;
  }}
  .modal-box {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    max-width: 1320px;
    width: 100%;
    max-height: 92vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7);
  }}
  .modal-header {{
    background: var(--bg-card-alt);
    border-bottom: 1px solid var(--border-color);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .modal-body {{
    padding: 20px;
    overflow-y: auto;
    display: flex;
    gap: 20px;
    flex-direction: column;
  }}
  .inspection-split {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .inspection-panel {{
    background: #080c14;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  .inspection-panel img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    image-rendering: auto;
  }}
  .inspection-panel.zoom-2x img {{
    transform: scale(1.5);
    transform-origin: top center;
    image-rendering: pixelated;
  }}
  .inspection-caption {{
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 600;
  }}
</style>
</head>
<body>

<header class="top-bar">
  <div class="brand">
    <div class="brand-badge">Gate 1</div>
    <div>
      <div class="brand-title">PipeAugment AI — Generative Selection Review</div>
      <div style="font-size: 11px; color: var(--text-muted);">Plain SD 1.5 vs Canny ControlNet Pilot | 48 Outputs Across 6 Parents</div>
    </div>
  </div>

  <div class="header-actions">
    <div class="reviewer-input-group">
      <label for="globalReviewer">Reviewer Name:</label>
      <input type="text" id="globalReviewer" placeholder="e.g. Shubham" oninput="syncGlobalReviewer(this.value)">
    </div>
    <button class="btn btn-primary" onclick="exportReviewJSON()">Export Review JSON</button>
    <button class="btn" onclick="exportReviewCSV()">Export Review CSV</button>
    <button class="btn" onclick="triggerImportFileInput()">Import Review File</button>
    <input type="file" id="importFileInput" accept=".json,.csv" style="display:none" onchange="handleImportFile(event)">
  </div>
</header>

<!-- Phase 4 Quick Filters Bar -->
<div class="quick-filters-bar">
  <div class="filter-group-inline">
    <span class="filter-group-title">Method:</span>
    <div class="pill-group" id="pillGroup_method">
      <button type="button" class="btn-pill active" data-val="all" onclick="setQuickFilter('method', 'all')">All Methods</button>
      <button type="button" class="btn-pill" data-val="controlnet_canny" onclick="setQuickFilter('method', 'controlnet_canny')">ControlNet Only</button>
      <button type="button" class="btn-pill" data-val="plain_img2img" onclick="setQuickFilter('method', 'plain_img2img')">Plain SD 1.5 Only</button>
    </div>
  </div>

  <div class="filter-group-inline">
    <span class="filter-group-title">Class:</span>
    <div class="pill-group" id="pillGroup_class">
      <button type="button" class="btn-pill active" data-val="all" onclick="setQuickFilter('class', 'all')">All Classes</button>
      <button type="button" class="btn-pill" data-val="corroded" onclick="setQuickFilter('class', 'corroded')">Corroded</button>
      <button type="button" class="btn-pill" data-val="non_corroded" onclick="setQuickFilter('class', 'non_corroded')">Non-Corroded</button>
    </div>
  </div>

  <div class="filter-group-inline">
    <span class="filter-group-title">Status:</span>
    <div class="pill-group" id="pillGroup_status">
      <button type="button" class="btn-pill active" data-val="all" onclick="setQuickFilter('status', 'all')">All Statuses</button>
      <button type="button" class="btn-pill" data-val="pending" onclick="setQuickFilter('status', 'pending')">Pending Only</button>
      <button type="button" class="btn-pill" data-val="reviewed" onclick="setQuickFilter('status', 'reviewed')">Reviewed Only</button>
    </div>
  </div>
</div>

<div class="stats-bar">
  <div class="stats-items">
    <div class="stat-pill"><span class="dot dot-approved"></span> <span id="countApproved">0</span> Approved</div>
    <div class="stat-pill"><span class="dot dot-rejected"></span> <span id="countRejected">0</span> Rejected</div>
    <div class="stat-pill"><span class="dot dot-needs"></span> <span id="countNeeds">0</span> Needs Review</div>
    <div class="stat-pill"><span class="dot dot-pending"></span> <span id="countPending">48</span> Pending</div>
    <div style="color: var(--text-dim);">|</div>
    <div id="progressText" style="font-weight: 600; color: var(--accent-amber);">Reviewed: 0 / 48 (0%)</div>
  </div>

  <div class="filter-controls">
    <label style="color: var(--text-dim); font-size: 11px;">Condition Filter:</label>
    <select id="filterCondition" onchange="applyFilters()">
      <option value="all">All Lighting Conditions</option>
      <option value="low_light_inspection">Low-Light Only</option>
      <option value="directional_industrial_lighting">Directional Only</option>
    </select>
  </div>
</div>

<main class="container">
  <div class="banner-notice">
    <strong>Evaluation Guidelines & Ethical Notice:</strong>
    <p>1. This is an exploratory visual inspection, <strong>not</strong> pixel-level proof or certified industrial metallurgical validation.</p>
    <p>2. <strong>Reject</strong> outputs with changed pipe geometry/joints, hallucinated rust defects on non-corroded pipes, erased existing corrosion, or severe artifacts.</p>
    <p>3. <strong>Approval Rule:</strong> An "Approved" decision strictly requires all 4 criteria to be <strong>Yes</strong> and a valid Reviewer Name.</p>
    <p>4. <em>Phase 4 Screening Badges:</em> Displayed SSIM, LPIPS, and Edge IoU metrics represent automated quality validation gates, not human approval.</p>
    <p>5. <em>Notice on drafts:</em> Browser storage is temporary. Always click <strong>Export Review JSON</strong> when finished to persist your decisions to disk.</p>
  </div>

  <div id="groupsContainer">
    <!-- Groups populated dynamically via JavaScript -->
  </div>
</main>

<!-- High-Res Side-by-Side Modal -->
<div class="modal-backdrop" id="inspectModal">
  <div class="modal-box">
    <div class="modal-header">
      <div style="display: flex; align-items: center; gap: 10px;">
        <span class="group-badge" id="modalBadge">MODAL</span>
        <span style="font-size: 14px; font-weight: 700; color: #fff;" id="modalTitle">Comparison</span>
      </div>
      <div style="display: flex; gap: 10px; align-items: center;">
        <button class="btn" onclick="toggleModalZoom()">Toggle 2x Zoom</button>
        <button class="btn" onclick="toggleModalCannyOverlay()">Show Canny Edges</button>
        <button class="btn" onclick="closeModal()">Close (Esc)</button>
      </div>
    </div>
    <div class="modal-body">
      <div class="inspection-split">
        <div class="inspection-panel" id="modalLeftPanel">
          <div class="inspection-caption">Original Source Parent (512x512)</div>
          <img id="modalParentImg" src="" alt="Parent">
        </div>
        <div class="inspection-panel" id="modalRightPanel">
          <div class="inspection-caption" id="modalCandidateCaption">Candidate Output (512x512)</div>
          <img id="modalCandidateImg" src="" alt="Candidate">
        </div>
      </div>
      <div id="modalControlsSlot" style="background: #111827; border: 1px solid var(--border-color); border-radius: 8px; padding: 14px;">
        <!-- Synchronized controls cloned here -->
      </div>
    </div>
  </div>
</div>

<script>
// Embedded inventory data
const INVENTORY = {inv_json_str};

// Working review state: key = output_id -> review_state
let reviewState = {{}};
let activeModalOutputId = null;
let modalZoom = false;
let modalShowingCanny = false;

// Quick Filter State
let quickFilters = {{
  method: "all",
  class: "all",
  status: "all"
}};

// Initialize review state from inventory or localStorage draft
function initReviewState() {{
  // 1. Fill defaults from INVENTORY
  for (const group of INVENTORY.groups) {{
    for (const out of group.outputs) {{
      reviewState[out.output_id] = Object.assign({{}}, out.review_state);
    }}
  }}

  // 2. Check localStorage draft (merge user work)
  try {{
    const saved = localStorage.getItem("pipeaugment_review_draft_v1");
    if (saved) {{
      const parsed = JSON.parse(saved);
      if (parsed && typeof parsed === "object") {{
        for (const [k, v] of Object.entries(parsed)) {{
          if (reviewState[k]) {{
            if (v.decision && v.decision !== "pending") {{
              reviewState[k] = Object.assign(reviewState[k], v);
            }} else if (reviewState[k].decision === "pending") {{
              reviewState[k] = Object.assign(reviewState[k], v);
            }}
          }}
        }}
      }}
    }}
  }} catch (e) {{
    console.warn("Could not read localStorage draft:", e);
  }}

  // 3. Check saved reviewer
  const savedReviewer = localStorage.getItem("pipeaugment_reviewer_name") || INVENTORY.reviewer_default || "";
  if (savedReviewer) {{
    const el = document.getElementById("globalReviewer");
    if (el) el.value = savedReviewer;
  }}
}}

function saveDraftToStorage() {{
  try {{
    localStorage.setItem("pipeaugment_review_draft_v1", JSON.stringify(reviewState));
  }} catch (e) {{
    console.warn("Failed to save draft to localStorage:", e);
  }}
}}

function syncGlobalReviewer(val) {{
  const trimmed = (val || "").trim();
  localStorage.setItem("pipeaugment_reviewer_name", trimmed);
  for (const k in reviewState) {{
    if (!reviewState[k].reviewer) {{
      reviewState[k].reviewer = trimmed;
    }}
  }}
}}

function setQuickFilter(dim, val) {{
  quickFilters[dim] = val;
  const group = document.getElementById("pillGroup_" + dim);
  if (group) {{
    const btns = group.querySelectorAll(".btn-pill");
    btns.forEach(b => {{
      if (b.getAttribute("data-val") === val) {{
        b.classList.add("active");
      }} else {{
        b.classList.remove("active");
      }}
    }});
  }}
  applyFilters();
}}

function updateCriteria(outputId, criterion, value) {{
  if (!reviewState[outputId]) return;
  reviewState[outputId][criterion] = value;
  reviewState[outputId].review_timestamp = new Date().toISOString();

  const reviewerInput = document.getElementById("globalReviewer");
  if (reviewerInput && reviewerInput.value.trim() && !reviewState[outputId].reviewer) {{
    reviewState[outputId].reviewer = reviewerInput.value.trim();
  }}

  if (reviewState[outputId].decision === "approved") {{
    if (value !== "yes") {{
      reviewState[outputId].decision = "needs_review";
    }}
  }}

  saveDraftToStorage();
  renderCardControls(outputId);
  updateStats();
  if (activeModalOutputId === outputId) {{
    renderModalControls(outputId);
  }}
}}

function setDecision(outputId, decision) {{
  if (!reviewState[outputId]) return;
  const state = reviewState[outputId];

  if (decision === "approved") {{
    const g = state.geometry_preserved === "yes";
    const c = state.corrosion_preserved === "yes";
    const v = state.condition_visible === "yes";
    const r = state.realism_acceptable === "yes";
    const rev = (state.reviewer || document.getElementById("globalReviewer").value || "").trim();

    if (!g || !c || !v || !r || !rev) {{
      alert(
        "Cannot Approve Output:\\n" +
        "- All 4 criteria (Geometry, Corrosion, Condition, Realism) must be 'Yes'.\\n" +
        "- A Reviewer Name must be entered.\\n\\n" +
        "Please complete criteria or mark as 'Needs Review'."
      );
      return;
    }}
    state.reviewer = rev;
  }}

  state.decision = decision;
  state.review_timestamp = new Date().toISOString();

  saveDraftToStorage();
  renderCardControls(outputId);
  updateStats();
  if (activeModalOutputId === outputId) {{
    renderModalControls(outputId);
  }}
}}

function updateNotes(outputId, notes) {{
  if (!reviewState[outputId]) return;
  reviewState[outputId].review_notes = notes;
  reviewState[outputId].review_timestamp = new Date().toISOString();
  saveDraftToStorage();
}}

function updateStats() {{
  let approved = 0, rejected = 0, needs = 0, pending = 0;
  const total = Object.keys(reviewState).length;

  for (const k in reviewState) {{
    const d = reviewState[k].decision;
    if (d === "approved") approved++;
    else if (d === "rejected") rejected++;
    else if (d === "needs_review") needs++;
    else pending++;
  }}

  document.getElementById("countApproved").textContent = approved;
  document.getElementById("countRejected").textContent = rejected;
  document.getElementById("countNeeds").textContent = needs;
  document.getElementById("countPending").textContent = pending;

  const reviewed = total - pending;
  const pct = total > 0 ? Math.round((reviewed / total) * 100) : 0;
  document.getElementById("progressText").textContent = `Reviewed: ${{reviewed}} / ${{total}} (${{pct}}%)`;
}}

function renderCardControls(outputId) {{
  const container = document.getElementById("controls_" + outputId);
  if (!container) return;
  const state = reviewState[outputId];

  const criteria = [
    {{ key: "geometry_preserved", label: "Geometry Preserved?" }},
    {{ key: "corrosion_preserved", label: "Corrosion Preserved?" }},
    {{ key: "condition_visible", label: "Condition Visible?" }},
    {{ key: "realism_acceptable", label: "Realism Acceptable?" }}
  ];

  let html = "";
  for (const cr of criteria) {{
    const cur = state[cr.key] || "unreviewed";
    html += `
      <div class="control-group">
        <div class="control-label">${{cr.label}}</div>
        <div class="criteria-radios">
          <button type="button" class="btn-check ${{cur === 'yes' ? 'selected-yes' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'yes')">Yes</button>
          <button type="button" class="btn-check ${{cur === 'no' ? 'selected-no' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'no')">No</button>
          <button type="button" class="btn-check ${{cur === 'uncertain' ? 'selected-uncertain' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'uncertain')">Uncertain</button>
        </div>
      </div>
    `;
  }}

  const curDec = state.decision || "pending";
  html += `
    <div class="control-group" style="margin-top: 6px;">
      <div class="control-label">Decision:</div>
      <div class="decision-row">
        <button type="button" class="btn-decision ${{curDec === 'approved' ? 'active-approved' : ''}}" onclick="setDecision('${{outputId}}', 'approved')">Approve</button>
        <button type="button" class="btn-decision ${{curDec === 'rejected' ? 'active-rejected' : ''}}" onclick="setDecision('${{outputId}}', 'rejected')">Reject</button>
        <button type="button" class="btn-decision ${{curDec === 'needs_review' ? 'active-needs' : ''}}" onclick="setDecision('${{outputId}}', 'needs_review')">Review</button>
      </div>
    </div>
    <div class="control-group" style="margin-top: 4px;">
      <textarea class="notes-input" placeholder="Review notes (optional)..." oninput="updateNotes('${{outputId}}', this.value)">${{escapeHtml(state.review_notes || '')}}</textarea>
    </div>
  `;

  container.innerHTML = html;
}}

function renderAllCards() {{
  const container = document.getElementById("groupsContainer");
  let html = "";

  for (const group of INVENTORY.groups) {{
    const pUri = group.parent_data_uri || group.parent_relative_path;
    const cannyUri = group.canny_data_uri || group.canny_map_relative_path;

    html += `
      <div class="group-card" id="${{group.group_id}}" data-class="${{group.parent_label}}" data-condition="${{group.condition}}">
        <div class="group-header">
          <div class="group-title-block">
            <span class="group-badge">Parent #${{group.group_index}}</span>
            <div>
              <div class="group-title">${{group.parent_prefix}}... &bull; ${{group.condition_display}}</div>
              <div class="group-subtitle">${{group.parent_label_display}}</div>
            </div>
          </div>
          <div style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);">
            SHA: ${{group.parent_sha256.substring(0, 12)}}...
          </div>
        </div>

        <div class="group-grid">
          <!-- Col 0: Parent Reference -->
          <div class="candidate-box is-parent">
            <div style="font-size: 12px; font-weight: 700; color: #fff;">1. Original Parent</div>
            <div class="thumb-wrap" onclick="openModal('${{group.outputs[0].output_id}}')">
              <img src="${{pUri}}" alt="Parent Reference">
              <button class="thumb-overlay-btn" type="button">Inspect</button>
            </div>
            <div class="meta-tag-row">
              <span class="meta-tag">Class: ${{group.parent_label}}</span>
              <span class="meta-tag">512x512</span>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
              <strong>Canny Edge Condition:</strong>
              <div class="thumb-wrap" style="aspect-ratio: 16/9; max-height: 100px; margin-top: 4px;" onclick="openModal('${{group.outputs[2].output_id}}', true)">
                <img src="${{cannyUri}}" alt="Canny Condition Map">
              </div>
              <div style="font-size: 10px; color: var(--text-dim); margin-top: 2px;">Thresholds: [100, 200]</div>
            </div>
          </div>
    `;

    // Cols 1..4: Outputs
    for (const out of group.outputs) {{
      const oUri = out.data_uri || out.relative_image_path;
      const ssim = out.ssim_rgb ? out.ssim_rgb.toFixed(4) : "—";
      const lpips = (out.lpips !== null && out.lpips !== undefined) ? out.lpips.toFixed(4) : "—";
      const edgeIoU = (out.edge_overlap_iou_heuristic !== null && out.edge_overlap_iou_heuristic !== undefined) ? out.edge_overlap_iou_heuristic.toFixed(4) : "—";

      let qClass = "pending";
      let qStatusText = "Unscreened";
      let qFlagsTooltip = "Automated quality screening pending";
      if (out.quality_status === "pass") {{
        qClass = "pass";
        qStatusText = "Pass";
        qFlagsTooltip = "Automated screening passed all thresholds";
      }} else if (out.quality_status === "flag_for_review") {{
        qClass = "flag";
        qStatusText = "Flagged";
        qFlagsTooltip = "Flagged: " + (out.quality_flags || "threshold exceeded");
      }} else if (out.quality_status === "fail") {{
        qClass = "fail";
        qStatusText = "Fail";
        qFlagsTooltip = "Failed: " + (out.quality_flags || "fatal quality violation");
      }}

      html += `
        <div class="candidate-box" id="box_${{out.output_id}}" data-method="${{out.method}}" data-class="${{group.parent_label}}">
          <div style="font-size: 12px; font-weight: 700; color: var(--accent-amber);">${{out.method_label}}</div>
          <div class="thumb-wrap" onclick="openModal('${{out.output_id}}')">
            <img src="${{oUri}}" alt="${{out.output_id}}">
            <button class="thumb-overlay-btn" type="button">Zoom & Inspect</button>
          </div>
          <div class="meta-tag-row">
            <span class="meta-tag" title="RGB Structural Similarity Index">SSIM: ${{ssim}}</span>
            <span class="meta-tag" title="AlexNet LPIPS perceptual distance">LPIPS: ${{lpips}}</span>
            <span class="meta-tag" title="Canny Edge Retention IoU">Edge: ${{edgeIoU}}</span>
            <span class="meta-tag meta-tag-${{qClass}}" title="${{escapeHtml(qFlagsTooltip)}}">Status: ${{qStatusText}}</span>
          </div>
          <div class="automated-notice">Automated screening metrics (not approval)</div>

          <div id="controls_${{out.output_id}}">
            <!-- Populated by renderCardControls -->
          </div>
        </div>
      `;
    }}

    html += `
        </div>
      </div>
    `;
  }}

  container.innerHTML = html;

  for (const group of INVENTORY.groups) {{
    for (const out of group.outputs) {{
      renderCardControls(out.output_id);
    }}
  }}
  updateStats();
}}

// Modal Inspection functions
function openModal(outputId, showCannyOnly = false) {{
  activeModalOutputId = outputId;
  const modal = document.getElementById("inspectModal");
  let candidateRecord = null;
  let parentGroup = null;

  for (const g of INVENTORY.groups) {{
    for (const out of g.outputs) {{
      if (out.output_id === outputId) {{
        candidateRecord = out;
        parentGroup = g;
        break;
      }}
    }}
    if (candidateRecord) break;
  }}

  if (!candidateRecord || !parentGroup) return;

  const pImg = document.getElementById("modalParentImg");
  const cImg = document.getElementById("modalCandidateImg");
  const leftCap = document.getElementById("modalLeftPanel").querySelector(".inspection-caption");
  const rightCap = document.getElementById("modalCandidateCaption");

  if (showCannyOnly) {{
    pImg.src = parentGroup.canny_data_uri || parentGroup.canny_map_relative_path;
    leftCap.textContent = "Source Canny Edge Map (100, 200)";
  }} else {{
    pImg.src = parentGroup.parent_data_uri || parentGroup.parent_relative_path;
    leftCap.textContent = `Original Parent (${{parentGroup.parent_label}})`;
  }}

  cImg.src = candidateRecord.data_uri || candidateRecord.relative_image_path;
  const ssim = candidateRecord.ssim_rgb ? candidateRecord.ssim_rgb.toFixed(4) : "—";
  const lpips = (candidateRecord.lpips !== null && candidateRecord.lpips !== undefined) ? candidateRecord.lpips.toFixed(4) : "—";
  const edgeIoU = (candidateRecord.edge_overlap_iou_heuristic !== null && candidateRecord.edge_overlap_iou_heuristic !== undefined) ? candidateRecord.edge_overlap_iou_heuristic.toFixed(4) : "—";
  rightCap.textContent = `${{candidateRecord.method_label}} | SSIM: ${{ssim}} | LPIPS: ${{lpips}} | Edge IoU: ${{edgeIoU}} | Screen: ${{candidateRecord.quality_status || 'unscreened'}}`;

  document.getElementById("modalTitle").textContent = `${{parentGroup.parent_prefix}}... &bull; ${{parentGroup.condition_display}}`;
  document.getElementById("modalBadge").textContent = candidateRecord.output_id;

  renderModalControls(outputId);
  modal.classList.add("is-open");
}}

function closeModal() {{
  document.getElementById("inspectModal").classList.remove("is-open");
  activeModalOutputId = null;
}}

function toggleModalZoom() {{
  modalZoom = !modalZoom;
  const leftP = document.getElementById("modalLeftPanel");
  const rightP = document.getElementById("modalRightPanel");
  if (modalZoom) {{
    leftP.classList.add("zoom-2x");
    rightP.classList.add("zoom-2x");
  }} else {{
    leftP.classList.remove("zoom-2x");
    rightP.classList.remove("zoom-2x");
  }}
}}

function toggleModalCannyOverlay() {{
  if (!activeModalOutputId) return;
  modalShowingCanny = !modalShowingCanny;
  openModal(activeModalOutputId, modalShowingCanny);
}}

function renderModalControls(outputId) {{
  const slot = document.getElementById("modalControlsSlot");
  const state = reviewState[outputId];
  if (!state) return;

  const criteria = [
    {{ key: "geometry_preserved", label: "Geometry Preserved?" }},
    {{ key: "corrosion_preserved", label: "Corrosion Preserved?" }},
    {{ key: "condition_visible", label: "Condition Visible?" }},
    {{ key: "realism_acceptable", label: "Realism Acceptable?" }}
  ];

  let html = `<div style="display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end;">`;
  for (const cr of criteria) {{
    const cur = state[cr.key] || "unreviewed";
    html += `
      <div style="flex: 1; min-width: 140px;">
        <div class="control-label">${{cr.label}}</div>
        <div class="criteria-radios" style="margin-top: 4px;">
          <button type="button" class="btn-check ${{cur === 'yes' ? 'selected-yes' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'yes')">Yes</button>
          <button type="button" class="btn-check ${{cur === 'no' ? 'selected-no' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'no')">No</button>
          <button type="button" class="btn-check ${{cur === 'uncertain' ? 'selected-uncertain' : ''}}" onclick="updateCriteria('${{outputId}}', '${{cr.key}}', 'uncertain')">Uncertain</button>
        </div>
      </div>
    `;
  }}

  const curDec = state.decision || "pending";
  html += `
    <div style="flex: 1; min-width: 200px;">
      <div class="control-label">Decision:</div>
      <div class="decision-row" style="margin-top: 4px;">
        <button type="button" class="btn-decision ${{curDec === 'approved' ? 'active-approved' : ''}}" onclick="setDecision('${{outputId}}', 'approved')">Approve</button>
        <button type="button" class="btn-decision ${{curDec === 'rejected' ? 'active-rejected' : ''}}" onclick="setDecision('${{outputId}}', 'rejected')">Reject</button>
        <button type="button" class="btn-decision ${{curDec === 'needs_review' ? 'active-needs' : ''}}" onclick="setDecision('${{outputId}}', 'needs_review')">Review</button>
      </div>
    </div>
  </div>
  <div style="margin-top: 10px;">
    <textarea class="notes-input" placeholder="Notes for this candidate..." oninput="updateNotes('${{outputId}}', this.value)">${{escapeHtml(state.review_notes || '')}}</textarea>
  </div>
  `;

  slot.innerHTML = html;
}}

// Keyboard navigation
window.addEventListener("keydown", (e) => {{
  if (e.key === "Escape") closeModal();
}});

// Unified Filtering
function applyFilters() {{
  const condFilter = document.getElementById("filterCondition").value;

  for (const group of INVENTORY.groups) {{
    const el = document.getElementById(group.group_id);
    if (!el) continue;

    // Class filter (Quick filter)
    const matchesClass = (quickFilters.class === "all" || group.parent_label === quickFilters.class);
    // Condition filter (Dropdown)
    const matchesCond = (condFilter === "all" || group.condition === condFilter);

    let hasVisibleOutputs = false;
    for (const out of group.outputs) {{
      const bEl = document.getElementById("box_" + out.output_id);
      if (!bEl) continue;
      const dec = reviewState[out.output_id] ? reviewState[out.output_id].decision : "pending";

      // Method match (Quick filter)
      let matchesMethod = (quickFilters.method === "all" || out.method === quickFilters.method);

      // Status match (Quick filter)
      let matchesStatus = true;
      if (quickFilters.status === "pending") {{
        matchesStatus = (dec === "pending");
      }} else if (quickFilters.status === "reviewed") {{
        matchesStatus = (dec !== "pending");
      }}

      if (matchesMethod && matchesStatus && matchesCond && matchesClass) {{
        bEl.style.display = "flex";
        hasVisibleOutputs = true;
      }} else {{
        bEl.style.display = "none";
      }}
    }}

    if (hasVisibleOutputs && matchesCond && matchesClass) {{
      el.style.display = "block";
    }} else {{
      el.style.display = "none";
    }}
  }}
}}

// Export Functions
function exportReviewJSON() {{
  const exportPayload = {{
    schema_version: "1.0",
    exported_at_utc: new Date().toISOString(),
    inventory_created_at: INVENTORY.created_at_utc,
    total_records: Object.keys(reviewState).length,
    reviewer_default: (document.getElementById("globalReviewer").value || "").trim(),
    reviews: []
  }};

  for (const group of INVENTORY.groups) {{
    for (const out of group.outputs) {{
      const st = reviewState[out.output_id] || {{}};
      exportPayload.reviews.push({{
        output_id: out.output_id,
        parent_id: out.parent_id,
        condition: out.condition,
        method: out.method,
        strength: out.strength,
        control_scale: out.control_scale,
        output_sha256: out.output_sha256,
        geometry_preserved: st.geometry_preserved || "unreviewed",
        corrosion_preserved: st.corrosion_preserved || "unreviewed",
        condition_visible: st.condition_visible || "unreviewed",
        realism_acceptable: st.realism_acceptable || "unreviewed",
        decision: st.decision || "pending",
        reviewer: st.reviewer || (document.getElementById("globalReviewer").value || "").trim(),
        review_notes: st.review_notes || "",
        review_timestamp: st.review_timestamp || null
      }});
    }}
  }}

  const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `generative_review_export_${{new Date().toISOString().replace(/[:.]/g, "-")}}.json`;
  a.click();
  URL.revokeObjectURL(url);
}}

function exportReviewCSV() {{
  const headers = [
    "output_id", "parent_id", "condition", "method", "strength", "control_scale",
    "output_sha256", "geometry_preserved", "corrosion_preserved", "condition_visible",
    "realism_acceptable", "decision", "reviewer", "review_notes", "review_timestamp"
  ];

  const rows = [headers.join(",")];
  for (const group of INVENTORY.groups) {{
    for (const out of group.outputs) {{
      const st = reviewState[out.output_id] || {{}};
      const row = [
        out.output_id,
        out.parent_id,
        out.condition,
        out.method,
        out.strength,
        out.control_scale !== null ? out.control_scale : "",
        out.output_sha256,
        st.geometry_preserved || "unreviewed",
        st.corrosion_preserved || "unreviewed",
        st.condition_visible || "unreviewed",
        st.realism_acceptable || "unreviewed",
        st.decision || "pending",
        `"${{(st.reviewer || "").replace(/"/g, '""')}}"`,
        `"${{(st.review_notes || "").replace(/"/g, '""')}}"`,
        st.review_timestamp || ""
      ];
      rows.push(row.join(","));
    }}
  }}

  const blob = new Blob([rows.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `generative_review_export_${{new Date().toISOString().replace(/[:.]/g, "-")}}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}}

function triggerImportFileInput() {{
  document.getElementById("importFileInput").click();
}}

function handleImportFile(event) {{
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {{
    const content = e.target.result;
    try {{
      if (file.name.endsWith(".json")) {{
        const parsed = JSON.parse(content);
        const list = parsed.reviews || (Array.isArray(parsed) ? parsed : []);
        let count = 0;
        for (const item of list) {{
          if (item && item.output_id && reviewState[item.output_id]) {{
            reviewState[item.output_id] = Object.assign(reviewState[item.output_id], item);
            count++;
          }}
        }}
        alert(`Successfully imported ${{count}} review entries from ${{file.name}}.`);
      }} else if (file.name.endsWith(".csv")) {{
        const lines = content.split(/\\r?\\n/).filter(l => l.trim().length > 0);
        const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
        let count = 0;
        for (let i = 1; i < lines.length; i++) {{
          const parts = parseCsvLine(lines[i]);
          const obj = {{}};
          for (let j = 0; j < headers.length; j++) {{
            obj[headers[j]] = parts[j] || "";
          }}
          if (obj.output_id && reviewState[obj.output_id]) {{
            reviewState[obj.output_id] = Object.assign(reviewState[obj.output_id], obj);
            count++;
          }}
        }}
        alert(`Successfully imported ${{count}} review records from ${{file.name}}.`);
      }}
      saveDraftToStorage();
      renderAllCards();
      applyFilters();
    }} catch (err) {{
      alert("Error parsing file: " + err.message);
    }}
  }};
  reader.readAsText(file);
}}

function parseCsvLine(text) {{
  const re_value = /(?!\\s*$)\\s*(?:"([^"]*(?:""[^"]*)*)"|([^,]*))\\s*(?:,|$)/g;
  const a = [];
  text.replace(re_value, function(m0, m1, m2) {{
    if (m1 !== undefined) a.push(m1.replace(/""/g, '"'));
    else if (m2 !== undefined) a.push(m2);
    return "";
  }});
  return a;
}}

function escapeHtml(str) {{
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}}

// Initialize
initReviewState();
renderAllCards();
</script>
</body>
</html>
"""
    return html


def build_readme() -> str:
    return """# PipeAugment AI — Generative Selection Review Utility

This directory contains the self-contained offline review utility for evaluating and selecting condition-controlled generative augmentation candidates for PipeAugment AI.

## Artifacts in this Directory

- `index.html`: Fully self-contained, offline HTML/CSS/JavaScript review application. Can be opened directly in any modern web browser without starting a web server or installing node dependencies.
- `review_inventory.json`: Machine-readable manifest inventory of all 48 generated candidates across the 6 training parents (3 non-corroded, 3 corroded) and 2 lighting conditions, including Phase 4 automated quality screening metrics.
- `reviewed_manifest_v1.json` / `reviewed_manifest_v2.json`: Official versioned reviewed manifests produced after importing an exported human review.

## Quick Start: How to Open the Review Tool

Run in PowerShell:
```powershell
Invoke-Item "D:\\vscode\\Gen AI Project\\pipeaugment-ai\\experiments\\reviews\\generative_pilot_v1\\index.html"
```
Or double-click `index.html` in Windows File Explorer.

## Quick Filters (Phase 4 Upgrade)
- **Method**: `[All Methods]` | `[ControlNet Only]` | `[Plain SD 1.5 Only]`
- **Class**: `[All Classes]` | `[Corroded]` | `[Non-Corroded]`
- **Status**: `[All Statuses]` | `[Pending Only]` | `[Reviewed Only]`

To review the remaining 22 ControlNet candidates:
1. Click `[ControlNet Only]`.
2. Click `[Pending Only]`.
3. Inspect and mark each candidate.
4. Click **Export Review JSON** to persist your decisions.

## Evaluation Criteria

Each candidate output must be reviewed across four structured criteria:
1. **Geometry Preserved?** (Yes / No / Uncertain): Does the pipe retain its cylindrical shape, seams, joints, flanges, and background perspective?
2. **Corrosion Preserved?** (Yes / No / Uncertain): Are pit locations and rust textures retained without hallucinating new defects or erasing existing corrosion?
3. **Condition Visible?** (Yes / No / Uncertain): Is the requested lighting condition (low-light dimming or directional beam) clearly perceptible?
4. **Realism Acceptable?** (Yes / No / Uncertain): Does the image look like an authentic industrial pipeline inspection capture without generative artifacts or blur?

### Decision Rules
- **Approved**: Allowed ONLY when all four criteria are **Yes**, a valid **Reviewer Name** is supplied, and the reviewer explicitly confirms.
- **Rejected**: Use if pipe geometry breaks, corrosion is invented/removed, or unphysical artifacts occur.
- **Needs Review**: Use when uncertain or when secondary domain verification is needed.

## Export & Import Workflow

1. In the web interface, enter your Reviewer Name in the top bar.
2. Review the candidate outputs.
3. Click **Export Review JSON** to download `generative_review_export_<timestamp>.json`.
4. Save the exported JSON into `experiments/reviews/generative_pilot_v1/`.
5. Import and validate your review using the Python verification CLI:
```powershell
& ".\\.venv-ai\\Scripts\\python.exe" ai/scripts/import_generative_review.py --input experiments/reviews/generative_pilot_v1/generative_review_export_<timestamp>.json
```
6. The import script will verify all hashes, reject invalid approvals, and write the SHA-256 integrity-verified manifest `reviewed_manifest_v3.json`.
"""


def main():
    parser = argparse.ArgumentParser(description="Build PipeAugment AI Generative Review Utility.")
    parser.add_argument(
        "--pilots-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "pilots"),
        help="Directory containing diffusion_v1 and controlnet_canny_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1"),
        help="Output directory for index.html and review_inventory.json",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Do not embed base64 image data into HTML (use relative file paths)",
    )
    args = parser.parse_args()

    pilots_dir = Path(args.pilots_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_base64 = not args.no_embed

    print(f"Building review inventory from {pilots_dir}...")
    inventory = build_inventory(PROJECT_ROOT, pilots_dir, embed_base64=embed_base64)

    # 1. Save review_inventory.json (without huge base64 strings for clean readability)
    inventory_clean = dict(inventory)
    groups_clean = []
    for g in inventory["groups"]:
        g_c = dict(g)
        g_c.pop("parent_data_uri", None)
        g_c.pop("canny_data_uri", None)
        outs_clean = []
        for o in g["outputs"]:
            o_c = dict(o)
            o_c.pop("data_uri", None)
            outs_clean.append(o_c)
        g_c["outputs"] = outs_clean
        groups_clean.append(g_c)
    inventory_clean["groups"] = groups_clean

    inv_json_path = output_dir / "review_inventory.json"
    with open(inv_json_path, "w", encoding="utf-8") as jf:
        json.dump(inventory_clean, jf, indent=2)
    print(f"Saved review inventory: {inv_json_path.name} ({len(inventory['groups'])} groups, {inventory['total_candidates']} candidates)")

    # 2. Generate and save index.html
    print(f"Generating self-contained review interface (embed_base64={embed_base64})...")
    html_content = generate_html_content(inventory, embed_base64=embed_base64)
    index_html_path = output_dir / "index.html"
    with open(index_html_path, "w", encoding="utf-8") as hf:
        hf.write(html_content)
    print(f"Saved review interface: {index_html_path.name} ({len(html_content)/(1024*1024):.2f} MB)")

    # 3. Save README.md
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as rf:
        rf.write(build_readme())
    print(f"Saved README: {readme_path.name}")

    print("\nGenerative Review Utility successfully reconstructed.")
    print(f"Review interface path: {index_html_path}")


if __name__ == "__main__":
    main()
