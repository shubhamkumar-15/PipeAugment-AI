#!/usr/bin/env python3
"""
Build high-resolution side-by-side comparison sheets for PipeAugment AI Canny-ControlNet Pilot.
Creates 12 comparative sheets (6 parents x 2 conditions) comparing:
Panel 1: Original Source Image
Panel 2: Source Canny Edge Map (100, 200)
Panel 3: Matched Plain SD 1.5 img2img (strength=0.35) from diffusion_v1
Panel 4: ControlNet Canny (scale=0.50, strength=0.35)
Panel 5: ControlNet Canny (scale=0.80, strength=0.35)
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))


def build_comparison_sheet(
    parent_id: str,
    parent_label: str,
    condition: str,
    parent_img_path: Path,
    canny_map_path: Path,
    plain_img_path: Optional[Path],
    plain_meta: Optional[Dict[str, Any]],
    cnet_05_path: Optional[Path],
    cnet_05_meta: Optional[Dict[str, Any]],
    cnet_08_path: Optional[Path],
    cnet_08_meta: Optional[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Renders a 5-panel side-by-side horizontal comparison sheet for a parent and condition.
    """
    thumb_w, thumb_h = 360, 360
    caption_h = 75
    cell_h = thumb_h + caption_h
    cell_w = thumb_w
    padding_x = 24
    padding_y = 20
    panel_gap = 16
    header_h = 70
    footer_h = 65

    sheet_w = padding_x * 2 + cell_w * 5 + panel_gap * 4
    sheet_h = header_h + cell_h + footer_h + padding_y * 2

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(11, 16, 26))
    draw = ImageDraw.Draw(sheet)

    # Header
    cond_display = condition.replace("_", " ").title()
    title = f"PipeAugment AI — ControlNet Canny Comparison | Parent: {parent_id[:20]} | Class: {parent_label.upper()} | Condition: {cond_display}"
    draw.text((padding_x, padding_y), title, fill=(255, 90, 0))
    subtitle = "Side-by-side evaluation: Plain img2img vs Canny structural conditioning (scales 0.50, 0.80) at strength 0.35"
    draw.text((padding_x, padding_y + 24), subtitle, fill=(160, 180, 205))

    # Load images
    def load_thumb(p: Optional[Path]) -> Optional[Image.Image]:
        if p and p.exists():
            with Image.open(p) as img:
                return img.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        return None

    orig_thumb = load_thumb(parent_img_path)
    canny_thumb = load_thumb(canny_map_path)
    plain_thumb = load_thumb(plain_img_path)
    c05_thumb = load_thumb(cnet_05_path)
    c08_thumb = load_thumb(cnet_08_path)

    # Panel definitions
    plain_ssim = f"{plain_meta.get('ssim_rgb', 0.0):.4f}" if plain_meta else "N/A"
    c05_ssim = f"{cnet_05_meta.get('ssim_rgb', 0.0):.4f}" if cnet_05_meta else "N/A"
    c05_iou = f"{cnet_05_meta.get('edge_overlap_iou_heuristic', 0.0):.4f}" if cnet_05_meta else "N/A"
    c08_ssim = f"{cnet_08_meta.get('ssim_rgb', 0.0):.4f}" if cnet_08_meta else "N/A"
    c08_iou = f"{cnet_08_meta.get('edge_overlap_iou_heuristic', 0.0):.4f}" if cnet_08_meta else "N/A"

    panels = [
        (
            orig_thumb,
            "1. Original Source Image",
            f"Class: {parent_label} | Size: 512x512",
            f"ID: {parent_id[:16]}...",
            (255, 255, 255)
        ),
        (
            canny_thumb,
            "2. Canny Edge Map",
            "Thresholds: [100, 200]",
            "Condition input to ControlNet",
            (0, 220, 255)
        ),
        (
            plain_thumb,
            "3. Plain SD 1.5 img2img",
            f"Strength: 0.35 | Steps: 10",
            f"SSIM: {plain_ssim}",
            (255, 180, 50)
        ),
        (
            c05_thumb,
            "4. ControlNet Canny (c=0.50)",
            f"Strength: 0.35 | Scale: 0.50",
            f"SSIM: {c05_ssim} | Edge IoU: {c05_iou}",
            (100, 220, 100)
        ),
        (
            c08_thumb,
            "5. ControlNet Canny (c=0.80)",
            f"Strength: 0.35 | Scale: 0.80",
            f"SSIM: {c08_ssim} | Edge IoU: {c08_iou}",
            (50, 255, 150)
        ),
    ]

    base_y = header_h + padding_y

    for idx, (thumb, p_title, line1, line2, color) in enumerate(panels):
        x = padding_x + idx * (cell_w + panel_gap)
        y = base_y

        # Draw panel box border
        draw.rectangle([x - 2, y - 2, x + cell_w + 1, y + thumb_h + 1], outline=(40, 55, 80), width=1)

        if thumb is not None:
            sheet.paste(thumb, (x, y))
        else:
            draw.rectangle([x, y, x + cell_w, y + thumb_h], fill=(20, 25, 35))
            draw.text((x + 20, y + thumb_h // 2), "[Image Not Available]", fill=(120, 140, 160))

        # Draw caption
        cap_y = y + thumb_h + 8
        draw.text((x, cap_y), p_title, fill=color)
        draw.text((x, cap_y + 20), line1, fill=(200, 215, 235))
        draw.text((x, cap_y + 38), line2, fill=(140, 160, 185))

    # Footer banner
    footer_y = base_y + cell_h + 10
    draw.rectangle([padding_x, footer_y, sheet_w - padding_x, footer_y + footer_h], fill=(16, 24, 38), outline=(35, 50, 75))
    f_line1 = "NOTE: SSIM and Edge Overlap IoU are descriptive heuristics, NOT proof of structural or corrosion preservation."
    f_line2 = "Human domain review criteria: (1) Geometry preserved (2) Corrosion preserved without hallucination (3) Lighting condition visible (4) Realism acceptable."
    draw.text((padding_x + 16, footer_y + 12), f_line1, fill=(255, 160, 30))
    draw.text((padding_x + 16, footer_y + 34), f_line2, fill=(180, 200, 220))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path
