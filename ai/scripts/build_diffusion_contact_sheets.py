#!/usr/bin/env python3
"""
Build high-resolution contact sheets for PipeAugment AI Generative Pilot.
Creates one detailed sheet per parent (6 sheets total) showing:
- Original Source Image
- Low-Light Inspection @ Strength 0.20
- Low-Light Inspection @ Strength 0.35
- Directional Industrial Lighting @ Strength 0.20
- Directional Industrial Lighting @ Strength 0.35
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))


def build_parent_contact_sheet(
    parent_id: str,
    parent_label: str,
    parent_img_path: Path,
    variants: List[Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Renders a high-resolution contact sheet for a single parent.
    Layout: 1 row of 5 high-res panels (or 2 rows of 3 panels).
    Here we use a 2-row layout with 384x384 thumbnails:
    Row 1: [Original], [Low-Light s=0.20], [Low-Light s=0.35]
    Row 2: [Empty/Legend], [Directional s=0.20], [Directional s=0.35]
    """
    thumb_w, thumb_h = 400, 400
    caption_h = 56
    cell_h = thumb_h + caption_h
    cell_w = thumb_w
    padding = 24
    header_h = 60

    sheet_w = padding * 2 + cell_w * 3 + padding * 2
    sheet_h = header_h + cell_h * 2 + padding * 2

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(11, 16, 26))
    draw = ImageDraw.Draw(sheet)

    # Header
    title = f"PipeAugment AI — Diffusion Pilot: Parent {parent_id[:24]} ({parent_label.upper()})"
    draw.text((padding, 18), title, fill=(255, 90, 0))
    subtitle = "Pretrained SD 1.5 Image-to-Image | Preservation vs. Environmental Condition Inspection"
    draw.text((padding, 38), subtitle, fill=(160, 180, 205))

    # Load parent image
    with Image.open(parent_img_path) as p_img:
        p_thumb = p_img.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

    # Panel definitions: (row, col, img, title_text, detail_text)
    panels = []
    # Panel 0: Original
    panels.append((
        0, 0, p_thumb,
        f"Original: {parent_label}",
        f"512x512 | SHA: {parent_id[:12]}"
    ))

    # Map variants
    cond_order = [
        ("low_light_inspection", 0.20, 0, 1),
        ("low_light_inspection", 0.35, 0, 2),
        ("directional_industrial_lighting", 0.20, 1, 1),
        ("directional_industrial_lighting", 0.35, 1, 2),
    ]

    for cond, str_val, r_idx, c_idx in cond_order:
        match = next((v for v in variants if v["condition"] == cond and abs(v["strength"] - str_val) < 0.01), None)
        if match and Path(match["image_path"]).exists():
            with Image.open(match["image_path"]) as v_img:
                v_thumb = v_img.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            c_name = "Low-Light" if "low_light" in cond else "Directional"
            steps = match.get("effective_steps", int(30 * str_val))
            ssim_val = match.get("ssim", 0.0)
            panels.append((
                r_idx, c_idx, v_thumb,
                f"{c_name} (strength={str_val:.2f})",
                f"Steps: {steps} | Seed: {match.get('seed', 'N/A')} | SSIM: {ssim_val:.4f}"
            ))

    # Draw Legend in (1, 0)
    legend_x = padding
    legend_y = header_h + cell_h + padding
    draw.rectangle([legend_x, legend_y, legend_x + cell_w, legend_y + thumb_h], fill=(16, 24, 38), outline=(30, 45, 70))
    draw.text((legend_x + 20, legend_y + 30), "PILOT INSPECTION CRITERIA", fill=(255, 155, 0))
    legend_text = (
        "1. Geometry Preserved?\n"
        "   Does pipe shape, curve, joints\n"
        "   and background remain intact?\n\n"
        "2. Corrosion Preserved?\n"
        "   Are pit locations and rust\n"
        "   textures retained without\n"
        "   invented or erased damage?\n\n"
        "3. Condition Visible?\n"
        "   Is low-light or directional\n"
        "   illumination perceptible?\n\n"
        "Decision: Record in review.csv"
    )
    draw.text((legend_x + 20, legend_y + 60), legend_text, fill=(200, 215, 235))

    # Paste panels
    for r_idx, c_idx, img_obj, title_txt, detail_txt in panels:
        x = padding + c_idx * (cell_w + padding)
        y = header_h + r_idx * (cell_h + padding)

        # Image border
        draw.rectangle([x - 1, y - 1, x + thumb_w + 1, y + thumb_h + 1], outline=(40, 60, 90))
        sheet.paste(img_obj, (x, y))

        # Captions
        draw.text((x + 2, y + thumb_h + 6), title_txt, fill=(255, 255, 255))
        draw.text((x + 2, y + thumb_h + 28), detail_txt, fill=(150, 175, 205))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path
