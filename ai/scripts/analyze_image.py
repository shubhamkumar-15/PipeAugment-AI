"""
PipeAugment AI — Command-Line Image Corrosion Analysis & Layout Map Generator.
Executes ResNet-18 inspection on a target image and generates 3 visual layout maps.
Outputs JSON summary to stdout.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

# Ensure ai directory is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_ml.inspector import PipeAugmentInspector


def main():
    parser = argparse.ArgumentParser(description="PipeAugment AI Corrosion Image Inspector")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image file")
    parser.add_argument("--image-id", type=str, default=None, help="Identifier for the image")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for analysis results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to ResNet-18 checkpoint .pth")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device ('cuda' or 'cpu')")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        sys.stderr.write(f"Image file not found: {image_path}\n")
        sys.exit(1)

    image_id = args.image_id or image_path.stem

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = PROJECT_ROOT / "datasets" / "metadata" / "analysis"

    analysis_dir = base_output_dir / image_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess image
    with Image.open(image_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img_rgb = np.array(pil_img)

    inspector = PipeAugmentInspector(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    analysis_data, cam_map, rust_mask = inspector.analyze(img_rgb)
    maps = inspector.generate_layout_maps(
        image_rgb=img_rgb,
        output_dir=analysis_dir,
        prefix="",
        cam_map=cam_map,
        rust_mask=rust_mask,
    )

    result_payload = {
        "image_id": image_id,
        **analysis_data,
        "map_files": {k: str(v) for k, v in maps.items()},
        "map_urls": {
            "pipeline_canny": f"/api/v1/images/{image_id}/maps/pipeline_canny",
            "corrosion_layout": f"/api/v1/images/{image_id}/maps/corrosion_layout",
            "corrosion_overlay": f"/api/v1/images/{image_id}/maps/corrosion_overlay",
        },
    }

    # Save to both analysis_dir/analysis.json and base_output_dir/{image_id}.json for fast lookup
    json_path1 = analysis_dir / "analysis.json"
    with open(json_path1, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    json_path2 = base_output_dir / f"{image_id}.json"
    with open(json_path2, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    # Print JSON payload to stdout
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
