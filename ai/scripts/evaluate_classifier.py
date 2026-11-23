#!/usr/bin/env python3
"""
Model Evaluation Script for PipeAugment AI.
Loads a saved checkpoint and evaluates performance against an arbitrary manifest
(validation or locked test manifest).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_ml.dataset import PipeAugmentDataset, get_baseline_transforms
from pipeaugment_ml.model import build_classifier
from pipeaugment_ml.metrics import compute_classification_metrics, plot_confusion_matrix


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int = 16,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoint on device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    class_names = ckpt.get("class_names", ["non_corroded", "corroded"])

    # Build model architecture and load weights
    model, _, transforms = build_classifier(
        model_name=config.get("model", {}).get("architecture", "resnet18"),
        num_classes=len(class_names),
        pretrained=False
    )
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transforms = get_baseline_transforms()
    dataset = PipeAugmentDataset(manifest_path, dataset_root, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    detailed_preds: List[Dict[str, Any]] = []

    for images, targets, metas in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds_cpu = preds.cpu().tolist()
        targets_cpu = targets.cpu().tolist()
        probs_cpu = probs.cpu().tolist()

        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)

        for i in range(len(preds_cpu)):
            detailed_preds.append({
                "image_id": metas["image_id"][i],
                "relative_path": metas["relative_path"][i],
                "ground_truth_label": metas["label"][i],
                "ground_truth_index": targets_cpu[i],
                "predicted_index": preds_cpu[i],
                "predicted_label": class_names[preds_cpu[i]],
                "correct": preds_cpu[i] == targets_cpu[i],
                "prob_non_corroded": probs_cpu[i][0],
                "prob_corroded": probs_cpu[i][1],
            })

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    metrics["loss"] = avg_loss

    # Save evaluation outputs
    eval_name = manifest_path.stem
    metrics_file = output_dir / f"{eval_name}_metrics.json"
    preds_file = output_dir / f"{eval_name}_predictions.csv"
    cm_file = output_dir / f"{eval_name}_confusion_matrix.png"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(detailed_preds).to_csv(preds_file, index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_file, title=f"Evaluation ({eval_name})")

    print(f"Evaluation complete on {manifest_path.name}: Loss={avg_loss:.4f}, Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier checkpoint on manifest.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path")
    parser.add_argument("--dataset-root", type=str, default=str(PROJECT_ROOT / "datasets" / "raw" / "k-pipelines"), help="Dataset raw root")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )
