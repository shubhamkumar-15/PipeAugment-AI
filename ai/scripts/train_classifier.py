#!/usr/bin/env python3
"""
Baseline Classifier Training Script for PipeAugment AI.
Trains ResNet-18 baseline on K-Pipelines Standard dataset.
Implements 2-epoch head warmup, fine-tuning with AdamW, early stopping,
checkpoint saving, and comprehensive metric logging.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure ai directory is on python path
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_ml.reproducibility import set_seed
from pipeaugment_ml.dataset import PipeAugmentDataset, get_baseline_transforms
from pipeaugment_ml.model import build_classifier, freeze_backbone, unfreeze_backbone
from pipeaugment_ml.metrics import compute_classification_metrics, plot_confusion_matrix, plot_training_curves


def get_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_environment_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": str(device),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    return info


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    detailed_preds: List[Dict[str, Any]] = []

    for images, targets, metas in dataloader:
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

        # Batch metadata
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
    return avg_loss, metrics, detailed_preds


def run_training(
    config_path: Path,
    output_dir: Optional[Path] = None,
    smoke_test: bool = False,
    resume_checkpoint: Optional[Path] = None,
    seed_override: Optional[int] = None,
) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Initialize seed
    seed = seed_override if seed_override is not None else config["training"].get("seed", 42)
    config["training"]["seed"] = seed
    set_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Paths
    raw_root = PROJECT_ROOT / config["dataset"]["raw_root"]
    train_manifest = PROJECT_ROOT / config["dataset"]["train_manifest"]
    val_manifest = PROJECT_ROOT / config["dataset"]["val_manifest"]
    class_mapping_path = PROJECT_ROOT / config["dataset"]["class_mapping"]

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_mapping_data = json.load(f)
    class_names = [class_mapping_data["index_to_class"][str(i)] for i in range(len(class_mapping_data["index_to_class"]))]

    # Setup Run Directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment_name", "baseline_resnet18")
    if output_dir is None:
        run_name = f"smoke_test_{timestamp}" if smoke_test else f"{exp_name}_seed{seed}_{timestamp}"
        run_dir = PROJECT_ROOT / "experiments" / "runs" / run_name
    else:
        run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "training.log"
    def log_print(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")

    log_print(f"Starting PipeAugment AI training run: {run_dir.name}")
    log_print(f"Smoke test mode: {smoke_test}")

    # Build Model
    model, weights_id, transforms = build_classifier(
        model_name=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    )
    model = model.to(device)
    log_print(f"Model architecture: {config['model']['architecture']} with weights: {weights_id}")

    # Datasets
    train_dataset = PipeAugmentDataset(train_manifest, raw_root, transform=transforms)
    val_dataset = PipeAugmentDataset(val_manifest, raw_root, transform=transforms)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    if smoke_test:
        train_dataset = Subset(train_dataset, list(range(min(16, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(8, len(val_dataset)))))
        max_epochs = 1
        warmup_epochs = 1
        batch_size = 4
    else:
        max_epochs = config["training"]["max_epochs"]
        warmup_epochs = config["training"]["warmup_epochs"]

    samples_per_epoch = config["training"].get("samples_per_epoch", None)
    if samples_per_epoch is not None and not smoke_test:
        log_print(f"Matched-budget training enabled: {samples_per_epoch} samples per epoch (balanced 50/50 across classes) drawn from pool of {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    log_print(f"Train samples (or pool): {len(train_dataset)}, Validation samples: {len(val_dataset)}, Batch size: {batch_size}")

    # Save Pre-Run Metadata
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(get_environment_info(device), f, indent=2)

    with open(run_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping_data, f, indent=2)

    with open(run_dir / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_manifest_sha256": get_file_sha256(train_manifest),
            "val_manifest_sha256": get_file_sha256(val_manifest),
        }, f, indent=2)

    criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler Setup
    warmup_lr = config["training"]["warmup_lr"]
    finetune_lr = config["training"]["finetune_lr"]
    weight_decay = config["training"]["weight_decay"]

    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    best_preds: List[Dict[str, Any]] = []
    patience_counter = 0
    early_stopping_patience = config["training"]["early_stopping_patience"]

    history: List[Dict[str, Any]] = []
    metrics_csv = run_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        f.write("epoch,phase,train_loss,val_loss,val_accuracy,val_macro_f1,val_macro_precision,val_macro_recall,learning_rate,epoch_time_seconds\n")

    current_phase = "warmup"
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=warmup_lr, weight_decay=weight_decay)

    total_optimizer_steps = 0
    start_epoch = 1
    # Checkpoint restoration if resuming
    if resume_checkpoint and Path(resume_checkpoint).exists():
        log_print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        total_optimizer_steps = ckpt.get("total_optimizer_steps", 0)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # Switch from Warmup to Fine-Tuning
        if epoch > warmup_epochs and current_phase == "warmup":
            current_phase = "finetune"
            log_print(f"--- Epoch {epoch}: Transitioning to full fine-tuning. Unfreezing backbone parameters ---")
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)

        curr_lr = optimizer.param_groups[0]["lr"]

        # Determine train dataloader for this epoch (budget matching if enabled)
        if samples_per_epoch is not None and not smoke_test:
            epoch_rng = np.random.default_rng(seed + epoch * 1000)
            k_per_class = samples_per_epoch // 2
            selected_indices = []

            has_source_types = (
                "source_type" in train_dataset.df.columns
                and len(train_dataset.df["source_type"].dropna().unique()) > 1
            )

            for c_idx in [0, 1]:
                class_mask = (train_dataset.df["class_index"] == c_idx)
                if has_source_types:
                    source_types = sorted(train_dataset.df.loc[class_mask, "source_type"].dropna().unique())
                    target_per_source = k_per_class // len(source_types)
                    c_selected = []
                    for st in source_types:
                        st_indices = np.where(class_mask & (train_dataset.df["source_type"] == st))[0]
                        n_sample = min(len(st_indices), target_per_source)
                        c_selected.extend(epoch_rng.choice(st_indices, size=n_sample, replace=False).tolist())

                    if len(c_selected) < k_per_class:
                        rem_pool = list(set(np.where(class_mask)[0]) - set(c_selected))
                        needed = k_per_class - len(c_selected)
                        c_selected.extend(epoch_rng.choice(rem_pool, size=needed, replace=False).tolist())

                    selected_indices.extend(c_selected)
                else:
                    c_all = np.where(class_mask)[0]
                    sub = epoch_rng.choice(c_all, size=k_per_class, replace=False)
                    selected_indices.extend(sub.tolist())

            epoch_idx = np.array(selected_indices)
            epoch_rng.shuffle(epoch_idx)
            current_train_loader = DataLoader(
                Subset(train_dataset, epoch_idx.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
        else:
            current_train_loader = train_loader

        # Train & Evaluate
        train_loss = train_one_epoch(model, current_train_loader, criterion, optimizer, device)
        total_optimizer_steps += len(current_train_loader)
        val_loss, val_metrics, val_preds = evaluate(model, val_loader, criterion, device, class_names)
        epoch_time = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        log_print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{current_phase}] - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log epoch record
        epoch_record = {
            "epoch": epoch,
            "phase": current_phase,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "learning_rate": curr_lr,
            "epoch_time_seconds": epoch_time,
        }
        history.append(epoch_record)

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_phase},{train_loss:.6f},{val_loss:.6f},"
                f"{val_acc:.6f},{val_f1:.6f},{val_metrics['macro_precision']:.6f},"
                f"{val_metrics['macro_recall']:.6f},{curr_lr:.6e},{epoch_time:.2f}\n"
            )

        # Update best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            best_preds = val_preds
            patience_counter = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "config": config,
                "class_names": class_names,
                "total_optimizer_steps": total_optimizer_steps,
            }, run_dir / "best_model.pth")
            log_print(f"  >>> Best model saved at epoch {epoch} with validation macro-F1: {val_f1:.4f}")
        else:
            if current_phase == "finetune":
                patience_counter += 1
                log_print(f"  Patience counter: {patience_counter}/{early_stopping_patience}")

        # Always save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "config": config,
            "history": history,
            "total_optimizer_steps": total_optimizer_steps,
        }, run_dir / "last_checkpoint.pth")

        # Early stopping check
        if current_phase == "finetune" and patience_counter >= early_stopping_patience:
            log_print(f"Early stopping triggered after {patience_counter} fine-tuning epochs without improvement.")
            break

    # Save Final Artifacts
    log_print(f"Training completed. Best validation macro-F1: {best_val_f1:.4f} at epoch {best_epoch}.")

    with open(run_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    pd.DataFrame(best_preds).to_csv(run_dir / "validation_predictions.csv", index=False)

    if best_metrics and "confusion_matrix" in best_metrics:
        plot_confusion_matrix(
            best_metrics["confusion_matrix"],
            class_names,
            run_dir / "validation_confusion_matrix.png",
            title=f"Validation Confusion Matrix (Epoch {best_epoch})"
        )

    if history:
        plot_training_curves(history, run_dir / "training_curves.png")

    steps_per_epoch = len(current_train_loader) if 'current_train_loader' in locals() else len(train_loader)
    cm = best_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    val_errors = int(np.sum(cm) - np.trace(cm))

    training_summary = {
        "experiment_name": exp_name,
        "dataset_name": config["dataset"]["name"],
        "training_pool_size": len(train_dataset),
        "samples_per_epoch": samples_per_epoch if (samples_per_epoch is not None and not smoke_test) else len(train_dataset),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "actual_total_steps": total_optimizer_steps,
        "val_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "best_val_accuracy": best_metrics.get("accuracy", 0.0),
        "corrosion_precision": best_metrics.get("per_class", {}).get("corroded", {}).get("precision", 0.0),
        "corrosion_recall": best_metrics.get("per_class", {}).get("corroded", {}).get("recall", 0.0),
        "validation_errors_count": val_errors,
        "total_epochs_trained": len(history),
        "early_stopping_triggered": bool(current_phase == "finetune" and patience_counter >= early_stopping_patience),
    }

    with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    log_print("All artifacts successfully saved.")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline ResNet-18 classifier on K-Pipelines.")
    parser.add_argument("--config", type=str, default=str(AI_DIR / "configs" / "baseline_resnet18.json"), help="Config JSON path")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test on subset")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    args = parser.parse_args()

    run_training(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        smoke_test=args.smoke_test,
        resume_checkpoint=Path(args.resume) if args.resume else None,
        seed_override=args.seed,
    )
