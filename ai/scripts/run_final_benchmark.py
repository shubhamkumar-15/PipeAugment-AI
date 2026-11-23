"""
PipeAugment AI — Phase 7 Final Evaluation Benchmark & Robustness Stress Suite.
Evaluates 9 trained checkpoints (A, B, C x seeds 42, 101, 202) on:
  1. Clean held-out test split (60 images)
  2. 6 controlled synthetic perturbation stress conditions (applied strictly at inference time)
Exports:
  - experiments/final_evaluation/final_benchmark_results.json
  - experiments/final_evaluation/per_run_metrics.csv
  - experiments/final_evaluation/aggregated_summary.csv
  - experiments/final_evaluation/stress_suite_preview.png
  - experiments/final_evaluation/charts/abc_clean_vs_stress_f1.png
  - experiments/final_evaluation/charts/abc_corrosion_recall_robustness.png
  - experiments/final_evaluation/charts/abc_confusion_matrices_clean.png
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights

# Path resolution
CURRENT_DIR = Path(__file__).resolve().parent
AI_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_cv.stress_suite import (
    apply_stress_condition,
    CONDITIONS,
    CONDITION_KEYS,
)
from pipeaugment_ml.model import build_classifier
from pipeaugment_ml.metrics import compute_classification_metrics


EXPERIMENTS = {
    "A": {
        "key": "A",
        "name": "Baseline A",
        "prefix": "baseline_resnet18",
        "description": "Clean industrial originals only (ResNet-18 ImageNet-1K V1)",
        "pool_size": 480,
    },
    "B": {
        "key": "B",
        "name": "Experiment B",
        "prefix": "traditional_resnet18",
        "description": "Originals + Traditional CV augmentations (960 transforms)",
        "pool_size": 1440,
    },
    "C": {
        "key": "C",
        "name": "Experiment C (PipeAugment AI)",
        "prefix": "pipeaugment_resnet18",
        "description": "Originals + Traditional CV + Quality-Screened ControlNet Canny",
        "pool_size": 1609,
    },
}

SEEDS = [42, 101, 202]
CLASS_NAMES = ["non_corroded", "corroded"]


class MemoryTestDataset(Dataset):
    """Loads pre-transformed in-memory test images and applies torchvision eval transforms."""

    def __init__(self, images: List[np.ndarray], labels: List[int], metas: List[Dict[str, Any]], transform):
        self.images = images
        self.labels = labels
        self.metas = metas
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        img_np = self.images[idx]
        pil_img = Image.fromarray(img_np)
        tensor = self.transform(pil_img)
        return tensor, self.labels[idx], self.metas[idx]


def verify_dataset_isolation(test_path: Path) -> Dict[str, Any]:
    """Strictly checks SHA-256 and zero leakage before unsealing test set."""
    with open(test_path, "rb") as f:
        test_sha256 = hashlib.sha256(f.read()).hexdigest()

    test_df = pd.read_csv(test_path)
    test_ids = set(test_df["image_id"])
    test_hashes = set(test_df["sha256"])

    manifests = [
        ("train.csv", PROJECT_ROOT / "datasets" / "manifests" / "train.csv"),
        ("val.csv", PROJECT_ROOT / "datasets" / "manifests" / "val.csv"),
        ("train_traditional_v1.csv", PROJECT_ROOT / "datasets" / "manifests" / "train_traditional_v1.csv"),
        ("train_pipeaugment_v1.csv", PROJECT_ROOT / "datasets" / "manifests" / "train_pipeaugment_v1.csv"),
    ]

    leakage_report = {}
    for name, path in manifests:
        df = pd.read_csv(path)
        id_overlap = len(test_ids.intersection(set(df["image_id"])))
        hash_overlap = len(test_hashes.intersection(set(df["sha256"])))
        if id_overlap > 0 or hash_overlap > 0:
            raise RuntimeError(f"LEAKAGE VIOLATION: {name} shares {id_overlap} IDs / {hash_overlap} hashes with test.csv!")
        leakage_report[name] = {"rows": len(df), "id_overlap": id_overlap, "hash_overlap": hash_overlap}

    return {
        "test_sha256": test_sha256,
        "sample_count": len(test_df),
        "class_distribution": test_df["label"].value_counts().to_dict(),
        "leakage_audit": leakage_report,
    }


def find_checkpoints(runs_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Discovers and validates all 9 checkpoint directories."""
    checkpoints = {}
    for exp_key, exp_info in EXPERIMENTS.items():
        checkpoints[exp_key] = {}
        for s in SEEDS:
            prefix = exp_info["prefix"]
            matches = sorted(list(runs_dir.glob(f"{prefix}_seed{s}_*")))
            if not matches:
                raise FileNotFoundError(f"Missing run directory for {prefix} seed {s}")
            run_dir = matches[-1]
            ckpt_file = run_dir / "best_model.pth"
            if not ckpt_file.exists():
                raise FileNotFoundError(f"Missing checkpoint file: {ckpt_file}")
            
            # Inspect checkpoint
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            checkpoints[exp_key][s] = {
                "dir_name": run_dir.name,
                "path": str(ckpt_file),
                "best_epoch": int(ckpt.get("epoch", -1)),
                "val_macro_f1": float(ckpt.get("val_macro_f1", 0.0)),
                "val_accuracy": float(ckpt.get("val_accuracy", 0.0)),
            }
    return checkpoints


def generate_stress_preview(
    test_df: pd.DataFrame,
    raw_root: Path,
    output_path: Path,
) -> None:
    """Creates a 4-sample reference strip showing all 7 evaluation conditions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Pick 2 non_corroded and 2 corroded samples
    nc_samples = test_df[test_df["label"] == "non_corroded"].head(2)
    c_samples = test_df[test_df["label"] == "corroded"].head(2)
    selected = pd.concat([nc_samples, c_samples])

    n_samples = len(selected)
    n_conds = len(CONDITION_KEYS)

    fig, axes = plt.subplots(n_samples, n_conds, figsize=(n_conds * 2.6, n_samples * 2.6), dpi=150)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for row_idx, (_, row) in enumerate(selected.iterrows()):
        img_p = raw_root / row["relative_path"]
        pil_img = Image.open(img_p)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        raw_rgb = np.array(pil_img)
        img_id = row["image_id"]
        label = row["label"]

        for col_idx, cond in enumerate(CONDITION_KEYS):
            ax = axes[row_idx, col_idx]
            transformed = apply_stress_condition(raw_rgb, cond, img_id)
            ax.imshow(transformed)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                short_title = CONDITIONS[cond]["display_name"]
                ax.set_title(short_title, fontsize=8, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n({img_id[:8]}...)", fontsize=8, fontweight="bold")

    fig.suptitle(
        "PipeAugment AI — Evaluation Conditions Preview\n(Clean Held-Out Test vs. 6 Controlled Stress Perturbations)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Stress preview saved to: {output_path}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs model inference and computes classification metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, metas in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds, CLASS_NAMES)
    cm = metrics["confusion_matrix"]
    # cm: row=true [non_corroded, corroded], col=pred [non_corroded, corroded]
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    corr_prec = metrics["per_class"]["corroded"]["precision"]
    corr_rec = metrics["per_class"]["corroded"]["recall"]
    corr_f1 = metrics["per_class"]["corroded"]["f1_score"]

    nc_prec = metrics["per_class"]["non_corroded"]["precision"]
    nc_rec = metrics["per_class"]["non_corroded"]["recall"]
    nc_f1 = metrics["per_class"]["non_corroded"]["f1_score"]

    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "corrosion_precision": float(corr_prec),
        "corrosion_recall": float(corr_rec),
        "corrosion_f1": float(corr_f1),
        "non_corroded_precision": float(nc_prec),
        "non_corroded_recall": float(nc_rec),
        "non_corroded_f1": float(nc_f1),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": cm,
    }


def plot_clean_vs_stress_f1(aggregated_summary: List[Dict[str, Any]], output_path: Path) -> None:
    """Grouped bar chart showing Macro-F1 across all 7 conditions for A, B, C."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    cond_labels = [CONDITIONS[c]["display_name"] for c in CONDITION_KEYS]
    x = np.arange(len(CONDITION_KEYS))
    width = 0.26

    # Map records by (exp, cond)
    data_map = {(r["experiment_key"], r["condition_name"]): r for r in aggregated_summary}

    colors = {"A": "#64748B", "B": "#3B82F6", "C": "#F59E0B"}

    for i, exp_k in enumerate(["A", "B", "C"]):
        means = [data_map[(exp_k, c)]["macro_f1_mean"] * 100 for c in CONDITION_KEYS]
        stds = [data_map[(exp_k, c)]["macro_f1_std"] * 100 for c in CONDITION_KEYS]
        label = EXPERIMENTS[exp_k]["name"]
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            color=colors[exp_k],
            edgecolor="#1E293B",
            linewidth=1.0,
            alpha=0.92,
        )

    ax.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "PipeAugment AI — Macro-F1 Under Clean and Stress Conditions (Mean ± Std, 3 Seeds)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=20, ha="right", fontsize=9, fontweight="medium")
    ax.set_ylim(50, 105)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_corrosion_recall_robustness(aggregated_summary: List[Dict[str, Any]], output_path: Path) -> None:
    """Chart displaying Corrosion Recall (detection sensitivity) across all conditions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    cond_labels = [CONDITIONS[c]["display_name"] for c in CONDITION_KEYS]
    x = np.arange(len(CONDITION_KEYS))
    data_map = {(r["experiment_key"], r["condition_name"]): r for r in aggregated_summary}

    colors = {"A": "#64748B", "B": "#3B82F6", "C": "#F59E0B"}
    markers = {"A": "o", "B": "s", "C": "D"}

    for exp_k in ["A", "B", "C"]:
        means = [data_map[(exp_k, c)]["corrosion_recall_mean"] * 100 for c in CONDITION_KEYS]
        stds = [data_map[(exp_k, c)]["corrosion_recall_std"] * 100 for c in CONDITION_KEYS]
        label = EXPERIMENTS[exp_k]["name"]
        ax.errorbar(
            x,
            means,
            yerr=stds,
            label=label,
            color=colors[exp_k],
            marker=markers[exp_k],
            markersize=8,
            linewidth=2.2,
            capsize=4,
            alpha=0.95,
        )

    ax.set_ylabel("Corrosion Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "PipeAugment AI — Corrosion Recall Across Environmental Stress Conditions (Mean ± Std)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=20, ha="right", fontsize=9, fontweight="medium")
    ax.set_ylim(60, 105)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(axis="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_side_by_side_confusion_matrices(
    clean_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plots side-by-side aggregated confusion matrices for A, B, C on clean test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    for i, exp_k in enumerate(["A", "B", "C"]):
        ax = axes[i]
        cm = np.array(clean_summary[exp_k]["summed_confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues if exp_k != "C" else plt.cm.Oranges)

        ax.set_title(f"{EXPERIMENTS[exp_k]['name']}\n(3 Seeds Summed: N={cm.sum()})", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Corroded", "Corroded"], fontsize=9)
        ax.set_yticklabels(["Non-Corroded", "Corroded"], fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
        if i == 0:
            ax.set_ylabel("True Label", fontsize=9, fontweight="bold")

        thresh = cm.max() / 2.0
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = cm[r, c]
                ax.text(
                    c, r, f"{val}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold", fontsize=12
                )

    fig.suptitle("Clean Test Split — Summed Confusion Matrices (Seeds 42, 101, 202)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    print("=" * 65)
    print("PIPEAUGMENT AI — PHASE 7 FINAL EVALUATION BENCHMARK")
    print("=" * 65)

    test_csv = PROJECT_ROOT / "datasets" / "manifests" / "test.csv"
    raw_root = PROJECT_ROOT / "datasets" / "raw" / "k-pipelines"
    runs_dir = PROJECT_ROOT / "experiments" / "runs"
    eval_out_dir = PROJECT_ROOT / "experiments" / "final_evaluation"
    charts_dir = eval_out_dir / "charts"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Isolation and Leakage Audit
    print("\n[Step 1/5] Verifying test dataset isolation and leakage assertions...")
    isolation_info = verify_dataset_isolation(test_csv)
    print(f"  Test CSV SHA-256: {isolation_info['test_sha256']}")
    print(f"  Test Sample Count: {isolation_info['sample_count']} images")
    print(f"  Class Distribution: {isolation_info['class_distribution']}")
    print("  Zero leakage confirmed across all training and validation manifests.")

    # 2. Discover Checkpoints
    print("\n[Step 2/5] Locating and verifying all 9 trained checkpoints...")
    checkpoints = find_checkpoints(runs_dir)
    for exp_k, seeds_dict in checkpoints.items():
        for s, info in seeds_dict.items():
            print(f"  Exp {exp_k} (seed {s:3d}): best epoch {info['best_epoch']} in {info['dir_name']}")

    # 3. Load Raw Test Images in Memory
    print("\n[Step 3/5] Loading 60 test images into memory...")
    test_df = pd.read_csv(test_csv)
    raw_images = []
    labels = []
    metas = []
    for _, row in test_df.iterrows():
        img_p = raw_root / row["relative_path"]
        pil_img = Image.open(img_p)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        raw_images.append(np.array(pil_img))
        labels.append(int(row["class_index"]))
        metas.append(row.to_dict())

    # Generate Stress Suite Preview Sheet
    preview_path = eval_out_dir / "stress_suite_preview.png"
    print(f"  Generating visual reference sheet: {preview_path}")
    generate_stress_preview(test_df, raw_root, preview_path)

    # Setup Torch Device and Transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Evaluation compute device: {device}")
    _, _, eval_transforms = build_classifier("resnet18", num_classes=2, pretrained=True)

    # 4. Evaluation Loop across Conditions and Models
    print("\n[Step 4/5] Executing evaluation benchmark across 7 conditions & 9 checkpoints...")
    per_run_records = []

    # Cache pre-transformed images for each condition so every checkpoint evaluates identical pixels
    condition_loaders = {}
    for cond_key in CONDITION_KEYS:
        cond_transformed_images = [
            apply_stress_condition(img_rgb, cond_key, metas[idx]["image_id"])
            for idx, img_rgb in enumerate(raw_images)
        ]
        cond_ds = MemoryTestDataset(cond_transformed_images, labels, metas, eval_transforms)
        cond_loader = DataLoader(cond_ds, batch_size=16, shuffle=False)
        condition_loaders[cond_key] = cond_loader

    # Iterate through all models and conditions
    for exp_k, exp_info in EXPERIMENTS.items():
        for s in SEEDS:
            ckpt_info = checkpoints[exp_k][s]
            ckpt_path = Path(ckpt_info["path"])

            # Load model weights
            model, _, _ = build_classifier("resnet18", num_classes=2, pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)

            for cond_key in CONDITION_KEYS:
                loader = condition_loaders[cond_key]
                m_res = evaluate_model(model, loader, device)

                record = {
                    "experiment_key": exp_k,
                    "experiment_name": exp_info["name"],
                    "seed": s,
                    "condition_name": cond_key,
                    "condition_display": CONDITIONS[cond_key]["display_name"],
                    "is_stress": CONDITIONS[cond_key]["is_stress"],
                    "checkpoint_dir": ckpt_info["dir_name"],
                    "best_epoch": ckpt_info["best_epoch"],
                    **m_res
                }
                per_run_records.append(record)
                print(f"  [{exp_k} s{s:3d}] {cond_key:22s} -> Acc: {m_res['accuracy']*100:.2f}% | F1: {m_res['macro_f1']*100:.2f}% | Corr Recall: {m_res['corrosion_recall']*100:.2f}% | FN: {m_res['false_negatives']}")

    # Save per_run_metrics.csv
    per_run_df = pd.DataFrame(per_run_records)
    per_run_csv_path = eval_out_dir / "per_run_metrics.csv"
    per_run_df.to_csv(per_run_csv_path, index=False)
    print(f"\nSaved per-run metrics: {per_run_csv_path} ({len(per_run_df)} rows)")

    # 5. Aggregate Metrics (Mean ± Std) across Seeds
    print("\n[Step 5/5] Computing aggregated statistics and generating charts...")
    aggregated_records = []
    
    clean_summary = {}

    for exp_k in ["A", "B", "C"]:
        # Extract clean test results
        clean_runs = per_run_df[(per_run_df["experiment_key"] == exp_k) & (per_run_df["condition_name"] == "clean_test")]
        clean_f1_mean = float(clean_runs["macro_f1"].mean())

        # Sum confusion matrices on clean_test
        summed_cm = np.zeros((2, 2), dtype=int)
        for cm_list in clean_runs["confusion_matrix"]:
            summed_cm += np.array(cm_list)

        clean_summary[exp_k] = {
            "experiment_name": EXPERIMENTS[exp_k]["name"],
            "accuracy_mean": float(clean_runs["accuracy"].mean()),
            "accuracy_std": float(clean_runs["accuracy"].std()),
            "macro_f1_mean": clean_f1_mean,
            "macro_f1_std": float(clean_runs["macro_f1"].std()),
            "corrosion_recall_mean": float(clean_runs["corrosion_recall"].mean()),
            "corrosion_recall_std": float(clean_runs["corrosion_recall"].std()),
            "corrosion_precision_mean": float(clean_runs["corrosion_precision"].mean()),
            "corrosion_precision_std": float(clean_runs["corrosion_precision"].std()),
            "corrosion_f1_mean": float(clean_runs["corrosion_f1"].mean()),
            "corrosion_f1_std": float(clean_runs["corrosion_f1"].std()),
            "false_negatives_mean": float(clean_runs["false_negatives"].mean()),
            "false_positives_mean": float(clean_runs["false_positives"].mean()),
            "summed_confusion_matrix": summed_cm.tolist(),
            "per_seed_runs": clean_runs[["seed", "accuracy", "macro_f1", "corrosion_recall", "corrosion_precision", "false_negatives", "false_positives"]].to_dict(orient="records"),
        }

        for cond_key in CONDITION_KEYS:
            sub_df = per_run_df[(per_run_df["experiment_key"] == exp_k) & (per_run_df["condition_name"] == cond_key)]
            f1_mean = float(sub_df["macro_f1"].mean())
            robustness_gap = clean_f1_mean - f1_mean if cond_key != "clean_test" else 0.0

            agg_rec = {
                "experiment_key": exp_k,
                "experiment_name": EXPERIMENTS[exp_k]["name"],
                "condition_name": cond_key,
                "condition_display": CONDITIONS[cond_key]["display_name"],
                "is_stress": CONDITIONS[cond_key]["is_stress"],
                "accuracy_mean": float(sub_df["accuracy"].mean()),
                "accuracy_std": float(sub_df["accuracy"].std()),
                "macro_f1_mean": f1_mean,
                "macro_f1_std": float(sub_df["macro_f1"].std()),
                "corrosion_recall_mean": float(sub_df["corrosion_recall"].mean()),
                "corrosion_recall_std": float(sub_df["corrosion_recall"].std()),
                "corrosion_precision_mean": float(sub_df["corrosion_precision"].mean()),
                "corrosion_precision_std": float(sub_df["corrosion_precision"].std()),
                "corrosion_f1_mean": float(sub_df["corrosion_f1"].mean()),
                "corrosion_f1_std": float(sub_df["corrosion_f1"].std()),
                "false_negatives_mean": float(sub_df["false_negatives"].mean()),
                "false_positives_mean": float(sub_df["false_positives"].mean()),
                "macro_f1_robustness_gap": float(robustness_gap),
            }
            aggregated_records.append(agg_rec)

    agg_df = pd.DataFrame(aggregated_records)
    agg_csv_path = eval_out_dir / "aggregated_summary.csv"
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"Saved aggregated summary: {agg_csv_path} ({len(agg_df)} rows)")

    # Compute Overall Robustness Summary (across 6 stress conditions)
    overall_robustness = {}
    for exp_k in ["A", "B", "C"]:
        stress_subset = agg_df[(agg_df["experiment_key"] == exp_k) & (agg_df["is_stress"] == True)]
        clean_rec = agg_df[(agg_df["experiment_key"] == exp_k) & (agg_df["condition_name"] == "clean_test")].iloc[0]
        avg_stress_f1 = float(stress_subset["macro_f1_mean"].mean())
        avg_stress_acc = float(stress_subset["accuracy_mean"].mean())
        avg_stress_recall = float(stress_subset["corrosion_recall_mean"].mean())
        avg_stress_prec = float(stress_subset["corrosion_precision_mean"].mean())
        avg_gap = float(clean_rec["macro_f1_mean"] - avg_stress_f1)

        overall_robustness[exp_k] = {
            "experiment_name": EXPERIMENTS[exp_k]["name"],
            "clean_macro_f1": float(clean_rec["macro_f1_mean"]),
            "average_stress_macro_f1": avg_stress_f1,
            "average_stress_accuracy": avg_stress_acc,
            "average_stress_corrosion_recall": avg_stress_recall,
            "average_stress_corrosion_precision": avg_stress_prec,
            "average_macro_f1_robustness_gap": avg_gap,
        }

    # Group robustness conditions for dashboard consumption
    robustness_by_condition = []
    for cond_key in CONDITION_KEYS:
        cond_dict = {
            "condition_name": cond_key,
            "display_name": CONDITIONS[cond_key]["display_name"],
            "description": CONDITIONS[cond_key]["description"],
            "is_stress": CONDITIONS[cond_key]["is_stress"],
            "experiments": {}
        }
        for exp_k in ["A", "B", "C"]:
            r = agg_df[(agg_df["experiment_key"] == exp_k) & (agg_df["condition_name"] == cond_key)].iloc[0]
            cond_dict["experiments"][exp_k] = {
                "accuracy": {"mean": r["accuracy_mean"], "std": r["accuracy_std"]},
                "macro_f1": {"mean": r["macro_f1_mean"], "std": r["macro_f1_std"]},
                "corrosion_recall": {"mean": r["corrosion_recall_mean"], "std": r["corrosion_recall_std"]},
                "corrosion_precision": {"mean": r["corrosion_precision_mean"], "std": r["corrosion_precision_std"]},
                "false_negatives_mean": r["false_negatives_mean"],
                "robustness_gap": r["macro_f1_robustness_gap"],
            }
        robustness_by_condition.append(cond_dict)

    # Master JSON Schema for FastAPI / Next.js Dashboard
    final_json_payload = {
        "metadata": {
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_sample_count": isolation_info["sample_count"],
            "test_sha256": isolation_info["test_sha256"],
            "class_distribution": isolation_info["class_distribution"],
            "seeds_evaluated": SEEDS,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "dataset_provenance": "K-Pipelines Standard test split (held-out sealed set)",
            "caveats": [
                "Perturbation tests (conditions 2-7) are synthetic controlled stress simulations applied at inference time.",
                "They are not physical subsea inspection captures.",
                "Dataset size is 60 images (29 non-corroded, 31 corroded) reflecting K-Pipelines test allocation.",
                "Test set is now unsealed; no further model tuning is permitted on this split."
            ]
        },
        "experiments": EXPERIMENTS,
        "clean_test_summary": clean_summary,
        "overall_robustness_summary": overall_robustness,
        "robustness_by_condition": robustness_by_condition,
        "artifacts": {
            "per_run_metrics_csv": "experiments/final_evaluation/per_run_metrics.csv",
            "aggregated_summary_csv": "experiments/final_evaluation/aggregated_summary.csv",
            "stress_preview_png": "experiments/final_evaluation/stress_suite_preview.png",
            "charts": {
                "f1_clean_vs_stress": "experiments/final_evaluation/charts/abc_clean_vs_stress_f1.png",
                "corrosion_recall": "experiments/final_evaluation/charts/abc_corrosion_recall_robustness.png",
                "confusion_matrices": "experiments/final_evaluation/charts/abc_confusion_matrices_clean.png",
            }
        }
    }

    json_path = eval_out_dir / "final_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json_payload, f, indent=2)
    print(f"Saved master benchmark JSON: {json_path}")

    # Generate Publication Charts
    print("\nGenerating evaluation charts...")
    plot_clean_vs_stress_f1(aggregated_records, charts_dir / "abc_clean_vs_stress_f1.png")
    plot_corrosion_recall_robustness(aggregated_records, charts_dir / "abc_corrosion_recall_robustness.png")
    plot_side_by_side_confusion_matrices(clean_summary, charts_dir / "abc_confusion_matrices_clean.png")

    print("\n" + "=" * 65)
    print("PHASE 7 FINAL BENCHMARK EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
