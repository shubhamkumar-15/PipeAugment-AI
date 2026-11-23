from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ["non_corroded", "corroded"]
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics with explicit class order and support counts.
    Guards against division-by-zero by setting zero_division=0.0.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0.0))

    # Per-class scores
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0.0)
    per_class_prec = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0.0)
    per_class_rec = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0.0)

    # Confusion matrix with explicit labels [0, 1]
    labels_order = list(range(len(class_names)))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_order)

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        support_cnt = int(np.sum(y_true_arr == idx))
        pred_cnt = int(np.sum(y_pred_arr == idx))
        per_class_dict[name] = {
            "class_index": idx,
            "precision": float(per_class_prec[idx]) if idx < len(per_class_prec) else 0.0,
            "recall": float(per_class_rec[idx]) if idx < len(per_class_rec) else 0.0,
            "f1_score": float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0,
            "ground_truth_support": support_cnt,
            "predicted_count": pred_cnt,
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "total_samples": len(y_true),
        "class_order": class_names,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Path,
    title: str = "Validation Confusion Matrix"
) -> None:
    """Plots and saves a publication-quality confusion matrix image."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Oranges)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, format(cm_arr[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Plots training and validation loss, accuracy, and macro-F1 over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Loss Curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#ff5a00", marker="o", markersize=4)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#132038", linestyle="--", marker="s", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Metric Curves
    ax2.plot(epochs, val_acc, label="Val Accuracy", color="#34d399", marker="^", markersize=4)
    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="#ff9b00", marker="D", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Accuracy & Macro-F1")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
