import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Setup paths
AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_ml.reproducibility import set_seed
from pipeaugment_ml.dataset import PipeAugmentDataset, get_baseline_transforms
from pipeaugment_ml.model import build_classifier, freeze_backbone, unfreeze_backbone
from pipeaugment_ml.metrics import compute_classification_metrics


MANIFESTS_DIR = PROJECT_ROOT / "datasets" / "manifests"
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw" / "k-pipelines"


def test_manifest_reading_and_structure():
    """Verify train, val, and test manifests exist, have required columns, and non-empty rows."""
    required_cols = {"image_id", "dataset", "relative_path", "label", "class_index", "split", "group_id", "sha256"}
    for split in ["train", "val", "test"]:
        csv_path = MANIFESTS_DIR / f"{split}.csv"
        assert csv_path.exists(), f"Missing manifest {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"Empty manifest: {split}.csv"
        assert required_cols.issubset(set(df.columns)), f"Columns missing in {split}.csv"


def test_class_mapping():
    """Verify class_mapping.json aligns with binary corrosion definitions."""
    mapping_file = MANIFESTS_DIR / "class_mapping.json"
    assert mapping_file.exists(), "class_mapping.json missing"
    with open(mapping_file, "r") as f:
        data = json.load(f)
    assert data["class_to_index"] == {"non_corroded": 0, "corroded": 1}
    assert data["index_to_class"] == {"0": "non_corroded", "1": "corroded"}


def test_disjoint_splits_and_no_leakage():
    """Verify splits are strictly disjoint by image_id, relative_path, and sha256."""
    train_df = pd.read_csv(MANIFESTS_DIR / "train.csv")
    val_df = pd.read_csv(MANIFESTS_DIR / "val.csv")
    test_df = pd.read_csv(MANIFESTS_DIR / "test.csv")

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    # Disjoint ID check
    assert train_ids.isdisjoint(val_ids), "Train and Val share image_ids!"
    assert train_ids.isdisjoint(test_ids), "Train and Test share image_ids!"
    assert val_ids.isdisjoint(test_ids), "Val and Test share image_ids!"

    # Disjoint SHA-256 check
    train_shas = set(train_df["sha256"])
    val_shas = set(val_df["sha256"])
    test_shas = set(test_df["sha256"])

    assert train_shas.isdisjoint(val_shas), "Train and Val share identical image SHA-256!"
    assert train_shas.isdisjoint(test_shas), "Train and Test share identical image SHA-256!"
    assert val_shas.isdisjoint(test_shas), "Val and Test share identical image SHA-256!"


def test_duplicate_group_isolation():
    """Verify that any group_id appears in at most one split."""
    train_df = pd.read_csv(MANIFESTS_DIR / "train.csv")
    val_df = pd.read_csv(MANIFESTS_DIR / "val.csv")
    test_df = pd.read_csv(MANIFESTS_DIR / "test.csv")

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert train_groups.isdisjoint(val_groups), "group_id leak between train and val"
    assert train_groups.isdisjoint(test_groups), "group_id leak between train and test"
    assert val_groups.isdisjoint(test_groups), "group_id leak between val and test"


def test_dataset_item_and_label_loading():
    """Verify PipeAugmentDataset loads valid tensors and correct integer labels."""
    transforms = get_baseline_transforms()
    val_dataset = PipeAugmentDataset(MANIFESTS_DIR / "val.csv", RAW_ROOT, transform=transforms)
    assert len(val_dataset) == 60

    img_tensor, label_idx, meta = val_dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert label_idx in (0, 1)
    assert meta["label"] in ("non_corroded", "corroded")


def test_model_output_shape():
    """Verify ResNet-18 output logits shape is (batch_size, 2)."""
    model, weights_id, _ = build_classifier(model_name="resnet18", num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (4, 2)


def test_model_freezing():
    """Verify backbone freezing leaves only fc parameters trainable."""
    model, _, _ = build_classifier(model_name="resnet18", num_classes=2, pretrained=False)
    freeze_backbone(model)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    unfreeze_backbone(model)
    for param in model.parameters():
        assert param.requires_grad is True


def test_metrics_on_known_examples():
    """Verify metrics calculation on deterministic known inputs."""
    # Perfect match: 2 non_corroded (0), 2 corroded (1)
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    m = compute_classification_metrics(y_true, y_pred, ["non_corroded", "corroded"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]

    # 1 mistake:
    y_pred_err = [0, 1, 1, 1]
    m_err = compute_classification_metrics(y_true, y_pred_err, ["non_corroded", "corroded"])
    assert m_err["accuracy"] == 0.75
    assert m_err["confusion_matrix"] == [[1, 1], [0, 2]]


def test_checkpoint_restoration(tmp_path):
    """Verify model and optimizer states can be saved and restored accurately."""
    set_seed(42)
    model, _, _ = build_classifier(model_name="resnet18", num_classes=2, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "test_ckpt.pth"
    torch.save({
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_val_f1": 0.88,
    }, ckpt_path)

    # Re-instantiate and load
    model_loaded, _, _ = build_classifier(model_name="resnet18", num_classes=2, pretrained=False)
    ckpt = torch.load(ckpt_path)
    model_loaded.load_state_dict(ckpt["model_state_dict"])

    for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
        assert torch.equal(p1, p2)
