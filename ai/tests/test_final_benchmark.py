"""
Unit tests for Phase 7 Final Benchmark & Robustness Suite.
Tests:
  - Deterministic stress transform reproducibility (identical pixels given same seed)
  - Pixel, shape, and dtype validity across all 7 conditions
  - Aggregation math (mean/std) on known synthetic results
  - Dashboard JSON schema completeness and contract integrity
  - Strict zero-leakage assertion between train manifests and test.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.pipeaugment_cv.stress_suite import (
    apply_stress_condition,
    CONDITIONS,
    CONDITION_KEYS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_deterministic_stress_transforms_reproducibility():
    """Verify that every stress condition is 100% deterministic and reproducible."""
    dummy_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    image_id = "test_pipe_sample_42"

    for cond in CONDITION_KEYS:
        out1 = apply_stress_condition(dummy_img, cond, image_id)
        out2 = apply_stress_condition(dummy_img, cond, image_id)
        assert np.array_equal(out1, out2), f"Condition {cond} produced non-identical outputs for same image_id"


def test_stress_transform_pixel_shape_and_dtype():
    """Verify output shape (H, W, 3), dtype uint8, and valid pixel bounds [0, 255]."""
    h, w = 48, 48
    dummy_img = np.random.randint(20, 220, size=(h, w, 3), dtype=np.uint8)
    image_id = "test_pipe_bounds_99"

    for cond in CONDITION_KEYS:
        out = apply_stress_condition(dummy_img, cond, image_id)
        assert out.shape == (h, w, 3), f"Condition {cond} changed shape to {out.shape}"
        assert out.dtype == np.uint8, f"Condition {cond} changed dtype to {out.dtype}"
        assert out.min() >= 0, f"Condition {cond} has negative pixel values"
        assert out.max() <= 255, f"Condition {cond} exceeded 255 pixel values"


def test_aggregation_math_on_mock_results():
    """Verify that mean and std calculation logic is mathematically accurate."""
    mock_runs = [
        {"macro_f1": 0.90, "accuracy": 0.90},
        {"macro_f1": 0.92, "accuracy": 0.92},
        {"macro_f1": 0.94, "accuracy": 0.94},
    ]
    f1_vals = [r["macro_f1"] for r in mock_runs]
    mean_val = float(np.mean(f1_vals))
    # Sample std (ddof=1)
    std_val = float(np.std(f1_vals, ddof=1))

    assert np.isclose(mean_val, 0.92)
    assert np.isclose(std_val, 0.02)


def test_json_schema_completeness_for_dashboard():
    """Verify that final_benchmark_results.json conforms strictly to API/Dashboard schema."""
    json_path = PROJECT_ROOT / "experiments" / "final_evaluation" / "final_benchmark_results.json"
    assert json_path.exists(), f"Missing benchmark JSON: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top level keys
    assert "metadata" in data
    assert "experiments" in data
    assert "clean_test_summary" in data
    assert "overall_robustness_summary" in data
    assert "robustness_by_condition" in data
    assert "artifacts" in data

    # Metadata fields
    meta = data["metadata"]
    assert meta["test_sample_count"] == 60
    assert len(meta["seeds_evaluated"]) == 3
    assert len(meta["caveats"]) >= 3

    # Clean test summary
    for exp_k in ["A", "B", "C"]:
        assert exp_k in data["clean_test_summary"]
        clean_exp = data["clean_test_summary"][exp_k]
        for field in ["accuracy_mean", "macro_f1_mean", "corrosion_recall_mean", "summed_confusion_matrix"]:
            assert field in clean_exp

    # Robustness by condition
    assert len(data["robustness_by_condition"]) == 7
    for cond_rec in data["robustness_by_condition"]:
        assert "condition_name" in cond_rec
        assert "experiments" in cond_rec
        for exp_k in ["A", "B", "C"]:
            assert exp_k in cond_rec["experiments"]
            exp_metrics = cond_rec["experiments"][exp_k]
            assert "accuracy" in exp_metrics
            assert "macro_f1" in exp_metrics
            assert "corrosion_recall" in exp_metrics
            assert "robustness_gap" in exp_metrics


def test_zero_leakage_assertion():
    """Verify zero overlap between sealed test set and any training/val manifest."""
    test_csv = PROJECT_ROOT / "datasets" / "manifests" / "test.csv"
    assert test_csv.exists()
    test_df = pd.read_csv(test_csv)
    test_ids = set(test_df["image_id"])
    test_hashes = set(test_df["sha256"])

    manifests = [
        PROJECT_ROOT / "datasets" / "manifests" / "train.csv",
        PROJECT_ROOT / "datasets" / "manifests" / "val.csv",
        PROJECT_ROOT / "datasets" / "manifests" / "train_traditional_v1.csv",
        PROJECT_ROOT / "datasets" / "manifests" / "train_pipeaugment_v1.csv",
    ]

    for m_path in manifests:
        assert m_path.exists(), f"Missing manifest: {m_path}"
        df = pd.read_csv(m_path)
        id_overlap = test_ids.intersection(set(df["image_id"]))
        hash_overlap = test_hashes.intersection(set(df["sha256"]))
        assert len(id_overlap) == 0, f"Leakage: {len(id_overlap)} overlapping IDs in {m_path.name}"
        assert len(hash_overlap) == 0, f"Leakage: {len(hash_overlap)} overlapping hashes in {m_path.name}"
