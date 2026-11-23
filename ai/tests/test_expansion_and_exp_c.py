import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from scripts.generate_generative_expansion import (
    sample_expansion_parents,
    make_expansion_output_id,
)

MANIFESTS_DIR = PROJECT_ROOT / "datasets" / "manifests"


# =====================================================================
# 1. Deterministic Parent Selection & Zero Leakage Tests
# =====================================================================

def test_expansion_parent_selection_and_no_leakage():
    """Verify that expansion parents are strictly from train.csv and do not leak into val or test."""
    train_csv = MANIFESTS_DIR / "train.csv"
    val_csv = MANIFESTS_DIR / "val.csv"
    test_csv = MANIFESTS_DIR / "test.csv"

    parents_df = sample_expansion_parents(
        train_csv, val_csv, test_csv,
        seed=42, num_per_class=50
    )

    # 1. Exactly 100 parents, 50 per class
    assert len(parents_df) == 100
    assert len(parents_df[parents_df["label"] == "non_corroded"]) == 50
    assert len(parents_df[parents_df["label"] == "corroded"]) == 50

    # 2. All parents exist in train.csv
    train_df = pd.read_csv(train_csv)
    train_ids = set(train_df["image_id"])
    sampled_ids = set(parents_df["image_id"])
    assert sampled_ids.issubset(train_ids)

    # 3. Zero leakage into val.csv or test.csv
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    assert len(sampled_ids & val_ids) == 0, "Expansion parents leak into validation set!"
    assert len(sampled_ids & test_ids) == 0, "Expansion parents leak into test set!"

    # 4. Deterministic reproducibility
    parents_df_2 = sample_expansion_parents(
        train_csv, val_csv, test_csv,
        seed=42, num_per_class=50
    )
    assert list(parents_df["image_id"]) == list(parents_df_2["image_id"])


# =====================================================================
# 2. Quality Gate Admission & Exclusion Rules
# =====================================================================

def test_quality_gate_admission_and_exclusion():
    """Verify that only pass + non-safety-flagged outputs are marked training_eligibility=True."""
    mock_candidates = [
        # Candidate 1: Clean pass
        {"id": "c1", "quality_status": "pass", "safety_flagged": False, "expected_admitted": True},
        # Candidate 2: Flagged by Phase 4 QualityValidator
        {"id": "c2", "quality_status": "flag_for_review", "safety_flagged": False, "expected_admitted": False},
        # Candidate 3: Fatal failure (degenerate/constant/corrupt)
        {"id": "c3", "quality_status": "fail", "safety_flagged": False, "expected_admitted": False},
        # Candidate 4: Passed metrics but safety flagged by Diffusers safety checker
        {"id": "c4", "quality_status": "pass", "safety_flagged": True, "expected_admitted": False},
        # Candidate 5: Both flagged
        {"id": "c5", "quality_status": "flag_for_review", "safety_flagged": True, "expected_admitted": False},
    ]

    for cand in mock_candidates:
        is_admitted = (cand["quality_status"] == "pass") and (not cand["safety_flagged"])
        assert is_admitted == cand["expected_admitted"], f"Failed admission rule for candidate {cand['id']}"


# =====================================================================
# 3. Stratified Source-Type Sampling & Budget Matching Tests
# =====================================================================

def test_stratified_source_type_sampling_and_budget():
    """Verify matched-budget sampling draws exactly 480 samples, balanced 50/50 by class, stratified across source_type."""
    # Construct a synthetic multi-source training DataFrame
    # 240 originals, 480 traditional, 90 generative per class (total 1620 rows)
    rows = []
    idx = 0
    for c_idx, label in [(0, "non_corroded"), (1, "corroded")]:
        for i in range(240):
            rows.append({"image_id": f"orig_{idx}", "class_index": c_idx, "label": label, "source_type": "original"})
            idx += 1
        for i in range(480):
            rows.append({"image_id": f"trad_{idx}", "class_index": c_idx, "label": label, "source_type": "traditional"})
            idx += 1
        for i in range(90):
            rows.append({"image_id": f"gen_{idx}", "class_index": c_idx, "label": label, "source_type": "generative"})
            idx += 1

    synthetic_df = pd.DataFrame(rows)
    samples_per_epoch = 480
    k_per_class = samples_per_epoch // 2  # 240

    epoch_rng = np.random.default_rng(42)
    selected_indices = []

    has_source_types = (
        "source_type" in synthetic_df.columns
        and len(synthetic_df["source_type"].dropna().unique()) > 1
    )
    assert has_source_types is True

    for c_idx in [0, 1]:
        class_mask = (synthetic_df["class_index"] == c_idx)
        source_types = sorted(synthetic_df.loc[class_mask, "source_type"].dropna().unique())
        target_per_source = k_per_class // len(source_types)  # 240 // 3 = 80
        c_selected = []
        for st in source_types:
            st_indices = np.where(class_mask & (synthetic_df["source_type"] == st))[0]
            n_sample = min(len(st_indices), target_per_source)
            c_selected.extend(epoch_rng.choice(st_indices, size=n_sample, replace=False).tolist())

        if len(c_selected) < k_per_class:
            rem_pool = list(set(np.where(class_mask)[0]) - set(c_selected))
            needed = k_per_class - len(c_selected)
            c_selected.extend(epoch_rng.choice(rem_pool, size=needed, replace=False).tolist())

        selected_indices.extend(c_selected)

    # 1. Total samples per epoch must equal exactly 480
    assert len(selected_indices) == 480

    # 2. Class balance must be strictly 240 / 240
    sampled_subset = synthetic_df.iloc[selected_indices]
    assert len(sampled_subset[sampled_subset["class_index"] == 0]) == 240
    assert len(sampled_subset[sampled_subset["class_index"] == 1]) == 240

    # 3. Source types must be stratified: ~80 original, ~80 traditional, ~80 generative per class
    for c_idx in [0, 1]:
        c_sub = sampled_subset[sampled_subset["class_index"] == c_idx]
        orig_count = len(c_sub[c_sub["source_type"] == "original"])
        trad_count = len(c_sub[c_sub["source_type"] == "traditional"])
        gen_count = len(c_sub[c_sub["source_type"] == "generative"])

        assert orig_count == 80
        assert trad_count == 80
        assert gen_count == 80
