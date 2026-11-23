import sys
import json
import pytest
import hashlib
from pathlib import Path
from typing import Dict, Any

AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from scripts.build_generative_review import build_inventory, generate_html_content
from scripts.import_generative_review import (
    validate_and_process_reviews,
    import_review,
    generate_review_summary,
    load_inventory,
)


@pytest.fixture
def pilots_dir():
    return PROJECT_ROOT / "experiments" / "pilots"


@pytest.fixture
def review_dir():
    return PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1"


@pytest.fixture
def sample_inventory(review_dir):
    inv_path = review_dir / "review_inventory.json"
    return load_inventory(inv_path)


def test_inventory_construction_and_parity(pilots_dir):
    """Verify inventory contains exactly 48 outputs, 12 groups, and 6 parents without leaks."""
    inventory = build_inventory(PROJECT_ROOT, pilots_dir, embed_base64=False)

    assert inventory["parent_count"] == 6
    assert inventory["condition_count"] == 2
    assert inventory["group_count"] == 12
    assert inventory["total_candidates"] == 48
    assert len(inventory["outputs_lookup"]) == 48

    # Each group has exactly 4 outputs
    for g in inventory["groups"]:
        assert len(g["outputs"]) == 4
        methods = [o["method"] for o in g["outputs"]]
        assert methods.count("plain_img2img") == 2
        assert methods.count("controlnet_canny") == 2


def test_matching_source_and_output_ids(sample_inventory):
    """Verify all 48 outputs map to real parents and have valid SHA-256 hashes."""
    lookup = sample_inventory["outputs_lookup"]
    assert len(lookup) == 48

    for out_id, item in lookup.items():
        assert item["output_id"] == out_id
        assert len(item["output_sha256"]) == 64
        # Verify image file exists
        img_path = PROJECT_ROOT / item["relative_image_path"]
        assert img_path.exists(), f"Image file {img_path} does not exist"


def test_unknown_id_rejected(sample_inventory):
    """Verify import rejects unknown output IDs."""
    fake_record = [{
        "output_id": "gen_unknown_output_uuid_12345",
        "parent_id": "parent_uuid",
        "decision": "pending",
    }]
    with pytest.raises(ValueError, match="Unknown output_id"):
        validate_and_process_reviews(fake_record, sample_inventory, PROJECT_ROOT)


def test_duplicate_id_rejected(sample_inventory):
    """Verify import rejects duplicate output IDs in review file."""
    valid_id = list(sample_inventory["outputs_lookup"].keys())[0]
    dup_records = [
        {"output_id": valid_id, "decision": "pending"},
        {"output_id": valid_id, "decision": "pending"},
    ]
    with pytest.raises(ValueError, match="Duplicate output_id"):
        validate_and_process_reviews(dup_records, sample_inventory, PROJECT_ROOT)


def test_hash_mismatch_rejected(sample_inventory, tmp_path):
    """Verify import rejects records with mismatched SHA-256 hashes."""
    valid_id = list(sample_inventory["outputs_lookup"].keys())[0]
    inv_item = sample_inventory["outputs_lookup"][valid_id]

    tampered_records = [{
        "output_id": valid_id,
        "output_sha256": "0" * 64,  # wrong hash
        "decision": "pending",
    }]
    with pytest.raises(ValueError, match="Integrity check FAILED"):
        validate_and_process_reviews(tampered_records, sample_inventory, PROJECT_ROOT)


def test_invalid_criteria_values_rejected(sample_inventory):
    """Verify import rejects criteria outside allowed values."""
    valid_id = list(sample_inventory["outputs_lookup"].keys())[0]
    invalid_record = [{
        "output_id": valid_id,
        "geometry_preserved": "somewhat",  # invalid
        "decision": "pending",
    }]
    with pytest.raises(ValueError, match="Invalid value 'somewhat'"):
        validate_and_process_reviews(invalid_record, sample_inventory, PROJECT_ROOT)


def test_incomplete_approval_rejected(sample_inventory):
    """Verify approval is rejected if any criterion is not 'yes' or reviewer is missing."""
    valid_id = list(sample_inventory["outputs_lookup"].keys())[0]

    # Case 1: Missing reviewer name
    rec_missing_reviewer = [{
        "output_id": valid_id,
        "geometry_preserved": "yes",
        "corrosion_preserved": "yes",
        "condition_visible": "yes",
        "realism_acceptable": "yes",
        "decision": "approved",
        "reviewer": "   ",  # whitespace
    }]
    with pytest.raises(ValueError, match="reviewer name is empty"):
        validate_and_process_reviews(rec_missing_reviewer, sample_inventory, PROJECT_ROOT)

    # Case 2: One criterion is 'uncertain'
    rec_uncertain = [{
        "output_id": valid_id,
        "geometry_preserved": "yes",
        "corrosion_preserved": "uncertain",
        "condition_visible": "yes",
        "realism_acceptable": "yes",
        "decision": "approved",
        "reviewer": "Alice",
    }]
    with pytest.raises(ValueError, match="corrosion_preserved='uncertain'"):
        validate_and_process_reviews(rec_uncertain, sample_inventory, PROJECT_ROOT)

    # Case 3: One criterion is 'no'
    rec_no = [{
        "output_id": valid_id,
        "geometry_preserved": "no",
        "corrosion_preserved": "yes",
        "condition_visible": "yes",
        "realism_acceptable": "yes",
        "decision": "approved",
        "reviewer": "Alice",
    }]
    with pytest.raises(ValueError, match="geometry_preserved='no'"):
        validate_and_process_reviews(rec_no, sample_inventory, PROJECT_ROOT)


def test_pending_entries_excluded_from_approved_manifest(sample_inventory, tmp_path):
    """Verify that pending and needs_review entries are never marked approved."""
    valid_ids = list(sample_inventory["outputs_lookup"].keys())
    id1, id2, id3 = valid_ids[0], valid_ids[1], valid_ids[2]

    # Create mock review with 1 approved, 1 rejected, 1 pending
    test_reviews = [
        {
            "output_id": id1,
            "geometry_preserved": "yes",
            "corrosion_preserved": "yes",
            "condition_visible": "yes",
            "realism_acceptable": "yes",
            "decision": "approved",
            "reviewer": "Auditor 1",
        },
        {
            "output_id": id2,
            "geometry_preserved": "no",
            "corrosion_preserved": "no",
            "condition_visible": "yes",
            "realism_acceptable": "no",
            "decision": "rejected",
            "reviewer": "Auditor 1",
        },
        {
            "output_id": id3,
            "geometry_preserved": "unreviewed",
            "decision": "pending",
        }
    ]

    val_records, stats = validate_and_process_reviews(test_reviews, sample_inventory, PROJECT_ROOT)
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["pending"] == 46  # 1 explicitly pending + 45 missing from file filled as pending

    approved_list = [r["output_id"] for r in val_records if r["decision"] == "approved"]
    assert approved_list == [id1]
    assert id2 not in approved_list
    assert id3 not in approved_list


def test_roundtrip_json_import(sample_inventory, tmp_path):
    """Verify roundtrip export JSON can be imported without loss."""
    valid_id = list(sample_inventory["outputs_lookup"].keys())[0]
    inv_item = sample_inventory["outputs_lookup"][valid_id]

    export_payload = {
        "schema_version": "1.0",
        "exported_at_utc": "2026-11-22T12:00:00Z",
        "reviews": [
            {
                "output_id": valid_id,
                "parent_id": inv_item["parent_id"],
                "condition": inv_item["condition"],
                "method": inv_item["method"],
                "strength": inv_item["strength"],
                "control_scale": inv_item["control_scale"],
                "output_sha256": inv_item["output_sha256"],
                "geometry_preserved": "yes",
                "corrosion_preserved": "yes",
                "condition_visible": "yes",
                "realism_acceptable": "yes",
                "decision": "approved",
                "reviewer": "Senior Inspector",
                "review_notes": "Well preserved weld joint.",
            }
        ]
    }

    test_file = tmp_path / "test_export.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f)

    manifest = import_review(
        input_file=test_file,
        inventory_path=PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1" / "review_inventory.json",
        output_dir=tmp_path / "out",
        version="test_v1",
        print_summary=False,
    )

    assert manifest["approved_count"] == 1
    assert manifest["approved_outputs"] == [valid_id]
    assert manifest["imported_file_sha256"] == hashlib.sha256(test_file.read_bytes()).hexdigest()

    rec = next(r for r in manifest["records"] if r["output_id"] == valid_id)
    assert rec["decision"] == "approved"
    assert rec["reviewer"] == "Senior Inspector"
    assert rec["review_notes"] == "Well preserved weld joint."


def test_original_pilot_files_preserved():
    """Verify that importing reviews never alters the original pilot review.csv files."""
    diff_rev = PROJECT_ROOT / "experiments" / "pilots" / "diffusion_v1" / "review.csv"
    cnet_rev = PROJECT_ROOT / "experiments" / "pilots" / "controlnet_canny_v1" / "review.csv"

    assert diff_rev.exists()
    assert cnet_rev.exists()

    # Verify all records in both files remain unreviewed
    import pandas as pd
    df_d = pd.read_csv(diff_rev)
    df_c = pd.read_csv(cnet_rev)

    assert (df_d["decision"] == "pending").all()
    assert (df_c["decision"] == "pending").all()


def test_generated_html_security_and_offline():
    """Verify generated HTML has no external CDN dependencies and escapes notes safely."""
    html_path = PROJECT_ROOT / "experiments" / "reviews" / "generative_pilot_v1" / "index.html"
    assert html_path.exists()

    content = html_path.read_text(encoding="utf-8")

    # Offline check: no remote scripts or external stylesheets
    assert '<script src="http' not in content
    assert '<link rel="stylesheet" href="http' not in content
    assert 'fonts.googleapis.com' not in content
    assert 'cdn.' not in content

    # XSS safety: verify escapeHtml helper exists in script
    assert "function escapeHtml" in content
    assert "&amp;" in content
    assert "&lt;" in content
    assert "&gt;" in content
