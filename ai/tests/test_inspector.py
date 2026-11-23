"""
Unit and Integration Tests for PipeAugment AI ResNet-18 Corrosion Inspector
and Visual Layout Map Generation.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# Ensure ai directory is on sys.path
AI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from pipeaugment_ml.inspector import PipeAugmentInspector, find_default_checkpoint


def test_checkpoint_discovery():
    """Verify default trained checkpoint is found."""
    ckpt = find_default_checkpoint()
    assert ckpt is not None, "Trained ResNet-18 checkpoint must exist in experiments/runs/"
    assert ckpt.exists()


def test_inspector_initialization():
    """Verify PipeAugmentInspector initializes with eval mode and correct classes."""
    inspector = PipeAugmentInspector(device="cpu")
    assert inspector.class_names == ["non_corroded", "corroded"]
    assert not inspector.model.training


def test_inspector_invalid_input():
    """Verify invalid image shapes and dtypes are rejected."""
    inspector = PipeAugmentInspector(device="cpu")
    # Float array instead of uint8
    invalid_float = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        inspector.predict_and_cam(invalid_float)

    # 2D array instead of 3D
    invalid_2d = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        inspector.predict_and_cam(invalid_2d)


def test_inspector_inference_and_cam():
    """Verify predict_and_cam produces valid probabilities and Grad-CAM map."""
    inspector = PipeAugmentInspector(device="cpu")
    dummy_img = np.full((256, 256, 3), 120, dtype=np.uint8)
    # Add a warm rust-like spot in the center
    dummy_img[100:150, 100:150, 0] = 180  # R
    dummy_img[100:150, 100:150, 1] = 80   # G
    dummy_img[100:150, 100:150, 2] = 20   # B

    pred_info, cam_map = inspector.predict_and_cam(dummy_img)

    assert pred_info["predicted_class"] in ["corroded", "non_corroded"]
    assert 0.0 <= pred_info["confidence"] <= 1.0
    assert 0.0 <= pred_info["corrosion_probability"] <= 1.0
    assert 0.0 <= pred_info["non_corrosion_probability"] <= 1.0
    total_prob = pred_info["corrosion_probability"] + pred_info["non_corrosion_probability"]
    assert pytest.approx(total_prob, abs=1e-3) == 1.0

    assert cam_map.shape == (256, 256)
    assert cam_map.min() >= 0.0
    assert cam_map.max() <= 1.0


def test_inspector_analysis_severity_levels():
    """Verify analyze returns expected schema, severity level, and disclaimer."""
    inspector = PipeAugmentInspector(device="cpu")
    dummy_clean = np.full((256, 256, 3), 100, dtype=np.uint8)

    analysis, cam_map, rust_mask = inspector.analyze(dummy_clean)

    assert "predicted_class" in analysis
    assert "confidence" in analysis
    assert "severity_level" in analysis
    assert analysis["severity_level"] in ["None", "Low", "Moderate", "Severe"]
    assert "rust_coverage_percent" in analysis
    assert isinstance(analysis["rust_coverage_percent"], float)
    assert "disclaimer" in analysis
    assert "Diagnostic" in analysis["disclaimer"]


def test_layout_maps_generation(tmp_path):
    """Verify 3 layout map PNGs are generated and saved correctly."""
    inspector = PipeAugmentInspector(device="cpu")
    dummy_img = np.full((256, 256, 3), 120, dtype=np.uint8)
    dummy_img[80:160, 80:160, 0] = 200
    dummy_img[80:160, 80:160, 1] = 90
    dummy_img[80:160, 80:160, 2] = 30

    maps = inspector.generate_layout_maps(dummy_img, tmp_path, prefix="test_")

    assert "pipeline_canny" in maps
    assert "corrosion_layout" in maps
    assert "corrosion_overlay" in maps

    for map_key, file_path in maps.items():
        p = Path(file_path)
        assert p.exists()
        assert p.stat().st_size > 0
        with Image.open(p) as img:
            assert img.size == (256, 256)
