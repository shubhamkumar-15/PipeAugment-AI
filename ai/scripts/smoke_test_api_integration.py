"""
End-to-End Live Integration Smoke Test for PipeAugment AI Background Worker.
Tests:
  1. Traditional CV job execution via run_api_job.py
  2. Generative ControlNet Canny diffusion execution on RTX 5060 GPU
  3. QualityValidator Phase 4 scoring verification
  4. Dataset ZIP export verification containing the outputs
  5. Clean artifact teardown
"""

import sys
import json
import subprocess
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AI_DIR = PROJECT_ROOT / "ai"
DATASET_ROOT = PROJECT_ROOT / "datasets"
AI_PYTHON = PROJECT_ROOT / ".venv-ai" / "Scripts" / "python.exe"

print("=" * 65)
print("PIPEAUGMENT AI — LIVE INTEGRATION SMOKE TEST")
print("=" * 65)

# 1. Create a dummy parent pipe image
parent_id = str(uuid4())
orig_dir = DATASET_ROOT / "original"
meta_dir = DATASET_ROOT / "metadata"
jobs_dir = DATASET_ROOT / "metadata" / "jobs"
orig_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)
jobs_dir.mkdir(parents=True, exist_ok=True)

# Generate a 128x128 synthetic pipeline cylinder with rust texture
img_arr = np.full((128, 128, 3), 120, dtype=np.uint8)
img_arr[40:88, :, 0] = 160  # reddish rust band
img_arr[40:88, :, 1] = 90
img_arr[40:88, :, 2] = 50

parent_file = orig_dir / f"{parent_id}.png"
Image.fromarray(img_arr).save(parent_file, format="PNG")

parent_meta = {
    "image_id": parent_id,
    "parent_image_id": None,
    "original_filename": "smoke_pipe.png",
    "stored_filename": f"{parent_id}.png",
    "relative_path": f"original/{parent_id}.png",
    "image_format": "PNG",
    "content_type": "image/png",
    "width": 128,
    "height": 128,
    "size_bytes": parent_file.stat().st_size,
    "sha256": "smoke_sha256",
    "created_at": "2026-11-22T21:00:00Z",
    "source_type": "user_upload",
    "generation_method": None,
    "label": "corroded",
    "annotation_status": "unannotated",
    "split": "unassigned",
}
with open(meta_dir / f"{parent_id}.json", "w", encoding="utf-8") as f:
    json.dump(parent_meta, f, indent=2)

print(f"[Step 1/5] Created test parent image: {parent_id}")

created_artifacts = [parent_file, meta_dir / f"{parent_id}.json"]

try:
    # 2. Traditional Job Execution
    trad_job_id = str(uuid4())
    trad_status_file = jobs_dir / f"{trad_job_id}.json"
    trad_status_data = {
        "job_id": trad_job_id,
        "job_type": "traditional",
        "status": "queued",
        "total_items": 1,
        "completed_items": 0,
        "failed_items": 0,
        "parent_image_ids": [parent_id],
        "config": {"conditions": ["brightness"], "intensity": 0.5, "variations_per_image": 1, "seed": 42},
        "created_at": "2026-11-22T21:00:00Z",
        "started_at": None,
        "finished_at": None,
        "generated_image_ids": [],
        "item_results": [],
        "error_message": None,
    }
    with open(trad_status_file, "w", encoding="utf-8") as f:
        json.dump(trad_status_data, f, indent=2)
    created_artifacts.append(trad_status_file)

    trad_spec = {
        "job_id": trad_job_id,
        "job_type": "traditional",
        "status_file": str(trad_status_file),
        "parent_image_ids": [parent_id],
        "parent_images": [
            {
                "image_id": parent_id,
                "file_path": str(parent_file),
                "relative_path": f"original/{parent_id}.png",
                "sha256": "smoke_sha256",
                "label": "corroded",
                "original_filename": "smoke_pipe.png",
            }
        ],
        "config": {"conditions": ["brightness"], "intensity": 0.5, "variations_per_image": 1, "seed": 42},
        "output_dirs": {
            "augmented_dir": str(DATASET_ROOT / "augmented" / "app_generated"),
            "metadata_dir": str(meta_dir),
        },
    }
    trad_spec_file = jobs_dir / f"spec_{trad_job_id}.json"
    with open(trad_spec_file, "w", encoding="utf-8") as f:
        json.dump(trad_spec, f, indent=2)
    created_artifacts.append(trad_spec_file)

    print("\n[Step 2/5] Running traditional augmentation job via run_api_job.py...")
    cmd_trad = [str(AI_PYTHON), str(AI_DIR / "scripts" / "run_api_job.py"), "--job-spec", str(trad_spec_file)]
    res = subprocess.run(cmd_trad, capture_output=True, text=True)
    if res.returncode != 0:
        print("Traditional worker stderr:", res.stderr)
        raise RuntimeError("Traditional job worker failed.")

    with open(trad_status_file, "r", encoding="utf-8") as f:
        trad_out = json.load(f)
    print(f"  Traditional Job Status: {trad_out['status']}")
    assert trad_out["status"] == "completed"
    assert trad_out["completed_items"] == 1
    assert len(trad_out["generated_image_ids"]) == 1

    trad_gen_id = trad_out["generated_image_ids"][0]
    trad_png = DATASET_ROOT / "augmented" / "app_generated" / "traditional" / f"{trad_gen_id}.png"
    trad_meta_file = meta_dir / f"{trad_gen_id}.json"
    assert trad_png.exists()
    assert trad_meta_file.exists()
    created_artifacts.extend([trad_png, trad_meta_file])

    with open(trad_meta_file, "r", encoding="utf-8") as f:
        trad_meta = json.load(f)
    print(f"  Traditional Quality Status: {trad_meta['quality_status']} | SSIM: {trad_meta['quality_metrics']['ssim']:.4f}")

    # 3. ControlNet Generative Job Execution
    gen_job_id = str(uuid4())
    gen_status_file = jobs_dir / f"{gen_job_id}.json"
    gen_status_data = {
        "job_id": gen_job_id,
        "job_type": "generative",
        "status": "queued",
        "total_items": 1,
        "completed_items": 0,
        "failed_items": 0,
        "parent_image_ids": [parent_id],
        "config": {
            "mode": "controlnet_canny",
            "conditions": ["low_light_inspection"],
            "strength": 0.35,
            "control_scale": 0.80,
            "variations_per_image": 1,
            "seed": 42,
        },
        "created_at": "2026-11-22T21:00:00Z",
        "started_at": None,
        "finished_at": None,
        "generated_image_ids": [],
        "item_results": [],
        "error_message": None,
    }
    with open(gen_status_file, "w", encoding="utf-8") as f:
        json.dump(gen_status_data, f, indent=2)
    created_artifacts.append(gen_status_file)

    gen_spec = {
        "job_id": gen_job_id,
        "job_type": "generative",
        "status_file": str(gen_status_file),
        "parent_image_ids": [parent_id],
        "parent_images": [
            {
                "image_id": parent_id,
                "file_path": str(parent_file),
                "relative_path": f"original/{parent_id}.png",
                "sha256": "smoke_sha256",
                "label": "corroded",
                "original_filename": "smoke_pipe.png",
            }
        ],
        "config": {
            "mode": "controlnet_canny",
            "conditions": ["low_light_inspection"],
            "strength": 0.35,
            "control_scale": 0.80,
            "variations_per_image": 1,
            "seed": 42,
        },
        "output_dirs": {
            "augmented_dir": str(DATASET_ROOT / "augmented" / "app_generated"),
            "metadata_dir": str(meta_dir),
        },
    }
    gen_spec_file = jobs_dir / f"spec_{gen_job_id}.json"
    with open(gen_spec_file, "w", encoding="utf-8") as f:
        json.dump(gen_spec, f, indent=2)
    created_artifacts.append(gen_spec_file)

    print("\n[Step 3/5] Running ControlNet generative job via run_api_job.py on RTX 5060...")
    cmd_gen = [str(AI_PYTHON), str(AI_DIR / "scripts" / "run_api_job.py"), "--job-spec", str(gen_spec_file)]
    res_gen = subprocess.run(cmd_gen, capture_output=True, text=True)
    if res_gen.returncode != 0:
        print("Generative worker stderr:", res_gen.stderr)
        raise RuntimeError("Generative job worker failed.")

    with open(gen_status_file, "r", encoding="utf-8") as f:
        gen_out = json.load(f)
    print(f"  Generative Job Status: {gen_out['status']}")
    assert gen_out["status"] == "completed"
    assert gen_out["completed_items"] == 1
    assert len(gen_out["generated_image_ids"]) == 1

    gen_img_id = gen_out["generated_image_ids"][0]
    gen_png = DATASET_ROOT / "augmented" / "app_generated" / "generative" / f"{gen_img_id}.png"
    gen_meta_file = meta_dir / f"{gen_img_id}.json"
    assert gen_png.exists()
    assert gen_meta_file.exists()
    created_artifacts.extend([gen_png, gen_meta_file])

    with open(gen_meta_file, "r", encoding="utf-8") as f:
        gen_meta = json.load(f)
    print(f"  Generative Quality Status: {gen_meta['quality_status']} | SSIM: {gen_meta['quality_metrics']['ssim']:.4f} | LPIPS: {gen_meta['quality_metrics']['lpips']:.4f}")

    # 4. Verify Dataset ZIP Export
    print("\n[Step 4/5] Testing Dataset ZIP Export with generated images...")
    backend_python = PROJECT_ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
    backend_dir = PROJECT_ROOT / "backend"
    export_test_script = f"""
import sys
sys.path.insert(0, r"{str(backend_dir)}")
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
resp = client.post('/api/v1/datasets/export', json={{
    'include_originals': True,
    'include_traditional': True,
    'include_generative': True,
}})
assert resp.status_code == 201
data = resp.json()
print('Export success:', data['filename'], 'Image count:', data['image_count'])

dl_resp = client.get(f'/api/v1/datasets/exports/{{data[\"export_id\"]}}/download')
assert dl_resp.status_code == 200
assert len(dl_resp.content) > 1000
print('Download stream success, bytes:', len(dl_resp.content))
"""
    res_export = subprocess.run([str(backend_python), "-c", export_test_script], cwd=str(backend_dir), capture_output=True, text=True)
    if res_export.returncode != 0:
        print("Export test stderr:", res_export.stderr)
        raise RuntimeError("Export test failed.")
    print("  " + res_export.stdout.strip())

    print("\n[Step 5/5] Live integration smoke test completed successfully!")

finally:
    # Cleanup smoke artifacts
    print("\nCleaning up temporary smoke test artifacts...")
    for p in created_artifacts:
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    print("Teardown complete.")
