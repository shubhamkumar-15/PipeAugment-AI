# Baseline Classifier Training & Validation Report

## Overview
This document records the automated dataset setup, audit, and baseline classification experiment for PipeAugment AI using the K-Pipelines Standard dataset.

---

## 1. Environment & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **OS** | Windows 11 |
| **Python Environment** | Python 3.14.5 in `.venv-ai` (protected from `backend/.venv`) |
| **PyTorch Version** | `2.14.1+cu130` (CUDA 13.0 build) |
| **Torchvision Version** | `0.29.1+cu130` |
| **GPU Model** | NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM) |
| **Compute Capability** | SM 12.0 (Blackwell Architecture) |
| **NVIDIA Driver** | 621.24 (CUDA UMD 13.5) |
| **CUDA Verification** | Finite matrix multiplication verified; absolute diff vs CPU reference: `6.58e-05` |

---

## 2. Dataset Provenance & Structure

- **Source Repository:** `https://github.com/leoxthomas/K-Pipelines`
- **Resolved Commit:** `14e2ecb6e4014959fe2935d767c41bd9d5d53cd3` (branch `main`)
- **Retrieval Timestamp:** `2026-11-22T11:04:39Z`
- **Local Location:** `datasets/raw/k-pipelines/`
- **Reported License:** GNU General Public License v3.0 (GPL-3.0)
- **Paper Citation:** Ramos, Casas, and Rivas-Echeverría (2025). *Synthetic generated data for intelligent corrosion classification in oil and gas pipelines*, Intelligent Systems with Applications, Vol. 25, 200463. DOI: `10.1016/j.iswa.2024.200463`.
- **Usage Scope:** Baseline experiment strictly uses the **Standard** version (600 images). The Augmented release (1,080 images) is excluded from the baseline.

---

## 3. Dataset Audit Findings

Audited via `ai/scripts/audit_dataset.py`:

- **Total Readable Images:** 600 (100% valid, 0 corrupt files)
- **Formats:** 600 JPEG images
- **Dimensions:** 512 × 512 pixels across all images
- **Source Splits & Counts:**
  - `train`: 480 images (80%)
  - `valid`: 60 images (10%)
  - `test`: 60 images (10%)
- **Class Balance:** Exactly 300 `corrosion` (corroded) and 300 `no_corrosion` (non_corroded)
  - Train: 242 non_corroded, 238 corroded
  - Validation: 29 non_corroded, 31 corroded
  - Test: 29 non_corroded, 31 corroded
- **Exact Duplicates (SHA-256):** 0 exact duplicate groups found
- **Cross-Split Duplicates:** 0 cross-split duplicates (completely leakage-free)
- **Label Conflicts:** 0 conflicting labels
- **Near-Duplicate Candidates (pHash distance <= 4):** 0 candidates flagged

---

## 4. Reproducible Manifests

Manifests generated via `ai/scripts/prepare_dataset.py` with seed 42:
- `datasets/manifests/train.csv` (480 rows)
- `datasets/manifests/val.csv` (60 rows)
- `datasets/manifests/test.csv` (60 rows) — **LOCKED / UNTOUCHED**
- `datasets/manifests/class_mapping.json` (`non_corroded`: 0, `corroded`: 1)
- `datasets/manifests/split_metadata.json`

Manifest columns: `image_id`, `dataset`, `relative_path`, `label`, `class_index`, `split`, `group_id`, `sha256`.

---

## 5. Model Architecture & Training Recipe (Baseline A)

- **Model:** torchvision ResNet-18
- **Pretrained Weights:** Official torchvision `ResNet18_Weights.IMAGENET1K_V1` (downloaded directly from `https://download.pytorch.org/models/resnet18-f37072fd.pth`)
- **Classification Head:** Linear layer replacing `model.fc` (`in_features=512`, `out_features=2`)
- **Preprocessing:** Deterministic evaluation preprocessing (Resize to 256 bilinear, CenterCrop to 224, Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`). No stochastic augmentations applied to Baseline A.
- **Batch Size:** 16
- **Max Epochs:** 20
- **Warmup Schedule:** 2 epochs head-only warmup (`freeze_backbone()`, `warmup_lr = 1e-3`)
- **Fine-Tuning Schedule:** Epochs 3–20 full backbone fine-tuning (`unfreeze_backbone()`, `finetune_lr = 1e-4`)
- **Optimizer:** `AdamW` (`weight_decay = 1e-4`)
- **Loss Function:** `CrossEntropyLoss`
- **Seed:** 42
- **Workers:** `num_workers = 0` (Windows multi-process safe)
- **Early Stopping:** Patience of 5 fine-tuning epochs monitored on validation macro-F1

---

## 6. Smoke Test Verification

Before full execution, a 1-epoch smoke test was run on a small training/validation subset:
- **Command:** `python ai/scripts/train_classifier.py --smoke-test --output-dir experiments/runs/smoke_test`
- **Outcome:** Finite loss confirmed (`0.7359`), output shape `(4, 2)` verified, checkpoint saving and reloading verified.

---

## 7. Actual Training & Validation Results

- **Run Directory:** `experiments/runs/baseline_resnet18_seed42_20261122_111437/`
- **Training Progression:**
  - Epoch 1 [warmup]: Train Loss = 0.5940, Val Loss = 0.3657, Val Acc = 0.9500, Val Macro-F1 = 0.9500
  - Epoch 2 [warmup]: Train Loss = 0.4888, Val Loss = 0.2695, Val Acc = 0.8833, Val Macro-F1 = 0.8806
  - Epoch 3 [finetune]: Train Loss = 0.2888, Val Loss = 0.3250, Val Acc = 0.9333, Val Macro-F1 = 0.9333
  - **Epoch 4 [finetune]: Train Loss = 0.1176, Val Loss = 0.2306, Val Acc = 0.9667, Val Macro-F1 = 0.9667 (BEST)**
  - Epoch 5 [finetune]: Train Loss = 0.0388, Val Loss = 0.2458, Val Acc = 0.9500, Val Macro-F1 = 0.9500
  - Epoch 6 [finetune]: Train Loss = 0.0365, Val Loss = 0.1606, Val Acc = 0.9667, Val Macro-F1 = 0.9667
  - Epoch 7 [finetune]: Train Loss = 0.0420, Val Loss = 0.1529, Val Acc = 0.9667, Val Macro-F1 = 0.9667
  - Epoch 8 [finetune]: Train Loss = 0.0161, Val Loss = 0.2372, Val Acc = 0.9667, Val Macro-F1 = 0.9667
  - Epoch 9 [finetune]: Train Loss = 0.0193, Val Loss = 0.1355, Val Acc = 0.9500, Val Macro-F1 = 0.9499
  - Early stopping triggered at Epoch 9 (5 fine-tuning epochs without improving upon Epoch 4 Macro-F1).

### Best Validation Metrics (Epoch 4)
- **Validation Accuracy:** `0.9667` (58 / 60 correct)
- **Validation Macro-F1:** `0.9667`
- **Validation Macro-Precision:** `0.9677`
- **Validation Macro-Recall:** `0.9677`
- **Per-Class Breakdown:**
  - `non_corroded`: Precision: `0.9355`, Recall: `1.0000`, F1: `0.9667`, Support: 29
  - `corroded`: Precision: `1.0000`, Recall: `0.9355`, F1: `0.9667`, Support: 31
- **Confusion Matrix:**
  - True `non_corroded`: 29 predicted `non_corroded`, 0 predicted `corroded`
  - True `corroded`: 2 predicted `non_corroded`, 29 predicted `corroded`
- **Test Set Status:** **NOT EVALUATED** (locked for final benchmarking after traditional and generative augmentation experiments).

---

## 8. Saved Run Artifacts

All artifacts reside in `experiments/runs/baseline_resnet18_seed42_20261122_111437/`:
- `config.json`: Complete training hyperparameter settings
- `environment.json`: Hardware, PyTorch, and CUDA runtime fingerprint
- `dataset_summary.json`: Training and validation sample statistics
- `manifest_hashes.json`: SHA-256 hashes of input manifests
- `class_mapping.json`: Explicit label-to-index mappings
- `metrics.csv`: Per-epoch training/validation losses, accuracies, F1 scores, learning rates, and wall-clock times
- `best_model.pth`: PyTorch model checkpoint from Epoch 4
- `last_checkpoint.pth`: Full resumable checkpoint from Epoch 9 (model, optimizer, history)
- `validation_metrics.json`: Detailed validation metrics at best checkpoint
- `validation_predictions.csv`: Per-sample prediction log with ground truth and probabilities
- `validation_confusion_matrix.png`: Heatmap plot of validation classification confusion matrix
- `training_curves.png`: Loss and metric curves over epochs
- `training.log`: Timestamped terminal log of execution

---

## 9. Exact Reproduction Commands

All commands run with the explicit virtual environment Python executable:

```powershell
# 1. Dataset Audit
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\audit_dataset.py"

# 2. Manifest Preparation
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\prepare_dataset.py"

# 3. Unit Tests
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" -m pytest "d:\vscode\Gen AI Project\pipeaugment-ai\ai\tests\test_baseline.py" -v

# 4. Smoke Test
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\train_classifier.py" --smoke-test --output-dir "d:\vscode\Gen AI Project\pipeaugment-ai\experiments\runs\smoke_test"

# 5. Full Baseline Training
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\train_classifier.py"

# 6. Resume Training (if needed)
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\train_classifier.py" --resume "d:\vscode\Gen AI Project\pipeaugment-ai\experiments\runs\baseline_resnet18_seed42_20261122_111437\last_checkpoint.pth"

# 7. Checkpoint Evaluation (Validation or Future Test)
& "d:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "d:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\evaluate_classifier.py" --checkpoint "d:\vscode\Gen AI Project\pipeaugment-ai\experiments\runs\baseline_resnet18_seed42_20261122_111437\best_model.pth" --manifest "d:\vscode\Gen AI Project\pipeaugment-ai\datasets\manifests\val.csv" --output-dir "d:\vscode\Gen AI Project\pipeaugment-ai\experiments\runs\eval_val"
```

---

## 10. Known Limitations & Caveats

1. **Synthetic Image Origin:** K-Pipelines Standard dataset images were synthetically generated via Stable Diffusion. High validation accuracy on synthetic images does not prove field robustness against complex physical optical scattering, heavy marine growth, or pipeline coatings in real subsea or onshore pipelines.
2. **Locked Test Set:** The 60-image test set is strictly withheld and has not been evaluated. Test evaluation will be conducted only once after all comparison models (Baseline A, Traditional Augmentation B, Generative Augmentation C) are finalized.
