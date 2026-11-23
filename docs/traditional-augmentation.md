# Traditional Image Augmentation & Experiment B Report

## Overview
This document records the design, implementation, verification, batch generation, and experimental validation of the **Traditional Image Augmentation Engine (`traditional_v1`)** for PipeAugment AI. It benchmarks **Experiment B (Traditional ResNet-18)** directly against **Baseline A (Standard ResNet-18)** under a strictly matched training budget on the shared validation split.

> [!IMPORTANT]
> **Evaluation Disclaimer**:
> This is a **development comparison on a shared validation split**. It is **not** final test performance. The final test manifest (`datasets/manifests/test.csv`, 60 images) remains strictly sealed and un-evaluated. No claims of real-field robustness or statistical significance from a single seed (seed 42) are made.

---

## 1. Engine Design and Transformation Formulas

The traditional engine is implemented modularly in `ai/pipeaugment_cv/`:
- `pipeaugment_cv/config.py`: Parameter ranges, validation constraints, and dataclasses.
- `pipeaugment_cv/transforms.py`: Stateless, pure-function OpenCV/NumPy transformations.
- `pipeaugment_cv/engine.py`: `TraditionalAugmentationEngine` with deterministic seed derivation, parameter resolution, and quantitative sanity checking.

### Transformation Formulas and Parameter Ranges

All transformations preserve 512×512 resolution and uint8 RGB channels. No geometric transformations (no flips, rotations, crops, or affine warps) are applied in this version.

#### A. Lighting Variations (5 Transforms)

1. **Brightness Multiplier (`brightness`)**
   - **Formula**: $I_{\text{out}} = \text{clip}(I_{\text{in}} \times f, 0, 255)$, where $f \in [0.80, 1.20]$.
   - Implemented via `cv2.convertScaleAbs(img, alpha=factor, beta=0)`.
   - Identity exclusion: $|f - 1.0| \ge 0.02$.

2. **Contrast Adjustment (`contrast`)**
   - **Formula**: $I_{\text{out}} = \text{clip}(128.0 + f \times (I_{\text{in}} - 128.0), 0, 255)$, where $f \in [0.85, 1.15]$.
   - Implemented via `cv2.convertScaleAbs(img, alpha=factor, beta=128.0 * (1.0 - factor))`.
   - Identity exclusion: $|f - 1.0| \ge 0.02$.

3. **Gamma Correction (`gamma`)**
   - **Formula**: $I_{\text{out}} = \text{round}\left(255 \times \left(\frac{I_{\text{in}}}{255}\right)^\gamma\right)$, where $\gamma \in [0.85, 1.20]$.
   - Precomputed via 256-element uint8 lookup table (LUT) and applied via `cv2.LUT(img, lut)`.
   - Identity exclusion: $|\gamma - 1.0| \ge 0.02$.

4. **Smooth Multiplicative Uneven Illumination (`uneven_illumination`)**
   - **Formula**: Generates a 2D linear gradient ramp along angle $\theta \in [0^\circ, 360^\circ]$, normalized to $[1.0 - \Delta, 1.0 + \Delta]$ where $\Delta = \text{intensity} - 1.0$, $\text{intensity} \in [0.85, 1.15]$.
   - Blurred with Gaussian filter $\sigma = \min(H, W) / 6$ to ensure low-frequency spatial variation without harsh edges:
     $$I_{\text{out}} = \text{clip}(I_{\text{in}} \odot M_{\text{illum}}, 0, 255)$$

5. **Soft Spatial Shadow (`soft_shadow`)**
   - **Formula**: Samples shadow center $(c_x, c_y) \in [0.15W, 0.85W] \times [0.15H, 0.85H]$, radius $R \in [0.25, 0.65] \times \min(H, W)$, and shadow attenuation $\alpha \in [0.75, 0.90]$.
   - Mask is generated via Euclidean distance falloff $d = \sqrt{(x - c_x)^2 + (y - c_y)^2}$ and smoothed with a Gaussian kernel ($\sigma = R / 2$):
     $$M_{\text{shadow}}(x, y) = 1.0 - (1.0 - \alpha) \cdot \exp\left(-\frac{d^2}{2 (R/2)^2}\right)$$
     $$I_{\text{out}} = \text{clip}(I_{\text{in}} \odot M_{\text{shadow}}, 0, 255)$$

#### B. Image Quality Variations (4 Transforms)

6. **Gaussian Blur (`gaussian_blur`)**
   - **Formula**: Spatial 2D Gaussian convolution with $\sigma \in [0.40, 0.90]$ pixels.
   - Kernel size computed dynamically: $k = 2 \lceil 3\sigma \rceil + 1$ (always odd).
   - Applied via `cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)`.

7. **Gaussian Additive Noise (`gaussian_noise`)**
   - **Formula**: $I_{\text{out}} = \text{clip}(I_{\text{in}} + \mathcal{N}(0, \sigma_{\text{noise}}^2), 0, 255)$, where $\sigma_{\text{noise}} \in [2.0, 6.0]$ on the 0–255 scale.
   - Drawn strictly from an isolated local NumPy RNG (`np.random.default_rng(seed)`), avoiding global state pollution.

8. **JPEG Compression Simulation (`jpeg_compression`)**
   - **Formula**: Converts RGB $\to$ BGR, encodes to in-memory JPEG byte buffer via `cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])` with quality $q \in [75, 90]$, decodes back via `cv2.imdecode`, and restores to RGB.
   - Result is saved losslessly as PNG on disk.

9. **Downsampling and Restoration (`downsample_restore`)**
   - **Formula**: Downsamples to low resolution $(sW, sH)$ with $s \in [0.65, 0.90]$ using area interpolation (`cv2.INTER_AREA`), followed by restoration to original $(W, H)$ using bicubic interpolation (`cv2.INTER_CUBIC`).
   - Simulates sensor resolution mismatch and transmission downscaling.

---

## 2. Technical Checks vs. Human Review

### Automated Technical Sanity Checks
Every generated image undergoes five automated quantitative checks in `TraditionalAugmentationEngine.execute()`:
1. **Dimension Preservation**: $H \times W \times C$ exactly matches parent dimensions ($512 \times 512 \times 3$).
2. **Data Type and Range**: Output dtype is strictly `np.uint8` in $[0, 255]$.
3. **Finite Pixel Check**: All pixel values are finite numbers (no NaN or Inf values).
4. **Non-Constant Check**: Verifies image is not degenerate all-black, all-white, or constant ($I \ne I_{0,0,0}$).
5. **Non-Zero Variation Check**: Verifies that mean absolute pixel difference $\Delta_{\text{pixel}} > 0.0$ relative to parent.

### Quantitative Pilot Verification
- **Pilot Dataset**: 12 training parents (6 non-corroded, 6 corroded) $\times$ 2 variations = 24 images.
- **Pilot Contact Sheet**: Saved at `experiments/pilots/traditional_v1/traditional_pilot_contact_sheet.png` (808×3660 PNG).
- **Summary Metrics (`pilot_sanity_report.json`)**:
  - Technical checks passed: 24/24 (100%)
  - Mean absolute pixel difference average: 4.9615
  - Mean absolute pixel difference range: [0.3524, 14.5335]
  - Max clipping rate (low / 0): 10.52% (in localized soft shadow)
  - Max clipping rate (high / 255): 5.37% (in localized brightness increase)

### Human Review Status
- **Status**: `not_reviewed`
- The quantitative checks confirm mathematical and channel validity. They do **not** constitute expert validation of corrosion preservation or visual plausibility. All generated manifests explicitly record `"human_review_status": "not_reviewed"`.

---

## 3. Label Inheritance Assumption and Provenance

- **Provenance Model**: Every generated variation inherits the parent image's ground-truth classification label under the explicit label preservation assumption:
  `"label_provenance": "inherited_under_label_preservation_assumption"`
- **Lineage Metadata**: Recorded in `datasets/manifests/traditional_v1_generated.csv` (960 rows):
  - `generated_image_id` (deterministic ID e.g. `aug_trad1_005f6893..._brightness`)
  - `parent_image_id`
  - `parent_sha256`
  - `output_sha256`
  - `relative_output_path`
  - `transform_category` (`lighting` or `image_quality`)
  - `transform_name`
  - `exact_parameters` (JSON-encoded)
  - `seed`
  - `creation_timestamp_utc`

---

## 4. Leakage Prevention and Manifest Integrity

### Manifest Verification Results
- **Parent Split Verification**: 100% (960/960) of generated parents originate strictly from `datasets/manifests/train.csv`.
- **Validation Disjointness**: 0 generated variations or parent IDs exist in `val.csv`.
- **Test Disjointness**: 0 generated variations or parent IDs exist in `test.csv`.
- **Duplicate Isolation**: Group IDs (`group_id`) from parent records are preserved and assigned to generated images, ensuring no cross-split leakage.
- **Raw File Immutability**: All original raw images and baseline manifests remained untouched (verified via SHA-256 hashes).
- **Generation Failures**: 0 failed generations. All 960 attempted augmentations succeeded and passed all technical checks.

### Manifest File Summary
- `datasets/manifests/traditional_v1_generated.csv`: 960 augmented entries.
- `datasets/manifests/train_traditional_v1.csv`: 1,440 entries (480 original training images + 960 augmented variations).
- `datasets/manifests/traditional_v1_generation_summary.json`: Complete distribution summary.

---

## 5. Training Budget Parity Control

To evaluate the effect of augmentation fairly without introducing a compute-budget confound:
- **Baseline A Budget**: 480 training samples per epoch, batch size 16 = 30 optimizer steps per epoch.
- **Experiment B Matched Budget**:
  - Training pool: 1,440 images.
  - Samples drawn per epoch: **480 samples** (exactly matched to Baseline A).
  - Class proportions per epoch: Exactly 240 non-corroded and 240 corroded (50/50 balance).
  - Optimizer steps per epoch: **30 steps** (identical to Baseline A).
  - Epoch sampling: Drawn reproducibly using `np.random.default_rng(seed + epoch * 1000)`. In any single epoch, only 1/3 (480/1440) of the pool is sampled; across epochs, the network experiences diverse augmentations without multiplying training gradient updates.

---

## 6. Development Comparison: Baseline A vs. Experiment B

### Summary Comparison Table (`experiments/comparisons/baseline_vs_traditional_v1.csv`)

| Metric | Baseline A | Experiment B (Traditional v1) | Delta | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Evaluation Context** | Shared Validation Split | Shared Validation Split | Identical | 60 images (29 non-corroded, 31 corroded); test set locked |
| **Training Pool Size** | 480 | 1,440 | +960 | 480 originals + 960 traditional variations |
| **Samples Drawn / Epoch** | 480 | 480 | 0 | Enforced compute budget matching |
| **Optimizer Steps / Epoch** | 30 | 30 | 0 | Batch size 16 across 480 samples |
| **Best Checkpoint Epoch** | **Epoch 4** | **Epoch 3** | -1 epoch | Peak validation Macro-F1 reached earlier in fine-tuning |
| **Total Epochs Trained** | 9 | 8 | -1 epoch | Both stopped after 5 epochs without improvement |
| **Actual Total Optimizer Steps** | 270 | 240 | -30 steps | Shorter convergence to identical validation plateau |
| **Validation Accuracy** | **0.9667** (58/60) | **0.9667** (58/60) | 0.0000 | Identical top-line accuracy |
| **Validation Macro-F1** | **0.9667** | **0.9667** | 0.0000 | Harmonic mean of per-class F1 |
| **Corrosion Precision** | **1.0000** (29/29) | **1.0000** (29/29) | 0.0000 | Zero false-positive corrosion alerts |
| **Corrosion Recall** | **0.9355** (29/31) | **0.9355** (29/31) | 0.0000 | 29/31 corroded images correctly caught |
| **Non-Corroded Precision** | **0.9355** (29/31) | **0.9355** (29/31) | 0.0000 | 29 true non-corroded out of 31 predicted |
| **Non-Corroded Recall** | **1.0000** (29/29) | **1.0000** (29/29) | 0.0000 | All non-corroded correctly identified |
| **Validation Errors Count** | **2** | **2** | 0 | Identical failure modes |
| **Random Seed** | 42 | 42 | 0 | Single seed |

### Analysis of Error Cases
Both Baseline A and Experiment B failed on the exact same two validation images:
1. `4efaeb5e-65c8-4abd-83e8-e3414b56726d...`: Ground truth `corroded`, misclassified as `non_corroded`.
   - Baseline A $P(\text{corroded}) = 0.1867$
   - Experiment B $P(\text{corroded}) = 0.1373$
2. `74144b87-062e-4671-97c7-1fec2e1a0496...`: Ground truth `corroded`, misclassified as `non_corroded`.
   - Baseline A $P(\text{corroded}) = 0.0000$
   - Experiment B $P(\text{corroded}) = 0.0039$

### Honest Evaluation Assessment
Traditional lighting and quality augmentations under matched training budget did not improve or degrade validation accuracy on this dataset split. The model reached its peak performance (Macro-F1 0.9667) one epoch earlier (Epoch 3 vs Epoch 4) with 240 total steps vs 270 steps, but encountered the identical failure cases on low-contrast corrosion features. This highlights the motivation for subsequent generative (diffusion-based) augmentation to synthesize novel pipeline condition variations.

---

## 7. Commands Executed

```powershell
# 1. Run traditional augmentation unit test suite
& "D:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" -m pytest "D:\vscode\Gen AI Project\pipeaugment-ai\ai\tests\test_traditional.py" -v

# 2. Run 12-parent pilot generation and sanity check
& "D:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "D:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\generate_traditional.py" --pilot

# 3. Run full batch generation (960 variations)
& "D:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "D:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\generate_traditional.py" --full

# 4. Run 1-epoch smoke test for Experiment B
& "D:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "D:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\train_classifier.py" --config "D:\vscode\Gen AI Project\pipeaugment-ai\ai\configs\traditional_resnet18.json" --smoke-test --output-dir "D:\vscode\Gen AI Project\pipeaugment-ai\experiments\runs\smoke_test_b"

# 5. Run full Experiment B training run
& "D:\vscode\Gen AI Project\pipeaugment-ai\.venv-ai\Scripts\python.exe" "D:\vscode\Gen AI Project\pipeaugment-ai\ai\scripts\train_classifier.py" --config "D:\vscode\Gen AI Project\pipeaugment-ai\ai\configs\traditional_resnet18.json"

# 6. Run backend regression test suite
& "D:\vscode\Gen AI Project\pipeaugment-ai\backend\.venv\Scripts\python.exe" -m pytest "D:\vscode\Gen AI Project\pipeaugment-ai\backend\tests" -v
```

---

## 8. Known Limitations
1. **Single Random Seed**: Both runs evaluated seed 42 only. Differences cannot be claimed as statistically significant.
2. **Fixed Geometric Coordinates**: Geometric transforms were deliberately omitted to prevent border interpolation artifacts, but this limits structural variability.
3. **Synthetic Lighting Approximation**: Low-amplitude linear gradients and 2D Gaussians simulate illumination variations but do not model 3D cylindrical pipeline geometry or optical reflectance.
4. **Unreviewed Visuals**: Augmentations are technically and mathematically validated, but domain experts have not reviewed individual variations.
5. **Class Balance Sampling Difference**: Experiment B draws exactly 240/240 samples per epoch (50/50 balance) versus Baseline A's original 242 non-corroded / 238 corroded split.
6. **Convergence Rate Interpretation**: Earlier best epoch in one run (Epoch 3 vs Epoch 4) does not establish faster convergence.
7. **Leakage Scope Caveat**: Duplicate checks do not prove all forms of leakage absent; they confirm no exact duplicate overlap or near-duplicate candidates were found by the documented checks.
