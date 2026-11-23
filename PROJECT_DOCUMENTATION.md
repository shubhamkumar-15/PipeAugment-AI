# PipeAugment AI — Comprehensive Project Documentation & Technical Ledger

> **Platform**: PipeAugment AI — Condition-Controlled, Corrosion-Preserving Pipeline Image Augmentation & Inspection System  
> **Repository Location**: `D:\vscode\Gen AI Project\pipeaugment-ai\`  
> **Documentation Scope**: Full Project Lifecycle, Dataset Provenance, Model Training, Traditional & Generative Augmentation Pipelines, Human Review Gate, Backend REST Services, Frontend Design System, Experimental Results, and Audit Verifications.  
> **Document Version**: `1.0.0` (Milestone: Human Review Gate Complete)

---

## 1. Executive Summary & Core Mission

### 1.1 The Industrial Challenge
Visual inspection of oil, gas, and municipal pipeline infrastructure is critical to preventing catastrophic structural failures, environmental contamination, and multi-million-dollar outages. While deep learning computer vision models (such as convolutional neural networks and object detectors) are widely deployed to detect surface corrosion and structural pitting, their real-world reliability suffers severely under domain shifts:
- **Harsh Environmental Lighting**: Deep subterranean, subsea, or indoor pipeline inspections occur under extreme low-light conditions, uneven ambient illumination, or harsh directional spotlighting with specular reflections.
- **Sensor & Transmission Degradation**: Downhole camera sensors frequently suffer from Gaussian noise, high-compression transmission artifacts, and optical blurring.
- **Micro-Texture Fragility**: Unlike general image synthesis where coarse appearance suffices, corrosion inspection models depend on the preservation of high-frequency surface pitting, rust boundary contours, and pipe geometry. Standard generative diffusion models tend to smooth, erase, or hallucinate corrosion when altering lighting.

### 1.2 The PipeAugment AI Solution
**PipeAugment AI** is a full-stack, condition-controlled, corrosion-preserving image augmentation platform. It systematically investigates whether generative AI (specifically Stable Diffusion and ControlNet structural conditioning) can synthesize physically plausible environmental inspection conditions while preserving corrosion morphology better than deterministic computer vision transforms.

### 1.3 Key Accomplishments to Date
1. **Full-Stack Architecture**: Built a production-grade FastAPI REST API backend with chunked streaming and Pillow image validation, paired with a Next.js 16 (Turbopack) frontend featuring a custom **Cyber-Saffron** dark design system.
2. **Dataset Audit & Leakage-Free Manifests**: Ingested and audited the 600-image K-Pipelines Standard dataset, establishing deterministic train/val/test splits with a **strictly sealed 60-image test set**.
3. **Baseline Classifier (Baseline A)**: Trained a ResNet-18 model achieving **0.9667 Validation Macro-F1** (58/60 correct) on the shared validation split.
4. **Traditional Augmentation Engine (Experiment B)**: Engineered 9 OpenCV deterministic lighting and quality transforms, generated 960 augmented images (1,440 pool), and evaluated under an exact compute-budget parity control, reaching 0.9667 Macro-F1 with identical failure cases.
5. **Generative Diffusion Pilot (`diffusion_pilot_v1`)**: Deployed pretrained Stable Diffusion v1.5 img2img on local CUDA hardware, generating 24 environmental lighting variations across 6 training parents.
6. **Canny-ControlNet Adapter Pilot (`controlnet_canny_v1`)**: Engineered structural conditioning adapters (`control_v11p_sd15_canny`), achieving a **+13.7% increase in SSIM** (0.6345 vs 0.5581) and **+66.8% increase in Edge IoU** (0.2527 vs 0.1515) over plain diffusion.
7. **Human Review & Generative Selection Gate**: Created a 100% self-contained, offline HTML visual inspection portal (29.01 MB) evaluating 48 candidate outputs across 4 physical criteria, backed by automated cryptographic validation and CLI summary tooling.
8. **Automated Testing & Verification**: Maintained **71 total passing automated tests** (49 AI unit tests + 22 backend API tests) with zero dependency conflicts.

---

## 2. System Architecture & Directory Ledger

The project is structured as a decoupled monorepo where API services, machine learning engines, raw datasets, experimental runs, and frontend interfaces are strictly separated.

```
pipeaugment-ai/
├── .gitignore
├── README.md                                # Platform setup and quickstart guide
├── PROJECT_DOCUMENTATION.md                 # This master technical documentation
│
├── datasets/                                # Data storage layer
│   ├── raw/
│   │   └── k-pipelines/                     # Cloned upstream K-Pipelines repository
│   │       ├── Standard/                    # 600 images (train: 480, valid: 60, test: 60)
│   │       │   ├── train/ [corrosion / no_corrosion]
│   │       │   ├── valid/ [corrosion / no_corrosion]
│   │       │   └── test/  [corrosion / no_corrosion] (SEALED)
│   │       └── Augmented/                   # Upstream augmented set (excluded from baseline)
│   ├── original/                            # Immutable uploads persisted via FastAPI
│   ├── metadata/                            # Atomic JSON metadata records for uploads
│   ├── augmented/
│   │   └── traditional_v1/                  # 960 deterministic OpenCV generated PNGs
│   └── manifests/
│       ├── train.csv                        # 480 original training samples
│       ├── val.csv                          # 60 validation samples (evaluation split)
│       ├── test.csv                         # 60 test samples (LOCKED & SEALED)
│       ├── train_traditional_v1.csv         # 1,440 samples (480 original + 960 augmented)
│       ├── traditional_v1_generated.csv     # 960 augmented image lineage records
│       ├── traditional_v1_generation_summary.json
│       ├── class_mapping.json               # {"non_corroded": 0, "corroded": 1}
│       └── split_metadata.json              # Dataset partition hashes and statistics
│
├── ai/                                      # AI & Computer Vision Core
│   ├── requirements.txt                     # High-level ML requirements
│   ├── requirements-pre-diffusion.txt       # Frozen dependencies prior to diffusers
│   ├── requirements-lock.txt                # Exact resolved pip lock for .venv-ai
│   ├── configs/
│   │   ├── baseline_resnet18.json           # Baseline A training recipe
│   │   ├── traditional_resnet18.json        # Experiment B training recipe
│   │   ├── traditional_v1.json              # Traditional CV transform parameters
│   │   ├── diffusion_pilot_v1.json          # Pretrained SD 1.5 img2img configuration
│   │   └── controlnet_pilot_v1.json         # ControlNet Canny adapter configuration
│   ├── pipeaugment_ml/                      # Classification & evaluation library
│   │   ├── dataset.py                       # PyTorch Dataset with deterministic sampling
│   │   ├── metrics.py                       # Accuracy, Macro-F1, Precision, Recall, Confusion Matrix
│   │   ├── model.py                         # ResNet-18 builder with head replacement
│   │   └── reproducibility.py               # Deterministic seed locker (Python, NumPy, PyTorch)
│   ├── pipeaugment_cv/                      # Traditional Computer Vision engine
│   │   ├── config.py                        # Dataclasses and parameter boundary rules
│   │   ├── transforms.py                    # 9 stateless OpenCV/NumPy transformations
│   │   └── engine.py                        # TraditionalAugmentationEngine with sanity checks
│   ├── pipeaugment_gen/                     # Generative AI & Diffusion engine
│   │   ├── config.py                        # Diffusion & ControlNet configuration schemas
│   │   ├── engine.py                        # Hugging Face Diffusers CUDA execution engine
│   │   ├── metadata.py                      # JSON metadata and SHA-256 lineage tracking
│   │   └── prompts.py                       # Condition prompt templates and negative prompts
│   ├── scripts/
│   │   ├── audit_dataset.py                 # K-Pipelines integrity, checksum & pHash audit
│   │   ├── prepare_dataset.py               # Reproducible manifest generator (seed 42)
│   │   ├── train_classifier.py              # Model training runner with warmup & early stopping
│   │   ├── evaluate_classifier.py           # Model checkpoint evaluation CLI
│   │   ├── generate_traditional.py          # Batch generation CLI for traditional transforms
│   │   ├── run_diffusion_pilot.py           # SD 1.5 pilot runner (24 outputs)
│   │   ├── run_controlnet_pilot.py          # ControlNet Canny pilot runner (24 outputs)
│   │   ├── build_diffusion_contact_sheets.py
│   │   ├── build_controlnet_comparison_sheets.py
│   │   ├── build_generative_review.py       # Standalone HTML review portal builder
│   │   ├── import_generative_review.py      # Review validator & certified manifest signer
│   │   └── summarize_generative_review.py   # Terminal summary of review decisions
│   └── tests/                               # 49 Pytest Unit Tests
│       ├── test_baseline.py                 # 9 ML dataset & model architecture tests
│       ├── test_traditional.py              # 13 OpenCV transform & engine tests
│       ├── test_generative.py               # 10 Diffusers pipeline & memory tests
│       ├── test_controlnet.py               # 6 ControlNet adapter & metric tests
│       └── test_review.py                   # 11 Review gate & HTML security tests
│
├── backend/                                 # FastAPI REST API Gateway
│   ├── pytest.ini
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements-lock.txt
│   ├── app/
│   │   ├── main.py                          # FastAPI application initialization & mounting
│   │   ├── core/
│   │   │   └── config.py                    # Pydantic BaseSettings configuration
│   │   ├── schemas/
│   │   │   ├── health.py                    # Health check response schemas
│   │   │   └── image.py                     # Image metadata & pagination schemas
│   │   ├── services/
│   │   │   ├── image_validation.py          # Format, dimension, stream & pixel validation
│   │   │   └── image_storage.py             # Atomic file writes & UUIDv4 persistence
│   │   └── api/
│   │       └── routes/
│   │           ├── health.py                # GET /api/v1/health
│   │           └── images.py                # POST /upload, GET /images, GET /{id}, GET /{id}/file
│   └── tests/                               # 22 Pytest API Tests
│       ├── test_health.py
│       └── test_images.py
│
├── frontend/                                # Next.js 16 Presentation Layer
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── public/
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                   # Root layout with fonts & theme provider
│       │   ├── page.tsx                     # Main application workspace dashboard
│       │   ├── globals.css                  # Tailwind CSS v4 & Cyber-Saffron theme tokens
│       │   └── design-system/
│       │       └── page.tsx                 # Interactive UI component showcase
│       ├── components/
│       │   ├── layout/                      # AppShell, Sidebar, TopBar, MobileNav
│       │   └── ui/                          # 17 design system UI primitives
│       └── lib/
│           ├── motion.ts                    # Framer Motion animation configurations
│           └── utils.ts                     # Class merge & formatting utilities
│
└── experiments/                             # Experimental Artifacts & Run History
    ├── runs/
    │   ├── baseline_resnet18_seed42_20261122_111437/  # Baseline A run artifacts
    │   └── traditional_resnet18_seed42_20261122_113419/ # Experiment B run artifacts
    ├── pilots/
    │   ├── traditional_v1/                  # 24-image pilot & contact sheet
    │   ├── diffusion_v1/                    # 24 SD 1.5 outputs, metadata & contact sheets
    │   └── controlnet_canny_v1/             # 24 ControlNet outputs, maps & 5-panel sheets
    ├── reviews/
    │   └── generative_pilot_v1/             # Offline HTML review portal & signed manifests
    └── comparisons/
        └── baseline_vs_traditional_v1.csv   # Metric delta table (Baseline A vs B)
```

---

## 3. Hardware, Execution Environment & Dependencies

To ensure zero cross-contamination between web microservices and heavy GPU deep learning frameworks, the system maintains two strictly isolated Python virtual environments on Windows 11.

### 3.1 Hardware Specifications
- **Host Machine**: Windows 11 Home 64-bit
- **Graphics Processing Unit**: NVIDIA GeForce RTX 5060 Laptop GPU
- **Dedicated Video Memory (VRAM)**: 8,188 MiB (8.19 GB) GDDR6
- **GPU Architecture / Compute Capability**: SM 12.0 (Blackwell Architecture)
- **NVIDIA Display Driver**: Version 621.24 (CUDA UMD 13.5)

### 3.2 Dual Environment Architecture

| Environment Role | Directory | Python Executable | Key Packages | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Backend Gateway** | `backend/.venv` | `backend/.venv/Scripts/python.exe` | FastAPI, Pydantic v2, Uvicorn, Pillow, Pytest | Lightweight REST API, image ingestion, metadata persistence |
| **AI / Deep Learning** | `.venv-ai` | `.venv-ai/Scripts/python.exe` | PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`, Diffusers `0.43.0`, Transformers `5.24.0`, Accelerate `1.16.0`, OpenCV Headless `5.1.0.95` | Model training, CV transforms, GPU diffusion inference, review tooling |
| **Presentation Layer**| `frontend/` | Node.js v24.16.0 / npm 11.13.0 | Next.js 16.4.6 (Turbopack), React 19.3.0, Tailwind CSS 4.3.3, Motion 14.14.0 | Responsive web interface, design system, interactive inspection UI |

### 3.3 CUDA 13.0 & SM 12.0 Blackwell Verification
The NVIDIA Blackwell architecture (SM 12.0) requires explicit runtime verification with CUDA 13.0 wheels. A numerical sanity check was executed in `.venv-ai`:
- **Verification Method**: Deterministic $1024 \times 1024$ float32 matrix multiplication on `cuda:0` compared against CPU reference.
- **Maximum Absolute Discrepancy**: $6.58 \times 10^{-5}$ (well within IEEE 754 float32 precision limits).
- **GPU Tensor Allocation**: Confirmed fully functional without driver fallback or kernel panics.

---

## 4. Dataset Ingestion, Provenance & Comprehensive Audit

### 4.1 Upstream Dataset Provenance
- **Dataset Title**: K-Pipelines Dataset
- **Upstream Git Repository**: `https://github.com/leoxthomas/K-Pipelines`
- **Resolved Git Commit SHA**: `14e2ecb6e4014959fe2935d767c41bd9d5d53cd3` (`main` branch)
- **Retrieval Timestamp**: `2026-11-22T11:04:39Z`
- **License**: GNU General Public License v3.0 (GPL-3.0)
- **Academic Citation**:  
  > Ramos, Casas, and Rivas-Echeverría (2025). *Synthetic generated data for intelligent corrosion classification in oil and gas pipelines*. **Intelligent Systems with Applications**, Volume 25, Article 200463. DOI: `10.1016/j.iswa.2024.200463`.
- **Dataset Nature**: High-resolution synthetic pipeline inspection imagery generated via text-to-image diffusion models for academic corrosion detection research.

### 4.2 Comprehensive Audit Findings (`ai/scripts/audit_dataset.py`)
A rigorous forensic audit was conducted on the **Standard** release (600 images). The **Augmented** release (1,080 images) was deliberately quarantined to prevent uncontrolled synthetic bias in baseline benchmarks.

| Audit Check | Result | Verification Details |
| :--- | :---: | :--- |
| **Total Readable Images** | **600 / 600 (100%)** | All files successfully opened and decoded via Pillow without truncation. |
| **Corrupted or Truncated Files** | **0** | Zero I/O errors or decoding exceptions. |
| **Image Dimensions** | **$512 \times 512$** | Exactly $512 \times 512 \times 3$ across all 600 files. |
| **Color Space / Channels** | **RGB (8-bit)** | Standard 3-channel 24-bit sRGB. |
| **Container Formats** | **100% JPEG** | All 600 images stored as JPEG. |
| **Exact SHA-256 Duplicates** | **0** | 600 unique SHA-256 cryptographic hashes; zero identical duplicates. |
| **Cross-Split Duplicates** | **0** | Zero hash intersections between `train`, `valid`, and `test`. |
| **Conflicting Labels** | **0** | No image file appears under both `corrosion` and `no_corrosion`. |
| **Perceptual Near-Duplicates** | **0** | Perceptual hashing (pHash) with Hamming distance threshold $\le 4$ found 0 near-duplicate candidate pairs. |

### 4.3 Partition and Class Distribution

```
Total Standard Dataset: 600 Images
├── Training Split (80%): 480 Images
│   ├── Corroded: 238 images (49.6%)
│   └── Non-Corroded: 242 images (50.4%)
├── Validation Split (10%): 60 Images
│   ├── Corroded: 31 images (51.7%)
│   └── Non-Corroded: 29 images (48.3%)
└── Test Split (10%): 60 Images [STRICTLY SEALED]
    ├── Corroded: 31 images (51.7%)
    └── Non-Corroded: 29 images (48.3%)
```

### 4.4 Data Lineage & Cryptographic Manifests
Manifests were generated reproducibly via `ai/scripts/prepare_dataset.py` using fixed seed `42`:
- `datasets/manifests/train.csv` (480 rows)
- `datasets/manifests/val.csv` (60 rows)
- `datasets/manifests/test.csv` (60 rows) — **LOCKED & SEALED**
- `datasets/manifests/class_mapping.json`:
  ```json
  {
    "non_corroded": 0,
    "corroded": 1
  }
  ```
- **Manifest Fields Recorded**: `image_id`, `dataset`, `relative_path`, `label`, `class_index`, `split`, `group_id`, `sha256`.

---

## 5. Experiment A: Baseline Classifier (ResNet-18)

### 5.1 Architecture & Training Recipe
The baseline model provides a clean, reproducible empirical benchmark using unmodified standard images:
- **Architecture**: Deep Residual Network 18 (ResNet-18).
- **Pretrained Weights**: Official torchvision `ResNet18_Weights.IMAGENET1K_V1` (`resnet18-f37072fd.pth`).
- **Classification Head**: Single linear layer replacing `model.fc`:
  $$\text{Head}: \mathbb{R}^{512} \longrightarrow \mathbb{R}^{2}$$
- **Deterministic Preprocessing**:
  - Bilinear Resize to $256 \times 256$.
  - Deterministic CenterCrop to $224 \times 224$.
  - ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - *No stochastic augmentations* (no random crops, no horizontal flips) to establish a pure baseline.
- **Optimization Strategy**:
  - Two-Stage Transfer Learning:
    - **Stage 1 (Epochs 1–2 Warmup)**: Backbone frozen (`requires_grad = False`), head learning rate $\eta_{\text{warmup}} = 1.0 \times 10^{-3}$.
    - **Stage 2 (Epochs 3–20 Fine-Tuning)**: Backbone unfrozen, full model fine-tuning with $\eta_{\text{finetune}} = 1.0 \times 10^{-4}$.
  - Optimizer: `AdamW` with weight decay $\lambda = 1.0 \times 10^{-4}$.
  - Loss Function: `CrossEntropyLoss`.
  - Batch Size: 16 (30 optimizer updates per epoch).
  - Early Stopping: Patience of 5 fine-tuning epochs monitored on Validation Macro-F1.

### 5.2 Training Progression & Validation Results
- **Run Directory**: `experiments/runs/baseline_resnet18_seed42_20261122_111437/`

| Epoch | Phase | Train Loss | Val Loss | Val Accuracy | Val Macro-F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Warmup | 0.5940 | 0.3657 | 0.9500 (57/60) | 0.9500 | Head initialization |
| **2** | Warmup | 0.4888 | 0.2695 | 0.8833 (53/60) | 0.8806 | Head convergence |
| **3** | Fine-Tune | 0.2888 | 0.3250 | 0.9333 (56/60) | 0.9333 | Unfrozen backbone |
| **4** | Fine-Tune | **0.1176** | **0.2306** | **0.9667 (58/60)** | **0.9667** | **BEST CHECKPOINT** |
| **5** | Fine-Tune | 0.0388 | 0.2458 | 0.9500 (57/60) | 0.9500 | Patience counter = 1 |
| **6** | Fine-Tune | 0.0365 | 0.1606 | 0.9667 (58/60) | 0.9667 | Patience counter = 2 |
| **7** | Fine-Tune | 0.0420 | 0.1529 | 0.9667 (58/60) | 0.9667 | Patience counter = 3 |
| **8** | Fine-Tune | 0.0161 | 0.2372 | 0.9667 (58/60) | 0.9667 | Patience counter = 4 |
| **9** | Fine-Tune | 0.0193 | 0.1355 | 0.9500 (57/60) | 0.9499 | **Early Stop Triggered** |

### 5.3 Best Validation Metrics Breakdown (Epoch 4)
- **Top-Line Accuracy**: **0.9667** (58 out of 60 correct)
- **Validation Macro-F1**: **0.9667**
- **Validation Macro-Precision**: **0.9677**
- **Validation Macro-Recall**: **0.9677**
- **Class Breakdown**:
  - `non_corroded`: Precision: `0.9355` (29/31), Recall: `1.0000` (29/29), F1: `0.9667` (Support: 29)
  - `corroded`: Precision: `1.0000` (29/29), Recall: `0.9355` (29/31), F1: `0.9667` (Support: 31)
- **Validation Confusion Matrix**:
  $$\begin{pmatrix} \text{TN: } 29 & \text{FP: } 0 \\ \text{FN: } 2 & \text{TP: } 29 \end{pmatrix}$$
- **Zero False Positives**: Not a single non-corroded pipe was falsely flagged as corroded.
- **False Negatives**: Exactly 2 corroded validation images misclassified as non-corroded:
  1. `4efaeb5e-65c8-4abd-83e8-e3414b56726d...`: $P(\text{corroded}) = 0.1867$
  2. `74144b87-062e-4671-97c7-1fec2e1a0496...`: $P(\text{corroded}) = 0.0000$

---

## 6. Experiment B: Traditional Augmentation Engine (`traditional_v1`)

### 6.1 Transformation Mathematics & Implementation
The deterministic computer vision engine is implemented in `ai/pipeaugment_cv/transforms.py`. All transformations operate on 8-bit unsigned integer RGB images ($I \in [0, 255]^{512 \times 512 \times 3}$) and guarantee strict dimension and dtype preservation.

#### A. Environmental Lighting Transforms (5 Transforms)
1. **Brightness Multiplier (`brightness`)**:
   $$I_{\text{out}} = \text{clip}(I_{\text{in}} \times f, 0, 255), \quad f \in [0.80, 1.20], \quad |f - 1.0| \ge 0.02$$
   Implemented via `cv2.convertScaleAbs(img, alpha=factor, beta=0)`.
2. **Contrast Adjustment (`contrast`)**:
   $$I_{\text{out}} = \text{clip}(128.0 + f \times (I_{\text{in}} - 128.0), 0, 255), \quad f \in [0.85, 1.15], \quad |f - 1.0| \ge 0.02$$
3. **Gamma Correction (`gamma`)**:
   $$I_{\text{out}} = \text{round}\left(255 \times \left(\frac{I_{\text{in}}}{255}\right)^\gamma\right), \quad \gamma \in [0.85, 1.20], \quad |\gamma - 1.0| \ge 0.02$$
   Executed via precomputed 256-element uint8 lookup tables (`cv2.LUT`).
4. **Smooth Uneven Illumination (`uneven_illumination`)**:
   Generates a 2D linear gradient ramp across random orientation $\theta \in [0^\circ, 360^\circ]$, normalized to $[1.0 - \Delta, 1.0 + \Delta]$ ($\Delta = |\text{intensity} - 1.0|$, intensity $\in [0.85, 1.15]$), smoothed by Gaussian convolution ($\sigma = \min(H, W)/6$):
   $$I_{\text{out}} = \text{clip}(I_{\text{in}} \odot M_{\text{illum}}, 0, 255)$$
5. **Soft Spatial Shadow (`soft_shadow`)**:
   Simulates occlusions by sampling shadow centroid $(c_x, c_y) \in [0.15W, 0.85W] \times [0.15H, 0.85H]$, radius $R \in [0.25, 0.65] \times \min(H, W)$, and attenuation $\alpha \in [0.75, 0.90]$ with Gaussian distance falloff:
   $$M_{\text{shadow}}(x, y) = 1.0 - (1.0 - \alpha) \cdot \exp\left(-\frac{(x - c_x)^2 + (y - c_y)^2}{2 (R/2)^2}\right)$$
   $$I_{\text{out}} = \text{clip}(I_{\text{in}} \odot M_{\text{shadow}}, 0, 255)$$

#### B. Image Quality Transforms (4 Transforms)
6. **Gaussian Blur (`gaussian_blur`)**:
   Spatial 2D Gaussian kernel with $\sigma \in [0.40, 0.90]$ pixels, kernel width $k = 2\lceil 3\sigma \rceil + 1$.
7. **Additive Gaussian Noise (`gaussian_noise`)**:
   $$I_{\text{out}} = \text{clip}(I_{\text{in}} + \mathcal{N}(0, \sigma_{\text{noise}}^2), 0, 255), \quad \sigma_{\text{noise}} \in [2.0, 6.0]$$
   Sampled using isolated NumPy generator `np.random.default_rng(seed)`.
8. **JPEG Compression Simulation (`jpeg_compression`)**:
   Encodes to memory buffer with quality $q \in [75, 90]$ via `cv2.imencode`, decodes back via `cv2.imdecode`.
9. **Downsampling & Bicubic Restoration (`downsample_restore`)**:
   Downsamples to scale $s \in [0.65, 0.90]$ using area interpolation (`cv2.INTER_AREA`), followed by restoration to original $(512, 512)$ using bicubic interpolation (`cv2.INTER_CUBIC`).

### 6.2 Automated Technical Sanity Checks
Every single generated image is evaluated against 5 quantitative sanity constraints:
1. Shape invariant: Output $(H, W, C)$ exactly matches parent $(512, 512, 3)$.
2. Dtype check: Output is strictly `np.uint8` in $[0, 255]$.
3. Finite pixel check: Zero `NaN` or `Inf` values.
4. Non-constant check: Image is not solid black, solid white, or flat constant.
5. Non-zero perturbation: Mean absolute pixel difference $\Delta > 0.0$ relative to parent.

### 6.3 Batch Generation Ledger
- **Input Parents**: 480 training images from `datasets/manifests/train.csv`.
- **Generation Strategy**: Exactly 2 variations per parent:
  - Variant 1: Sampled from the 5 environmental lighting transforms.
  - Variant 2: Sampled from the 4 image quality transforms.
- **Output Count**: Exactly 960 augmented images generated into `datasets/augmented/traditional_v1/`.
- **Sanity Pass Rate**: **960 / 960 (100.0%)** passed all checks. Zero failed generations.
- **Combined Training Pool**: 1,440 images (`datasets/manifests/train_traditional_v1.csv`).
- **Leakage Audit**: 0 generated images or parent IDs present in `val.csv` or `test.csv`.

### 6.4 Training Budget Parity Control
To rigorously test whether data diversity helps without confounding compute budget:
- **Baseline A Budget**: 480 samples/epoch, batch size 16 = **30 optimizer steps/epoch**.
- **Experiment B Matched Budget**:
  - Pool size: 1,440 images.
  - Samples drawn per epoch: **480 samples** (drawn reproducibly via `seed + epoch * 1000`).
  - Class proportions: Exactly 240 non-corroded and 240 corroded per epoch.
  - Optimizer updates per epoch: **30 steps** (strictly identical to Baseline A).

### 6.5 Benchmark Results: Baseline A vs. Experiment B

| Metric | Baseline A (Standard) | Experiment B (Traditional v1) | Delta | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Training Pool** | 480 images | 1,440 images | +960 images | Augmented diversity |
| **Epoch Budget** | 480 samples / 30 steps | 480 samples / 30 steps | 0 | Strict parity |
| **Best Checkpoint Epoch** | **Epoch 4** | **Epoch 3** | **-1 epoch** | Peak performance reached 1 epoch earlier |
| **Total Steps Trained** | 270 steps | 240 steps | -30 steps | Shorter convergence window |
| **Validation Accuracy** | **0.9667** (58/60) | **0.9667** (58/60) | **0.0000** | Identical top-line accuracy |
| **Validation Macro-F1** | **0.9667** | **0.9667** | **0.0000** | Identical F1 score |
| **Corrosion Precision** | **1.0000** (29/29) | **1.0000** (29/29) | 0.0000 | Zero false-positive corrosion alerts |
| **Corrosion Recall** | **0.9355** (29/31) | **0.9355** (29/31) | 0.0000 | 2 false negatives |
| **Identical Failure Samples** | `4efaeb5e...`, `74144b87...` | `4efaeb5e...`, `74144b87...` | 0 | Failed on exact same images |

#### Key Empirical Insight
Traditional deterministic augmentations accelerated convergence by 30 steps but did **not** resolve the model's blind spot on subtle, low-contrast corrosion defects. This provided empirical justification for investigating generative diffusion models capable of synthesizing contextual surface illumination.

---

## 7. Generative Modeling Pilot 1: Stable Diffusion v1.5 (`diffusion_pilot_v1`)

### 7.1 Architecture & Pipeline Setup
- **Base Model**: Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)
- **Resolved Commit**: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- **License**: CreativeML OpenRAIL-M
- **Pipeline**: `StableDiffusionImg2ImgPipeline` (via `AutoPipelineForImage2Image`)
- **Precision**: `float16` (`variant="fp16"`)
- **Scheduler**: `PNDMScheduler` (1,000 training timesteps, scaled linear beta)
- **Safety Checker**: Active (`StableDiffusionSafetyChecker`, 0 NSFW triggers)

### 7.2 Hardware Execution & Memory Footprint
- **Device**: NVIDIA GeForce RTX 5060 Laptop GPU (8,188 MB dedicated VRAM).
- **Placement**: Direct CUDA allocation (`pipeline.to("cuda")`).
- **Peak GPU Allocated VRAM**: **3,280.44 MB** (~3.28 GB).
- **VRAM Headroom Remaining**: ~4.9 GB headroom on local hardware.
- **Inference Latency**:
  - Initial cold start (JIT & CUDA graph compile): 17.54s.
  - Steady-state ($s=0.20$, 6 denoising steps): **~0.97s – 0.98s per image**.
  - Steady-state ($s=0.35$, 10 denoising steps): **~1.41s – 1.43s per image**.

### 7.3 Conditioning & Prompt Engineering
- **Source Selection**: 6 deterministically sampled training parents (3 corroded, 3 non-corroded; seeds derived via SHA-256).
- **Environmental Conditions Tested**:
  - `low_light_inspection`: *"dim, low-light inspection conditions, underexposed ambient lighting, weak inspection illumination."*
  - `directional_industrial_lighting`: *"harsh directional industrial work light casting focused illumination and distinct directional shadows."*
- **Standardized Negative Prompt**:
  > *"changed pipe geometry, added or missing pipes, changed joints, altered structural features, painted-over surfaces, invented surface defects, removed corrosion, added corrosion, text, watermarks, signature, logo, illustration, rendering, 3d render, cartoon, blurry, heavy blur, out of focus, distorted, artifact"*
- **Target Outputs**: 6 parents $\times$ 2 conditions $\times$ 2 strengths ($0.20, 0.35$) = **24 outputs**.

### 7.4 Quantitative Diagnostics & Limitations

| Denoising Strength | Effective Steps | Mean RGB SSIM | SSIM Range | Mean Pixel Difference |
| :---: | :---: | :---: | :---: | :---: |
| **0.20** | 6 steps | **0.6397** | [0.5782, 0.7416] | 13.48 |
| **0.35** | 10 steps | **0.5578** | [0.5032, 0.6756] | 17.72 |
| **Overall Pilot** | — | **0.5988** | [0.5032, 0.7416] | 15.60 |

#### Observed Failure Modes
At denoising strength 0.35, the plain img2img pipeline experienced structural drifting:
1. **Pipe Boundary Warping**: Cylindrical silhouettes and weld seams lost strict alignment.
2. **Corrosion Smoothing / Hallucination**: Fine rust textures softened into smooth paint-like patches, or new rust appeared on non-corroded pipes.
3. **Absence of Dense Guidance**: Text prompts alone proved incapable of enforcing spatial boundary constraints.

---

## 8. Generative Modeling Pilot 2: Canny-ControlNet Adapter (`controlnet_canny_v1`)

### 8.1 Architecture & Structural Guidance
To eliminate the geometric drift observed in Plain SD 1.5, structural edge conditioning was introduced:
- **ControlNet Model**: `lllyasviel/control_v11p_sd15_canny`
- **Resolved Commit**: `115a470d547982438f70198e353a921996e2e819`
- **Model Checkpoint Size**: 723 MB (`diffusion_pytorch_model.fp16.safetensors`)
- **Pipeline**: `StableDiffusionControlNetImg2ImgPipeline` (fp16 on CUDA)
- **Structural Feature Extractor**: Canny edge detection ($T_{\text{low}} = 100$, $T_{\text{high}} = 200$) executed on $512 \times 512$ source images, replicated to 3 RGB channels.

### 8.2 Hardware & Memory Metrics
- **Peak GPU Allocated VRAM**: **4,012.16 MB** (~4.01 GB).
- **Peak GPU Reserved VRAM**: **4,280.00 MB** (~4.28 GB).
- **Minimum Free VRAM Headroom**: **2,729.00 MB** (~2.73 GB).
- **Net Memory Overhead**: ControlNet added only 731.72 MB of active tensor allocation over plain img2img.
- **Inference Runtime**: **~1.87s – 1.88s per image** (10 steps @ $s=0.35$). Zero CPU offloading required.

### 8.3 Rigorous Head-to-Head Comparison (Strength = 0.35)
All 24 outputs matched the exact same 6 training parents, conditions, and seeds:

| Metric | Plain SD 1.5 (`s=0.35`) | ControlNet Canny (`c=0.50`) | ControlNet Canny (`c=0.80`) | ControlNet $c=0.80$ Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Mean RGB SSIM** | **0.5581** | **0.6106** | **0.6345** | **+0.0764 (+13.7%)** |
| **Min RGB SSIM** | 0.5032 | 0.5623 | 0.5832 | +0.0800 |
| **Max RGB SSIM** | 0.6756 | 0.7172 | 0.7317 | +0.0561 |
| **Mean Edge IoU** | **0.1515** | **0.2128** | **0.2527** | **+0.1012 (+66.8%)** |
| **Min Edge IoU** | 0.1304 | 0.1808 | 0.2081 | +0.0777 |
| **Max Edge IoU** | 0.1693 | 0.2476 | 0.2829 | +0.1136 |
| **Mean Edge Dice** | 0.2632 | 0.3505 | 0.4029 | **+0.1397 (+53.1%)** |
| **Steady Runtime** | ~1.42s | ~1.87s | ~1.88s | +0.46s |
| **VRAM Allocated** | 3,280.44 MB | 4,012.16 MB | 4,012.16 MB | +731.72 MB |

### 8.4 Per-Parent Structural Diagnostic Breakdown

| Parent ID | Class | Edge Density | Plain SSIM | CNet 0.50 SSIM | CNet 0.80 SSIM | Plain Edge IoU | CNet 0.80 Edge IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `27a30c35...` | non_corroded | 0.0631 | 0.6756 | 0.7164 | 0.7311 | 0.1648 | 0.2829 |
| `0888523b...` | non_corroded | 0.1225 | 0.5573 | 0.6335 | 0.6558 | 0.1693 | 0.2818 |
| `a0b0ecba...` | non_corroded | 0.0706 | 0.5186 | 0.5793 | 0.6010 | 0.1304 | 0.2173 |
| `796e406a...` | corroded | 0.1077 | 0.5032 | 0.5633 | 0.5832 | 0.1395 | 0.2393 |
| `13d2a961...` | corroded | 0.0798 | 0.5369 | 0.5648 | 0.5847 | 0.1583 | 0.2617 |
| `d6dc47f8...` | corroded | 0.0694 | 0.5568 | 0.6067 | 0.6514 | 0.1466 | 0.2333 |

### 8.5 5-Panel Comparative Contact Sheets
Generated 12 side-by-side comparison sheets (`experiments/pilots/controlnet_canny_v1/contact_sheets/`):
- Panel 1: Original Source Image ($512 \times 512$)
- Panel 2: Canny Edge Condition Map ($T = [100, 200]$)
- Panel 3: Plain SD 1.5 img2img ($s=0.35$)
- Panel 4: ControlNet Canny ($c=0.50, s=0.35$)
- Panel 5: ControlNet Canny ($c=0.80, s=0.35$)

---

## 9. Human Review & Generative Selection Gate (`generative_pilot_v1`)

### 9.1 The Scientific Rationale
In safety-critical industrial vision, higher SSIM and Edge IoU scores are merely descriptive mathematical heuristics. They **do not prove** that physical corrosion morphology (pit depth, rust distribution, micro-cracking) is authentically preserved. Therefore, an explicit, auditable **Human Review Gate** was instituted as a mandatory prerequisite before admitting any generative variation to training manifests.

### 9.2 The Offline Interactive Review Portal
- **Location**: `experiments/reviews/generative_pilot_v1/index.html`
- **Architecture**: A single, 29.01 MB self-contained HTML/CSS/JavaScript web application generated by `ai/scripts/build_generative_review.py`.
- **Zero-Dependency Guarantee**: Contains zero npm dependencies, zero remote CDN scripts, zero external stylesheet requests, and zero remote font downloads. All candidate images and reference parents are embedded directly as base64 data URIs.
- **Evaluation Cards**: 12 parent-condition comparison cards containing all **48 candidate outputs** (24 Plain SD 1.5 + 24 ControlNet Canny).
- **Inspection Modal**: Interactive high-resolution viewer offering 1:1 and $2\times$ pixelated inspection to inspect micro-texture rust boundaries.
- **Local State & Export**: Automatically saves drafts to browser `localStorage` and provides one-click export to signed JSON and CSV.

### 9.3 Structured Evaluation Criteria

| Criterion | Evaluation Question | Options | Strict Approval Rule |
| :--- | :--- | :---: | :--- |
| **Geometry Preserved?** | Does the pipe retain cylindrical boundaries, joints, welds, and background alignment? | Yes / No / Uncertain | Must be **Yes** |
| **Corrosion Preserved?** | Are rust pits retained without hallucinating new defects or erasing existing corrosion? | Yes / No / Uncertain | Must be **Yes** |
| **Condition Visible?** | Is the requested lighting (dim ambient or directional spotlight) clearly perceptible? | Yes / No / Uncertain | Must be **Yes** |
| **Realism Acceptable?** | Does the image appear authentic without AI blurring, comic rendering, or latent artifacts? | Yes / No / Uncertain | Must be **Yes** |

> **Approval Invariant**: The system strictly forbids marking an output `"approved"` unless **all 4 criteria are "Yes"** and a non-empty Reviewer Name is provided. Any missing positive answer triggers validation failure.

### 9.4 Certification CLI & Certified Manifests
- **Validation Script**: `ai/scripts/import_generative_review.py`
  - Re-computes on-disk SHA-256 hashes of all 48 image files (halts if modified).
  - Enforces schema validity and non-empty reviewer identity.
  - Generates immutable signed manifests:
    - `experiments/reviews/generative_pilot_v1/reviewed_manifest_v1.json`
    - `experiments/reviews/generative_pilot_v1/reviewed_manifest_v1.csv`
- **Summary Reporter**: `ai/scripts/summarize_generative_review.py`

### 9.5 Current Human Review Status & Findings (`reviewed_summary_v1.txt`)
- **Total Records Evaluated**: 48 candidate outputs
- **Overall Decisions**:
  - **Approved**: 1 / 48 (2.1%)
  - **Rejected**: 3 / 48 (6.2%)
  - **Needs Review**: 2 / 48 (4.2%)
  - **Pending**: 42 / 48 (87.5%)
- **Method Breakdown**:
  - Plain SD 1.5 ($s=0.20$): 0 approved, 1 rejected
  - Plain SD 1.5 ($s=0.35$): 0 approved, 2 rejected
  - ControlNet Canny ($c=0.50$): 0 approved, 0 rejected
  - ControlNet Canny ($c=0.80$): **1 approved**, 0 rejected
- **Failure Mode Statistics for Rejected Outputs**:
  - Pipe Geometry Distorted: 2 / 3
  - Corrosion Hallucinated or Erased: 3 / 3
  - Condition Insufficiently Visible: 3 / 3
  - Visual Artifacts / Low Realism: 3 / 3
- **Governance Gate**: Experiment C (Classifier training on generative data) remains **strictly blocked** until pending evaluations are completed and certified.

---

## 10. Backend Architecture & REST API Service (`pipeaugment-ai/backend`)

### 10.1 Framework & Core Design
- **Framework**: FastAPI with Pydantic v2 and `pydantic-settings`.
- **Modularity**: Decoupled service architecture (`app/services/image_storage.py` and `app/services/image_validation.py`) isolating HTTP transport logic from filesystem operations.
- **Base Prefix**: `/api/v1`

### 10.2 REST Endpoints Ledger

| HTTP Method | Endpoint Path | Functionality | Status Codes |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/v1/health` | Health probe returning service status, name, and version | 200 |
| `POST` | `/api/v1/images/upload` | Validates and stores original inspection image | 201, 413, 415, 422 |
| `GET` | `/api/v1/images` | Paginated listing of uploaded images (newest first, 1–100 limit) | 200, 422 |
| `GET` | `/api/v1/images/{id}` | Retrieves metadata record for a specific image by UUID | 200, 404, 422 |
| `GET` | `/api/v1/images/{id}/file` | Streams raw immutable image bytes with canonical `Content-Type` | 200, 404, 422 |

### 10.3 Security, Validation & Storage Guarantees
1. **Chunked Streaming Ingestion**: Uploads are consumed in 64 KB chunks. If total bytes exceed 10 MB (`MAX_UPLOAD_BYTES = 10 * 1024 * 1024`), the stream aborts immediately with HTTP 413 without reading the remaining payload into RAM.
2. **Strict Format & Header Verification**:
   - Supported Formats: JPEG (`.jpg`), PNG (`.png`), WEBP (`.webp`).
   - Mismatches between file extension and detected byte header (e.g., JPEG renamed to `.png`) are rejected with HTTP 415.
3. **Pillow Pixel Decoding & Dimension Bounds**:
   - Files are actively decoded via Pillow. Renamed text files or corrupt byte streams are rejected with HTTP 422.
   - Images exceeding 25,000,000 pixels (`MAX_IMAGE_PIXELS = 25_000_000`) are rejected with HTTP 422 to prevent decompression bomb denial-of-service.
   - Multi-frame or animated files (GIFs/animated WEBPs) are strictly rejected with HTTP 422.
4. **Atomic Immutability & Safe Rollback**:
   - Original images are stored under `datasets/original/<uuid>.<ext>`.
   - Metadata is stored under `datasets/metadata/<uuid>.json`.
   - Writes use hidden temporary files (`.tmp_*`) and atomic operating system rename (`os.replace`). If metadata generation fails, all temporary and committed files are immediately unlinked, preventing orphaned data.

---

## 11. Frontend Architecture & Cyber-Saffron Design System (`frontend/`)

### 11.1 Technology Stack
- **Framework**: Next.js 16.4.6 using Turbopack engine and React 19.3.0.
- **Styling**: Tailwind CSS v4 with custom CSS variable design tokens.
- **Component Primitives**: Radix UI headless components (`@radix-ui/react-slot`, `popover`, `dropdown-menu`, `scroll-area`).
- **Icons**: Lucide React (`lucide-react`).
- **Animation**: Motion v14 (`motion`) spring dynamics.

### 11.2 The Cyber-Saffron Dark Theme
Designed specifically for industrial inspection contexts where high visual contrast and low eye fatigue are required during extended inspection shifts:
- **Base Surfaces**:
  - Canvas Base: `#0c0d0e`
  - Low Surface: `#141618`
  - High Surface: `#1e2124`
- **Accent Tokens (Cyber-Saffron)**:
  - Saffron Base: `#f59e0b` (Amber-500)
  - Saffron Bright: `#eab308` (Yellow-500)
  - Saffron Muted / Glow: `rgba(245, 158, 11, 0.15)`
- **Typography**:
  - Headings & Structural Titles: **Space Grotesk** (geometric industrial display)
  - Body Prose & Data Tables: **DM Sans** (clean, highly legible grotesque)
  - Metrics, Timestamps & Identifiers: **JetBrains Mono**

### 11.3 Component Library Ledger
The design system library (`frontend/src/components/`) provides a full suite of interactive UI components:
- **Layout**: `AppShell` (responsive sidebar + topbar layout), `Sidebar`, `TopBar`, `MobileNav`.
- **Primitives**: `Button` (primary, secondary, danger, ghost with loading/icon slots), `Badge` (with pulse dot indicator), `Card` & `SpotlightCard` (interactive hover lighting), `Chip`, `Dialog`, `Drawer`, `DropdownMenu`, `EmptyState`, `Input`, `Popover`, `RevealText`, `ScrollArea`, `SegmentedControl`, `Skeleton`, `Slider`, and `Table`.
- **Interactive Component Showcase**: Accessible at `http://localhost:3000/design-system` (`frontend/src/app/design-system/page.tsx`).

---

## 12. Automated Testing, Quality Assurance & Verification

The project enforces continuous verification across both Python environments.

```
Total Automated Tests: 71 Passing
├── AI Core Test Suite (.venv-ai): 49 Tests Passed (7.55s)
│   ├── Baseline ML Suite (test_baseline.py): 9 tests
│   ├── Traditional CV Suite (test_traditional.py): 13 tests
│   ├── Generative Diffusion Suite (test_generative.py): 10 tests
│   ├── ControlNet Adapter Suite (test_controlnet.py): 6 tests
│   └── Review Gate & Security Suite (test_review.py): 11 tests
└── Backend API Test Suite (backend/.venv): 22 Tests Passed (0.48s)
    ├── Health Probe Suite (test_health.py): 3 tests
    └── Image Ingestion & Storage Suite (test_images.py): 19 tests
```

### 12.1 Detailed AI Test Coverage (`ai/tests/`)
1. `test_baseline.py`: Verifies manifest schema, deterministic PyTorch dataset loading, batch generation, ResNet-18 head replacement, freeze/unfreeze mechanisms, and metric calculations.
2. `test_traditional.py`: Verifies all 9 pure-function OpenCV transformations, parameter boundary enforcement, uint8 range preservation, and the 5 automated sanity checks.
3. `test_generative.py`: Verifies prompt formatting, negative prompt injection, deterministic seed derivation via SHA-256, memory allocation guards, and metadata tracking.
4. `test_controlnet.py`: Verifies Canny edge map generation, ControlNet pipeline loading, tensor dimensionality matching, SSIM computation, and Edge IoU / Dice overlap metrics.
5. `test_review.py`: Verifies review inventory serialization, rejection of unauthorized candidate IDs, rejection of tampered file hashes, strict validation of positive criteria for approvals, exclusion of pending outputs, and offline HTML security (zero remote URL requests).

### 12.2 Detailed Backend Test Coverage (`backend/tests/`)
1. `test_health.py`: Validates `/api/v1/health` response schema and runtime versioning.
2. `test_images.py`: Validates valid JPEG, PNG, and WEBP uploads; tests 413 payload truncation on 10 MB limits; tests 415 extension/MIME mismatches; tests 422 rejection of corrupt and animated files; tests atomic storage and rollback; tests pagination and image binary streaming.

---

## 13. Comprehensive Master Results Matrix

| Metric / Dimension | Baseline A (Standard) | Experiment B (Traditional) | Plain SD 1.5 Pilot | ControlNet Canny Pilot |
| :--- | :---: | :---: | :---: | :---: |
| **Model Architecture** | ResNet-18 (ImageNet-1K) | ResNet-18 (ImageNet-1K) | Stable Diffusion v1.5 | SD 1.5 + Canny ControlNet |
| **Input Data Source** | 480 Original Standard | 480 Orig + 960 Trad (1,440 Pool) | 6 Deterministic Parents | Same 6 Deterministic Parents |
| **Augmentation Type** | None (Deterministic CenterCrop) | 9 OpenCV Lighting/Quality | Latent Text-to-Image img2img | Dense Canny Edge Guided img2img |
| **Evaluation Split** | Shared Val Split (60 imgs) | Shared Val Split (60 imgs) | 24 Pilot Variations | 24 Pilot Variations |
| **Test Set Status** | **LOCKED (60 imgs)** | **LOCKED (60 imgs)** | **LOCKED (60 imgs)** | **LOCKED (60 imgs)** |
| **Best Epoch / Steps** | Epoch 4 (270 steps) | Epoch 3 (240 steps) | N/A (Generative Pilot) | N/A (Generative Pilot) |
| **Validation Accuracy** | **0.9667** (58/60) | **0.9667** (58/60) | N/A | N/A |
| **Validation Macro-F1** | **0.9667** | **0.9667** | N/A | N/A |
| **Corrosion Precision** | **1.0000** (29/29) | **1.0000** (29/29) | N/A | N/A |
| **Corrosion Recall** | **0.9355** (29/31) | **0.9355** (29/31) | N/A | N/A |
| **Mean RGB SSIM ($s=0.35$)** | N/A | N/A | **0.5581** | **0.6345 (+13.7%)** |
| **Mean Edge IoU ($s=0.35$)** | N/A | N/A | **0.1515** | **0.2527 (+66.8%)** |
| **Peak GPU VRAM (Allocated)** | ~1.1 GB | ~1.1 GB | **3,280.44 MB** | **4,012.16 MB** |
| **Inference Time / Image** | ~0.005s | ~0.005s | ~1.42s ($s=0.35$) | ~1.88s ($s=0.35$) |
| **Human Review Status** | N/A | Mathematically Validated | Pending / Evaluated in Gate | 1 Certified Approved |
| **Training Eligibility** | Baseline Reference | Approved for Training | Strictly False | Strictly False (Pending Review) |

---

## 14. CLI Commands & Complete Reproduction Ledger

All commands are designed for Windows PowerShell and execute against the explicit virtual environment Python binaries.

### 14.1 Dataset & Baseline Classification (Baseline A)
```powershell
# 1. Forensic Audit of K-Pipelines Dataset
& ".\.venv-ai\Scripts\python.exe" ai/scripts/audit_dataset.py

# 2. Manifest Generation with Fixed Seed 42
& ".\.venv-ai\Scripts\python.exe" ai/scripts/prepare_dataset.py

# 3. Execute Baseline ML Unit Tests
& ".\.venv-ai\Scripts\python.exe" -m pytest ai/tests/test_baseline.py -v

# 4. Train Baseline ResNet-18 Model
& ".\.venv-ai\Scripts\python.exe" ai/scripts/train_classifier.py --config ai/configs/baseline_resnet18.json

# 5. Evaluate Best Baseline Checkpoint on Validation Set
& ".\.venv-ai\Scripts\python.exe" ai/scripts/evaluate_classifier.py `
  --checkpoint experiments/runs/baseline_resnet18_seed42_20261122_111437/best_model.pth `
  --manifest datasets/manifests/val.csv `
  --output-dir experiments/runs/eval_val
```

### 14.2 Traditional Augmentation & Benchmark (Experiment B)
```powershell
# 1. Execute Traditional CV Unit Tests
& ".\.venv-ai\Scripts\python.exe" -m pytest ai/tests/test_traditional.py -v

# 2. Generate 960 Traditional Augmentations
& ".\.venv-ai\Scripts\python.exe" ai/scripts/generate_traditional.py --full

# 3. Train Experiment B Under Matched Budget
& ".\.venv-ai\Scripts\python.exe" ai/scripts/train_classifier.py --config ai/configs/traditional_resnet18.json
```

### 14.3 Generative Modeling Pilots (Diffusion & ControlNet)
```powershell
# 1. Execute Generative & ControlNet Unit Tests
& ".\.venv-ai\Scripts\python.exe" -m pytest ai/tests/test_generative.py ai/tests/test_controlnet.py -v

# 2. Run Pretrained Stable Diffusion v1.5 Pilot (24 images)
& ".\.venv-ai\Scripts\python.exe" ai/scripts/run_diffusion_pilot.py --config ai/configs/diffusion_pilot_v1.json

# 3. Run Canny ControlNet Structural Adapter Pilot (24 images)
& ".\.venv-ai\Scripts\python.exe" ai/scripts/run_controlnet_pilot.py --config ai/configs/controlnet_pilot_v1.json

# 4. Build 5-Panel Side-by-Side Comparison Sheets
& ".\.venv-ai\Scripts\python.exe" ai/scripts/build_controlnet_comparison_sheets.py
```

### 14.4 Human Review Gate & Verification
```powershell
# 1. Build Standalone Offline HTML Review Application
& ".\.venv-ai\Scripts\python.exe" ai/scripts/build_generative_review.py

# 2. Open Review Portal in Default Browser
Invoke-Item "experiments/reviews/generative_pilot_v1/index.html"

# 3. Import and Cryptographically Validate Exported Review
& ".\.venv-ai\Scripts\python.exe" ai/scripts/import_generative_review.py `
  --input experiments/reviews/generative_pilot_v1/generative_review_export_2026-11-22T12-53-06-615Z.json `
  --summary

# 4. Inspect Real-Time Review Certification Status
& ".\.venv-ai\Scripts\python.exe" ai/scripts/summarize_generative_review.py

# 5. Run Full AI Test Suite (49 Tests)
& ".\.venv-ai\Scripts\python.exe" -m pytest ai/tests -v
```

### 14.5 Backend & Frontend Service Operations
```powershell
# 1. Run Backend Pytest Suite (22 Tests)
& ".\backend\.venv\Scripts\python.exe" -m pytest backend/tests -v

# 2. Launch FastAPI Backend Server
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 3. Launch Frontend Development Server
Set-Location frontend
npm run dev
# Access UI at http://localhost:3000 and Design System at http://localhost:3000/design-system
```

---

## 15. Research Roadmap & Next Experimental Phases

```
Completed Milestones:
[x] Full-Stack Service & Presentation Foundation (FastAPI + Next.js 16 Cyber-Saffron)
[x] K-Pipelines Dataset Ingestion, Audit & Leak-Free Split Manifests
[x] Experiment A: Baseline Classifier (ResNet-18, Macro-F1 = 0.9667)
[x] Experiment B: Traditional CV Augmentation Engine (9 Transforms, 960 Images, Macro-F1 = 0.9667)
[x] Generative Pilot 1: Plain SD 1.5 Image-to-Image (SSIM = 0.5581, Edge IoU = 0.1515)
[x] Generative Pilot 2: Canny ControlNet Adapter (SSIM = 0.6345, Edge IoU = 0.2527)
[x] Human Review & Generative Selection Gate (Offline Portal, Cryptographic Verification, 48 Candidates)

Upcoming Milestones:
[ ] Milestone 1: Complete Human Certification of Remaining 42 Generative Candidates
[ ] Milestone 2: Batch Generative Augmentation using Certified ControlNet Parameters
[ ] Milestone 3: Experiment C — Train Classifier on Generatively Augmented Dataset
[ ] Milestone 4: Comprehensive Multi-Seed Comparison (Baseline A vs. Traditional B vs. Generative C)
[ ] Milestone 5: Unseal and Evaluate Locked 60-Image Test Split (Final Generalization Benchmark)
[ ] Milestone 6: Integrate Generative Augmentation Job Pipeline into Backend & Frontend UI
```

### Concluding Note
PipeAugment AI maintains strict scientific and data governance standards. Every synthetic image carries full cryptographic lineage back to its source parent image, and no generative image is admitted to downstream training without passing both automated quantitative checks and certified human inspection. All code, configuration files, checkpoints, and manifests are fully reproducible under the procedures detailed in this document.
