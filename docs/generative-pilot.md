# Pretrained Diffusion Image-to-Image Pilot Report (`diffusion_pilot_v1`)

## Overview
This document records the architecture, environment setup, model provenance, execution metrics, and technical diagnostics of the **first condition-controlled generative image-to-image pilot** for PipeAugment AI. The pilot benchmarks pretrained Stable Diffusion v1.5 on 6 training-only pipeline inspection parents (3 non-corroded, 3 corroded) across two environmental lighting conditions at two denoising strengths (0.20 and 0.35), producing 24 target outputs.

> [!IMPORTANT]
> **Experimental Governance & Training Policy**:
> - **Training Eligibility**: All 24 pilot outputs are strictly marked `"training_eligibility": false`. None of these outputs have been added to any training manifest.
> - **Human Review Status**: Marked as `"human_review_status": "pending"` in all metadata records and in [review.csv](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/review.csv). Technical validation metrics confirm image integrity and non-trivial generation, but do **not** prove corrosion preservation or visual fidelity to field environments.
> - **Sealed Test Set**: The test manifest (`datasets/manifests/test.csv`, 60 images) remains strictly locked and un-evaluated.

---

## 1. Model Provenance, Architecture, and License

| Parameter | Value | Details / Verification |
| :--- | :--- | :--- |
| **Model Repository** | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Official public Hugging Face Hub repository |
| **Resolved Commit SHA** | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | Fixed immutable commit hash |
| **Model License** | CreativeML OpenRAIL-M | Gated access: `False` (public access, no tokens required) |
| **Pipeline Class** | `StableDiffusionImg2ImgPipeline` | Loaded via `AutoPipelineForImage2Image` |
| **Weights Precision** | `float16` (`variant="fp16"`) | Minimal component download: only fp16 SafeTensors fetched |
| **Scheduler Class** | `PNDMScheduler` | 1000 train timesteps, `beta_schedule="scaled_linear"` |
| **Safety Checker** | `StableDiffusionSafetyChecker` | Active; flags recorded in metadata (0 NSFW flags encountered) |
| **Local Cache Location** | `C:\Users\shubh\.cache\huggingface\hub\` | Standard Hugging Face hub cache |

---

## 2. Minimal Dependencies and Environment

The `.venv-ai` virtual environment (Python 3.14.5) was updated with only the minimal runtime requirements for inference, without compiling from source:
- `diffusers==0.43.0`
- `transformers==5.24.0`
- `accelerate==1.16.0`
- `safetensors==0.8.0`
- `huggingface_hub==2.4.0`

**Preserved Installations**:
- PyTorch: `2.14.1+cu130` (CUDA 13.0, sm_120 Blackwell support)
- Torchvision: `0.29.1+cu130`
- OpenCV Headless: `5.1.0.95`
- Dependency verification: `pip check` confirmed zero broken requirements.

---

## 3. Memory Strategy and GPU Execution

- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8,188 MB dedicated VRAM).
- **Placement Strategy**: Direct CUDA placement (`pipeline.to("cuda")`).
- **Memory Consumption**:
  - Peak GPU VRAM allocated during inference: **3,280.44 MB** (~3.28 GB).
  - Dedicated GPU memory headroom remaining: ~4.7 GB.
  - Offload fallback (`enable_model_cpu_offload()`): Implemented as automatic OOM recovery handler, but never triggered because VRAM headroom was abundant.
- **Inference Runtime**:
  - Smoke test / first image (kernel JIT & graph initialization): 17.54s.
  - Steady-state per-image generation runtime:
    - Strength 0.20 (6 effective denoising steps): **~0.97s – 0.98s**
    - Strength 0.35 (10 effective denoising steps): **~1.41s – 1.43s**
  - Total batch elapsed time for 23 variations: **46.72s**.

---

## 4. Prompt Engineering and Conditioning

### Environmental Condition Templates

Prompts are designed to alter ambient lighting neutrally across both classes without introducing class-correlated distortions or instructions.

- **Base Intent**:
  > *"Photorealistic photograph of the same industrial pipeline, same viewpoint, same pipe shape, same joints and background, retaining existing surface texture, rust locations and surface defects"*

- **Condition 1: Low-Light Inspection (`low_light_inspection`)**:
  > *"dim, low-light inspection conditions, underexposed ambient lighting, weak inspection illumination."*

- **Condition 2: Directional Industrial Lighting (`directional_industrial_lighting`)**:
  > *"harsh directional industrial work light casting focused illumination and distinct directional shadows."*

- **Standardized Negative Prompt**:
  > *"changed pipe geometry, added or missing pipes, changed joints, altered structural features, painted-over surfaces, invented surface defects, removed corrosion, added corrosion, text, watermarks, signature, logo, illustration, rendering, 3d render, cartoon, blurry, heavy blur, out of focus, distorted, artifact"*

### Prompt Limitations
Prompts and negative prompts are soft guidance requests to the latent diffusion model; they do **not** enforce hard geometric, physical, or morphological constraints. Generative models can hallucinate textures, soften rust edges, or shift pipeline seams unless constrained by dense conditioning (e.g., ControlNet depth or edge guidance).

---

## 5. Pilot Design & Source Parents

Six training parents deterministically selected from `datasets/manifests/train.csv` (`seed=42`):
1. `27a30c35-ace6-401d-86ba-e75895d9f016...` (`non_corroded`, Seed: `1059455960`)
2. `0888523b-18fd-44f8-8582-77109ba0cf78...` (`non_corroded`, Seed: `1522751532`)
3. `a0b0ecba-c1de-4117-834c-b881792caba3...` (`non_corroded`, Seed: `427368755`)
4. `796e406a-4bcc-4cd9-9319-aa89b5709907...` (`corroded`, Seed: `1386301971`)
5. `13d2a961-7952-4181-bde3-ccc2e349df3c...` (`corroded`, Seed: `281685160`)
6. `d6dc47f8-a720-4a66-b17f-1d52846e2d09...` (`corroded`, Seed: `884480466`)

- **Fixed Seed Rule**: One fixed deterministic seed was derived per parent (via SHA-256) and held constant across all 4 configurations for that parent to isolate the effects of condition and strength.
- **Target Outputs**: 6 parents $\times$ 2 conditions $\times$ 2 strengths = 24 outputs.

---

## 6. Execution Results and Diagnostic Metrics

- **Targeted Outputs**: 24
- **Attempted**: 23 (1 completed during smoke test)
- **Successful Generations**: 23 (100% technical check pass rate)
- **Failed Generations**: 0
- **Total Valid Outputs**: 24

### Diagnostic SSIM and Pixel Differences
Multi-channel RGB Structural Similarity Index (SSIM) computed with Gaussian window size 11, $\sigma = 1.5$, and `data_range=255.0`:

| Denoising Strength | Effective Steps | Mean SSIM | SSIM Range | Mean Pixel Difference |
| :---: | :---: | :---: | :---: | :---: |
| **0.20** | 6 steps | **0.6397** | [0.5782, 0.7416] | **13.48** |
| **0.35** | 10 steps | **0.5578** | [0.5032, 0.6756] | **17.72** |
| **Overall Pilot** | — | **0.5988** | [0.5032, 0.7416] | **15.60** |

> [!NOTE]
> **SSIM Diagnostic Role**:
> SSIM is a descriptive similarity metric, **not** automated proof of corrosion preservation. Lower SSIM at strength 0.35 indicates stronger semantic and luminance divergence from the source. It does not indicate whether that divergence preserved or degraded pit morphology.

---

## 7. Contact Sheets and Review Artifacts

All outputs and contact sheets are stored under `experiments/pilots/diffusion_v1/`:
- **Parent Contact Sheets (`contact_sheets/`)**:
  - [contact_sheet_parent_01_27a30c35-ace6-40.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_01_27a30c35-ace6-40.png) (Non-corroded, 1296×1020)
  - [contact_sheet_parent_02_0888523b-18fd-44.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_02_0888523b-18fd-44.png) (Non-corroded, 1296×1020)
  - [contact_sheet_parent_03_a0b0ecba-c1de-41.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_03_a0b0ecba-c1de-41.png) (Non-corroded, 1296×1020)
  - [contact_sheet_parent_04_796e406a-4bcc-4c.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_04_796e406a-4bcc-4c.png) (Corroded, 1296×1020)
  - [contact_sheet_parent_05_13d2a961-7952-41.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_05_13d2a961-7952-41.png) (Corroded, 1296×1020)
  - [contact_sheet_parent_06_d6dc47f8-a720-4a.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/contact_sheets/contact_sheet_parent_06_d6dc47f8-a720-4a.png) (Corroded, 1296×1020)
- **Review Table**: [review.csv](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/review.csv) (24 entries initialized with `unreviewed` fields).
- **Manifest of Sources**: [sources.csv](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/sources.csv).
- **Full Run Summary**: [summary.json](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/diffusion_v1/summary.json).

---

## 8. Critical Assessment & Next Steps

### Strengths Observed
1. **Plausible Lighting Realism**: Stable Diffusion 1.5 successfully translates global illumination to convincingly simulate low-light inspection and directional spotlighting.
2. **High GPU Efficiency**: Consuming only 3.28 GB VRAM with sub-second execution per output on RTX 5060, large-scale inference is technically viable.

### Key Limitations
1. **Subtle Surface Drift**: At strength 0.35, high-frequency rust textures and fine edge boundaries begin to blur or take on generative painterly artifacts.
2. **Lack of Dense Structural Conditioning**: Standard image-to-image relies solely on initial latent noise and text prompts; it lacks pixel-level geometric or defect preservation boundaries.
3. **Lack of Reliable Corrosion Masks**: Without segmentation masks to protect corroded zones explicitly, text prompts alone cannot guarantee that pit depth or surface loss is preserved.

### Strategic Recommendation for Next Milestone
Before generating full datasets or training Experiment C, investigate **structural conditioning adapters (e.g. ControlNet Depth/Canny or masked inpainting)** to lock pipeline geometry and corrosion contours while allowing the diffusion model to vary only the ambient inspection environment.
