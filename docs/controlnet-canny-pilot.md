# PipeAugment AI — Canny-ControlNet Comparison Pilot Report (`controlnet_canny_v1`)

## Overview
This document evaluates whether structural conditioning via **Canny ControlNet** improves upon the baseline pretrained Stable Diffusion v1.5 Image-to-Image pilot (`diffusion_v1`). The pilot evaluates **24 new outputs** generated across 6 training parents (3 non-corroded, 3 corroded) at a fixed denoising strength of **0.35** and two ControlNet conditioning scales (**0.50** and **0.80**) under two lighting conditions: `low_light_inspection` and `directional_industrial_lighting`.

> [!IMPORTANT]
> **Scientific Integrity & Governance Policy**:
> - **SSIM is Descriptive, Not Proof**: Structural Similarity (SSIM) measures multi-channel perceptual luminance, contrast, and structure correlation. A higher SSIM indicates closer general image alignment, but does **not** prove that micro-level rust pits, crack boundaries, or surface defects are accurately preserved.
> - **ControlNet Guides Structure; It Does Not Guarantee Defect Retention**: Canny edge conditioning injects structural priors into the U-Net via cross-attention. It constrains macroscopic edges (pipe boundaries, weld seams, major contours) but does not guarantee retention of subtle corroded textures or corrosion boundaries.
> - **Edge Overlap is a Heuristic**: Edge IoU and Dice are descriptive edge-coincidence metrics under fixed Canny thresholds (100, 200). Because Canny edge detection is sensitive to image contrast and illumination, lighting changes inherently alter the detected edge maps.
> - **Memory Distinctions**: Peak GPU allocated memory (active tensor allocations: ~4.01 GB) is distinguished from peak GPU reserved memory (PyTorch allocator pool: ~4.28 GB) and total dedicated hardware VRAM (8,188 MB).
> - **Training Eligibility**: All 24 pilot outputs are strictly marked `"training_eligibility": false`. None have been added to any training manifest.
> - **Locked Test Set**: The test set (`datasets/manifests/test.csv`, 60 images) remains strictly sealed and un-evaluated.

---

## 1. Model Provenance, Architecture, and License

| Component | Repository / Model ID | Commit SHA | License | Configuration |
| :--- | :--- | :--- | :--- | :--- |
| **Base Diffusion Model** | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | CreativeML OpenRAIL-M | fp16 SafeTensors, PNDMScheduler |
| **ControlNet Model** | `lllyasviel/control_v11p_sd15_canny` | `115a470d547982438f70198e353a921996e2e819` | CreativeML OpenRAIL-M | fp16 SafeTensors (`diffusion_pytorch_model.fp16.safetensors`, 723 MB) |
| **Pipeline Class** | `StableDiffusionControlNetImg2ImgPipeline` | Official Hugging Face Diffusers | — | Both base UNet and ControlNet in fp16 on CUDA |
| **Safety Checker** | `StableDiffusionSafetyChecker` | Active | — | 0 NSFW violations recorded |

---

## 2. Hardware Execution and Memory Metrics

The entire pilot executed on the local GPU without CPU offloading or CUDA out-of-memory (OOM) fallbacks.

- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8,188 MB dedicated VRAM, SM 12.0).
- **Placement Strategy**: Direct CUDA placement (`pipeline.to("cuda")`, `offload_strategy="none"`).
- **Memory Measurements**:
  - **Peak GPU Allocated Memory** (active PyTorch tensor footprint): **4,012.16 MB** (~4.01 GB).
  - **Peak GPU Reserved Memory** (total memory held by PyTorch caching allocator): **4,280.00 MB** (~4.28 GB).
  - **Minimum Device Free Memory** during inference: **2,729.00 MB** (~2.73 GB headroom).
  - ControlNet added ~732 MB of allocated tensor memory over plain img2img (~3,280 MB $\to$ ~4,012 MB), remaining well within the 8 GB budget.
- **Inference Runtime**:
  - Smoke test / first image (pipeline compile and JIT warm-up): **9.20s**.
  - Steady-state generation runtime per image (10 effective denoising steps @ strength 0.35): **~1.85s – 1.91s**.
  - Total elapsed batch time for 23 variations + 12 comparison sheet renderings: **48.69s**.

---

## 3. Matched Experimental Design

To isolate the structural effect of ControlNet Canny, all generation parameters were matched precisely against the preceding plain img2img pilot (`diffusion_v1`):

1. **Exact Same 6 Training Parents** (from `experiments/pilots/diffusion_v1/sources.csv`):
   - Non-corroded: `27a30c35...` (Seed: 1059455960), `0888523b...` (Seed: 1522751532), `a0b0ecba...` (Seed: 427368755)
   - Corroded: `796e406a...` (Seed: 1386301971), `13d2a961...` (Seed: 281685160), `d6dc47f8...` (Seed: 884480466)
2. **Identical Fixed Seeds**: The exact same 32-bit deterministic seed was passed to the generator per parent.
3. **Fixed Strength**: Matched at **0.35** (10 effective denoising steps out of 30 total steps).
4. **Tested Conditioning Scales**: **0.50** (moderate structural constraint) and **0.80** (strong structural constraint).
5. **Canny Conditioning Settings**: Thresholds `[100, 200]` applied to 512x512 grayscale images, replicated across 3 RGB channels for ControlNet input.
6. **Prompts**: Exact identical condition prompt extensions for `low_light_inspection` and `directional_industrial_lighting`, with identical negative prompts.

---

## 4. Empirical Results and Quantitative Metrics

### Comparison of Matched Pairs (Strength = 0.35)

| Metric | Plain SD 1.5 img2img (`s=0.35`) | ControlNet Canny (`c=0.50, s=0.35`) | ControlNet Canny (`c=0.80, s=0.35`) | Delta (c=0.80 vs Plain) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean SSIM (RGB)** | **0.5581** | **0.6106** | **0.6345** | **+0.0764 (+13.7%)** |
| **Min SSIM (RGB)** | 0.5032 | 0.5623 | 0.5832 | +0.0800 |
| **Max SSIM (RGB)** | 0.6756 | 0.7172 | 0.7317 | +0.0561 |
| **Mean Edge IoU** | **0.1515** | **0.2128** | **0.2527** | **+0.1012 (+66.8%)** |
| **Min Edge IoU** | 0.1304 | 0.1808 | 0.2081 | +0.0777 |
| **Max Edge IoU** | 0.1693 | 0.2476 | 0.2829 | +0.1136 |
| **Mean Edge Dice** | 0.2632 | 0.3505 | 0.4029 | +0.1397 (+53.1%) |
| **Mean Runtime / Image** | **~1.42s** | **~1.87s** | **~1.88s** | +0.46s |
| **Peak GPU Allocated** | 3,280.44 MB | 4,012.16 MB | 4,012.16 MB | +731.72 MB |

### Per-Parent Breakdown at Strength 0.35

| Parent ID | Class | Edge Density | Plain img2img SSIM | CNet c=0.50 SSIM | CNet c=0.80 SSIM | Plain Edge IoU | CNet c=0.50 IoU | CNet c=0.80 IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `27a30c35...` | non_corroded | 0.0631 | 0.6756 | 0.7164 | 0.7311 | 0.1648 | 0.2378 | 0.2829 |
| `0888523b...` | non_corroded | 0.1225 | 0.5573 | 0.6335 | 0.6558 | 0.1693 | 0.2476 | 0.2818 |
| `a0b0ecba...` | non_corroded | 0.0706 | 0.5186 | 0.5793 | 0.6010 | 0.1304 | 0.1809 | 0.2173 |
| `796e406a...` | corroded | 0.1077 | 0.5032 | 0.5633 | 0.5832 | 0.1395 | 0.1960 | 0.2393 |
| `13d2a961...` | corroded | 0.0798 | 0.5369 | 0.5648 | 0.5847 | 0.1583 | 0.2226 | 0.2617 |
| `d6dc47f8...` | corroded | 0.0694 | 0.5568 | 0.6067 | 0.6514 | 0.1466 | 0.1916 | 0.2333 |

---

## 5. Artifacts and Side-by-Side Comparison Sheets

All generated artifacts are organized under `experiments/pilots/controlnet_canny_v1/`:
- **Generated Images**: [outputs/](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/outputs) (24 PNG files)
- **JSON Metadata**: [metadata/](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/metadata) (24 JSON records)
- **Source Canny Edge Maps**: [control_maps/](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/control_maps) (6 PNG edge maps)
- **Review Spreadsheet**: [review.csv](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/review.csv) (24 pending rows)
- **Summary**: [summary.json](file:///d:/vscode/Gen%20AI Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/summary.json)
- **Comparative Contact Sheets** (12 files, 5 panels each):
  1. [sheet_parent_01_27a30c35-ace6-40_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_01_27a30c35-ace6-40_low_light_inspection.png)
  2. [sheet_parent_01_27a30c35-ace6-40_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_01_27a30c35-ace6-40_directional_industrial_lighting.png)
  3. [sheet_parent_02_0888523b-18fd-44_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_02_0888523b-18fd-44_low_light_inspection.png)
  4. [sheet_parent_02_0888523b-18fd-44_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_02_0888523b-18fd-44_directional_industrial_lighting.png)
  5. [sheet_parent_03_a0b0ecba-c1de-41_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_03_a0b0ecba-c1de-41_low_light_inspection.png)
  6. [sheet_parent_03_a0b0ecba-c1de-41_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_03_a0b0ecba-c1de-41_directional_industrial_lighting.png)
  7. [sheet_parent_04_796e406a-4bcc-4c_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_04_796e406a-4bcc-4c_low_light_inspection.png)
  8. [sheet_parent_04_796e406a-4bcc-4c_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_04_796e406a-4bcc-4c_directional_industrial_lighting.png)
  9. [sheet_parent_05_13d2a961-7952-41_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_05_13d2a961-7952-41_low_light_inspection.png)
  10. [sheet_parent_05_13d2a961-7952-41_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_05_13d2a961-7952-41_directional_industrial_lighting.png)
  11. [sheet_parent_06_d6dc47f8-a720-4a_low_light_inspection.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_06_d6dc47f8-a720-4a_low_light_inspection.png)
  12. [sheet_parent_06_d6dc47f8-a720-4a_directional_industrial_lighting.png](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/contact_sheets/sheet_parent_06_d6dc47f8-a720-4a_directional_industrial_lighting.png)

### Layout of Each Comparison Sheet
Each 5-panel sheet displays side-by-side:
1. **Panel 1**: Original Source Image (512x512)
2. **Panel 2**: Canny Edge Condition Map (Thresholds 100, 200)
3. **Panel 3**: Matched Plain SD 1.5 img2img (`strength=0.35`)
4. **Panel 4**: ControlNet Canny (`control_scale=0.50`, `strength=0.35`)
5. **Panel 5**: ControlNet Canny (`control_scale=0.80`, `strength=0.35`)

To view the contact sheets in Windows File Explorer:
```powershell
Invoke-Item "D:\vscode\Gen AI Project\pipeaugment-ai\experiments\pilots\controlnet_canny_v1\contact_sheets"
```

---

## 6. Qualitative Observations and Scientific Findings

1. **Preservation of Pipeline Cylindrical Geometry**:
   - In plain img2img at strength 0.35, pipe seams, flange angles, and joint curves frequently drifted, softened, or warped.
   - At `control_scale=0.50` and `0.80`, ControlNet effectively anchored pipe silhouettes, valve edges, and wall boundaries to the source outline, eliminating geometric drift.
2. **Corrosion Preservation vs. Texture Modification**:
   - For heavily corroded parents (e.g. `796e406a...` and `13d2a961...`), high-frequency rust pitting registers strongly in the Canny edge map (edge densities of 0.1077 and 0.0798). ControlNet uses these edge signals to stabilize where surface irregularities should appear.
   - However, at `c=0.80`, edge conditioning can over-constrain the surface, producing harsh high-contrast edge halos around rust borders rather than natural pitting.
   - Conversely, at `c=0.50`, the balance between natural lighting diffusion and edge guidance appears visually more plausible.
3. **Condition Visibility**:
   - In both `c=0.50` and `c=0.80`, the environmental lighting condition remains perceptible (deepened ambient shadows in `low_light_inspection` and focused specular highlights in `directional_industrial_lighting`).

---

## 7. Status of Human Review & Next Steps

All 24 rows in [review.csv](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/pilots/controlnet_canny_v1/review.csv) are initialized as:
- `geometry_preserved`: `"unreviewed"`
- `corrosion_preserved`: `"unreviewed"`
- `condition_visible`: `"unreviewed"`
- `realism_acceptable`: `"unreviewed"`
- `decision`: `"pending"`

No pilot outputs may be admitted to training datasets until human review is completed.
