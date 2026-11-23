# PipeAugment AI — Phase 4 Quality & Corrosion-Preservation Validation Engine

## 1. Executive Summary

Phase 4 implements a reusable, strictly tested image quality and corrosion-preservation validation module (`ai/pipeaugment_eval/`) designed to evaluate candidate synthetic variations (traditional CV, plain diffusion, and Canny-ControlNet) prior to training admission.

The engine establishes a **three-state decision gate**:
- `pass`: Meets all quantitative fidelity, edge retention, and color-stability thresholds.
- `flag_for_review`: Exceeds one or more secondary threshold boundaries (e.g. perceptual shift, subtle edge loss); flagged for human auditor inspection.
- `fail`: Violates fundamental technical validity constraints (non-finite pixels, shape mismatch, constant/blank image, identical unaugmented clone).

---

## 2. Architecture & Components

The engine is encapsulated in `ai/pipeaugment_eval/`:

```
ai/pipeaugment_eval/
├── __init__.py           # Unified exports
├── config.py             # QualityThresholdProfile & QualityValidationConfig dataclasses
├── metrics.py            # Image validity checks, multi-channel SSIM, LPIPS, Canny edge retention
├── corrosion_diag.py     # Heuristic rust-candidate mask extraction & region stability (HSV/Lab)
└── validator.py          # QualityValidator orchestrating the three-state evaluation gate
```

### Metrics & Algorithms
1. **Technical Image Validity (`evaluate_image_validity`)**:
   - Shape preservation (`(H, W, 3)`)
   - Pixel data type (`uint8`)
   - Finite pixel checks (no `NaN`, `Inf`)
   - Degeneracy screening (`is_all_black`, `is_all_white`, `is_constant`)
   - Identity check (`mean_pixel_abs_diff > 0.0`, catching identical copies)
   - Dynamic range clipping rates (`clipping_low_rate` at 0, `clipping_high_rate` at 255)

2. **Multi-Channel RGB SSIM (`compute_ssim_rgb`)**:
   - Computes Structural Similarity Index across R, G, and B color channels independently with `data_range=255.0` (using an 11×11 Gaussian filter with $\sigma=1.5$).
   - Averages channel SSIM scores into a single scalar in $[0.0, 1.0]$.

3. **Perceptual Distance (`LPIPSEvaluator`)**:
   - Learned Perceptual Image Patch Similarity using an AlexNet trunk (`lpips.LPIPS(net='alex')`).
   - Normalizes uint8 $[0, 255]$ inputs into float32 tensors scaled to $[-1.0, 1.0]$ with shape $(1, 3, H, W)$.
   - Evaluated in inference mode (`torch.inference_mode()`) on CUDA when available.

4. **Edge Retention IoU / Dice (`compute_edge_retention`)**:
   - Applies Canny edge detection at thresholds $[100, 200]$ on grayscale source and augmented pairs.
   - Measures edge map overlap:
     $$\text{IoU} = \frac{|E_{\text{src}} \cap E_{\text{gen}}|}{|E_{\text{src}} \cup E_{\text{gen}}|}, \quad \text{Dice} = \frac{2 \cdot |E_{\text{src}} \cap E_{\text{gen}}|}{|E_{\text{src}}| + |E_{\text{gen}}|}$$
   - Safely returns `1.0` if both edge maps are empty (avoiding zero-division).

5. **Diagnostic Rust-Region Stability (`evaluate_rust_stability`)**:
   - Combines HSV color thresholding ($H \in [6, 26]$, $S \in [45, 255]$, $V \in [35, 210]$) with CIE-Lab refinement ($a^* \ge 130$, $b^* \ge 135$) to segment candidate rust pixels.
   - Computes `source_rust_ratio`, `output_rust_ratio`, `rust_area_delta` ($\text{ratio}_{\text{gen}} - \text{ratio}_{\text{src}}$), and `rust_mask_iou`.
   - Safely returns `rust_mask_iou = None` if neither source nor generated image contains candidate rust pixels.

> [!CAUTION]
> **Scientific Caveat**: The HSV/Lab rust-region analysis is an exploratory color heuristic for stability tracking during image generation. It is **NOT** ground-truth metallurgical corrosion segmentation, ultrasound thickness measurement, or certified NDT defect validation.

---

## 3. Threshold Configuration Profiles

Configured in `ai/configs/quality_validation_v1.json`:

| Parameter | Traditional Profile | Generative Profile | Description |
| :--- | :---: | :---: | :--- |
| `min_ssim` | $\ge 0.65$ | $\ge 0.50$ | Minimum required structural similarity |
| `max_lpips` | $\le 0.25$ | $\le 0.35$ | Maximum acceptable perceptual distance |
| `min_edge_iou` | $\ge 0.20$ | $\ge 0.15$ | Minimum structural edge retention |
| `max_rust_area_delta` | $\le 0.08$ | $\le 0.10$ | Maximum allowable rust-area drift |
| `max_clipping_low_rate` | $\le 0.05$ | $\le 0.05$ | Max proportion of underexposed (0) pixels |
| `max_clipping_high_rate` | $\le 0.05$ | $\le 0.05$ | Max proportion of blown-out (255) pixels |
| `min_pixel_abs_diff` | $\ge 1.0$ | $\ge 1.0$ | Minimum mean absolute pixel difference |

---

## 4. Evaluation Results Across 72 Pilot Images

Evaluated by `ai/scripts/evaluate_pilot_quality.py` across all 72 generated pairs from the three completed pilots:

| Pilot Method | Count | Mean SSIM | Mean LPIPS | Mean Edge IoU | Mean Rust Delta | Pass | Flagged | Fail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Traditional CV Pilot** | 24 | **0.9618** | **0.0320** | **0.7978** | **0.0000** | 17 | 7 | 0 |
| **Plain SD 1.5 ($s=0.20$)** | 12 | 0.6396 | 0.1896 | 0.1925 | -0.0645 | 8 | 4 | 0 |
| **Plain SD 1.5 ($s=0.35$)** | 12 | 0.5581 | 0.2660 | 0.1515 | -0.0714 | 4 | 8 | 0 |
| **ControlNet Canny ($c=0.50, s=0.35$)** | 12 | 0.6106 | 0.2167 | 0.2128 | -0.0641 | 11 | 1 | 0 |
| **ControlNet Canny ($c=0.80, s=0.35$)** | 12 | **0.6345** | **0.1971** | **0.2527** | -0.0626 | **12** | **0** | 0 |

### Key Findings
1. **ControlNet Conditioning Prevents Edge Collapse**: At conditioning scale $0.80$, ControlNet achieves a **$66.8\%$ improvement in edge retention** (Edge IoU $0.2527$ vs $0.1515$) and a **$25.9\%$ reduction in perceptual distortion** (LPIPS $0.1971$ vs $0.2660$) compared to Plain SD 1.5 at the same denoising strength ($s=0.35$).
2. **Automated Screening Pass Rates**:
   - ControlNet $c=0.80$: **$100\%$ automated screening pass rate** (12/12 pass, 0 flagged).
   - ControlNet $c=0.50$: **$91.7\%$ automated screening pass rate** (11/11 pass, 1 flagged).
   - Plain SD 1.5 $s=0.35$: **$66.7\%$ flagged for review** (8/12 flagged due to severe structural divergence).
3. **Rust Drift Under Diffusion**: Diffusion-based pipelines cause a negative rust-area shift ($\Delta \approx -0.063$), reflecting a tendency of the generative backbone to slightly soften high-frequency rust pitting under heavy noise.

---

## 5. Offline Review Portal Upgrade

The offline review tool (`experiments/reviews/generative_pilot_v1/index.html`) has been rebuilt with Phase 4 capabilities:
- **Phase 4 Diagnostic Badges**: Every candidate card displays `SSIM`, `LPIPS`, `Edge IoU`, and `Status: Pass/Flagged` badges labeled as automated screening metrics.
- **Quick Filters**:
  - `Method`: `[All Methods]` | `[ControlNet Only]` | `[Plain SD 1.5 Only]`
  - `Class`: `[All Classes]` | `[Corroded]` | `[Non-Corroded]`
  - `Status`: `[All Statuses]` | `[Pending Only]` | `[Reviewed Only]`
- **Pending Review Scope**: Filtering by `[ControlNet Only]` and `[Pending Only]` isolates the **22 remaining ControlNet candidates** awaiting human review.
