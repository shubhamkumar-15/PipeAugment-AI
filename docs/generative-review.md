# PipeAugment AI — Human Review & Generative Selection Gate

## Overview

This guide documents the lightweight, self-contained offline review utility created for the completed **Plain Stable Diffusion v1.5** (`diffusion_v1`) and **Canny-ControlNet** (`controlnet_canny_v1`) pilots.

The tool provides an interactive, standalone browser-based interface to visually inspect and evaluate all **48 candidate outputs** across the 6 training parents (3 non-corroded, 3 corroded) and 2 lighting conditions (`low_light_inspection` and `directional_industrial_lighting`).

> [!IMPORTANT]
> **Governance & Integrity Rules**:
> - **Zero Fabricated Decisions**: Neither the tool nor the backend pre-selects "Yes" or pre-approves any generative variation based on descriptive metrics (such as SSIM or Edge IoU).
> - **Strict Approval Gate**: An output is permitted to be marked `"approved"` **only** if all 4 structured criteria are answered `"yes"` and a valid Reviewer Name is provided.
> - **No Overwriting**: The import script writes a versioned reviewed manifest (`reviewed_manifest_v1.json` / `.csv`) and **never** overwrites original pilot files or raw source metadata.
> - **Offline & Portable**: The generated `index.html` has zero remote CDN dependencies, zero external network calls, and runs directly from File Explorer.

---

## 1. How to Open the Review Tool

The review tool is a single self-contained HTML page located at:
[experiments/reviews/generative_pilot_v1/index.html](file:///d:/vscode/Gen%20AI%20Project/pipeaugment-ai/experiments/reviews/generative_pilot_v1/index.html)

To open it directly in your default browser on Windows, run in PowerShell:
```powershell
Invoke-Item "D:\vscode\Gen AI Project\pipeaugment-ai\experiments\reviews\generative_pilot_v1\index.html"
```
Or navigate to `pipeaugment-ai\experiments\reviews\generative_pilot_v1\` in Windows File Explorer and double-click `index.html`.

To re-build the review interface at any time:
```powershell
& ".\.venv-ai\Scripts\python.exe" ai/scripts/build_generative_review.py
```

---

## 2. Review Criteria & Failure Modes

Every candidate output is evaluated against four criteria:

| Criterion | Evaluation Question | Options | Rejection Condition |
| :--- | :--- | :---: | :--- |
| **Geometry Preserved?** | Does the pipe retain its cylindrical contour, weld seams, flanges, joints, and background alignment? | Yes / No / Uncertain | Pipe boundary warped, joints blurred, or new pipes invented. |
| **Corrosion Preserved?** | Are pit locations and rust textures retained without hallucinating new defects or erasing existing corrosion? | Yes / No / Uncertain | Rust hallucinated on clean pipe, or real rust patches erased/smoothed away. |
| **Condition Visible?** | Is the requested lighting condition clearly perceptible? | Yes / No / Uncertain | Image looks like normal daylight; low-light dimming or directional beam is absent. |
| **Realism Acceptable?** | Does the image appear authentic without AI blurring, comic rendering, or latent artifacts? | Yes / No / Uncertain | Unphysical smearing, dreamlike textures, or cartoonish rendering. |

### Decision Meanings
- **Approved**: High-quality, physically plausible variation. **Requires all 4 criteria to be "Yes" and a Reviewer Name.**
- **Rejected**: Fails geometry, erases/hallucinates corrosion, or introduces unphysical artifacts.
- **Needs Review**: Ambiguous boundary cases requiring secondary consultation.
- **Pending**: Unreviewed default state.

---

## 3. How to Save and Export Your Review

1. In `index.html`, enter your name in the **Reviewer Name** field in the top navigation bar.
2. Select your assessments and decisions for the candidate images.
3. *Local Drafts*: Your progress automatically saves to browser `localStorage`.
4. *Exporting*: When finished, click **Export Review JSON** (or **Export Review CSV**) in the top header.
5. Save the downloaded file (e.g. `generative_review_export_<timestamp>.json`) into:
   `D:\vscode\Gen AI Project\pipeaugment-ai\experiments\reviews\generative_pilot_v1\`

> [!WARNING]
> Browser `localStorage` is temporary local browser memory. It is **not** a persistent disk file. Always click **Export Review JSON** to produce your review file before closing your browser.

---

## 4. How to Import and Certify Your Review

To validate and import your exported review into the project manifest, run:

```powershell
& ".\.venv-ai\Scripts\python.exe" ai/scripts/import_generative_review.py --input experiments/reviews/generative_pilot_v1/generative_review_export.json --summary
```

### What the Import Script Enforces:
- Verifies all 48 output IDs match `review_inventory.json`.
- Computes SHA-256 of all on-disk output image files and halts if any file has been modified.
- Rejects any `"approved"` record where any criterion is not `"yes"` or where `reviewer` is empty.
- Writes signed, versioned manifests:
  - `experiments/reviews/generative_pilot_v1/reviewed_manifest_v1.json`
  - `experiments/reviews/generative_pilot_v1/reviewed_manifest_v1.csv`
- Generates a certified audit summary in `reviewed_summary_v1.txt`.

---

## 5. How to Check Current Review Summary

To inspect the review status at any time, run:

```powershell
& ".\.venv-ai\Scripts\python.exe" ai/scripts/summarize_generative_review.py
```

- If no human review has been imported, it explicitly reports:
  `HUMAN REVIEW STATUS: PENDING` and keeps Experiment C blocked.
- Once imported, it outputs approval rates by method, condition, and source class, along with structured failure mode statistics.

---

## 6. Limitations of Student Visual Assessment

1. **Exploratory Visual Heuristic**: Visual review by student researchers assesses perceptual realism, but does **not** replace certified non-destructive metallurgical evaluation or ground-truth ultrasonic thickness measurement.
2. **Sample Size Scope**: The pilot evaluates 6 training parents (48 outputs). Relative approval differences across methods provide design guidance for filtering, but do **not** constitute broad statistical significance.
3. **Class Label Inheritance**: Source labels (`corroded` / `non_corroded`) are inherited from the K-Pipelines dataset without independent laboratory verification.
