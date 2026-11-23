# PipeAugment AI — Generative Selection Review Utility

This directory contains the self-contained offline review utility for evaluating and selecting condition-controlled generative augmentation candidates for PipeAugment AI.

## Artifacts in this Directory

- `index.html`: Fully self-contained, offline HTML/CSS/JavaScript review application. Can be opened directly in any modern web browser without starting a web server or installing node dependencies.
- `review_inventory.json`: Machine-readable manifest inventory of all 48 generated candidates across the 6 training parents (3 non-corroded, 3 corroded) and 2 lighting conditions, including Phase 4 automated quality screening metrics.
- `reviewed_manifest_v1.json` / `reviewed_manifest_v2.json`: Official versioned reviewed manifests produced after importing an exported human review.

## Quick Start: How to Open the Review Tool

Run in PowerShell:
```powershell
Invoke-Item "D:\vscode\Gen AI Project\pipeaugment-ai\experiments\reviews\generative_pilot_v1\index.html"
```
Or double-click `index.html` in Windows File Explorer.

## Quick Filters (Phase 4 Upgrade)
- **Method**: `[All Methods]` | `[ControlNet Only]` | `[Plain SD 1.5 Only]`
- **Class**: `[All Classes]` | `[Corroded]` | `[Non-Corroded]`
- **Status**: `[All Statuses]` | `[Pending Only]` | `[Reviewed Only]`

To review the remaining 22 ControlNet candidates:
1. Click `[ControlNet Only]`.
2. Click `[Pending Only]`.
3. Inspect and mark each candidate.
4. Click **Export Review JSON** to persist your decisions.

## Evaluation Criteria

Each candidate output must be reviewed across four structured criteria:
1. **Geometry Preserved?** (Yes / No / Uncertain): Does the pipe retain its cylindrical shape, seams, joints, flanges, and background perspective?
2. **Corrosion Preserved?** (Yes / No / Uncertain): Are pit locations and rust textures retained without hallucinating new defects or erasing existing corrosion?
3. **Condition Visible?** (Yes / No / Uncertain): Is the requested lighting condition (low-light dimming or directional beam) clearly perceptible?
4. **Realism Acceptable?** (Yes / No / Uncertain): Does the image look like an authentic industrial pipeline inspection capture without generative artifacts or blur?

### Decision Rules
- **Approved**: Allowed ONLY when all four criteria are **Yes**, a valid **Reviewer Name** is supplied, and the reviewer explicitly confirms.
- **Rejected**: Use if pipe geometry breaks, corrosion is invented/removed, or unphysical artifacts occur.
- **Needs Review**: Use when uncertain or when secondary domain verification is needed.

## Export & Import Workflow

1. In the web interface, enter your Reviewer Name in the top bar.
2. Review the candidate outputs.
3. Click **Export Review JSON** to download `generative_review_export_<timestamp>.json`.
4. Save the exported JSON into `experiments/reviews/generative_pilot_v1/`.
5. Import and validate your review using the Python verification CLI:
```powershell
& ".\.venv-ai\Scripts\python.exe" ai/scripts/import_generative_review.py --input experiments/reviews/generative_pilot_v1/generative_review_export_<timestamp>.json
```
6. The import script will verify all hashes, reject invalid approvals, and write the SHA-256 integrity-verified manifest `reviewed_manifest_v3.json`.
