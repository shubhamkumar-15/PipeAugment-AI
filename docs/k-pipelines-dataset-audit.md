# K-Pipelines Dataset Audit Report

## Overview
- **Dataset Name:** K-Pipelines (Standard Version)
- **Total Readable Images:** 600
- **Corrupt Images:** 0
- **Unique SHA-256 Checksums:** 600
- **Exact Duplicate Groups:** 0
- **Cross-Split Exact Duplicates:** 0
- **Label Conflicts (Exact):** 0
- **Near-Duplicate Candidates (pHash <= 4):** 0

## Split and Class Distribution

| Split | Class | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `test` | `corrosion` | 31 | 5.2% |
| `test` | `no_corrosion` | 29 | 4.8% |
| `train` | `corrosion` | 238 | 39.7% |
| `train` | `no_corrosion` | 242 | 40.3% |
| `valid` | `corrosion` | 31 | 5.2% |
| `valid` | `no_corrosion` | 29 | 4.8% |
| **Total** | -- | **600** | **100%** |

## Formats and Dimensions

- **Image Formats:** JPEG: 600
- **Dimensions:** 512x512: 600

## Duplicate Analysis

No exact duplicates (SHA-256 identical) found.

### Near-Duplicate Candidates (0 pairs found with pHash distance <= 4)
Candidate pairs are recorded in `datasets/manifests/k_pipelines_duplicate_candidates.csv`.
*Note:* As mandated by audit policy, perceptual hash candidates are flagged as potential visual similarities rather than confirmed duplicates, and raw images remain strictly immutable.

## Task Suitability

- **Image Classification:** Supported out of the box. Image labels correspond to directory structure (`corrosion`, `no_corrosion`).
- **Bounding Box Object Detection:** Not directly annotated in the source repository (no YOLO / COCO / Pascal VOC XML bounding box files are provided).
- **Pixel-Level Segmentation:** Not annotated (no binary or polygon masks provided in Standard release).

## Provenance and Licensing

- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Citation:** Ramos, Casas, and Rivas-Echeverría (2025), *Intelligent Systems with Applications*, Vol. 25, 200463.
- **Downstream Notice:** Dataset images were generated via Stable Diffusion for pipeline research. Redistribution or commercial deployment requires legal review under GPL v3.0.
