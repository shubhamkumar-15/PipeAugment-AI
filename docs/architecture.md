# PipeAugment AI - System Architecture

## Overview
PipeAugment AI is a condition-controlled, corrosion-preserving pipeline image augmentation platform designed to generate high-fidelity, corrosion-preserved pipeline inspection training data.

## System Layers

### 1. Presentation Layer (Frontend)
- **Technology**: Next.js App Router with TypeScript.
- **Role**: Provides the user interface for dataset visualization, inspection condition controls, augmentation job dispatching, and comparison views.
- **Communication**: Communicates with the backend exclusively via RESTful JSON APIs (`/api/v1/*`).

### 2. Service & API Layer (Backend)
- **Technology**: FastAPI and Pydantic (using `pydantic-settings`).
- **Role**: Owns backend API routing, request validation, task orchestration, metadata persistence, and lifecycle management.
- **Modularity**: AI and Computer Vision modules are strictly isolated behind dedicated service abstractions. Backend API routes remain decoupled from underlying model implementations.

### 3. AI & Computer Vision Layer (Isolated Modules)
- **Traditional Augmentation**: OpenCV-based deterministic augmentations (lighting, rotation, geometric distortion).
- **Generative Augmentation**: PyTorch and Hugging Face Diffusers pipelines for condition-controlled generative augmentation preserving pipeline corrosion features.
- **Isolation Guarantee**: Model execution, GPU workers, and heavy compute workloads will be isolated in dedicated worker services/submodules without polluting the lightweight API gateway.

## Data Lineage & Immutability Principles

### Immutable Original Images
- All uploaded/imported source inspection images stored in `datasets/original/` are treated as strictly immutable raw records.
- No augmentation or processing pipeline is permitted to overwrite, mutate, or modify original image files in place.

### Provenance & Parent Reference
- Every generated/augmented image (stored in `datasets/augmented/`) carries lineage metadata linking directly back to its parent source image.
- Manifests and metadata records in `datasets/metadata/` and `datasets/manifests/` capture exact transformation parameters, condition settings, model checkpoints, random seeds, and parent image identifiers, ensuring full auditability and reproducible data pipelines.
