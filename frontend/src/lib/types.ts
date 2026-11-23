/**
 * PipeAugment AI — Canonical TypeScript Definitions
 * Verified strictly against backend schemas in backend/app/schemas/
 */

export type SourceType = "user_upload" | "traditional" | "generative";
export type QualityStatus = "pass" | "flag_for_review" | "fail";
export type ReviewStatus = "pending" | "approved" | "rejected" | "needs_review";
export type Split = "train" | "val" | "test" | "unassigned";
export type Label = "clean" | "corrosion" | null;

export interface QualityMetrics {
  ssim_rgb?: number;
  ssim?: number;
  lpips?: number;
  mean_pixel_abs_diff?: number;
  clipping_low_rate?: number;
  clipping_high_rate?: number;
  edge_overlap_iou?: number;
  edge_overlap_dice?: number;
  source_edge_density?: number;
  output_edge_density?: number;
  source_rust_ratio?: number;
  output_rust_ratio?: number;
  rust_area_delta?: number;
  rust_mask_iou?: number;
  corrosion_diagnostic?: {
    diagnostic_type?: string;
    scientific_caveat?: string;
    source_rust_ratio?: number;
    output_rust_ratio?: number;
    rust_area_delta?: number;
    rust_mask_iou?: number;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface ImageMetadata {
  image_id: string;
  parent_image_id: string | null;
  original_filename: string;
  stored_filename: string;
  relative_path: string;
  image_format: string;
  content_type: string;
  width: int;
  height: int;
  size_bytes: int;
  sha256: string;
  created_at: string;
  source_type: SourceType;
  generation_method: string | null;
  label: string | null;
  annotation_status: string;
  split: string;

  // Augmented & Quality Validation fields
  condition: string | null;
  parameters: Record<string, unknown> | null;
  seed: number | null;
  quality_metrics: QualityMetrics | null;
  quality_status: QualityStatus | null;
  quality_flags: string[] | null;
  review_status: ReviewStatus | null;
  review_notes: string | null;
  reviewer: string | null;
}

// In typescript int is number
type int = number;

export interface ImageListResponse {
  items: ImageMetadata[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReviewUpdateRequest {
  review_status: ReviewStatus;
  review_notes?: string | null;
  reviewer?: string | null;
}

export interface DeleteImageResponse {
  image_id: string;
  deleted: boolean;
  deleted_files: string[];
  deleted_children_count: number;
}

export interface TraditionalAugmentRequest {
  parent_image_ids: string[];
  conditions?: string[];
  intensity?: number;
  variations_per_image?: number;
  seed?: number | null;
}

export interface GenerativeAugmentRequest {
  parent_image_ids: string[];
  mode?: "controlnet_canny" | "img2img";
  conditions?: string[];
  strength?: number;
  control_scale?: number;
  variations_per_image?: number;
  seed?: number | null;
}

export type JobType = "traditional" | "generative";
export type JobStatusType = "queued" | "running" | "completed" | "partial_failed" | "failed";

export interface JobItemResult {
  parent_id: string;
  condition: string;
  status: "success" | "failed";
  output_id?: string;
  error?: string;
  [key: string]: unknown;
}

export interface JobStatus {
  job_id: string;
  job_type: JobType;
  status: JobStatusType;
  total_items: number;
  completed_items: number;
  failed_items: number;
  parent_image_ids: string[];
  config: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  generated_image_ids: string[];
  item_results: JobItemResult[];
  error_message: string | null;
}

export interface JobListResponse {
  items: JobStatus[];
  total: number;
  limit: number;
  offset: number;
}

export interface DatasetStatsResponse {
  total_images: number;
  total_originals: number;
  total_generated: number;
  by_method: Record<string, number>;
  by_condition: Record<string, number>;
  by_quality_status: Record<string, number>;
  by_review_status: Record<string, number>;
  total_jobs: number;
  recent_activity: Array<Record<string, unknown>>;
}

export interface DatasetExportRequest {
  include_originals?: boolean;
  include_traditional?: boolean;
  include_generative?: boolean;
  only_quality_passed?: boolean;
  only_human_approved?: boolean;
  conditions?: string[];
}

export interface DatasetExportResponse {
  export_id: string;
  filename: string;
  image_count: number;
  size_bytes: number;
  download_url: string;
  created_at: string;
  filters_applied: Record<string, unknown>;
}

export interface ExportListResponse {
  items: DatasetExportResponse[];
  total: number;
}

export interface HealthResponse {
  status: "ok" | "healthy" | string;
  app_name: string;
  version: string;
}

export interface GPUInfo {
  gpu_detected: boolean;
  model_name?: string;
  total_memory_mb?: number;
  free_memory_mb?: number;
  driver_version?: string;
  error?: string;
}

export interface SystemInfo {
  app_name: string;
  app_version: string;
  hardware: {
    gpu: GPUInfo;
  };
  ai_worker: {
    environment_ready: boolean;
    python_executable: string;
  };
  capabilities: {
    augmentation_methods: string[];
    traditional_transforms: string[];
    generative_conditions: string[];
    frozen_controlnet_config?: {
      base_model: string;
      controlnet_model: string;
      control_scale: number;
      strength: number;
      guidance_scale: number;
      num_inference_steps: number;
      canny_thresholds: [number, number];
    };
  };
  upload_limits: {
    max_upload_bytes: number;
    max_image_pixels: number;
    allowed_formats: string[];
  };
}

export interface BenchmarkMetrics {
  f1: number;
  f1_std?: number;
  corrosion_recall: number;
  corrosion_recall_std?: number;
  accuracy: number;
  accuracy_std?: number;
}

export interface BenchmarkData {
  test_clean?: Record<string, BenchmarkMetrics>;
  stress_suites?: Record<string, Record<string, BenchmarkMetrics>>;
  [key: string]: unknown;
}

export interface CorrosionMapUrls {
  pipeline_canny: string;
  corrosion_layout: string;
  corrosion_overlay: string;
}

export interface CorrosionAnalysisResponse {
  image_id: string;
  predicted_class: "corroded" | "non_corroded";
  confidence: number;
  corrosion_probability: number;
  non_corrosion_probability: number;
  severity_level: "None" | "Low" | "Moderate" | "Severe";
  rust_coverage_percent: number;
  model_checkpoint: string;
  disclaimer: string;
  map_urls: CorrosionMapUrls;
}

export type AnalysisMapType = "pipeline_canny" | "corrosion_layout" | "corrosion_overlay";
