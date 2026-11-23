/**
 * PipeAugment AI — Typed API Client
 * Interacts with FastAPI backend at http://127.0.0.1:8000/api/v1
 */

import {
  HealthResponse,
  SystemInfo,
  ImageMetadata,
  ImageListResponse,
  ReviewUpdateRequest,
  DeleteImageResponse,
  TraditionalAugmentRequest,
  GenerativeAugmentRequest,
  JobStatus,
  JobListResponse,
  DatasetStatsResponse,
  DatasetExportRequest,
  DatasetExportResponse,
  ExportListResponse,
  BenchmarkData,
  CorrosionAnalysisResponse,
  AnalysisMapType,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";
const API_V1 = `${API_BASE_URL}/api/v1`;

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  const url = `${API_V1}${cleanEndpoint}`;
  const headers = new Headers(options.headers || {});

  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    if (!res.ok) {
      let errorData: unknown;
      try {
        errorData = await res.json();
      } catch {
        errorData = await res.text();
      }
      const message =
        typeof errorData === "object" && errorData !== null && "detail" in errorData
          ? String((errorData as { detail: unknown }).detail)
          : `HTTP ${res.status}: ${res.statusText}`;
      throw new ApiError(message, res.status, errorData);
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      err instanceof Error ? err.message : "Failed to connect to backend",
      0,
      err
    );
  }
}

// URL Builders
export function getImageFileUrl(imageId: string): string {
  return `${API_V1}/images/${imageId}/file`;
}

export function getChartUrl(chartName: string): string {
  return `${API_V1}/evaluation/charts/${encodeURIComponent(chartName)}`;
}

export function getExportDownloadUrl(exportId: string): string {
  return `${API_V1}/datasets/exports/${exportId}/download`;
}

export function getAnalysisMapUrl(imageId: string, mapType: AnalysisMapType): string {
  return `${API_V1}/images/${imageId}/maps/${mapType}`;
}

// API Methods
export const api = {
  // System & Health
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  async getSystemInfo(): Promise<SystemInfo> {
    return request<SystemInfo>("/system/info");
  },

  // Images
  async uploadImage(file: File): Promise<ImageMetadata> {
    const formData = new FormData();
    formData.append("file", file);
    return request<ImageMetadata>("/images/upload", {
      method: "POST",
      body: formData,
    });
  },

  async listImages(params?: {
    source_type?: string;
    parent_image_id?: string;
    condition?: string;
    quality_status?: string;
    review_status?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<ImageListResponse> {
    const searchParams = new URLSearchParams();
    if (params) {
      if (params.source_type) searchParams.set("source_type", params.source_type);
      if (params.parent_image_id) searchParams.set("parent_image_id", params.parent_image_id);
      if (params.condition) searchParams.set("condition", params.condition);
      if (params.quality_status) searchParams.set("quality_status", params.quality_status);
      if (params.review_status) searchParams.set("review_status", params.review_status);
      if (params.search) searchParams.set("search", params.search);
      if (params.limit !== undefined) searchParams.set("limit", params.limit.toString());
      if (params.offset !== undefined) searchParams.set("offset", params.offset.toString());
    }
    const qs = searchParams.toString();
    return request<ImageListResponse>(`/images${qs ? `?${qs}` : ""}`);
  },

  async getImage(imageId: string): Promise<ImageMetadata> {
    return request<ImageMetadata>(`/images/${imageId}`);
  },

  async updateReview(
    imageId: string,
    req: ReviewUpdateRequest
  ): Promise<ImageMetadata> {
    return request<ImageMetadata>(`/images/${imageId}/review`, {
      method: "PATCH",
      body: JSON.stringify(req),
    });
  },

  async deleteImage(imageId: string): Promise<DeleteImageResponse> {
    return request<DeleteImageResponse>(`/images/${imageId}`, {
      method: "DELETE",
    });
  },

  // Augmentation Jobs
  async createTraditionalJob(req: TraditionalAugmentRequest): Promise<JobStatus> {
    return request<JobStatus>("/augment/traditional", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  async createGenerativeJob(req: GenerativeAugmentRequest): Promise<JobStatus> {
    return request<JobStatus>("/augment/generative", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  async listJobs(params?: {
    status?: string;
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<JobListResponse> {
    const searchParams = new URLSearchParams();
    if (params) {
      if (params.status) searchParams.set("status", params.status);
      if (params.type) searchParams.set("type", params.type);
      if (params.limit !== undefined) searchParams.set("limit", params.limit.toString());
      if (params.offset !== undefined) searchParams.set("offset", params.offset.toString());
    }
    const qs = searchParams.toString();
    return request<JobListResponse>(`/jobs${qs ? `?${qs}` : ""}`);
  },

  async getJob(jobId: string): Promise<JobStatus> {
    return request<JobStatus>(`/jobs/${jobId}`);
  },

  async retryJob(jobId: string): Promise<JobStatus> {
    return request<JobStatus>(`/jobs/${jobId}/retry`, {
      method: "POST",
    });
  },

  // Datasets
  async getDatasetStats(): Promise<DatasetStatsResponse> {
    return request<DatasetStatsResponse>("/datasets/stats");
  },

  async exportDataset(req: DatasetExportRequest): Promise<DatasetExportResponse> {
    return request<DatasetExportResponse>("/datasets/export", {
      method: "POST",
      body: JSON.stringify(req),
    });
  },

  async listDatasetExports(): Promise<ExportListResponse> {
    return request<ExportListResponse>("/datasets/exports");
  },

  // Evaluation
  async getBenchmarkResults(): Promise<BenchmarkData> {
    return request<BenchmarkData>("/evaluation/benchmark");
  },

  // Corrosion Analysis & Layout Maps
  async analyzeImage(imageId: string): Promise<CorrosionAnalysisResponse> {
    return request<CorrosionAnalysisResponse>(`/images/${imageId}/analyze`, {
      method: "POST",
    });
  },

  async getImageAnalysis(imageId: string): Promise<CorrosionAnalysisResponse> {
    return request<CorrosionAnalysisResponse>(`/images/${imageId}/analysis`);
  },
};
