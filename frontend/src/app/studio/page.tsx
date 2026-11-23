"use client";

import React, { Suspense, useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Sparkles,
  Sliders,
  Play,
  Upload,
  ChevronDown,
  ChevronUp,
  Loader2,
  Eye,
  Columns,
  Split,
  Image as ImageIcon,
  AlertTriangle,
  CheckCircle2,
  LayoutGrid,
  ShieldCheck,
} from "lucide-react";
import { AppShell } from "@/components/layout";
import { UploadModal } from "@/components/ui/UploadModal";
import { AnimatedButton, StaggerButton } from "@/components/ui";
import { api, getImageFileUrl, getAnalysisMapUrl, ApiError } from "@/lib/api";
import { ImageMetadata, JobStatus, SystemInfo, CorrosionAnalysisResponse } from "@/lib/types";

// 6 Generative Conditions from prompts.py
const GENERATIVE_CONDITIONS = [
  {
    id: "low_light_inspection",
    name: "Low-Light Inspection",
    desc: "Underexposed industrial tunnel with sensor noise",
  },
  {
    id: "directional_industrial_lighting",
    name: "Directional Industrial",
    desc: "Oblique inspection torch highlighting surface relief",
  },
  {
    id: "bright_daylight",
    name: "Bright Daylight",
    desc: "Overexposed outdoor daylight with metallic specular glare",
  },
  {
    id: "harsh_shadows",
    name: "Harsh Shadows",
    desc: "Deep cast shadows obscuring surface pitting geometry",
  },
  {
    id: "wet_surface",
    name: "Wet Surface",
    desc: "Reflective water condensation film across steel",
  },
  {
    id: "dusty_surface",
    name: "Dusty Surface",
    desc: "Fine particulate deposition softening structural edges",
  },
];

// 9 Traditional Transforms grouped into Lighting and Quality
const TRADITIONAL_LIGHTING = [
  { id: "brightness", name: "Brightness" },
  { id: "contrast", name: "Contrast" },
  { id: "gamma", name: "Gamma Curve" },
  { id: "uneven_illumination", name: "Uneven Vignette" },
  { id: "soft_shadow", name: "Soft Shadow" },
];

const TRADITIONAL_QUALITY = [
  { id: "gaussian_blur", name: "Gaussian Blur" },
  { id: "gaussian_noise", name: "Sensor Noise" },
  { id: "jpeg_compression", name: "JPEG Artifacts" },
  { id: "downsample_restore", name: "Downsample Restore" },
];

function StudioElapsedTimer({ isGenerating }: { isGenerating: boolean }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!isGenerating) return;
    const start = Date.now();
    const timer = setInterval(() => {
      setElapsed((Date.now() - start) / 1000);
    }, 100);
    return () => {
      clearInterval(timer);
      setElapsed(0);
    };
  }, [isGenerating]);

  return (
    <div className="flex items-center gap-1.5 text-white/80">
      <span className="text-[var(--saffron)] font-bold">{elapsed.toFixed(1)}s</span>
      <span className="text-white/40">elapsed</span>
    </div>
  );
}

function AugmentationStudioContent() {
  const searchParams = useSearchParams();
  const initialImageId = searchParams.get("imageId");

  // Selection state
  const [parentImages, setParentImages] = useState<ImageMetadata[]>([]);
  const [selectedParent, setSelectedParent] = useState<ImageMetadata | null>(null);
  const [variations, setVariations] = useState<ImageMetadata[]>([]);
  const [selectedVariation, setSelectedVariation] = useState<ImageMetadata | null>(null);

  // View state: split | side-by-side | original | maps
  const [viewMode, setViewMode] = useState<"split" | "side-by-side" | "original" | "maps">("split");
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage (0-100)
  const isDraggingSlider = useRef<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [fitMode, setFitMode] = useState<"fit" | "fill">("fit");

  // Configuration mode: "generative" | "traditional"
  const [augmentMode, setAugmentMode] = useState<"generative" | "traditional">("generative");

  // Generative config (calibrated sharpness: strength 0.45, controlScale 0.75)
  const [selectedGenCondition, setSelectedGenCondition] = useState<string>("low_light_inspection");
  const [genMode, setGenMode] = useState<"controlnet_canny" | "img2img">("controlnet_canny");
  const [strength, setStrength] = useState<number>(0.45);
  const [controlScale, setControlScale] = useState<number>(0.75);
  const [variationsCount, setVariationsCount] = useState<number>(1);
  const [seed, setSeed] = useState<string>("");
  const [showAdvancedSeed, setShowAdvancedSeed] = useState<boolean>(false);

  // Traditional config (calibrated intensity 0.55 for clear visibility)
  const [selectedTradCondition, setSelectedTradCondition] = useState<string>("brightness");
  const [intensity, setIntensity] = useState<number>(0.55);

  // Live ResNet-18 Corrosion Analysis State
  const [parentAnalysis, setParentAnalysis] = useState<CorrosionAnalysisResponse | null>(null);
  const [variationAnalysis, setVariationAnalysis] = useState<CorrosionAnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [mapTarget, setMapTarget] = useState<"parent" | "variation">("parent");

  // Execution & Job state
  const [activeJob, setActiveJob] = useState<JobStatus | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [reviewToast, setReviewToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  // Load parent images
  const loadParentImages = useCallback(async (preferredId?: string | null) => {
    try {
      const res = await api.listImages({ limit: 50 });
      setParentImages(res.items);

      if (res.items.length > 0) {
        if (preferredId) {
          const match = res.items.find((img) => img.image_id === preferredId);
          if (match) {
            setSelectedParent(match);
            return;
          }
        }
        const original = res.items.find((img) => img.source_type === "user_upload") || res.items[0];
        setSelectedParent(original);
      }
    } catch {
      // Graceful fallback
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      loadParentImages(initialImageId);
      api.getSystemInfo().then(setSystemInfo).catch(() => {});
    }, 0);
    return () => clearTimeout(timer);
  }, [initialImageId, loadParentImages]);

  // Load variations whenever selected parent changes
  const loadVariationsForParent = useCallback(async (parentId: string) => {
    try {
      const res = await api.listImages({ parent_image_id: parentId, limit: 30 });
      setVariations(res.items);
      if (res.items.length > 0) {
        setSelectedVariation(res.items[0]);
      } else {
        setSelectedVariation(null);
      }
    } catch {
      setVariations([]);
      setSelectedVariation(null);
    }
  }, []);

  useEffect(() => {
    if (selectedParent) {
      const parentId = selectedParent.image_id;
      const timer = setTimeout(() => {
        loadVariationsForParent(parentId);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [selectedParent, loadVariationsForParent]);

  // Auto-analyze selected parent with ResNet-18
  useEffect(() => {
    if (!selectedParent) return;
    const parentId = selectedParent.image_id;
    let isCancelled = false;

    const timer = setTimeout(() => {
      setIsAnalyzing(true);
      api
        .getImageAnalysis(parentId)
        .then((data) => {
          if (!isCancelled) setParentAnalysis(data);
        })
        .catch(() => {
          api
            .analyzeImage(parentId)
            .then((data) => {
              if (!isCancelled) setParentAnalysis(data);
            })
            .catch(() => {})
            .finally(() => {
              if (!isCancelled) setIsAnalyzing(false);
            });
        })
        .finally(() => {
          if (!isCancelled) setIsAnalyzing(false);
        });
    }, 0);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [selectedParent]);

  // Auto-analyze selected variation with ResNet-18
  useEffect(() => {
    if (!selectedVariation) return;
    const varId = selectedVariation.image_id;
    let isCancelled = false;

    const timer = setTimeout(() => {
      api
        .getImageAnalysis(varId)
        .then((data) => {
          if (!isCancelled) setVariationAnalysis(data);
        })
        .catch(() => {
          api
            .analyzeImage(varId)
            .then((data) => {
              if (!isCancelled) setVariationAnalysis(data);
            })
            .catch(() => {});
        });
    }, 0);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [selectedVariation]);

  // Handle Before/After slider dragging
  const handleSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pct);
  }, []);

  const handleMouseDown = () => {
    isDraggingSlider.current = true;
  };

  useEffect(() => {
    const handleMouseUp = () => {
      isDraggingSlider.current = false;
    };
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingSlider.current) {
        handleSliderMove(e.clientX);
      }
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDraggingSlider.current && e.touches.length > 0) {
        handleSliderMove(e.touches[0].clientX);
      }
    };

    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("touchend", handleMouseUp);
    window.addEventListener("touchmove", handleTouchMove);

    return () => {
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchend", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
    };
  }, [handleSliderMove]);

  // Keyboard accessibility for slider
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") {
      setSliderPosition((prev) => Math.max(0, prev - 5));
    } else if (e.key === "ArrowRight") {
      setSliderPosition((prev) => Math.min(100, prev + 5));
    }
  };

  // Poll Job Status (1s interval)
  const pollJobStatus = useCallback((jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const updated = await api.getJob(jobId);
        setActiveJob(updated);

        if (
          updated.status === "completed" ||
          updated.status === "failed" ||
          updated.status === "partial_failed"
        ) {
          clearInterval(interval);
          setIsSubmitting(false);

          if (updated.status === "completed" && selectedParent) {
            setJobError(null);
            await loadVariationsForParent(selectedParent.image_id);
            setViewMode("split");
          } else if (updated.status === "failed") {
            setJobError(updated.error_message || "Augmentation worker reported an error");
          } else if (updated.status === "partial_failed" && selectedParent) {
            await loadVariationsForParent(selectedParent.image_id);
            setViewMode("split");
          }
        }
      } catch {
        clearInterval(interval);
        setIsSubmitting(false);
        setJobError("Lost connection to job status endpoint");
      }
    }, 1000);
  }, [selectedParent, loadVariationsForParent]);

  // Run Augmentation Job
  const handleGenerate = async () => {
    if (!selectedParent) return;
    setIsSubmitting(true);
    setJobError(null);

    const parsedSeed = seed.trim() !== "" ? parseInt(seed, 10) : undefined;

    try {
      let job: JobStatus;
      if (augmentMode === "generative") {
        job = await api.createGenerativeJob({
          parent_image_ids: [selectedParent.image_id],
          conditions: [selectedGenCondition],
          mode: genMode,
          strength: strength,
          control_scale: controlScale,
          variations_per_image: variationsCount,
          seed: Number.isNaN(parsedSeed) ? undefined : parsedSeed,
        });
      } else {
        job = await api.createTraditionalJob({
          parent_image_ids: [selectedParent.image_id],
          conditions: [selectedTradCondition],
          intensity: intensity,
          variations_per_image: variationsCount,
          seed: Number.isNaN(parsedSeed) ? undefined : parsedSeed,
        });
      }

      setActiveJob(job);
      pollJobStatus(job.job_id);
    } catch (err) {
      if (err instanceof ApiError) {
        setJobError(err.message);
      } else {
        setJobError("Failed to initiate augmentation task.");
      }
      setIsSubmitting(false);
    }
  };

  // Quick Approval from Studio with Toast Confirmation
  const handleQuickReview = async (decision: "approved" | "needs_review" | "rejected") => {
    if (!selectedVariation) return;
    try {
      const updated = await api.updateReview(selectedVariation.image_id, {
        review_status: decision,
        reviewer: "Lead Inspector",
      });
      setSelectedVariation(updated);
      setVariations((prev) =>
        prev.map((v) => (v.image_id === updated.image_id ? updated : v))
      );
      const label =
        decision === "approved"
          ? "Approved"
          : decision === "needs_review"
          ? "Flagged for Review"
          : "Rejected";
      setReviewToast({ message: `Review status saved: ${label}`, type: "success" });
      setTimeout(() => setReviewToast(null), 3000);
      if (selectedParent) {
        loadVariationsForParent(selectedParent.image_id);
      }
    } catch {
      setReviewToast({ message: "Failed to persist review status", type: "error" });
      setTimeout(() => setReviewToast(null), 3000);
    }
  };

  // Guided 3-Step Wizard Indicators
  const isStep1Done = Boolean(selectedParent);
  const isStep2Done = Boolean(activeJob?.status === "completed" || variations.length > 0);
  const isStep3Active = Boolean(selectedVariation);

  const activeConditionName =
    augmentMode === "generative"
      ? GENERATIVE_CONDITIONS.find((c) => c.id === selectedGenCondition)?.name || selectedGenCondition
      : [...TRADITIONAL_LIGHTING, ...TRADITIONAL_QUALITY].find((c) => c.id === selectedTradCondition)?.name || selectedTradCondition;

  const gpuModel = systemInfo?.hardware?.gpu?.gpu_detected
    ? systemInfo.hardware.gpu.model_name?.replace("NVIDIA ", "") || "RTX 5060"
    : "CPU Node";
  const freeVramGb = systemInfo?.hardware?.gpu?.free_memory_mb
    ? (systemInfo.hardware.gpu.free_memory_mb / 1024).toFixed(1)
    : null;
  const gpuBadgeText = freeVramGb ? `${gpuModel} · ${freeVramGb} GB Free` : gpuModel;

  return (
    <AppShell pageTitle="Augmentation Studio">
      <div className="space-y-8 max-w-7xl mx-auto pb-16">
        {/* ================================================== */}
        {/* EDITORIAL TOP HEADER */}
        {/* ================================================== */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-[var(--border)]">
          <div className="space-y-1">
            <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--saffron)] font-bold">
              AUGMENTATION WORKSPACE
            </span>
            <h1 className="font-display text-4xl sm:text-5xl uppercase tracking-wider text-[var(--ink)] leading-none">
              UPLOAD, CONFIGURE, COMPARE
            </h1>
          </div>

          <div className="flex items-center gap-3">
            {/* Overview Back Pill */}
            <Link
              href="/"
              className="px-3.5 py-1.5 rounded-full border border-[var(--border)] bg-[var(--bg-2)] hover:bg-[var(--border)]/30 text-xs font-mono text-[var(--muted)] hover:text-[var(--ink)] transition-colors flex items-center gap-1.5"
            >
              <span>← OVERVIEW</span>
            </Link>

            {/* Live GPU Status Badge */}
            <div className="px-3 py-1.5 rounded-full bg-[var(--bg-2)] border border-[var(--border)] text-xs font-mono flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[var(--green-live)] animate-dot-ripple" />
              <span className="text-[var(--ink)] font-semibold">{gpuBadgeText}</span>
            </div>
          </div>
        </div>

        {/* ================================================== */}
        {/* CENTERED MAIN WIZARD SHEET */}
        {/* ================================================== */}
        <div className="max-w-6xl mx-auto rounded-3xl bg-[var(--bg)] border border-[var(--border)] p-6 lg:p-8 shadow-[var(--shadow-m)] space-y-8">
          {/* 3 Connected Step Indicator Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs select-none">
            {/* Step 1 */}
            <div
              className={`p-3.5 rounded-xl border flex items-center gap-3 transition-colors ${
                isStep1Done
                  ? "border-[var(--green)]/40 bg-[var(--green)]/10 text-[var(--green-live)] font-semibold"
                  : "border-[var(--saffron)] bg-[var(--saffron)]/10 text-[var(--saffron)] font-bold shadow-xs"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center font-display text-base shrink-0 ${
                  isStep1Done
                    ? "bg-[var(--green-live)] text-white"
                    : "bg-[var(--saffron)] text-white"
                }`}
              >
                {isStep1Done ? <CheckCircle2 className="w-4 h-4" /> : "01"}
              </div>
              <div className="flex flex-col">
                <span className="font-bold uppercase tracking-wider">STEP 1</span>
                <span className="text-[11px] text-[var(--muted)]">Source Image</span>
              </div>
            </div>

            {/* Step 2 */}
            <div
              className={`p-3.5 rounded-xl border flex items-center gap-3 transition-colors ${
                isSubmitting
                  ? "border-[var(--saffron)] bg-[var(--saffron)]/15 text-[var(--saffron)] font-bold animate-pulse"
                  : isStep2Done
                  ? "border-[var(--green)]/40 bg-[var(--green)]/10 text-[var(--green-live)] font-semibold"
                  : "border-[var(--border)] bg-[var(--bg-2)] text-[var(--muted)]"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center font-display text-base shrink-0 ${
                  isStep2Done
                    ? "bg-[var(--green-live)] text-white"
                    : isSubmitting
                    ? "bg-[var(--saffron)] text-white"
                    : "bg-[var(--border-2)] text-[var(--ink)]"
                }`}
              >
                {isSubmitting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isStep2Done ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : (
                  "02"
                )}
              </div>
              <div className="flex flex-col">
                <span className="font-bold uppercase tracking-wider">STEP 2</span>
                <span className="text-[11px] text-[var(--muted)]">Configure & Generate</span>
              </div>
            </div>

            {/* Step 3 */}
            <div
              className={`p-3.5 rounded-xl border flex items-center gap-3 transition-colors ${
                isStep3Active
                  ? "border-[var(--saffron)] bg-[var(--saffron)]/10 text-[var(--saffron)] font-bold shadow-xs"
                  : "border-[var(--border)] bg-[var(--bg-2)] text-[var(--muted)]"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-lg flex items-center justify-center font-display text-base shrink-0 ${
                  isStep3Active
                    ? "bg-[var(--saffron)] text-white"
                    : "bg-[var(--border-2)] text-[var(--ink)]"
                }`}
              >
                03
              </div>
              <div className="flex flex-col">
                <span className="font-bold uppercase tracking-wider">STEP 3</span>
                <span className="text-[11px] text-[var(--muted)]">Compare & Review</span>
              </div>
            </div>
          </div>

          {/* ================================================== */}
          {/* 2-COLUMN WORKSPACE (62% Dark Obsidian Vault / 38% Config Sheet) */}
          {/* ================================================== */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* LEFT: DARK OBSIDIAN INSPECTION VAULT (8 cols on lg ~ 65%) */}
          <div className="lg:col-span-8 space-y-4">
            <div className="rounded-3xl bg-[var(--vault-bg)] text-white p-5 sm:p-6 border border-[var(--vault-border)] relative overflow-hidden vault-glow shadow-[var(--shadow-m)]">
              {/* Minimal Top Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-[var(--vault-border)]">
                {/* View Mode Switcher */}
                <div className="flex items-center gap-1 p-1 rounded-xl bg-[var(--vault-card)] border border-[var(--vault-border)]">
                  <button
                    type="button"
                    onClick={() => setViewMode("split")}
                    disabled={!selectedVariation}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono tracking-wider transition-colors cursor-pointer flex items-center gap-1.5 ${
                      viewMode === "split"
                        ? "bg-[var(--saffron)] text-white shadow-xs"
                        : "text-white/60 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
                    }`}
                  >
                    <Split className="w-3.5 h-3.5" />
                    <span>COMPARE SLIDER</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setViewMode("side-by-side")}
                    disabled={!selectedVariation}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono tracking-wider transition-colors cursor-pointer flex items-center gap-1.5 ${
                      viewMode === "side-by-side"
                        ? "bg-[var(--saffron)] text-white shadow-xs"
                        : "text-white/60 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
                    }`}
                  >
                    <Columns className="w-3.5 h-3.5" />
                    <span>SIDE BY SIDE</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setViewMode("original")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono tracking-wider transition-colors cursor-pointer flex items-center gap-1.5 ${
                      viewMode === "original"
                        ? "bg-[var(--saffron)] text-white shadow-xs"
                        : "text-white/60 hover:text-white"
                    }`}
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>ORIGINAL</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setViewMode("maps")}
                    disabled={!selectedParent}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono tracking-wider transition-colors cursor-pointer flex items-center gap-1.5 ${
                      viewMode === "maps"
                        ? "bg-[var(--saffron)] text-white shadow-xs"
                        : "text-white/60 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
                    }`}
                  >
                    <LayoutGrid className="w-3.5 h-3.5" />
                    <span>LAYOUT MAPS</span>
                  </button>
                </div>

                {/* Right utility buttons: Fit toggle & Replace Source */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setFitMode(fitMode === "fit" ? "fill" : "fit")}
                    className="px-2.5 py-1.5 rounded-lg bg-[var(--vault-card)] border border-[var(--vault-border)] text-white/70 hover:text-white text-xs font-mono transition-colors cursor-pointer"
                    title="Toggle Aspect Fit / Fill"
                  >
                    {fitMode === "fit" ? "FIT" : "1:1"}
                  </button>

                  <StaggerButton
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setIsUploadOpen(true)}
                    leftIcon={<Upload className="w-3 h-3 text-[var(--saffron)]" />}
                    className="bg-[var(--vault-card)] border-[var(--vault-border)] text-white/80 hover:text-[var(--saffron)] hover:border-[var(--saffron)]"
                  >
                    REPLACE SOURCE
                  </StaggerButton>
                </div>
              </div>

              {/* Main Inspection Canvas Viewport */}
              <div
                ref={containerRef}
                tabIndex={0}
                onKeyDown={handleKeyDown}
                data-cursor="native"
                className="relative aspect-[16/10] w-full mt-4 rounded-2xl bg-[#070b12] border border-[var(--vault-border)] overflow-hidden select-none bg-inspection-grid flex items-center justify-center focus:outline-none focus:border-[var(--saffron)]/70"
              >
                {!selectedParent ? (
                  /* Empty State: Upload Canvas + Quick Picker Strip */
                  <div className="p-8 text-center space-y-4 max-w-md">
                    <div className="w-16 h-16 rounded-2xl bg-[var(--vault-card)] border border-[var(--vault-border)] flex items-center justify-center text-[var(--saffron)] mx-auto shadow-lg shadow-[var(--saffron)]/10">
                      <ImageIcon className="w-8 h-8" />
                    </div>

                    <div className="space-y-1">
                      <h3 className="font-display text-2xl uppercase tracking-wider text-white">
                        UPLOAD PIPELINE IMAGE
                      </h3>
                      <p className="text-xs text-white/60">
                        Drag and drop an inspection photograph or select from repository
                      </p>
                    </div>

                    <div className="pt-2">
                      <AnimatedButton
                        type="button"
                        variant="saffron"
                        size="default"
                        onClick={() => setIsUploadOpen(true)}
                        className="tracking-wider uppercase"
                      >
                        BROWSE IMAGE
                      </AnimatedButton>
                    </div>

                    <div className="inline-block px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-mono text-white/50">
                      Supported formats: JPEG, PNG, WEBP · Max 10 MB
                    </div>

                    {/* Quick Pick Originals Strip */}
                    {parentImages.length > 0 && (
                      <div className="pt-4 border-t border-[var(--vault-border)] space-y-2">
                        <span className="text-[10px] font-mono uppercase text-white/50 block">
                          Or pick from repository:
                        </span>
                        <div className="flex items-center justify-center gap-2 overflow-x-auto py-1">
                          {parentImages.slice(0, 5).map((img) => (
                            <button
                              key={img.image_id}
                              type="button"
                              onClick={() => setSelectedParent(img)}
                              className="w-12 h-12 rounded-lg overflow-hidden border border-white/20 hover:border-[var(--saffron)] transition-all shrink-0 cursor-pointer"
                              title={img.original_filename}
                            >
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={getImageFileUrl(img.image_id)}
                                alt={img.original_filename}
                                className="w-full h-full object-cover"
                              />
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : !selectedVariation && viewMode !== "original" && viewMode !== "maps" ? (
                  /* Parent loaded, awaiting variation generation */
                  <div className="relative w-full h-full flex flex-col items-center justify-center p-4">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={getImageFileUrl(selectedParent.image_id)}
                      alt="Source Parent"
                      className={`max-h-full max-w-full ${fitMode === "fill" ? "object-cover w-full h-full" : "object-contain"}`}
                    />
                    <div className="absolute bottom-4 inset-x-6 p-3 rounded-xl bg-[var(--vault-card)]/90 backdrop-blur-md border border-[var(--vault-border)] text-center text-xs text-white/70">
                      Displaying master asset <span className="font-mono text-white">{selectedParent.original_filename}</span>. Select conditions on the right and click <strong className="text-[var(--saffron)]">GENERATE VARIATIONS</strong>.
                    </div>
                  </div>
                ) : (
                  /* Active Inspection Viewport */
                  <>
                    {/* Mode 1: Split Comparison Slider */}
                    {viewMode === "split" && selectedVariation && (
                      <div className="relative w-full h-full overflow-hidden flex items-center justify-center">
                        {/* Base: Generated Variation */}
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getImageFileUrl(selectedVariation.image_id)}
                          alt="Augmented Variation"
                          className={`absolute inset-0 w-full h-full pointer-events-none ${fitMode === "fill" ? "object-cover" : "object-contain"}`}
                        />

                        {/* Top: Clipped Original Parent with modern CSS clip-path */}
                        <div
                          className="absolute inset-0 pointer-events-none flex items-center justify-center"
                          style={{
                            clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`,
                          }}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={getImageFileUrl(selectedParent.image_id)}
                            alt="Original Source"
                            className={`w-full h-full pointer-events-none ${fitMode === "fill" ? "object-cover" : "object-contain"}`}
                          />
                        </div>

                        {/* Draggable Saffron Divider Line & Handle */}
                        <div
                          className="absolute top-0 bottom-0 z-20 cursor-ew-resize flex items-center justify-center"
                          style={{ left: `${sliderPosition}%` }}
                          onMouseDown={handleMouseDown}
                          onTouchStart={handleMouseDown}
                        >
                          <div className="w-[2px] h-full bg-[var(--saffron)] shadow-[0_0_10px_rgba(240,78,0,0.9)]" />
                          <div className="absolute w-7 h-7 rounded-full bg-[var(--vault-bg)] border-2 border-[var(--saffron)] shadow-lg flex items-center justify-center text-[var(--saffron)] text-[10px] font-mono font-bold select-none cursor-ew-resize hover:scale-110 transition-transform">
                            ⬌
                          </div>
                        </div>

                        {/* Corner Badges */}
                        <div className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded bg-black/75 backdrop-blur-md text-[10px] font-mono text-white/90">
                          ORIGINAL ({Math.round(sliderPosition)}%)
                        </div>
                        <div className="absolute top-3 right-3 z-10 px-2.5 py-1 rounded bg-[var(--saffron)]/90 backdrop-blur-md text-[10px] font-mono text-white font-semibold">
                          SYNTHETIC: {selectedVariation.condition || selectedVariation.generation_method}
                        </div>
                      </div>
                    )}

                    {/* Mode 2: Side-by-Side */}
                    {viewMode === "side-by-side" && selectedVariation && (
                      <div className="grid grid-cols-2 w-full h-full gap-3 p-3">
                        <div className="relative rounded-xl overflow-hidden bg-black/40 border border-white/10 flex items-center justify-center">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={getImageFileUrl(selectedParent.image_id)}
                            alt="Original"
                            className="max-h-full max-w-full object-contain"
                          />
                          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-white">
                            Original
                          </div>
                        </div>

                        <div className="relative rounded-xl overflow-hidden bg-black/40 border border-[var(--saffron)]/40 flex items-center justify-center">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={getImageFileUrl(selectedVariation.image_id)}
                            alt="Variation"
                            className="max-h-full max-w-full object-contain"
                          />
                          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-[var(--saffron)] text-[10px] font-mono text-white font-semibold">
                            {selectedVariation.condition || "Synthetic"}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Mode 3: Original Only */}
                    {viewMode === "original" && (
                      <div className="relative w-full h-full flex items-center justify-center">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getImageFileUrl(selectedParent.image_id)}
                          alt="Original"
                          className="max-h-full max-w-full object-contain"
                        />
                        <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-white">
                          Master Asset ({selectedParent.width}×{selectedParent.height})
                        </div>
                      </div>
                    )}

                    {/* Mode 4: Visual Layout Maps (Pipeline Canny, Corrosion Mask, Grad-CAM Overlay) */}
                    {viewMode === "maps" && selectedParent && (
                      <div className="relative w-full h-full p-4 overflow-y-auto flex flex-col justify-between space-y-3">
                        {/* Target Switcher if variation exists */}
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-[11px] font-mono text-white/50 uppercase">Analysis Target:</span>
                            <div className="inline-flex rounded-lg bg-black/40 border border-white/10 p-0.5">
                              <button
                                type="button"
                                onClick={() => setMapTarget("parent")}
                                className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors cursor-pointer ${
                                  mapTarget === "parent"
                                    ? "bg-[var(--saffron)] text-white font-bold"
                                    : "text-white/60 hover:text-white"
                                }`}
                              >
                                Original Asset
                              </button>
                              {selectedVariation && (
                                <button
                                  type="button"
                                  onClick={() => setMapTarget("variation")}
                                  className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors cursor-pointer ${
                                    mapTarget === "variation"
                                      ? "bg-[var(--saffron)] text-white font-bold"
                                      : "text-white/60 hover:text-white"
                                  }`}
                                >
                                  Synthetic Variation
                                </button>
                              )}
                            </div>
                          </div>

                          {(() => {
                            const currentAnalysis = mapTarget === "variation" && variationAnalysis ? variationAnalysis : parentAnalysis;
                            if (!currentAnalysis) return null;
                            return (
                              <div className="flex items-center gap-2 text-[11px] font-mono">
                                <span className="text-white/50">ResNet-18 Verdict:</span>
                                <span
                                  className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                                    currentAnalysis.predicted_class === "corroded"
                                      ? "bg-[var(--saffron)]/20 text-[var(--saffron)] border border-[var(--saffron)]/40"
                                      : "bg-[var(--green)]/20 text-[var(--green-live)] border border-[var(--green)]/40"
                                  }`}
                                >
                                  {currentAnalysis.predicted_class.replace("_", " ")} ({(currentAnalysis.confidence * 100).toFixed(1)}%)
                                </span>
                              </div>
                            );
                          })()}
                        </div>

                        {/* 3 Layout Map Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 flex-1 min-h-0">
                          {(() => {
                            const currentId = mapTarget === "variation" && selectedVariation ? selectedVariation.image_id : selectedParent.image_id;
                            const currentAnalysis = mapTarget === "variation" && variationAnalysis ? variationAnalysis : parentAnalysis;
                            return (
                              <>
                                {/* Map 1: Structural Canny */}
                                <div className="flex flex-col rounded-xl overflow-hidden bg-black/50 border border-white/10 p-2.5 space-y-2">
                                  <div className="flex items-center justify-between text-[11px] font-mono">
                                    <span className="font-semibold text-white/90">1. Structural Layout</span>
                                    <span className="text-[9px] text-white/40">Canny Edge</span>
                                  </div>
                                  <div className="relative flex-1 min-h-[140px] rounded-lg overflow-hidden bg-[#0D0E15] flex items-center justify-center border border-white/5">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={getAnalysisMapUrl(currentId, "pipeline_canny")}
                                      alt="Pipeline Structural Layout"
                                      className="w-full h-full object-contain"
                                    />
                                  </div>
                                  <p className="text-[10px] text-white/50 leading-tight">
                                    Pipeline structural layout preserving boundaries and weld geometry.
                                  </p>
                                </div>

                                {/* Map 2: Corrosion Layout */}
                                <div className="flex flex-col rounded-xl overflow-hidden bg-black/50 border border-white/10 p-2.5 space-y-2">
                                  <div className="flex items-center justify-between text-[11px] font-mono">
                                    <span className="font-semibold text-white/90">2. Corrosion Layout</span>
                                    <span className="text-[9px] text-[var(--saffron)] font-bold">
                                      {currentAnalysis ? `${currentAnalysis.rust_coverage_percent.toFixed(1)}% Coverage` : "Mask"}
                                    </span>
                                  </div>
                                  <div className="relative flex-1 min-h-[140px] rounded-lg overflow-hidden bg-[#0F1117] flex items-center justify-center border border-white/5">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={getAnalysisMapUrl(currentId, "corrosion_layout")}
                                      alt="Corrosion Layout Map"
                                      className="w-full h-full object-contain"
                                    />
                                  </div>
                                  <p className="text-[10px] text-white/50 leading-tight">
                                    Isolated candidate corrosion mask with Cyber-Saffron contours.
                                  </p>
                                </div>

                                {/* Map 3: AI Attention Heatmap */}
                                <div className="flex flex-col rounded-xl overflow-hidden bg-black/50 border border-white/10 p-2.5 space-y-2">
                                  <div className="flex items-center justify-between text-[11px] font-mono">
                                    <span className="font-semibold text-white/90">3. AI Attention Heatmap</span>
                                    <span className="text-[9px] text-white/40">Grad-CAM layer4</span>
                                  </div>
                                  <div className="relative flex-1 min-h-[140px] rounded-lg overflow-hidden bg-black flex items-center justify-center border border-white/5">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={getAnalysisMapUrl(currentId, "corrosion_overlay")}
                                      alt="AI Attention Heatmap"
                                      className="w-full h-full object-contain"
                                    />
                                  </div>
                                  <p className="text-[10px] text-white/50 leading-tight">
                                    ResNet-18 neural attention overlay pinpointing corrosion features.
                                  </p>
                                </div>
                              </>
                            );
                          })()}
                        </div>

                        {/* Honest Diagnostic Disclaimer Footer */}
                        <div className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 flex items-center justify-between text-[10px] font-mono text-white/50">
                          <span>
                            Exploratory CV screening diagnostics · ResNet-18 Grad-CAM & joint HSV/Lab color heuristic
                          </span>
                          <span className="text-[var(--gold)] font-semibold uppercase">Not Certified NDT</span>
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* Processing State Overlay */}
                {isSubmitting && (
                  <div className="absolute inset-0 z-30 bg-[#070b12]/92 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center space-y-5 animate-page-reveal select-none">
                    {/* Animated Pulse Rings & Spinner */}
                    <div className="relative flex items-center justify-center">
                      <div className="w-16 h-16 rounded-full border-3 border-[var(--saffron)]/30 border-t-[var(--saffron)] animate-spin" />
                      <div className="absolute w-8 h-8 rounded-full bg-[var(--saffron)]/10 animate-ping" />
                      <Sparkles className="absolute w-6 h-6 text-[var(--saffron)]" />
                    </div>

                    {/* Title & Pipeline Badge */}
                    <div className="space-y-1.5 max-w-md">
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono">
                        <span className="w-2 h-2 rounded-full bg-[var(--green-live)] animate-dot-ripple" />
                        <span className="text-white font-semibold">
                          {activeJob?.status === "queued" ? "QUEUED IN ENGINE" : `RUNNING ON ${gpuModel}`}
                        </span>
                      </div>
                      <h3 className="font-display text-2xl sm:text-3xl tracking-wider uppercase text-white">
                        {augmentMode === "generative"
                          ? genMode === "controlnet_canny"
                            ? "SYNTHESIZING WITH CONTROLNET CANNY"
                            : "SYNTHESIZING WITH PLAIN SD 1.5"
                          : "APPLYING TRADITIONAL CV TRANSFORMS"}
                      </h3>
                      <p className="text-xs font-mono text-white/70">
                        Condition: {activeConditionName} ·{" "}
                        {augmentMode === "generative"
                          ? `Strength ${strength.toFixed(2)}`
                          : `Intensity ${intensity.toFixed(2)}`}
                      </p>
                    </div>

                    {/* Live Stopwatch & Real Progress Counter */}
                    <div className="flex items-center gap-4 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-xs font-mono">
                      <StudioElapsedTimer isGenerating={isSubmitting || activeJob?.status === "running"} />
                      <span className="text-white/20">|</span>
                      <div className="text-white/80">
                        {activeJob
                          ? `Item ${Math.min(activeJob.total_items, (activeJob.completed_items ?? 0) + 1)} of ${activeJob.total_items}`
                          : "Initializing CUDA pipeline..."}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Live ResNet-18 Corrosion Diagnostic Bar */}
              {selectedParent && (
                <div className="mt-4 p-3.5 rounded-2xl bg-[#070b12] border border-[var(--vault-border)] space-y-2.5 font-mono text-xs text-white">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2.5">
                      <span className="text-white/50 text-[11px] flex items-center gap-1.5">
                        <ShieldCheck className="w-3.5 h-3.5 text-[var(--saffron)]" />
                        <span>ResNet-18 Corrosion Verdict:</span>
                      </span>

                      {isAnalyzing ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-white/5 text-white/60 text-[11px]">
                          <Loader2 className="w-3 h-3 animate-spin text-[var(--saffron)]" />
                          Running ResNet-18 inference...
                        </span>
                      ) : parentAnalysis ? (
                        <>
                          <span
                            className={`px-2.5 py-0.5 rounded text-[11px] font-bold uppercase tracking-wider ${
                              parentAnalysis.predicted_class === "corroded"
                                ? "bg-[var(--saffron)]/20 text-[var(--saffron)] border border-[var(--saffron)]/40"
                                : "bg-[var(--green)]/20 text-[var(--green-live)] border border-[var(--green)]/40"
                            }`}
                          >
                            {parentAnalysis.predicted_class.replace("_", " ")} ({(parentAnalysis.confidence * 100).toFixed(1)}%)
                          </span>

                          <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                            Severity:{" "}
                            <strong
                              className={
                                parentAnalysis.severity_level === "Severe"
                                  ? "text-[var(--danger)]"
                                  : parentAnalysis.severity_level === "Moderate"
                                  ? "text-[var(--saffron)]"
                                  : parentAnalysis.severity_level === "Low"
                                  ? "text-sky-400"
                                  : "text-[var(--green-live)]"
                              }
                            >
                              {parentAnalysis.severity_level}
                            </strong>
                          </span>

                          <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                            Rust Coverage:{" "}
                            <strong className="text-white font-bold">
                              {parentAnalysis.rust_coverage_percent.toFixed(2)}%
                            </strong>
                          </span>
                        </>
                      ) : (
                        <span className="text-white/40 text-[11px]">Ready for inspection</span>
                      )}
                    </div>

                    {/* Label Preservation Check for Generated Variations */}
                    {selectedVariation && (
                      <div className="flex items-center gap-2">
                        <span className="text-white/50 text-[11px]">Consistency:</span>
                        {(() => {
                          const qm = (selectedVariation.quality_metrics || {}) as Record<string, unknown>;
                          const isPreserved = qm.label_preserved;
                          const parentV = qm.parent_verdict || parentAnalysis?.predicted_class;
                          const varV = qm.variation_verdict || variationAnalysis?.predicted_class;

                          if (isPreserved === true || (parentV && varV && parentV === varV)) {
                            return (
                              <span className="px-2.5 py-0.5 rounded bg-[var(--green)]/20 text-[var(--green-live)] border border-[var(--green)]/40 font-bold text-[10px] flex items-center gap-1">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>Preserved ✓ ({String(parentV).replace("_", " ")} → {String(varV).replace("_", " ")})</span>
                              </span>
                            );
                          } else if (isPreserved === false || (parentV && varV && parentV !== varV)) {
                            return (
                              <span className="px-2.5 py-0.5 rounded bg-[var(--danger)]/20 text-[var(--danger)] border border-[var(--danger)]/40 font-bold text-[10px] flex items-center gap-1">
                                <AlertTriangle className="w-3 h-3" />
                                <span>Shifted ⚠️ ({String(parentV).replace("_", " ")} → {String(varV).replace("_", " ")})</span>
                              </span>
                            );
                          }
                          return (
                            <span className="px-2.5 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-white/60">
                              Verified
                            </span>
                          );
                        })()}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Bottom Structured High-Contrast Telemetry & Review Console */}
              {selectedVariation ? (
                <div className="mt-4 p-4 rounded-2xl bg-[#070b12] border border-[var(--vault-border)] shadow-inner space-y-3 font-mono text-xs text-white">
                  {/* Top Row: File Provenance & Model Specs */}
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-2.5 border-b border-white/10">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-white/50 text-[11px]">Asset:</span>
                      <span
                        className="text-white font-semibold truncate max-w-[220px]"
                        title={selectedVariation.original_filename}
                      >
                        {selectedVariation.original_filename}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-white/10 text-[10px] text-white/90 font-bold">
                        {selectedVariation.width}×{selectedVariation.height} px
                      </span>
                      {(() => {
                        const p = (selectedVariation.parameters || {}) as Record<string, unknown>;
                        const sVal = String(p.strength ?? "0.35");
                        const cVal = String(p.control_scale ?? "0.80");
                        const isPlain = p.mode === "img2img" || selectedVariation.generation_method === "img2img";
                        const badgeText =
                          selectedVariation.source_type === "generative"
                            ? isPlain
                              ? `Plain SD 1.5 (s=${sVal})`
                              : `ControlNet Canny (c=${cVal}, s=${sVal})`
                            : selectedVariation.source_type === "traditional"
                            ? `Traditional CV (${selectedVariation.condition || "Transform"})`
                            : "Original Source Asset";

                        return (
                          <span className="px-2.5 py-0.5 rounded bg-[var(--saffron)]/20 border border-[var(--saffron)]/40 text-[var(--saffron)] text-[10px] font-bold">
                            {badgeText}
                          </span>
                        );
                      })()}
                    </div>

                    {/* Seed & Runtime Provenance */}
                    <div className="flex items-center gap-3 text-[11px] text-white/70">
                      {selectedVariation.seed !== undefined && selectedVariation.seed !== null && (
                        <span>
                          Seed: <strong className="text-white">{String(selectedVariation.seed)}</strong>
                        </span>
                      )}
                      {(() => {
                        const p = (selectedVariation.parameters || {}) as Record<string, unknown>;
                        const qm = (selectedVariation.quality_metrics || {}) as Record<string, unknown>;
                        const rawRt = p.runtime_seconds ?? qm.runtime_seconds;
                        const rtNum = typeof rawRt === "number" ? rawRt : typeof rawRt === "string" ? parseFloat(rawRt) : null;
                        if (rtNum === null || Number.isNaN(rtNum)) return null;
                        return (
                          <span>
                            Runtime: <strong className="text-[var(--saffron)]">{rtNum.toFixed(2)}s</strong>
                          </span>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Middle Row: Phase 4 Quality Validation Metrics Pills */}
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-white/50 text-[11px]">Phase 4 Metrics:</span>
                      {selectedVariation.quality_metrics?.ssim_rgb !== undefined ? (
                        <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                          SSIM:{" "}
                          <strong className="text-white font-bold">
                            {selectedVariation.quality_metrics.ssim_rgb.toFixed(4)}
                          </strong>
                        </span>
                      ) : selectedVariation.quality_metrics?.ssim !== undefined ? (
                        <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                          SSIM:{" "}
                          <strong className="text-white font-bold">
                            {selectedVariation.quality_metrics.ssim.toFixed(4)}
                          </strong>
                        </span>
                      ) : null}

                      {selectedVariation.quality_metrics?.lpips !== undefined && (
                        <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                          LPIPS:{" "}
                          <strong className="text-white font-bold">
                            {selectedVariation.quality_metrics.lpips.toFixed(4)}
                          </strong>
                        </span>
                      )}

                      {selectedVariation.quality_metrics?.edge_overlap_iou !== undefined && (
                        <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[11px]">
                          Edge IoU:{" "}
                          <strong className="text-white font-bold">
                            {selectedVariation.quality_metrics.edge_overlap_iou.toFixed(4)}
                          </strong>
                        </span>
                      )}

                      {selectedVariation.quality_status && (
                        <span
                          className={`px-2.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                            selectedVariation.quality_status === "pass"
                              ? "bg-[var(--green)]/20 text-[var(--green-live)] border border-[var(--green)]/40"
                              : selectedVariation.quality_status === "flag_for_review"
                              ? "bg-[var(--gold)]/20 text-[var(--gold)] border border-[var(--gold)]/40"
                              : "bg-[var(--danger)]/20 text-[var(--danger)] border border-[var(--danger)]/40"
                          }`}
                        >
                          Quality: {selectedVariation.quality_status.replace(/_/g, " ")}
                        </span>
                      )}
                    </div>

                    {/* Inline Human Review Gate with Crisp Pill Buttons */}
                    <div className="flex items-center gap-2">
                      <span className="text-white/50 text-[11px]">Review Gate:</span>
                      <button
                        type="button"
                        onClick={() => handleQuickReview("approved")}
                        className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
                          selectedVariation.review_status === "approved"
                            ? "bg-[var(--green)] text-white border-[var(--green)] ring-2 ring-[var(--green)]/30 font-bold shadow-xs"
                            : "border-white/20 text-[var(--green-live)] hover:bg-[var(--green)]/20 hover:border-[var(--green)]"
                        }`}
                      >
                        Approve
                      </button>
                      <button
                        type="button"
                        onClick={() => handleQuickReview("needs_review")}
                        className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
                          selectedVariation.review_status === "needs_review"
                            ? "bg-[var(--gold)] text-black border-[var(--gold)] ring-2 ring-[var(--gold)]/30 font-bold shadow-xs"
                            : "border-white/20 text-[var(--gold)] hover:bg-[var(--gold)]/20 hover:border-[var(--gold)]"
                        }`}
                      >
                        Needs Review
                      </button>
                      <button
                        type="button"
                        onClick={() => handleQuickReview("rejected")}
                        className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
                          selectedVariation.review_status === "rejected"
                            ? "bg-[var(--danger)] text-white border-[var(--danger)] ring-2 ring-[var(--danger)]/30 font-bold shadow-xs"
                            : "border-white/20 text-[var(--danger)] hover:bg-[var(--danger)]/20 hover:border-[var(--danger)]"
                        }`}
                      >
                        Reject
                      </button>
                    </div>
                  </div>

                  {/* Toast Feedback Row */}
                  {reviewToast && (
                    <div className="pt-1 flex items-center justify-end">
                      <span
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-mono font-semibold animate-page-reveal ${
                          reviewToast.type === "success"
                            ? "bg-[var(--green)]/20 text-[var(--green-live)] border border-[var(--green)]/40"
                            : "bg-[var(--danger)]/20 text-[var(--danger)] border border-[var(--danger)]/40"
                        }`}
                      >
                        {reviewToast.message}
                      </span>
                    </div>
                  )}
                </div>
              ) : selectedParent ? (
                <div className="mt-4 p-3.5 rounded-2xl bg-[#070b12] border border-[var(--vault-border)] flex items-center justify-between font-mono text-xs text-white/70">
                  <div className="flex items-center gap-2">
                    <span className="text-white/50 text-[11px]">Selected Source:</span>
                    <strong className="text-white">{selectedParent.original_filename}</strong>
                    <span className="px-2 py-0.5 rounded bg-white/10 text-[10px] text-white/90 font-bold">
                      {selectedParent.width}×{selectedParent.height} px
                    </span>
                  </div>
                  <span className="text-[11px] text-[var(--saffron)]">
                    Ready for synthesis · Configure parameters on right →
                  </span>
                </div>
              ) : null}
            </div>

            {/* Variations Filmstrip beneath main viewport */}
            <div className="p-4 rounded-3xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] space-y-3">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-[var(--muted)] uppercase tracking-wider font-semibold">
                  VARIATIONS FILMSTRIP ({variations.length})
                </span>
                <span className="text-[11px] text-[var(--muted)]">
                  Click thumbnail to load into comparison viewport
                </span>
              </div>

              {variations.length === 0 ? (
                <div className="py-4 text-center text-xs text-[var(--muted)] font-mono">
                  No variations generated for this master asset yet.
                </div>
              ) : (
                <div className="flex items-center gap-3 overflow-x-auto pb-1">
                  {variations.map((v) => {
                    const isSelected = selectedVariation?.image_id === v.image_id;
                    return (
                      <button
                        key={v.image_id}
                        type="button"
                        onClick={() => setSelectedVariation(v)}
                        className={`relative w-20 h-20 rounded-xl overflow-hidden shrink-0 border-2 transition-all cursor-pointer ${
                          isSelected
                            ? "border-[var(--saffron)] scale-105 shadow-[var(--shadow-s)]"
                            : "border-[var(--border)] hover:border-[var(--border-2)] opacity-80 hover:opacity-100"
                        }`}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={getImageFileUrl(v.image_id)}
                          alt={v.condition || "variation"}
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-x-0 bottom-0 bg-black/75 p-0.5 text-[9px] font-mono text-white text-center truncate">
                          {v.condition || v.generation_method}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: CRISP CONFIGURATION CARD (4 cols on lg ~ 35%) */}
          <div className="lg:col-span-4 space-y-6">
            <div className="p-6 rounded-3xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-m)] space-y-6">
              {/* Header */}
              <div className="space-y-1">
                <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--saffron)] font-semibold">
                  AUGMENTATION CONTROLS
                </span>
                <h2 className="font-display text-3xl uppercase tracking-wider text-[var(--ink)]">
                  CONFIGURE VARIATION
                </h2>
              </div>

              {/* Method Segmented Switch: ControlNet AI vs Traditional CV */}
              <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                <button
                  type="button"
                  onClick={() => {
                    setAugmentMode("generative");
                    setJobError(null);
                  }}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold tracking-wider transition-colors cursor-pointer flex items-center justify-center gap-1.5 ${
                    augmentMode === "generative"
                      ? "bg-[var(--bg)] text-[var(--saffron)] shadow-xs"
                      : "text-[var(--muted)] hover:text-[var(--ink)]"
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>ControlNet AI</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setAugmentMode("traditional");
                    setJobError(null);
                  }}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold tracking-wider transition-colors cursor-pointer flex items-center justify-center gap-1.5 ${
                    augmentMode === "traditional"
                      ? "bg-[var(--bg)] text-[var(--saffron)] shadow-xs"
                      : "text-[var(--muted)] hover:text-[var(--ink)]"
                  }`}
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span>Traditional CV</span>
                </button>
              </div>

              {/* Generative Controls */}
              {augmentMode === "generative" ? (
                <div className="space-y-5 animate-page-reveal">
                  {/* 6 Selectable Condition Chips (2-col grid) */}
                  <div className="space-y-2">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Target Environmental Condition
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {GENERATIVE_CONDITIONS.map((cond) => {
                        const isSelected = selectedGenCondition === cond.id;
                        return (
                          <div
                            key={cond.id}
                            onClick={() => setSelectedGenCondition(cond.id)}
                            className={`p-2.5 rounded-xl border text-left cursor-pointer transition-all ${
                              isSelected
                                ? "border-[var(--saffron)] bg-[var(--saffron)]/10 text-[var(--ink)] shadow-xs"
                                : "border-[var(--border)] bg-[var(--bg-2)]/60 hover:bg-[var(--bg-2)] text-[var(--muted)] hover:text-[var(--ink)]"
                            }`}
                          >
                            <span className="text-xs font-semibold block leading-tight">
                              {cond.name}
                            </span>
                            <span className="text-[10px] text-[var(--muted)] mt-1 line-clamp-1 block">
                              {cond.desc}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Mode Toggle: ControlNet Canny vs Plain SD 1.5 */}
                  <div className="space-y-1.5">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Diffusion Architecture
                    </label>
                    <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                      <button
                        type="button"
                        onClick={() => {
                          setGenMode("controlnet_canny");
                          setJobError(null);
                        }}
                        className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-colors cursor-pointer flex items-center justify-center text-center ${
                          genMode === "controlnet_canny"
                            ? "bg-[var(--bg)] text-[var(--saffron)] font-bold shadow-xs border border-[var(--saffron)]/30"
                            : "text-[var(--muted)] hover:text-[var(--ink)]"
                        }`}
                      >
                        ControlNet Canny
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setGenMode("img2img");
                          setJobError(null);
                        }}
                        className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-colors cursor-pointer flex items-center justify-center text-center ${
                          genMode === "img2img"
                            ? "bg-[var(--bg)] text-[var(--saffron)] font-bold shadow-xs border border-[var(--saffron)]/30"
                            : "text-[var(--muted)] hover:text-[var(--ink)]"
                        }`}
                      >
                        Plain SD 1.5
                      </button>
                    </div>
                  </div>

                  {/* Denoising Strength Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center text-xs font-mono">
                      <span className="text-[var(--muted)] uppercase font-semibold">
                        Denoising Strength
                      </span>
                      <strong className="text-[var(--saffron)] font-bold">
                        {strength.toFixed(2)}
                      </strong>
                    </div>
                    <input
                      type="range"
                      min={0.25}
                      max={0.65}
                      step={0.01}
                      value={strength}
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      className="w-full accent-[var(--saffron)] cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] font-mono text-[var(--muted)]">
                      <span>0.25</span>
                      <span className="text-[var(--green)]">0.45 (Optimal Sharpness)</span>
                      <span>0.65</span>
                    </div>
                  </div>

                  {/* ControlNet Scale Slider */}
                  {genMode === "controlnet_canny" && (
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center text-xs font-mono">
                        <span className="text-[var(--muted)] uppercase font-semibold">
                          ControlNet Scale
                        </span>
                        <strong className="text-[var(--saffron)] font-bold">
                          {controlScale.toFixed(2)}
                        </strong>
                      </div>
                      <input
                        type="range"
                        min={0.35}
                        max={1.00}
                        step={0.05}
                        value={controlScale}
                        onChange={(e) => setControlScale(parseFloat(e.target.value))}
                        className="w-full accent-[var(--saffron)] cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] font-mono text-[var(--muted)]">
                        <span>0.35</span>
                        <span className="text-[var(--green)]">0.75 (Optimal Structural Anchor)</span>
                        <span>1.00</span>
                      </div>
                    </div>
                  )}

                  {/* Variations Selector: 1 | 2 | 4 */}
                  <div className="space-y-1.5">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Variations Count
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {[1, 2, 4].map((count) => (
                        <button
                          key={count}
                          type="button"
                          onClick={() => setVariationsCount(count)}
                          className={`py-1.5 rounded-lg border text-xs font-mono font-bold transition-colors cursor-pointer ${
                            variationsCount === count
                              ? "border-[var(--saffron)] bg-[var(--saffron)] text-white shadow-xs"
                              : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-2)]"
                          }`}
                        >
                          {count}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Collapsible Advanced Seed Disclosure */}
                  <div className="pt-2 border-t border-[var(--border)]">
                    <button
                      type="button"
                      onClick={() => setShowAdvancedSeed(!showAdvancedSeed)}
                      className="w-full flex items-center justify-between text-xs font-mono text-[var(--muted)] hover:text-[var(--ink)] cursor-pointer py-1"
                    >
                      <span>Advanced Parameters (Seed)</span>
                      {showAdvancedSeed ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>
                    {showAdvancedSeed && (
                      <div className="pt-2 space-y-1">
                        <input
                          type="text"
                          placeholder="Random reproducible seed..."
                          value={seed}
                          onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))}
                          className="w-full bg-[var(--bg-2)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs font-mono text-[var(--ink)] focus:outline-none focus:border-[var(--saffron)]"
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* Traditional Controls */
                <div className="space-y-5 animate-page-reveal">
                  {/* Transforms Group 1: Lighting */}
                  <div className="space-y-2">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Physical Lighting Perturbations
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {TRADITIONAL_LIGHTING.map((tf) => {
                        const isSelected = selectedTradCondition === tf.id;
                        return (
                          <button
                            key={tf.id}
                            type="button"
                            onClick={() => setSelectedTradCondition(tf.id)}
                            className={`p-2 rounded-xl border text-left text-xs font-semibold truncate transition-colors cursor-pointer ${
                              isSelected
                                ? "border-[var(--saffron)] bg-[var(--saffron)]/10 text-[var(--ink)] shadow-xs"
                                : "border-[var(--border)] bg-[var(--bg-2)]/60 hover:bg-[var(--bg-2)] text-[var(--muted)] hover:text-[var(--ink)]"
                            }`}
                          >
                            {tf.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Transforms Group 2: Image Quality */}
                  <div className="space-y-2">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Sensor & Lens Degradations
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {TRADITIONAL_QUALITY.map((tf) => {
                        const isSelected = selectedTradCondition === tf.id;
                        return (
                          <button
                            key={tf.id}
                            type="button"
                            onClick={() => setSelectedTradCondition(tf.id)}
                            className={`p-2 rounded-xl border text-left text-xs font-semibold truncate transition-colors cursor-pointer ${
                              isSelected
                                ? "border-[var(--saffron)] bg-[var(--saffron)]/10 text-[var(--ink)] shadow-xs"
                                : "border-[var(--border)] bg-[var(--bg-2)]/60 hover:bg-[var(--bg-2)] text-[var(--muted)] hover:text-[var(--ink)]"
                            }`}
                          >
                            {tf.name}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Intensity Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center text-xs font-mono">
                      <span className="text-[var(--muted)] uppercase font-semibold">
                        Transform Intensity
                      </span>
                      <strong className="text-[var(--saffron)] font-bold">
                        {intensity.toFixed(2)}
                      </strong>
                    </div>
                    <input
                      type="range"
                      min={0.10}
                      max={1.00}
                      step={0.05}
                      value={intensity}
                      onChange={(e) => setIntensity(parseFloat(e.target.value))}
                      className="w-full accent-[var(--saffron)] cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] font-mono text-[var(--muted)]">
                      <span>0.10</span>
                      <span className="text-[var(--green)]">0.55 (Calibrated Visible)</span>
                      <span>1.00</span>
                    </div>
                  </div>

                  {/* Variations Selector: 1 | 2 | 4 */}
                  <div className="space-y-1.5">
                    <label className="text-[11px] font-mono tracking-wider uppercase text-[var(--muted)] font-semibold block">
                      Variations Count
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {[1, 2, 4].map((count) => (
                        <button
                          key={count}
                          type="button"
                          onClick={() => setVariationsCount(count)}
                          className={`py-1.5 rounded-lg border text-xs font-mono font-bold transition-colors cursor-pointer ${
                            variationsCount === count
                              ? "border-[var(--saffron)] bg-[var(--saffron)] text-white shadow-xs"
                              : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-2)]"
                          }`}
                        >
                          {count}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Error Box */}
              {jobError && (
                <div className="p-3 rounded-xl bg-[var(--danger)]/10 border border-[var(--danger)]/30 text-[var(--danger)] text-xs flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>{jobError}</span>
                </div>
              )}

              {/* Live Job Progress Bar */}
              {activeJob && isSubmitting && (
                <div className="p-3.5 rounded-2xl bg-[var(--bg-2)] border border-[var(--border)] space-y-2.5">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="flex items-center gap-1.5 text-[var(--ink)] font-semibold">
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--saffron)]" />
                      Worker Synthesizing ({activeJob.job_type})...
                    </span>
                    <span className="text-[var(--muted)]">
                      {activeJob.completed_items} / {activeJob.total_items}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-[var(--bg-3)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[var(--saffron)] transition-all duration-300"
                      style={{
                        width: `${
                          activeJob.total_items > 0
                            ? (activeJob.completed_items / activeJob.total_items) * 100
                            : 15
                        }%`,
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Prominent Saffron CTA Button */}
              <AnimatedButton
                type="button"
                variant="saffron"
                size="full"
                onClick={handleGenerate}
                disabled={!selectedParent || isSubmitting}
                loading={isSubmitting}
                leftIcon={!isSubmitting ? <Play className="w-4 h-4 fill-white" /> : undefined}
                className="py-3.5 tracking-wider uppercase shadow-[var(--shadow-s)] hover:shadow-[var(--shadow-m)]"
              >
                {isSubmitting ? "Synthesizing Variation..." : "GENERATE VARIATIONS →"}
              </AnimatedButton>
            </div>
          </div>
        </div>
      </div>
    </div>

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={(uploaded) => {
          loadParentImages(uploaded.image_id);
        }}
      />
    </AppShell>
  );
}

export default function AugmentationStudioPage() {
  return (
    <Suspense
      fallback={
        <AppShell pageTitle="Augmentation Studio">
          <div className="py-20 text-center font-mono text-xs text-[var(--muted)]">
            Loading guided studio workspace...
          </div>
        </AppShell>
      }
    >
      <AugmentationStudioContent />
    </Suspense>
  );
}
