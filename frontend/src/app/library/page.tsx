"use client";

import React, { Suspense, useState, useEffect, useCallback, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Image as ImageIcon,
  Search,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Download,
  Trash2,
  X,
  Layers,
  Copy,
  Check,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Upload,
} from "lucide-react";
import { AppShell } from "@/components/layout";
import { UploadModal } from "@/components/ui/UploadModal";
import { LayeredButton, StaggerButton, AnimatedButton } from "@/components/ui";
import { api, getImageFileUrl, ApiError } from "@/lib/api";
import { ImageMetadata, ReviewStatus } from "@/lib/types";

interface ImageThumbnailProps {
  imageId: string;
  filename: string;
  sourceType: string;
  globalIndex: number;
  reviewStatus?: string | null;
  onClick: () => void;
}

function ImageThumbnail({
  imageId,
  filename,
  sourceType,
  globalIndex,
  reviewStatus,
  onClick,
}: ImageThumbnailProps) {
  const [hasError, setHasError] = useState(false);

  return (
    <div
      onClick={onClick}
      className="aspect-[4/3] w-full rounded-xl overflow-hidden bg-black/40 relative flex items-center justify-center cursor-pointer group"
    >
      {hasError ? (
        <div className="flex flex-col items-center justify-center p-4 text-center text-[var(--muted)] select-none">
          <ImageIcon className="w-8 h-8 text-[var(--muted)]/40 mb-1" />
          <span className="text-[10px] font-mono uppercase tracking-wider text-white/50">Preview Unavailable</span>
        </div>
      ) : (
        /* eslint-disable-next-line @next/next/no-img-element */
        <img
          src={getImageFileUrl(imageId)}
          alt={filename}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          loading="lazy"
          onError={() => setHasError(true)}
        />
      )}

      {/* Source Pill */}
      <div className="absolute top-2.5 left-2.5">
        <span
          className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase font-semibold ${
            sourceType === "generative"
              ? "bg-[var(--amber)] text-black"
              : sourceType === "traditional"
              ? "bg-[var(--saffron)] text-white"
              : "bg-[var(--ink)] text-[var(--bg)]"
          }`}
        >
          {sourceType === "user_upload" ? "Original" : sourceType}
        </span>
      </div>

      {/* Top-Right Index Badge */}
      <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5">
        <span className="px-2 py-0.5 rounded bg-black/60 backdrop-blur-md text-white font-mono text-[10px] font-bold">
          #{globalIndex}
        </span>
        {reviewStatus === "approved" && (
          <CheckCircle2 className="w-4 h-4 text-[var(--green-live)] fill-black/60" />
        )}
      </div>
    </div>
  );
}

function ImageLibraryContent() {
  const searchParams = useSearchParams();
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  // Filters from URL or state
  const [sourceType, setSourceType] = useState<string>(searchParams.get("source_type") || "all");
  const [qualityStatus, setQualityStatus] = useState<string>(searchParams.get("quality_status") || "all");
  const [reviewStatus, setReviewStatus] = useState<string>(searchParams.get("review_status") || "all");
  const [searchQuery, setSearchQuery] = useState<string>(searchParams.get("search") || "");
  const [parentIdFilter, setParentIdFilter] = useState<string | null>(searchParams.get("parent_image_id") || null);

  // Pagination state
  const [page, setPage] = useState<number>(1);
  const limit = 18; // Clean 3-column page multiples
  const [totalCount, setTotalCount] = useState<number>(0);
  const [images, setImages] = useState<ImageMetadata[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Inspector Drawer State
  const [selectedImage, setSelectedImage] = useState<ImageMetadata | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  // Review Form state in drawer
  const [drawerReviewStatus, setDrawerReviewStatus] = useState<ReviewStatus>("pending");
  const [drawerReviewNotes, setDrawerReviewNotes] = useState<string>("");
  const [drawerReviewer, setDrawerReviewer] = useState<string>("Lead Inspector");
  const [isSavingReview, setIsSavingReview] = useState<boolean>(false);
  const [reviewMessage, setReviewMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Upload modal state
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);

  // Open Inspector Drawer
  const openInspector = useCallback((img: ImageMetadata) => {
    setSelectedImage(img);
    setDrawerReviewStatus(img.review_status || "pending");
    setDrawerReviewNotes(img.review_notes || "");
    setDrawerReviewer(img.reviewer || "Lead Inspector");
    setReviewMessage(null);
    setIsDrawerOpen(true);
  }, []);

  const closeInspector = () => {
    setIsDrawerOpen(false);
    setSelectedImage(null);
  };

  // Body scroll lock and Esc listener for Inspect & Review Modal
  useEffect(() => {
    if (!isDrawerOpen || !selectedImage) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeInspector();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isDrawerOpen, selectedImage]);

  // Fetch images list from API
  const fetchImages = useCallback(async () => {
    setIsLoading(true);
    try {
      const offset = (page - 1) * limit;
      const res = await api.listImages({
        source_type: sourceType === "all" ? undefined : sourceType,
        quality_status: qualityStatus === "all" ? undefined : qualityStatus,
        review_status: reviewStatus === "all" ? undefined : reviewStatus,
        search: searchQuery.trim() !== "" ? searchQuery.trim() : undefined,
        parent_image_id: parentIdFilter || undefined,
        limit,
        offset,
      });

      setImages(res.items);
      setTotalCount(res.total);
    } catch {
      setImages([]);
      setTotalCount(0);
    } finally {
      setIsLoading(false);
    }
  }, [page, limit, sourceType, qualityStatus, reviewStatus, searchQuery, parentIdFilter]);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchImages();
    }, 0);
    return () => clearTimeout(timer);
  }, [fetchImages]);

  // Handle URL selected image param on initial load
  useEffect(() => {
    const selectedParam = searchParams.get("selected");
    if (selectedParam && images.length > 0) {
      const found = images.find((img) => img.image_id === selectedParam);
      if (found) {
        const timer = setTimeout(() => {
          openInspector(found);
        }, 0);
        return () => clearTimeout(timer);
      }
    }
  }, [searchParams, images, openInspector]);

  // Copy helper
  const copyToClipboard = (text: string, fieldName: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(fieldName);
    setTimeout(() => setCopiedField(null), 1800);
  };

  // Save review in Drawer
  const handleSaveReview = async () => {
    if (!selectedImage) return;
    setIsSavingReview(true);
    setReviewMessage(null);

    try {
      const updated = await api.updateReview(selectedImage.image_id, {
        review_status: drawerReviewStatus,
        review_notes: drawerReviewNotes.trim() ? drawerReviewNotes.trim() : undefined,
        reviewer: drawerReviewer.trim() ? drawerReviewer.trim() : undefined,
      });

      setSelectedImage(updated);
      setReviewMessage({ type: "success", text: "Decision persisted to manifest." });

      // Update in local library list
      setImages((prev) =>
        prev.map((img) => (img.image_id === updated.image_id ? updated : img))
      );
    } catch (err) {
      setReviewMessage({
        type: "error",
        text: err instanceof ApiError ? err.message : "Failed to persist review decision.",
      });
    } finally {
      setIsSavingReview(false);
    }
  };

  // Delete image
  const handleDeleteImage = async () => {
    if (!selectedImage) return;
    if (
      !window.confirm(
        `Are you sure you want to permanently delete image ${selectedImage.image_id}? If this is a parent master asset, its variations will also be cascade deleted.`
      )
    ) {
      return;
    }

    try {
      await api.deleteImage(selectedImage.image_id);
      closeInspector();
      fetchImages();
    } catch (err) {
      alert(
        err instanceof ApiError ? err.message : "Failed to delete image from repository"
      );
    }
  };

  const totalPages = Math.ceil(totalCount / limit) || 1;

  return (
    <AppShell pageTitle="Image Library">
      <div className="space-y-8 max-w-7xl mx-auto pb-16">
        {/* ================================================== */}
        {/* EDITORIAL HEADER + QUICK UPLOAD */}
        {/* ================================================== */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 pb-6 border-b border-[var(--border)]">
          <div className="space-y-1">
            <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--saffron)] font-semibold">
              INSPECTION ARCHIVE
            </span>
            <div className="flex items-baseline gap-3">
              <h1 className="font-display text-4xl sm:text-5xl uppercase tracking-wider text-[var(--ink)]">
                IMAGE LIBRARY
              </h1>
              <span className="px-3 py-1 rounded-full bg-[var(--bg-2)] border border-[var(--border)] font-mono text-xs font-bold text-[var(--saffron)]">
                + {totalCount} IMAGES FOUND
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={fetchImages}
              className="p-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg)] hover:bg-[var(--bg-2)] text-[var(--muted)] hover:text-[var(--ink)] transition-colors cursor-pointer"
              title="Refresh repository listing"
            >
              <RefreshCw className="w-4 h-4" />
            </button>

            <LayeredButton
              type="button"
              variant="ink"
              size="sm"
              onClick={() => setIsUploadOpen(true)}
              leftIcon={<Upload className="w-3.5 h-3.5 text-[var(--saffron)]" />}
              className="tracking-wider uppercase"
            >
              UPLOAD IMAGE
            </LayeredButton>
          </div>
        </div>

        {/* ================================================== */}
        {/* SLEEK SINGLE FILTER TOOLBAR */}
        {/* ================================================== */}
        <div className="p-4 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] flex flex-wrap items-center gap-3">
          {/* Search Input */}
          <div className="relative min-w-[220px] flex-1">
            <Search className="w-4 h-4 text-[var(--muted)] absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              placeholder="Search by filename or condition..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              className="w-full bg-[var(--bg-2)] border border-[var(--border)] text-[var(--ink)] text-xs rounded-xl pl-9 pr-3 py-2 font-mono focus:outline-none focus:border-[var(--saffron)]"
            />
          </div>

          {/* Source Segmented Filter */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-[var(--bg-2)] border border-[var(--border)] text-xs font-mono font-semibold">
            {[
              { id: "all", label: "All" },
              { id: "user_upload", label: "Originals" },
              { id: "traditional", label: "Traditional" },
              { id: "generative", label: "Generative" },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => {
                  setSourceType(tab.id);
                  setPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
                  sourceType === tab.id
                    ? "bg-[var(--bg)] text-[var(--saffron)] shadow-xs font-bold"
                    : "text-[var(--muted)] hover:text-[var(--ink)]"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Quality Select */}
          <select
            value={qualityStatus}
            onChange={(e) => {
              setQualityStatus(e.target.value);
              setPage(1);
            }}
            className="bg-[var(--bg-2)] border border-[var(--border)] text-[var(--ink)] text-xs rounded-xl px-3 py-2 font-mono focus:outline-none focus:border-[var(--saffron)] cursor-pointer"
          >
            <option value="all">Quality: All</option>
            <option value="pass">Quality: Pass</option>
            <option value="flag_for_review">Quality: Flagged</option>
            <option value="fail">Quality: Fail</option>
          </select>

          {/* Review Select */}
          <select
            value={reviewStatus}
            onChange={(e) => {
              setReviewStatus(e.target.value);
              setPage(1);
            }}
            className="bg-[var(--bg-2)] border border-[var(--border)] text-[var(--ink)] text-xs rounded-xl px-3 py-2 font-mono focus:outline-none focus:border-[var(--saffron)] cursor-pointer"
          >
            <option value="all">Review: All</option>
            <option value="pending">Review: Pending</option>
            <option value="approved">Review: Approved</option>
            <option value="rejected">Review: Rejected</option>
            <option value="needs_review">Review: Needs Review</option>
          </select>

          {/* Active Parent ID filter pill */}
          {parentIdFilter && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[var(--saffron)]/10 border border-[var(--saffron)] text-xs font-mono text-[var(--saffron)] font-semibold">
              <span>Parent: {parentIdFilter.slice(0, 8)}...</span>
              <button
                type="button"
                onClick={() => setParentIdFilter(null)}
                className="hover:text-[var(--ink)] ml-1 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>

        {/* ================================================== */}
        {/* CLEAN 3-COLUMN RESPONSIVE CARD GRID */}
        {/* ================================================== */}
        <div className="space-y-6">
          <div className="flex items-center justify-between text-xs font-mono text-[var(--muted)] px-1">
            <span>
              Showing {images.length > 0 ? (page - 1) * limit + 1 : 0} –{" "}
              {Math.min(page * limit, totalCount)} of {totalCount} images
            </span>
            <span>Page {page} of {totalPages}</span>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="aspect-[4/3] rounded-2xl bg-[var(--bg-2)] border border-[var(--border)] animate-pulse"
                />
              ))}
            </div>
          ) : images.length === 0 ? (
            <div className="py-16 rounded-3xl bg-[var(--bg)] border border-[var(--border)] text-center space-y-4">
              <div className="w-12 h-12 rounded-xl bg-[var(--bg-2)] border border-[var(--border)] flex items-center justify-center mx-auto text-[var(--muted)]">
                <ImageIcon className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="font-display text-2xl uppercase tracking-wider text-[var(--ink)]">
                  No Matching Pipeline Images
                </h3>
                <p className="text-xs text-[var(--muted)] max-w-sm mx-auto">
                  Adjust active filter criteria or upload an original inspection image to begin condition synthesis.
                </p>
              </div>
              <StaggerButton
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  setSourceType("all");
                  setQualityStatus("all");
                  setReviewStatus("all");
                  setSearchQuery("");
                  setParentIdFilter(null);
                }}
              >
                Reset Filter Parameters
              </StaggerButton>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {images.map((img, idx) => {
                const globalIndex = (page - 1) * limit + idx + 1;
                const ssimScore = img.quality_metrics?.ssim_rgb ?? img.quality_metrics?.ssim;

                return (
                  <div
                    key={img.image_id}
                    className="relative p-5 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover flex flex-col justify-between gap-4 group overflow-hidden"
                  >
                    {/* Large Subtle Watermark Index */}
                    <div className="absolute top-2 right-4 font-display text-4xl sm:text-5xl text-[var(--muted)]/15 select-none pointer-events-none tracking-tight">
                      #{globalIndex}
                    </div>

                    {/* Unobstructed Image Thumbnail with Index & Source Pill (with Error Fallback) */}
                    <ImageThumbnail
                      imageId={img.image_id}
                      filename={img.original_filename}
                      sourceType={img.source_type}
                      globalIndex={globalIndex}
                      reviewStatus={img.review_status}
                      onClick={() => openInspector(img)}
                    />

                    {/* Metadata & Quality Bar */}
                    <div className="space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <div className="truncate">
                          <h4 className="font-sans font-bold text-sm text-[var(--ink)] truncate">
                            {img.condition || img.original_filename}
                          </h4>
                          <p className="font-mono text-[11px] text-[var(--muted)]">
                            {img.width}×{img.height} px · {(img.size_bytes / 1024).toFixed(1)} KB
                          </p>
                        </div>

                        {img.quality_status && (
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase shrink-0 ${
                              img.quality_status === "pass"
                                ? "bg-[var(--green)]/15 text-[var(--green)] border border-[var(--green)]/30"
                                : "bg-[var(--gold)]/15 text-[var(--gold)] border border-[var(--gold)]/30"
                            }`}
                          >
                            {img.quality_status}
                          </span>
                        )}
                      </div>

                      {/* Thin Quality Progress Bar */}
                      <div className="space-y-1 pt-1">
                        <div className="flex justify-between text-[10px] font-mono text-[var(--muted)]">
                          <span>
                            {img.source_type === "user_upload"
                              ? "Original Source"
                              : ssimScore != null
                              ? `SSIM: ${ssimScore.toFixed(4)}`
                              : "Baseline"}
                          </span>
                          <span className="font-bold text-[var(--green)]">
                            {img.source_type === "user_upload"
                              ? "MASTER"
                              : ssimScore != null
                              ? `${(ssimScore * 100).toFixed(1)}%`
                              : "N/A"}
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-[var(--bg-3)] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[var(--green)] transition-all duration-300"
                            style={{
                              width: `${
                                img.source_type === "user_upload"
                                  ? 100
                                  : ssimScore != null
                                  ? Math.min(100, Math.max(10, ssimScore * 100))
                                  : 0
                              }%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Two Crisp Rectangular Hairline Buttons */}
                    <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[var(--border)]">
                      <Link
                        href={`/studio?imageId=${img.parent_image_id || img.image_id}`}
                        className="py-2 px-3 rounded-lg border border-[var(--border)] hover:border-[var(--saffron)] bg-[var(--bg)] hover:bg-[var(--bg-2)] text-[var(--ink)] text-xs font-semibold text-center transition-colors tracking-wide"
                      >
                        OPEN IN STUDIO
                      </Link>

                      <LayeredButton
                        type="button"
                        variant="ink"
                        size="sm"
                        onClick={() => openInspector(img)}
                        className="w-full text-center tracking-wide"
                      >
                        INSPECT & REVIEW
                      </LayeredButton>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-6 border-t border-[var(--border)]">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-4 py-2 rounded-xl border border-[var(--border)] text-xs font-semibold text-[var(--ink)] hover:bg-[var(--bg-2)] transition-colors disabled:opacity-40 cursor-pointer flex items-center gap-1.5"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>

              <span className="text-xs font-mono text-[var(--muted)]">
                Page {page} of {totalPages}
              </span>

              <button
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-4 py-2 rounded-xl border border-[var(--border)] text-xs font-semibold text-[var(--ink)] hover:bg-[var(--bg-2)] transition-colors disabled:opacity-40 cursor-pointer flex items-center gap-1.5"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ================================================== */}
      {/* CENTERED BLURRED-BACKDROP INSPECTION & REVIEW MODAL */}
      {/* ================================================== */}
      {mounted && isDrawerOpen && selectedImage && createPortal(
        <div className="fixed inset-0 z-[90] overflow-y-auto flex items-center justify-center p-4 sm:p-6">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-md animate-page-reveal"
            onClick={closeInspector}
          />

          {/* Centered Modal Panel */}
          <div className="relative w-full max-w-2xl bg-[var(--bg)] rounded-3xl border border-[var(--border)] shadow-2xl flex flex-col max-h-[90vh] z-10 animate-page-reveal overflow-hidden my-auto">
            {/* Header */}
            <div className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 bg-[var(--bg)]/95 backdrop-blur-md border-b border-[var(--border)]">
              <div className="flex items-center gap-2.5 overflow-hidden">
                <div className="w-7 h-7 rounded-lg bg-[var(--ink)] text-[var(--saffron)] flex items-center justify-center shrink-0">
                  <ImageIcon className="w-4 h-4" />
                </div>
                <div className="truncate">
                  <h3 className="font-display text-xl uppercase tracking-wider text-[var(--ink)] truncate">
                    {selectedImage.stored_filename}
                  </h3>
                  <span className="text-[10px] font-mono text-[var(--muted)] uppercase">
                    INSPECTION METADATA & HUMAN REVIEW GATE
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={closeInspector}
                className="p-1.5 rounded-lg text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-2)] transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Body */}
            <div className="p-6 space-y-6 flex-1 overflow-y-auto">
              {/* Unobstructed Image Preview (Dark Vault Framing for Color Neutrality) */}
              <div className="relative aspect-video w-full rounded-2xl bg-[var(--vault-bg)] overflow-hidden border border-[var(--vault-border)] flex items-center justify-center bg-inspection-grid">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={getImageFileUrl(selectedImage.image_id)}
                  alt={selectedImage.original_filename}
                  className="max-h-full max-w-full object-contain"
                />
              </div>

              {/* Action Toolbar */}
              <div className="flex items-center justify-between gap-3">
                <Link
                  href={`/studio?imageId=${selectedImage.parent_image_id || selectedImage.image_id}`}
                  className="flex-1 py-2 px-4 rounded-xl bg-[var(--saffron)] text-white hover:bg-[var(--amber)] text-xs font-bold text-center transition-colors flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <Layers className="w-4 h-4 text-white" />
                  <span>OPEN IN STUDIO →</span>
                </Link>

                <a
                  href={getImageFileUrl(selectedImage.image_id)}
                  download={selectedImage.stored_filename}
                  target="_blank"
                  rel="noreferrer"
                  className="py-2 px-4 rounded-xl border border-[var(--border)] text-[var(--ink)] hover:bg-[var(--bg-2)] text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  <Download className="w-4 h-4 text-[var(--saffron)]" />
                  <span>Download</span>
                </a>

                <button
                  type="button"
                  onClick={handleDeleteImage}
                  className="py-2 px-3 rounded-xl border border-[var(--danger)]/30 text-[var(--danger)] hover:bg-[var(--danger)]/10 text-xs font-semibold transition-colors flex items-center gap-1 cursor-pointer"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              {/* Technical Specifications Table */}
              <div className="space-y-3">
                <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--muted)] font-semibold block">
                  TECHNICAL SPECIFICATIONS
                </span>

                <div className="divide-y divide-[var(--border)] rounded-2xl bg-[var(--bg-2)] border border-[var(--border)] overflow-hidden text-xs">
                  {/* Image ID */}
                  <div className="p-3 flex items-center justify-between gap-3">
                    <span className="text-[var(--muted)]">Image ID</span>
                    <div className="flex items-center gap-1.5 font-mono">
                      <span className="text-[var(--ink)] text-[11px] truncate max-w-[200px]">
                        {selectedImage.image_id}
                      </span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(selectedImage.image_id, "image_id")}
                        className="text-[var(--muted)] hover:text-[var(--saffron)] p-1 cursor-pointer"
                      >
                        {copiedField === "image_id" ? <Check className="w-3.5 h-3.5 text-[var(--green)]" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                  {/* Parent ID */}
                  {selectedImage.parent_image_id && (
                    <div className="p-3 flex items-center justify-between gap-3">
                      <span className="text-[var(--muted)]">Parent Asset</span>
                      <button
                        type="button"
                        onClick={() => {
                          setParentIdFilter(selectedImage.parent_image_id);
                          closeInspector();
                        }}
                        className="text-[var(--saffron)] hover:underline font-mono text-[11px] truncate max-w-[200px] cursor-pointer"
                      >
                        {selectedImage.parent_image_id}
                      </button>
                    </div>
                  )}

                  {/* Source & Condition */}
                  <div className="p-3 flex items-center justify-between">
                    <span className="text-[var(--muted)]">Source / Condition</span>
                    <span className="font-mono text-[var(--ink)] font-semibold">
                      {selectedImage.source_type} ({selectedImage.condition || selectedImage.generation_method || "Original"})
                    </span>
                  </div>

                  {/* Dimensions & Size */}
                  <div className="p-3 flex items-center justify-between">
                    <span className="text-[var(--muted)]">Dimensions & Format</span>
                    <span className="font-mono text-[var(--ink)]">
                      {selectedImage.width}×{selectedImage.height} px · {(selectedImage.size_bytes / 1024).toFixed(1)} KB ({selectedImage.image_format})
                    </span>
                  </div>

                  {/* SHA256 */}
                  <div className="p-3 flex items-center justify-between gap-3">
                    <span className="text-[var(--muted)]">SHA-256</span>
                    <div className="flex items-center gap-1.5 font-mono">
                      <span className="text-[var(--ink)] text-[11px] truncate max-w-[200px]">
                        {selectedImage.sha256}
                      </span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(selectedImage.sha256, "sha256")}
                        className="text-[var(--muted)] hover:text-[var(--saffron)] p-1 cursor-pointer"
                      >
                        {copiedField === "sha256" ? <Check className="w-3.5 h-3.5 text-[var(--green)]" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Phase 4 Quality Screening Breakdown */}
              {selectedImage.quality_metrics && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--muted)] font-semibold">
                      PHASE 4 QUALITY SCREENING
                    </span>
                    {selectedImage.quality_status && (
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                          selectedImage.quality_status === "pass"
                            ? "bg-[var(--green)]/15 text-[var(--green)]"
                            : "bg-[var(--gold)]/15 text-[var(--gold)]"
                        }`}
                      >
                        {selectedImage.quality_status}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                      <span className="text-[10px] text-[var(--muted)] uppercase block">SSIM (RGB)</span>
                      <strong className="text-sm font-bold text-[var(--ink)]">
                        {selectedImage.quality_metrics.ssim_rgb?.toFixed(4) ?? selectedImage.quality_metrics.ssim?.toFixed(4) ?? "N/A"}
                      </strong>
                    </div>

                    <div className="p-3 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                      <span className="text-[10px] text-[var(--muted)] uppercase block">LPIPS Distance</span>
                      <strong className="text-sm font-bold text-[var(--ink)]">
                        {selectedImage.quality_metrics.lpips?.toFixed(4) ?? "N/A"}
                      </strong>
                    </div>

                    <div className="p-3 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                      <span className="text-[10px] text-[var(--muted)] uppercase block">Edge IoU</span>
                      <strong className="text-sm font-bold text-[var(--ink)]">
                        {selectedImage.quality_metrics.edge_overlap_iou?.toFixed(4) ?? "N/A"}
                      </strong>
                    </div>

                    <div className="p-3 rounded-xl bg-[var(--bg-2)] border border-[var(--border)]">
                      <span className="text-[10px] text-[var(--muted)] uppercase block">Mean Pixel Diff</span>
                      <strong className="text-sm font-bold text-[var(--ink)]">
                        {selectedImage.quality_metrics.mean_pixel_abs_diff?.toFixed(2) ?? "N/A"}
                      </strong>
                    </div>
                  </div>

                  {/* Approximate Diagnostic Notice */}
                  {selectedImage.quality_metrics.corrosion_diagnostic && (
                    <div className="p-3 rounded-xl bg-[var(--bg-2)] border border-[var(--border)] text-[11px] text-[var(--muted)] space-y-1">
                      <strong className="text-[var(--ink)] block font-semibold">Corrosion Diagnostic Notice:</strong>
                      <p>{selectedImage.quality_metrics.corrosion_diagnostic.scientific_caveat || "Heuristic color candidate analysis only; not ground-truth metallurgical segmentation."}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Working Human Review Gate Form */}
              <div className="p-5 rounded-2xl bg-[var(--bg-2)] border border-[var(--border)] space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono tracking-widest uppercase text-[var(--saffron)] font-bold">
                    HUMAN REVIEW GATE
                  </span>
                  <span className="text-[11px] font-mono text-[var(--muted)]">
                    Status: {selectedImage.review_status || "Pending"}
                  </span>
                </div>

                {reviewMessage && (
                  <div
                    className={`p-2.5 rounded-xl text-xs flex items-center gap-2 ${
                      reviewMessage.type === "success"
                        ? "bg-[var(--green)]/15 border border-[var(--green)]/30 text-[var(--green)]"
                        : "bg-[var(--danger)]/15 border border-[var(--danger)]/30 text-[var(--danger)]"
                    }`}
                  >
                    {reviewMessage.type === "success" ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                    <span>{reviewMessage.text}</span>
                  </div>
                )}

                {/* Decision Buttons */}
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => setDrawerReviewStatus("approved")}
                    className={`p-2 rounded-xl text-xs font-semibold border transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                      drawerReviewStatus === "approved"
                        ? "bg-[var(--green)] text-white border-[var(--green)] shadow-xs"
                        : "border-[var(--border)] bg-[var(--bg)] text-[var(--green)] hover:bg-[var(--green)]/10"
                    }`}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Approve</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setDrawerReviewStatus("needs_review")}
                    className={`p-2 rounded-xl text-xs font-semibold border transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                      drawerReviewStatus === "needs_review"
                        ? "bg-[var(--gold)] text-black border-[var(--gold)] shadow-xs"
                        : "border-[var(--border)] bg-[var(--bg)] text-[var(--gold)] hover:bg-[var(--gold)]/10"
                    }`}
                  >
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Flag</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setDrawerReviewStatus("rejected")}
                    className={`p-2 rounded-xl text-xs font-semibold border transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                      drawerReviewStatus === "rejected"
                        ? "bg-[var(--danger)] text-white border-[var(--danger)] shadow-xs"
                        : "border-[var(--border)] bg-[var(--bg)] text-[var(--danger)] hover:bg-[var(--danger)]/10"
                    }`}
                  >
                    <XCircle className="w-3.5 h-3.5" />
                    <span>Reject</span>
                  </button>
                </div>

                {/* Reviewer Name */}
                <div className="space-y-1">
                  <label className="text-[11px] font-mono text-[var(--muted)] block">
                    Reviewer Identity
                  </label>
                  <input
                    type="text"
                    value={drawerReviewer}
                    onChange={(e) => setDrawerReviewer(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] text-xs rounded-xl px-3 py-1.5 font-mono focus:outline-none focus:border-[var(--saffron)]"
                  />
                </div>

                {/* Review Notes */}
                <div className="space-y-1">
                  <label className="text-[11px] font-mono text-[var(--muted)] block">
                    Observations / Defect Retention
                  </label>
                  <textarea
                    rows={2}
                    placeholder="Document structural preservation, pitting clarity, or reasons for rejection..."
                    value={drawerReviewNotes}
                    onChange={(e) => setDrawerReviewNotes(e.target.value)}
                    className="w-full bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] text-xs rounded-xl p-2.5 focus:outline-none focus:border-[var(--saffron)] resize-none"
                  />
                </div>

                {/* Save CTA */}
                <AnimatedButton
                  type="button"
                  variant="saffron"
                  size="full"
                  onClick={handleSaveReview}
                  disabled={isSavingReview}
                  loading={isSavingReview}
                  className="tracking-wider uppercase"
                >
                  {isSavingReview ? "Persisting Decision..." : "Save Review Decision"}
                </AnimatedButton>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={() => {
          fetchImages();
        }}
      />
    </AppShell>
  );
}

export default function ImageLibraryPage() {
  return (
    <Suspense
      fallback={
        <AppShell pageTitle="Image Library">
          <div className="py-20 text-center font-mono text-xs text-[var(--muted)]">
            Loading pipeline imagery repository...
          </div>
        </AppShell>
      }
    >
      <ImageLibraryContent />
    </Suspense>
  );
}
