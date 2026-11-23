"use client";

import React, { useState, useRef, useEffect, useCallback, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { Upload, X, CheckCircle2, AlertCircle, Image as ImageIcon } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { ImageMetadata } from "@/lib/types";
import { AnimatedButton } from "./AnimatedButton";
import { StaggerButton } from "./StaggerButton";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (uploadedImage: ImageMetadata) => void;
}

export function UploadModal({ isOpen, onClose, onSuccess }: UploadModalProps) {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successImage, setSuccessImage] = useState<ImageMetadata | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleReset = useCallback(() => {
    setFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setError(null);
    setSuccessImage(null);
  }, []);

  const handleClose = useCallback(() => {
    handleReset();
    onClose();
  }, [handleReset, onClose]);

  // Body scroll lock
  useEffect(() => {
    if (!isOpen) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [isOpen]);

  // Escape key listener
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isUploading) {
        handleClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, isUploading, handleClose]);

  if (!isOpen || !mounted) return null;

  const handleFileChange = (selected: File | null) => {
    if (!selected) return;

    // Validate size (10 MB max)
    if (selected.size > 10 * 1024 * 1024) {
      setError("File exceeds maximum allowed size of 10 MB");
      return;
    }

    // Validate format
    const validFormats = ["image/jpeg", "image/png", "image/webp"];
    if (!validFormats.includes(selected.type)) {
      setError("Only JPEG, PNG, and WEBP formats are supported");
      return;
    }

    setError(null);
    setFile(selected);
    const objectUrl = URL.createObjectURL(selected);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);

    try {
      const res = await api.uploadImage(file);
      setSuccessImage(res);
      if (onSuccess) {
        onSuccess(res);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Failed to upload image. Please ensure backend is reachable.");
      }
    } finally {
      setIsUploading(false);
    }
  };

  const modalContent = (
    <div className="fixed inset-0 z-[90] flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
      <div className="w-full max-w-lg bg-[var(--bg)] border border-[var(--border)] rounded-2xl shadow-2xl overflow-hidden flex flex-col relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-2)]/60">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[var(--ink)] text-[var(--saffron)] flex items-center justify-center">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-display text-xl uppercase tracking-wider text-[var(--ink)]">
                Upload Inspection Image
              </h3>
              <p className="text-xs text-[var(--muted)]">
                Add an original pipeline photo to the master asset repository
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="p-1 rounded-lg text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-3)] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {error && (
            <div className="p-3 rounded-xl bg-[var(--danger)]/10 border border-[var(--danger)]/30 flex items-start gap-2.5 text-xs text-[var(--danger)]">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {successImage ? (
            <div className="p-6 rounded-xl bg-[var(--green)]/10 border border-[var(--green)]/30 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-[var(--green)]/20 text-[var(--green)] flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h4 className="font-display text-2xl uppercase tracking-wider text-[var(--ink)]">
                  Image Uploaded Successfully
                </h4>
                <p className="text-xs font-mono text-[var(--muted)] truncate max-w-sm mx-auto">
                  ID: {successImage.image_id}
                </p>
                <p className="text-xs text-[var(--muted)]">
                  {successImage.width}×{successImage.height} px · {(successImage.size_bytes / 1024).toFixed(1)} KB · {successImage.image_format}
                </p>
              </div>

              <div className="flex justify-center gap-3 pt-2">
                <StaggerButton
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                >
                  Upload Another
                </StaggerButton>
                <AnimatedButton
                  type="button"
                  variant="ink"
                  size="sm"
                  onClick={handleClose}
                >
                  Done
                </AnimatedButton>
              </div>
            </div>
          ) : (
            <>
              {/* Dropzone */}
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3 min-h-[180px] ${
                  isDragging
                    ? "border-[var(--saffron)] bg-[var(--saffron)]/5"
                    : previewUrl
                    ? "border-[var(--border-2)] bg-[var(--bg-2)]"
                    : "border-[var(--border)] hover:border-[var(--border-2)] bg-[var(--bg-2)]/40 hover:bg-[var(--bg-2)]"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      handleFileChange(e.target.files[0]);
                    }
                  }}
                />

                {previewUrl ? (
                  <div className="relative group">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Upload preview"
                      className="max-h-40 rounded-lg object-contain shadow-sm"
                    />
                    <div className="mt-2 text-xs font-mono text-[var(--muted)]">
                      {file?.name} ({(file?.size ? file.size / 1024 : 0).toFixed(1)} KB)
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-xl bg-[var(--bg)] border border-[var(--border)] flex items-center justify-center text-[var(--muted)] group-hover:text-[var(--saffron)] transition-colors">
                      <ImageIcon className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-[var(--ink)]">
                        Click or drag & drop pipeline inspection image
                      </p>
                      <p className="text-xs text-[var(--muted)] mt-1">
                        JPEG, PNG, or WEBP up to 10 MB · Validated with Pillow
                      </p>
                    </div>
                  </>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center justify-end gap-3 pt-2">
                <StaggerButton
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={handleClose}
                  disabled={isUploading}
                >
                  Cancel
                </StaggerButton>
                <AnimatedButton
                  type="button"
                  variant="saffron"
                  size="sm"
                  onClick={handleUpload}
                  disabled={!file || isUploading}
                  loading={isUploading}
                  leftIcon={!isUploading ? <Upload className="w-3.5 h-3.5" /> : undefined}
                >
                  {isUploading ? "Uploading..." : "Upload & Register"}
                </AnimatedButton>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
}
