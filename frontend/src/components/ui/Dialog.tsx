"use client";

import React, { useEffect } from "react";
import { X } from "lucide-react";

export interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  maxWidth?: "sm" | "md" | "lg" | "xl";
}

export function Dialog({
  isOpen,
  onClose,
  title,
  description,
  children,
  maxWidth = "md",
}: DialogProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    if (isOpen) {
      document.body.style.overflow = "hidden";
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => {
      document.body.style.overflow = "unset";
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const maxWidthStyles = {
    sm: "max-w-sm",
    md: "max-w-md",
    lg: "max-w-lg",
    xl: "max-w-xl",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/75 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal Dialog Box */}
      <div
        role="dialog"
        aria-modal="true"
        className={`relative z-10 w-full ${maxWidthStyles[maxWidth]} bg-surface-low border border-border-subtle rounded-2xl shadow-2xl p-6 overflow-hidden animate-page-reveal`}
      >
        <div className="flex items-start justify-between gap-4 pb-4 border-b border-border-subtle/50">
          <div>
            {title && (
              <h2 className="font-heading font-semibold text-lg text-text-primary tracking-tight">
                {title}
              </h2>
            )}
            {description && (
              <p className="text-text-muted text-xs mt-1">{description}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-text-muted hover:text-white hover:bg-surface-base transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="pt-4">{children}</div>
      </div>
    </div>
  );
}
