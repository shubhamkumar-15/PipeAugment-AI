"use client";

import React, { useEffect } from "react";
import { X } from "lucide-react";

export interface DrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  position?: "right" | "left";
  width?: "sm" | "md" | "lg" | "xl";
}

export function Drawer({
  isOpen,
  onClose,
  title,
  description,
  children,
  position = "right",
  width = "md",
}: DrawerProps) {
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

  const widthStyles = {
    sm: "w-80",
    md: "w-96",
    lg: "w-[480px]",
    xl: "w-[600px]",
  };

  const positionStyles = {
    right: "right-0 top-0 bottom-0 animate-drawer-in-right border-l",
    left: "left-0 top-0 bottom-0 animate-drawer-in-left border-r",
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Drawer Panel */}
      <aside
        className={`fixed z-10 ${positionStyles[position]} ${widthStyles[width]} max-w-full bg-surface-low border-border-subtle shadow-2xl flex flex-col`}
      >
        <div className="p-5 flex items-center justify-between border-b border-border-subtle/50">
          <div>
            {title && (
              <h3 className="font-heading font-semibold text-text-primary text-base">
                {title}
              </h3>
            )}
            {description && (
              <p className="text-text-muted text-xs mt-0.5">{description}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-text-muted hover:text-white hover:bg-surface-base transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">{children}</div>
      </aside>
    </div>
  );
}
