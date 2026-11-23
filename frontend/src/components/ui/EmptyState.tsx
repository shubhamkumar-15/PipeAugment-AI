import React from "react";
import { Button } from "./Button";

export interface EmptyStateProps {
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
  hint?: string;
  children?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  title,
  description,
  actionLabel,
  onAction,
  icon,
  hint,
  children,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`relative overflow-hidden flex flex-col items-center justify-center p-12 text-center rounded-2xl bg-surface-low border border-border-subtle ${className}`}
    >
      {/* Saffron Glow Background */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-saffron/10 rounded-full blur-3xl pointer-events-none" />

      {/* Radar Target Sensor Graphic */}
      <div className="relative flex items-center justify-center w-28 h-28 mb-6 select-none">
        <svg
          className="absolute inset-0 w-full h-full text-saffron"
          fill="none"
          viewBox="0 0 120 120"
        >
          <circle
            cx="60"
            cy="60"
            r="54"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeDasharray="4 4"
            opacity="0.3"
          />
          <circle
            cx="60"
            cy="60"
            r="40"
            stroke="currentColor"
            strokeWidth="1"
            opacity="0.5"
          />
          <circle
            cx="60"
            cy="60"
            r="40"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="24 160"
            strokeLinecap="round"
            className="animate-spin"
            style={{ animationDuration: "12s" }}
          />
          <line
            x1="60"
            y1="6"
            x2="60"
            y2="18"
            stroke="currentColor"
            strokeWidth="1.5"
            opacity="0.7"
          />
          <line
            x1="60"
            y1="102"
            x2="60"
            y2="114"
            stroke="currentColor"
            strokeWidth="1.5"
            opacity="0.7"
          />
          <line
            x1="6"
            y1="60"
            x2="18"
            y2="60"
            stroke="currentColor"
            strokeWidth="1.5"
            opacity="0.7"
          />
          <line
            x1="102"
            y1="60"
            x2="114"
            y2="60"
            stroke="currentColor"
            strokeWidth="1.5"
            opacity="0.7"
          />
        </svg>

        <div className="relative w-14 h-14 rounded-xl bg-surface-base border border-border-accent flex items-center justify-center text-saffron shadow-lg shadow-saffron/20">
          {icon || (
            <span className="material-symbols-outlined text-[28px]">
              sensors
            </span>
          )}
        </div>
      </div>

      <h3 className="font-heading font-semibold text-text-primary text-xl tracking-tight mb-2">
        {title}
      </h3>
      <p className="text-text-muted text-sm max-w-md mb-6 leading-relaxed">
        {description}
      </p>

      {actionLabel && onAction && (
        <Button variant="primary" size="md" onClick={onAction}>
          {actionLabel}
        </Button>
      )}

      {hint && (
        <span className="text-xs text-text-muted/60 mt-3 font-mono">
          {hint}
        </span>
      )}

      {children && <div className="mt-6 w-full">{children}</div>}
    </div>
  );
}
