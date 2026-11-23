import React from "react";

export interface ChipProps {
  label: string;
  active?: boolean;
  icon?: React.ReactNode;
  onClick?: () => void;
  onRemove?: () => void;
  className?: string;
  count?: number | string;
}

export function Chip({
  label,
  active = false,
  icon,
  onClick,
  onRemove,
  className = "",
  count,
}: ChipProps) {
  const activeStyles = active
    ? "bg-surface-base border-saffron text-white shadow-[0_0_12px_rgba(255,90,0,0.18)]"
    : "bg-surface-low border-border-subtle text-text-muted hover:border-border-subtle-hover hover:text-text-body";

  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all duration-150 cursor-pointer ${activeStyles} ${className}`}
    >
      {icon && <span className="shrink-0 text-saffron">{icon}</span>}
      <span>{label}</span>
      {count !== undefined && (
        <span
          className={`ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
            active ? "bg-saffron/20 text-saffron" : "bg-surface-highest text-text-muted"
          }`}
        >
          {count}
        </span>
      )}
      {onRemove && (
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="ml-1 hover:text-white text-text-muted cursor-pointer"
        >
          ✕
        </span>
      )}
    </button>
  );
}
