import React from "react";

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: React.ReactNode;
  icon?: React.ReactNode;
  count?: number | string;
}

export interface SegmentedControlProps<T extends string | number> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
  size?: "sm" | "md";
}

export function SegmentedControl<T extends string | number>({
  options,
  value,
  onChange,
  className = "",
  size = "md",
}: SegmentedControlProps<T>) {
  const sizeStyles = {
    sm: "p-0.5 text-xs",
    md: "p-1 text-sm",
  };

  const itemSizeStyles = {
    sm: "py-1 px-2.5",
    md: "py-1.5 px-3.5",
  };

  return (
    <div
      className={`inline-flex items-center rounded-lg bg-canvas border border-border-subtle ${sizeStyles[size]} ${className}`}
    >
      {options.map((option) => {
        const isSelected = option.value === value;
        return (
          <button
            key={String(option.value)}
            type="button"
            onClick={() => onChange(option.value)}
            className={`flex items-center justify-center gap-1.5 rounded-md font-medium transition-all duration-150 select-none cursor-pointer ${
              itemSizeStyles[size]
            } ${
              isSelected
                ? "bg-surface-base text-white border border-saffron/30 shadow-[0_2px_8px_rgba(0,0,0,0.4)]"
                : "text-text-muted hover:text-text-body hover:bg-white/[0.03]"
            }`}
          >
            {option.icon && <span className="shrink-0">{option.icon}</span>}
            <span>{option.label}</span>
            {option.count !== undefined && (
              <span
                className={`ml-1 px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  isSelected
                    ? "bg-saffron/20 text-saffron"
                    : "bg-surface-highest text-text-muted"
                }`}
              >
                {option.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
