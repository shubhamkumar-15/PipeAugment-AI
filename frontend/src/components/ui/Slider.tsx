import React from "react";

export interface SliderProps {
  label?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  helperText?: string;
  onChange: (value: number) => void;
  className?: string;
  disabled?: boolean;
}

export function Slider({
  label,
  value,
  min = 0,
  max = 100,
  step = 1,
  unit = "%",
  helperText,
  onChange,
  className = "",
  disabled = false,
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      {(label || value !== undefined) && (
        <div className="flex items-center justify-between text-xs">
          {label && (
            <label className="font-medium text-text-muted uppercase tracking-wider">
              {label}
            </label>
          )}
          <span className="px-2 py-0.5 rounded bg-saffron text-white font-mono font-semibold text-xs shadow-sm">
            {value}
            {unit}
          </span>
        </div>
      )}

      <div className="relative py-1.5 flex items-center">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-1.5 bg-surface-highest rounded-lg appearance-none cursor-pointer accent-saffron focus:outline-none focus-visible:ring-1 focus-visible:ring-saffron disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: `linear-gradient(to right, #FF5A00 0%, #FF5A00 ${percentage}%, #262A33 ${percentage}%, #262A33 100%)`,
          }}
        />
      </div>

      {helperText && (
        <p className="text-xs text-text-muted/80 flex items-center gap-1">
          <span>{helperText}</span>
        </p>
      )}
    </div>
  );
}
