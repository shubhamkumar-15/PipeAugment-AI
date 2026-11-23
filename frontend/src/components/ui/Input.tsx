import React, { forwardRef } from "react";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: string;
  leftIcon?: React.ReactNode;
  rightElement?: React.ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      helperText,
      error,
      leftIcon,
      rightElement,
      className = "",
      disabled,
      ...props
    },
    ref
  ) => {
    return (
      <div className="flex flex-col gap-1.5 w-full">
        {label && (
          <label className="text-xs font-medium text-text-muted uppercase tracking-wider">
            {label}
          </label>
        )}
        <div className="relative flex items-center w-full">
          {leftIcon && (
            <span className="absolute left-3 text-text-muted pointer-events-none flex items-center justify-center">
              {leftIcon}
            </span>
          )}
          <input
            ref={ref}
            disabled={disabled}
            className={`w-full rounded-lg bg-canvas border text-sm text-white placeholder:text-text-muted/60 transition-all duration-150 outline-none ${
              leftIcon ? "pl-9" : "pl-3.5"
            } ${rightElement ? "pr-12" : "pr-3.5"} py-2 ${
              error
                ? "border-rose focus:border-rose focus:ring-1 focus:ring-rose/40"
                : "border-border-subtle focus:border-saffron focus:ring-1 focus:ring-saffron/30 hover:border-border-subtle-hover"
            } disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
            {...props}
          />
          {rightElement && (
            <div className="absolute right-2.5 flex items-center">
              {rightElement}
            </div>
          )}
        </div>
        {error ? (
          <span className="text-xs text-rose">{error}</span>
        ) : helperText ? (
          <span className="text-xs text-text-muted/70">{helperText}</span>
        ) : null}
      </div>
    );
  }
);

Input.displayName = "Input";
