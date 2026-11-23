import React, { forwardRef } from "react";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "primary",
      size = "md",
      isLoading = false,
      leftIcon,
      rightIcon,
      className = "",
      disabled,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center font-medium font-body transition-all duration-200 select-none cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-saffron/50 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none active:scale-[0.98]";

    const sizeStyles = {
      sm: "px-2.5 py-1.5 text-xs rounded-md gap-1.5",
      md: "px-4 py-2 text-sm rounded-lg gap-2",
      lg: "px-6 py-3 text-base rounded-xl gap-2.5",
    };

    const variantStyles = {
      primary:
        "bg-gradient-to-r from-saffron to-amber text-white font-semibold shadow-[0_4px_20px_-2px_rgba(255,90,0,0.35)] hover:shadow-[0_6px_24px_rgba(255,90,0,0.45)] hover:brightness-105",
      secondary:
        "bg-surface-low border border-border-subtle text-text-body hover:bg-surface-base hover:border-border-accent hover:text-white shadow-sm",
      outline:
        "bg-transparent border border-border-subtle text-text-body hover:bg-surface-base hover:border-border-accent hover:text-white shadow-sm",
      ghost:
        "bg-transparent text-text-muted hover:text-white hover:bg-white/[0.05]",
      danger:
        "bg-rose/10 border border-rose/30 text-rose hover:bg-rose/20 hover:border-rose/50",
    };

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={`${baseStyles} ${sizeStyles[size]} ${variantStyles[variant]} ${className}`}
        {...props}
      >
        {isLoading ? (
          <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-1" />
        ) : (
          leftIcon && <span className="shrink-0">{leftIcon}</span>
        )}
        <span>{children}</span>
        {!isLoading && rightIcon && <span className="shrink-0">{rightIcon}</span>}
      </button>
    );
  }
);

Button.displayName = "Button";
