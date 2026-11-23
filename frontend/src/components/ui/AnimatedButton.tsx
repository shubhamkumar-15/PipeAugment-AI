"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const animatedButtonVariants = cva(
  "group relative inline-flex items-center justify-center gap-2 whitespace-nowrap text-xs font-bold tracking-wider uppercase transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--saffron)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 select-none cursor-pointer overflow-hidden active:scale-[0.98]",
  {
    variants: {
      variant: {
        saffron:
          "bg-[var(--saffron)] text-white shadow-md shadow-[var(--saffron)]/20 hover:bg-[var(--amber)] border border-transparent",
        ink:
          "bg-[var(--ink)] text-[var(--bg)] shadow-md hover:bg-[var(--ink-2)] border border-transparent",
        outline:
          "border border-[var(--border)] bg-transparent text-[var(--ink)] hover:border-[var(--saffron)] hover:text-[var(--saffron)] hover:bg-[var(--saffron)]/5",
      },
      size: {
        default: "h-11 px-6 py-2.5 rounded-xl",
        sm: "h-8 px-4 py-1.5 text-[11px] rounded-lg",
        lg: "h-13 px-8 py-3.5 text-sm rounded-2xl",
        full: "w-full h-12 px-6 py-3 rounded-xl",
      },
    },
    defaultVariants: {
      variant: "saffron",
      size: "default",
    },
  }
);

export interface AnimatedButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof animatedButtonVariants> {
  asChild?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  hideShimmer?: boolean;
}

export const AnimatedButton = React.forwardRef<HTMLButtonElement, AnimatedButtonProps>(
  (
    {
      className,
      variant = "saffron",
      size = "default",
      asChild = false,
      loading = false,
      leftIcon,
      rightIcon,
      hideShimmer = false,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : "button";

    return (
      <Comp
        ref={ref}
        data-cursor="expand"
        className={cn(animatedButtonVariants({ variant, size }), className)}
        disabled={disabled || loading}
        {...props}
      >
        {/* Subtle Restrained Shimmer Light Sweep on Hover */}
        {!hideShimmer && !loading && (
          <span
            aria-hidden="true"
            className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out bg-gradient-to-r from-transparent via-white/25 to-transparent pointer-events-none"
          />
        )}

        {/* Button Content */}
        <span className="relative z-10 inline-flex items-center justify-center gap-2">
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin text-current" />
          ) : (
            leftIcon
          )}
          <span>{children}</span>
          {!loading && rightIcon}
        </span>
      </Comp>
    );
  }
);

AnimatedButton.displayName = "AnimatedButton";
