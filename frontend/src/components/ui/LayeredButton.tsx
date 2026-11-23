"use client";

import React, { useEffect, useRef, useState } from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const layeredButtonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap text-xs font-semibold tracking-wider uppercase transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--saffron)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 overflow-hidden box-border cursor-pointer select-none",
  {
    variants: {
      variant: {
        ink: "bg-[var(--ink)] text-[var(--bg)] border border-transparent shadow-sm",
        saffron: "bg-[var(--saffron)] text-white border border-transparent shadow-sm shadow-[var(--saffron)]/20",
        outline: "border border-[var(--border)] bg-transparent text-[var(--ink)] hover:border-[var(--border-2)]",
        ghost: "bg-transparent text-[var(--ink)] hover:bg-[var(--bg-2)]",
      },
      size: {
        default: "h-10 px-5 py-2 rounded-xl",
        sm: "h-8 px-3.5 py-1.5 text-[11px] rounded-lg",
        lg: "h-12 px-7 py-3 text-sm rounded-xl",
        icon: "h-9 w-9 p-0 rounded-lg",
      },
    },
    defaultVariants: {
      variant: "ink",
      size: "default",
    },
  }
);

export interface LayeredButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof layeredButtonVariants> {
  asChild?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const LayeredButton = React.forwardRef<HTMLButtonElement, LayeredButtonProps>(
  (
    {
      className,
      variant = "ink",
      size = "default",
      children,
      asChild = false,
      loading = false,
      leftIcon,
      rightIcon,
      disabled,
      ...props
    },
    ref
  ) => {
    const internalRef = useRef<HTMLButtonElement>(null);
    const circleRef = useRef<HTMLSpanElement>(null);
    const labelRef = useRef<HTMLSpanElement>(null);
    const hoverLabelRef = useRef<HTMLSpanElement>(null);
    const [reducedMotion, setReducedMotion] = useState(false);

    React.useImperativeHandle(ref, () => internalRef.current!);

    useEffect(() => {
      if (typeof window === "undefined") return;
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        setReducedMotion(true);
        return;
      }

      const button = internalRef.current;
      const circle = circleRef.current;
      const label = labelRef.current;
      const hoverLabel = hoverLabelRef.current;

      if (!button || !circle) return;

      const layout = () => {
        const rect = button.getBoundingClientRect();
        const { width: w, height: h } = rect;
        if (w === 0 || h === 0) return;

        const R = ((w * w) / 4 + h * h) / (2 * h);
        const D = Math.ceil(2 * R) + 4;
        const delta = Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 2;
        const originY = D - delta;

        circle.style.width = `${D}px`;
        circle.style.height = `${D}px`;
        circle.style.bottom = `-${delta}px`;
        circle.style.left = "50%";
        circle.style.transform = "translateX(-50%) scale(0)";
        circle.style.transformOrigin = `50% ${originY}px`;

        if (label) {
          label.style.transform = "translateY(0)";
        }
        if (hoverLabel) {
          hoverLabel.style.transform = `translateY(${h + 10}px)`;
          hoverLabel.style.opacity = "0";
        }

        circle.dataset.scale = "1.25";
        if (label) label.dataset.moveY = (-(h + 8)).toString();
        if (hoverLabel) {
          hoverLabel.dataset.startY = Math.ceil(h + 20).toString();
        }
      };

      layout();
      window.addEventListener("resize", layout);
      return () => window.removeEventListener("resize", layout);
    }, []);

    const handleMouseEnter = () => {
      if (reducedMotion || disabled || loading) return;
      const circle = circleRef.current;
      const label = labelRef.current;
      const hoverLabel = hoverLabelRef.current;

      if (circle) {
        circle.style.transition = "transform 0.35s cubic-bezier(0.25, 1, 0.5, 1)";
        circle.style.transform = `translateX(-50%) scale(${circle.dataset.scale || 1.25})`;
      }

      if (label) {
        label.style.transition = "transform 0.35s cubic-bezier(0.25, 1, 0.5, 1)";
        label.style.transform = `translateY(${label.dataset.moveY || -36}px)`;
      }

      if (hoverLabel) {
        hoverLabel.style.transition =
          "transform 0.35s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.35s cubic-bezier(0.25, 1, 0.5, 1)";
        hoverLabel.style.transform = "translateY(0)";
        hoverLabel.style.opacity = "1";
      }
    };

    const handleMouseLeave = () => {
      if (reducedMotion || disabled || loading) return;
      const circle = circleRef.current;
      const label = labelRef.current;
      const hoverLabel = hoverLabelRef.current;

      if (circle) {
        circle.style.transition = "transform 0.25s cubic-bezier(0.25, 1, 0.5, 1)";
        circle.style.transform = "translateX(-50%) scale(0)";
      }

      if (label) {
        label.style.transition = "transform 0.25s cubic-bezier(0.25, 1, 0.5, 1)";
        label.style.transform = "translateY(0)";
      }

      if (hoverLabel) {
        hoverLabel.style.transition =
          "transform 0.25s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.25s cubic-bezier(0.25, 1, 0.5, 1)";
        hoverLabel.style.transform = `translateY(${hoverLabel.dataset.startY || 40}px)`;
        hoverLabel.style.opacity = "0";
      }
    };

    // Expanding circle color based on variant
    const circleColor =
      variant === "ink"
        ? "bg-[var(--saffron)]"
        : variant === "saffron"
        ? "bg-[var(--amber)]"
        : variant === "outline"
        ? "bg-[var(--saffron)]/15"
        : "bg-[var(--saffron)]/10";

    const hoverTextColor =
      variant === "ink"
        ? "text-white"
        : variant === "saffron"
        ? "text-white"
        : "text-[var(--saffron)]";

    const Comp = asChild ? Slot : "button";

    return (
      <Comp
        ref={internalRef}
        data-cursor="expand"
        className={cn(
          layeredButtonVariants({ variant, size }),
          className
        )}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        disabled={disabled || loading}
        {...props}
      >
        {/* Radial Expanding Circle Layer */}
        {!reducedMotion && (
          <span
            ref={circleRef}
            aria-hidden="true"
            className={cn("absolute rounded-full pointer-events-none will-change-transform z-0", circleColor)}
          />
        )}

        {/* Content Container */}
        <span className="relative z-10 inline-flex items-center justify-center gap-2 w-full h-full">
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-current" />
              <span>{children}</span>
            </span>
          ) : reducedMotion ? (
            <span className="inline-flex items-center justify-center gap-2">
              {leftIcon}
              <span>{children}</span>
              {rightIcon}
            </span>
          ) : (
            <>
              {/* Primary Label */}
              <span
                ref={labelRef}
                className="inline-flex items-center justify-center gap-2 will-change-transform"
              >
                {leftIcon}
                <span>{children}</span>
                {rightIcon}
              </span>

              {/* Hover Label Layer */}
              <span
                ref={hoverLabelRef}
                aria-hidden="true"
                className={cn(
                  "absolute inset-0 inline-flex items-center justify-center gap-2 will-change-transform pointer-events-none font-bold",
                  hoverTextColor
                )}
              >
                {leftIcon}
                <span>{children}</span>
                {rightIcon}
              </span>
            </>
          )}
        </span>
      </Comp>
    );
  }
);

LayeredButton.displayName = "LayeredButton";
