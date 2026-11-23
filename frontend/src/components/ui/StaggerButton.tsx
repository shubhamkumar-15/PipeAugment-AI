"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import {
  AnimatePresence,
  motion,
  type Variants,
  useReducedMotion,
  type Easing,
} from "motion/react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const staggerButtonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap text-xs font-semibold tracking-wider uppercase transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--saffron)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 select-none cursor-pointer overflow-hidden",
  {
    variants: {
      variant: {
        ink: "bg-[var(--ink)] text-[var(--bg)] border border-transparent shadow-sm hover:opacity-95",
        saffron: "bg-[var(--saffron)] text-white border border-transparent shadow-sm shadow-[var(--saffron)]/20 hover:bg-[var(--amber)]",
        outline: "border border-[var(--border)] bg-transparent text-[var(--ink)] hover:border-[var(--saffron)] hover:text-[var(--saffron)] hover:bg-[var(--saffron)]/5",
        ghost: "bg-transparent text-[var(--ink)] hover:bg-[var(--bg-2)] hover:text-[var(--saffron)]",
      },
      size: {
        default: "h-10 px-5 py-2 rounded-xl",
        sm: "h-8 px-3.5 py-1.5 text-[11px] rounded-lg",
        lg: "h-12 px-7 py-3 text-sm rounded-xl",
        icon: "h-9 w-9 p-0 rounded-lg",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
    },
  }
);

export interface StaggerButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof staggerButtonVariants> {
  asChild?: boolean;
  text?: string;
  hoverText?: string;
  staggerDelay?: number;
  staggerDuration?: number;
  direction?: "up" | "down" | "alternate-even" | "alternate-odd";
  easing?: Easing;
  disableStagger?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const useProcessedChars = (text: string, hoverText?: string) =>
  React.useMemo(() => {
    const base = Array.from(text);
    const hover = Array.from(hoverText ?? text);
    const max = Math.max(base.length, hover.length);
    return {
      safeBase: Array.from({ length: max }, (_, i) => base[i] ?? " "),
      safeHover: Array.from({ length: max }, (_, i) => hover[i] ?? " "),
    };
  }, [text, hoverText]);

export const StaggerButton = React.forwardRef<HTMLButtonElement, StaggerButtonProps>(
  (
    {
      className,
      variant = "outline",
      size = "default",
      asChild = false,
      text,
      hoverText,
      staggerDelay = 0.02,
      staggerDuration = 0.35,
      direction = "up",
      easing = [0.25, 1, 0.5, 1] as Easing,
      disableStagger = false,
      loading = false,
      leftIcon,
      rightIcon,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const [isHovered, setIsHovered] = React.useState(false);
    const prefersReducedMotion = useReducedMotion();

    const displayText =
      text ??
      (typeof children === "string"
        ? children
        : React.Children.toArray(children)
            .map((c) => (typeof c === "string" ? c : ""))
            .join("") || "");

    const { safeBase, safeHover } = useProcessedChars(displayText, hoverText);

    const containerVariants: Variants = {
      initial: {},
      hover: {
        transition: {
          staggerChildren: prefersReducedMotion ? 0 : staggerDelay,
        },
      },
    };

    const charVariants: Variants = {
      initial: { y: "0%" },
      hover: ({ index }: { index: number }) =>
        prefersReducedMotion
          ? { y: "0%" }
          : {
              y: direction === "down" ? "100%" : "-100%",
              transition: {
                duration: staggerDuration,
                delay: index * staggerDelay,
                ease: easing,
              },
            },
    };

    const Comp = asChild ? Slot : "button";

    return (
      <Comp
        ref={ref}
        data-cursor="expand"
        className={cn(
          staggerButtonVariants({ variant, size }),
          className
        )}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        disabled={disabled || loading}
        {...props}
      >
        <span className="inline-flex items-center justify-center gap-2">
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin text-current" />
          ) : (
            leftIcon
          )}

          {prefersReducedMotion || disableStagger || displayText === "" ? (
            <span>{children ?? displayText}</span>
          ) : (
            <AnimatePresence mode="wait">
              <motion.span
                className="relative inline-flex leading-none select-none overflow-hidden"
                variants={containerVariants}
                initial="initial"
                animate={isHovered ? "hover" : "initial"}
              >
                {safeBase.map((char, index) => {
                  const nextChar = safeHover[index];
                  const isSpace = char === " " && nextChar === " ";

                  return (
                    <span
                      key={index}
                      className="inline-block h-[1.2em] overflow-hidden relative"
                      style={{ lineHeight: 1.2 }}
                    >
                      <motion.span
                        className="block relative"
                        variants={charVariants}
                        custom={{ index }}
                        style={{ lineHeight: 1.2 }}
                      >
                        <span className="block h-[1.2em]">
                          {isSpace ? "\u00A0" : char}
                        </span>
                        <span className="block h-[1.2em] text-[var(--saffron)] font-bold">
                          {isSpace ? "\u00A0" : nextChar}
                        </span>
                      </motion.span>
                    </span>
                  );
                })}
              </motion.span>
            </AnimatePresence>
          )}

          {rightIcon}
        </span>
      </Comp>
    );
  }
);

StaggerButton.displayName = "StaggerButton";
