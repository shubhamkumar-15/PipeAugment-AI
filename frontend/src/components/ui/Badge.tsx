import React from "react";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "neutral" | "saffron" | "amber" | "gold" | "success" | "danger" | "outline";
  dot?: boolean;
  pulse?: boolean;
}

export function Badge({
  variant = "neutral",
  dot = false,
  pulse = false,
  className = "",
  children,
  ...props
}: BadgeProps) {
  const variantStyles = {
    neutral: "bg-surface-base border-border-subtle text-text-muted",
    saffron: "bg-saffron/10 border-saffron/30 text-saffron",
    amber: "bg-amber/15 border-amber/30 text-amber",
    gold: "bg-gold/15 border-gold/30 text-gold",
    success: "bg-emerald/15 border-emerald/30 text-emerald",
    danger: "bg-rose/15 border-rose/30 text-rose",
    outline: "bg-transparent border-border-subtle text-text-muted",
  };

  const dotColors = {
    neutral: "bg-text-muted",
    saffron: "bg-saffron",
    amber: "bg-amber",
    gold: "bg-gold",
    success: "bg-emerald",
    danger: "bg-rose",
    outline: "bg-text-muted",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border font-medium select-none ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]} ${
            pulse ? "animate-pulse" : ""
          }`}
        />
      )}
      <span>{children}</span>
    </span>
  );
}
