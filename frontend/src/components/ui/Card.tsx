import React from "react";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "surface" | "elevated" | "glass";
  hover?: boolean;
  selected?: boolean;
}

export function Card({
  variant = "surface",
  hover = false,
  selected = false,
  className = "",
  children,
  ...props
}: CardProps) {
  const variantStyles = {
    surface: "bg-surface-low border border-border-subtle",
    elevated: "bg-surface-base border border-border-subtle",
    glass: "glass-panel",
  };

  const hoverStyle = hover
    ? "hover-lift hover:border-border-subtle-hover hover:shadow-lg cursor-pointer"
    : "";

  const selectedStyle = selected
    ? "border-saffron ring-1 ring-saffron shadow-[0_0_24px_rgba(255,90,0,0.22)]"
    : "";

  return (
    <div
      className={`rounded-xl transition-all duration-200 ${variantStyles[variant]} ${hoverStyle} ${selectedStyle} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`p-5 pb-3 flex flex-col gap-1 border-b border-border-subtle/50 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={`font-heading font-semibold text-text-primary text-lg tracking-tight ${className}`}
      {...props}
    >
      {children}
    </h3>
  );
}

export function CardDescription({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={`text-text-muted text-sm ${className}`} {...props}>
      {children}
    </p>
  );
}

export function CardContent({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`p-5 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`p-5 pt-3 border-t border-border-subtle/50 flex items-center justify-between ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
