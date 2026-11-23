"use client";

import React from "react";

interface PageRevealProps {
  children: React.ReactNode;
  className?: string;
  delayMs?: number;
}

export function PageReveal({
  children,
  className = "",
  delayMs = 0,
}: PageRevealProps) {
  return (
    <div
      className={`animate-page-reveal ${className}`}
      style={delayMs ? { animationDelay: `${delayMs}ms` } : undefined}
    >
      {children}
    </div>
  );
}
