"use client";

import React from "react";
import { motion, useReducedMotion } from "motion/react";

export default function Template({ children }: { children: React.ReactNode }) {
  const shouldReduceMotion = useReducedMotion();

  if (shouldReduceMotion) {
    return <>{children}</>;
  }

  return (
    <div className="relative w-full min-h-screen flex flex-col">
      {/* Saffron Hairline Light Sweep across top of page */}
      <motion.div
        aria-hidden="true"
        className="fixed top-0 left-0 right-0 h-[2.5px] bg-gradient-to-r from-transparent via-[var(--saffron)] to-transparent z-[60] pointer-events-none"
        initial={{ scaleX: 0, opacity: 1, transformOrigin: "left" }}
        animate={{ scaleX: 1, opacity: 0 }}
        transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
      />

      {/* Lightweight GPU-Composited Content Glide (No full-page blur repaints) */}
      <motion.div
        className="flex-1 flex flex-col w-full"
        initial={{ opacity: 0, y: 8, scale: 0.996 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
        style={{ willChange: "opacity, transform" }}
      >
        {children}
      </motion.div>
    </div>
  );
}
