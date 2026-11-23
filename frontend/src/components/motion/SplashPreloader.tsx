"use client";

import React, { useEffect, useRef, useState } from "react";

export function SplashPreloader() {
  const [mounted, setMounted] = useState(false);
  const [exiting, setExiting] = useState(false);
  const [finished, setFinished] = useState(false);

  const counterRef = useRef<HTMLSpanElement | null>(null);
  const barRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Avoid running on server
    if (typeof window === "undefined") return;

    // Check prefers-reduced-motion
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const t = setTimeout(() => setFinished(true), 0);
      return () => clearTimeout(t);
    }

    // Mount curtain immediately on hard reload / fresh page load
    const mountTimer = setTimeout(() => setMounted(true), 0);

    const duration = 900;
    let startTimestamp: number | null = null;
    let frameId: number;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const elapsed = timestamp - startTimestamp;
      const rawProgress = Math.min(elapsed / duration, 1);
      // Smooth cubic-out easing for natural countdown deceleration
      const easedProgress = Math.min(
        100,
        Math.floor((1 - Math.pow(1 - rawProgress, 2.5)) * 100)
      );

      // Direct DOM mutation for 120fps counter without React re-renders
      if (counterRef.current) {
        counterRef.current.textContent = String(easedProgress).padStart(2, "0");
      }
      if (barRef.current) {
        barRef.current.style.width = `${easedProgress}%`;
      }

      if (rawProgress < 1) {
        frameId = window.requestAnimationFrame(step);
      } else {
        if (counterRef.current) counterRef.current.textContent = "100";
        if (barRef.current) barRef.current.style.width = "100%";

        // Trigger curtain sweep upward via GPU transform
        setExiting(true);
        setTimeout(() => {
          setFinished(true);
        }, 650);
      }
    };

    frameId = window.requestAnimationFrame(step);

    return () => {
      clearTimeout(mountTimer);
      if (frameId) window.cancelAnimationFrame(frameId);
    };
  }, []);

  if (!mounted || finished) return null;

  return (
    <div
      aria-hidden="true"
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-center bg-[#080c14] text-white select-none overflow-hidden transition-transform ease-[cubic-bezier(0.77,0,0.18,1)] duration-[650ms] ${
        exiting ? "-translate-y-full pointer-events-none" : "translate-y-0"
      }`}
      style={{ willChange: "transform" }}
    >
      {/* Subtle radial ambient glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[radial-gradient(circle,rgba(255,90,0,0.12)_0%,transparent_70%)] pointer-events-none" />

      <div className="relative z-10 flex flex-col items-center text-center px-6">
        {/* Saffron-to-amber wordmark */}
        <div className="font-display text-5xl sm:text-7xl md:text-8xl tracking-[0.08em] uppercase bg-gradient-to-r from-[#ff5a00] via-[#ff8c00] to-[#ffb800] bg-clip-text text-transparent drop-shadow-[0_4px_24px_rgba(255,90,0,0.25)]">
          PIPEAUGMENT AI
        </div>

        {/* Quiet tracked uppercase subtitle */}
        <p className="mt-3 font-sans text-[10px] sm:text-xs tracking-[0.24em] text-[#8ea0be] uppercase max-w-lg font-medium">
          Condition-Controlled · Corrosion-Preserving Augmentation Engine
        </p>

        {/* Percentage Counter */}
        <div className="mt-10 sm:mt-12 flex items-baseline gap-1 font-display">
          <span
            ref={counterRef}
            className="text-5xl sm:text-6xl text-[#ff5a00] tracking-wider tabular-nums font-normal"
          >
            00
          </span>
          <span className="text-xl sm:text-2xl text-[#8ea0be] font-sans font-medium">
            %
          </span>
        </div>

        {/* Progress Bar Track */}
        <div className="mt-4 w-48 sm:w-64 h-[2px] bg-white/10 rounded-full overflow-hidden">
          <div
            ref={barRef}
            className="h-full bg-gradient-to-r from-[#ff5a00] to-[#ff9b00] w-0"
            style={{ willChange: "width" }}
          />
        </div>
      </div>
    </div>
  );
}
