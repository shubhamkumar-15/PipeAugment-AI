"use client";

import { useEffect, useState, useRef } from "react";

/**
 * Motion Presets & Animation Tokens for PipeAugment AI
 * Based on Cyber-Saffron Light-First Editorial Specification.
 * Honors prefers-reduced-motion automatically.
 */

export const motionPresets = {
  // Page entrance: optical blur upward reveal
  pageReveal: "animate-page-reveal",

  // Editorial hero headline entrance
  heroEntry: "animate-hero-entry",

  // Live status beacon dot ripple
  dotRipple: "animate-dot-ripple",

  // Slide-over drawer
  drawerIn: "animate-drawer-in",

  // Ticker marquee
  ticker: "animate-ticker",

  // Tactile card lift
  liftHover: "lift-hover",

  // Stagger delays
  stagger: {
    1: "stagger-1",
    2: "stagger-2",
    3: "stagger-3",
    4: "stagger-4",
  },
} as const;

/**
 * Eased Counter Hook
 * Smoothly animates a number from 0 to target using Quartic-Out easing (1 - (1 - t)^4).
 * Honors prefers-reduced-motion.
 */
export function useEasedCounter(
  target: number,
  durationMs: number = 1200,
  shouldStart: boolean = true
): number {
  const [count, setCount] = useState<number>(0);

  useEffect(() => {
    if (!shouldStart || target === 0) {
      if (target === 0) {
        const t = setTimeout(() => setCount(0), 0);
        return () => clearTimeout(t);
      }
      return;
    }

    if (
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      const timer = setTimeout(() => setCount(target), 0);
      return () => clearTimeout(timer);
    }

    let startTimestamp: number | null = null;
    const startValue = 0;
    let animationFrameId: number;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / durationMs, 1);
      // Quartic-out easing: 1 - (1 - progress)^4
      const easeOut = 1 - Math.pow(1 - progress, 4);
      const current = Math.round(startValue + (target - startValue) * easeOut);
      setCount(current);

      if (progress < 1) {
        animationFrameId = window.requestAnimationFrame(step);
      } else {
        setCount(target);
      }
    };

    animationFrameId = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(animationFrameId);
  }, [target, durationMs, shouldStart]);

  return count;
}

/**
 * Hook to trigger animation when an element enters the viewport
 */
export function useInView(options: IntersectionObserverInit = { threshold: 0.2 }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsInView(true);
        observer.unobserve(el);
      }
    }, options);

    observer.observe(el);
    return () => observer.disconnect();
  }, [options]);

  return { ref, isInView };
}
