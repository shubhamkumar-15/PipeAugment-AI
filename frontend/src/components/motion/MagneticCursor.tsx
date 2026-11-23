"use client";

import React, { useEffect, useState, useRef } from "react";

export function MagneticCursor() {
  const [enabled, setEnabled] = useState(false);

  const dotRef = useRef<HTMLDivElement | null>(null);
  const ringRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    // Only enable on desktop with fine pointer & without reduced motion
    const isFinePointer = window.matchMedia(
      "(hover: hover) and (pointer: fine) and (min-width: 901px)"
    ).matches;
    const isReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (!isFinePointer || isReducedMotion) {
      return;
    }

    const timer = setTimeout(() => {
      setEnabled(true);
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return;

    let targetX = -100;
    let targetY = -100;
    let ringX = -100;
    let ringY = -100;
    let isRunning = false;
    let isExpanded = false;
    let isVisible = false;
    let animId: number | null = null;

    const setExpanded = (expanded: boolean) => {
      if (isExpanded === expanded) return;
      isExpanded = expanded;
      if (expanded) {
        dot.style.transform = `translate3d(${targetX}px, ${targetY}px, 0) translate(-50%, -50%) scale(1.4)`;
        dot.style.boxShadow = "0 0 14px rgba(240, 78, 0, 0.55)";
        ring.style.width = "54px";
        ring.style.height = "54px";
        ring.style.borderColor = "var(--gold)";
        ring.style.backgroundColor = "rgba(240, 184, 0, 0.12)";
      } else {
        dot.style.transform = `translate3d(${targetX}px, ${targetY}px, 0) translate(-50%, -50%) scale(1.0)`;
        dot.style.boxShadow = "0 0 8px rgba(240, 78, 0, 0.35)";
        ring.style.width = "36px";
        ring.style.height = "36px";
        ring.style.borderColor = "var(--saffron)";
        ring.style.backgroundColor = "transparent";
      }
    };

    const setVisible = (visible: boolean) => {
      if (isVisible === visible) return;
      isVisible = visible;
      dot.style.opacity = visible ? "1" : "0";
      ring.style.opacity = visible ? "1" : "0";
    };

    const loop = () => {
      const dx = targetX - ringX;
      const dy = targetY - ringY;
      ringX += dx * 0.22;
      ringY += dy * 0.22;

      const scale = isExpanded ? "scale(1.4)" : "scale(1.0)";
      dot.style.transform = `translate3d(${targetX}px, ${targetY}px, 0) translate(-50%, -50%) ${scale}`;
      ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0) translate(-50%, -50%)`;

      // Check if ring has caught up to target
      if (Math.abs(dx) < 0.15 && Math.abs(dy) < 0.15) {
        isRunning = false;
        animId = null;
        return; // Idle sleep: save CPU/battery
      }

      animId = requestAnimationFrame(loop);
    };

    const startLoop = () => {
      if (!isRunning) {
        isRunning = true;
        animId = requestAnimationFrame(loop);
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      targetX = e.clientX;
      targetY = e.clientY;
      if (!isVisible) setVisible(true);
      startLoop();
    };

    // Event delegation on document: only checks interactive targets on mouseover/mouseout
    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;

      if (target.closest('[data-cursor="native"]')) {
        setVisible(false);
        return;
      }

      const interactive = target.closest(
        'a, button, input, select, textarea, [data-cursor="expand"], [role="button"], label, [tabindex="0"]'
      );
      setExpanded(!!interactive);
      if (!isVisible) setVisible(true);
    };

    const handleMouseOut = (e: MouseEvent) => {
      const related = e.relatedTarget as HTMLElement | null;
      if (!related || related === document.documentElement) {
        setVisible(false);
      }
    };

    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    document.addEventListener("mouseover", handleMouseOver, { passive: true });
    document.addEventListener("mouseout", handleMouseOut, { passive: true });

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("mouseover", handleMouseOver);
      document.removeEventListener("mouseout", handleMouseOut);
      if (animId) cancelAnimationFrame(animId);
    };
  }, [enabled]);

  if (!enabled) return null;

  return (
    <>
      {/* Crisp inner dot (GPU translate3d only) */}
      <div
        ref={dotRef}
        aria-hidden="true"
        className="fixed top-0 left-0 pointer-events-none z-[9999] rounded-full w-2.5 h-2.5 bg-[var(--saffron)] opacity-0 transition-opacity duration-150 ease-out"
        style={{
          boxShadow: "0 0 8px rgba(240, 78, 0, 0.35)",
          willChange: "transform, opacity",
        }}
      />

      {/* Trailing outer ring (GPU translate3d with smooth transition on hover) */}
      <div
        ref={ringRef}
        aria-hidden="true"
        className="fixed top-0 left-0 pointer-events-none z-[9998] rounded-full w-9 h-9 border-[1.5px] border-[var(--saffron)] bg-transparent opacity-0 transition-[width,height,border-color,background-color,opacity] duration-200 ease-out"
        style={{
          willChange: "transform, width, height, opacity",
        }}
      />
    </>
  );
}
