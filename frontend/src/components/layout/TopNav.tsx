"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, ArrowRight, Upload } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { UploadModal } from "@/components/ui/UploadModal";
import { api } from "@/lib/api";
import { SystemInfo } from "@/lib/types";

const NAV_LINKS = [
  { name: "OVERVIEW", href: "/" },
  { name: "STUDIO", href: "/studio" },
  { name: "LIBRARY", href: "/library" },
  { name: "BATCH", href: "/batch" },
  { name: "DATASETS", href: "/datasets" },
  { name: "EVALUATION", href: "/evaluation" },
  { name: "SETTINGS", href: "/settings" },
];

export function TopNav() {
  const pathname = usePathname();
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  // Poll backend health status & load system telemetry
  useEffect(() => {
    let isMounted = true;
    const checkStatus = async () => {
      try {
        const [healthRes, sysRes] = await Promise.allSettled([
          api.getHealth(),
          api.getSystemInfo(),
        ]);

        if (isMounted) {
          if (healthRes.status === "fulfilled") {
            const statusStr = String(healthRes.value?.status || "").toLowerCase();
            setIsHealthy(statusStr === "ok" || statusStr === "healthy");
          } else {
            setIsHealthy(false);
          }
          if (sysRes.status === "fulfilled") {
            setSystemInfo(sysRes.value);
          }
        }
      } catch {
        if (isMounted) setIsHealthy(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const gpuName = systemInfo?.hardware?.gpu?.gpu_detected
    ? systemInfo.hardware.gpu.model_name || "NVIDIA GPU"
    : "CPU Pipeline Mode";

  const isStudio = pathname === "/studio";

  return (
    <>
      <header className="sticky top-0 z-40 h-16 w-full bg-[var(--bg)]/90 backdrop-blur-md border-b border-[var(--border)] transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">
          {/* LEFT: Brand Wordmark */}
          <Link href="/" className="flex items-center gap-2.5 shrink-0 group">
            {/* Minimal Geometric Pipe-Ring SVG Mark */}
            <div className="w-8 h-8 rounded-lg bg-[var(--ink)] text-[var(--bg)] flex items-center justify-center transition-transform group-hover:scale-105">
              <svg className="w-4 h-4 text-[var(--saffron)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <circle cx="12" cy="12" r="8" strokeDasharray="3 3" />
                <circle cx="12" cy="12" r="3" fill="currentColor" />
                <line x1="12" y1="2" x2="12" y2="4" strokeLinecap="round" />
                <line x1="12" y1="20" x2="12" y2="22" strokeLinecap="round" />
                <line x1="2" y1="12" x2="4" y2="12" strokeLinecap="round" />
                <line x1="20" y1="12" x2="22" y2="12" strokeLinecap="round" />
              </svg>
            </div>
            <div className="flex flex-col">
              <span className="font-display text-2xl tracking-wider uppercase text-[var(--ink)] leading-none">
                PIPEAUGMENT <span className="text-[var(--saffron)]">AI</span>
              </span>
            </div>
          </Link>

          {/* CENTER: Tracked Horizontal Navigation (Desktop) */}
          <nav className="hidden lg:flex items-center gap-6 xl:gap-8">
            {NAV_LINKS.map((link) => {
              const isActive =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`text-xs font-semibold tracking-[0.12em] transition-colors relative py-1 ${
                    isActive
                      ? "text-[var(--saffron)] after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-[var(--saffron)]"
                      : "text-[var(--muted)] hover:text-[var(--ink)]"
                  }`}
                >
                  {link.name}
                </Link>
              );
            })}
          </nav>

          {/* RIGHT: Live Status Dot, Theme Toggle, CTA Button */}
          <div className="flex items-center gap-3 shrink-0">
            {/* Live API Status Dot */}
            <div
              className={`flex items-center gap-2 px-2.5 py-1 rounded-full border text-xs font-mono transition-colors cursor-help ${
                isHealthy === true
                  ? "border-[var(--green-live)]/30 bg-[var(--green-live)]/10 text-[var(--green)]"
                  : isHealthy === false
                  ? "border-[var(--danger)]/30 bg-[var(--danger)]/10 text-[var(--danger)]"
                  : "border-[var(--border)] bg-[var(--bg-2)] text-[var(--muted)]"
              }`}
              title={
                isHealthy === true
                  ? `Backend Healthy · Node: ${gpuName}`
                  : isHealthy === false
                  ? "FastAPI Backend Offline at http://127.0.0.1:8000"
                  : "Checking Backend Health..."
              }
            >
              <span className="relative flex h-2 w-2">
                {isHealthy === true && (
                  <span className="animate-dot-ripple absolute inline-flex h-full w-full rounded-full bg-[var(--green-live)] opacity-75" />
                )}
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    isHealthy === true
                      ? "bg-[var(--green-live)]"
                      : isHealthy === false
                      ? "bg-[var(--danger)]"
                      : "bg-[var(--muted)]"
                  }`}
                />
              </span>
              <span className="hidden sm:inline text-[11px] font-semibold tracking-wider uppercase">
                {isHealthy === true ? "API LIVE" : isHealthy === false ? "OFFLINE" : "CONNECTING"}
              </span>
            </div>

            {/* Light / Dark Mode Toggle */}
            <ThemeToggle />

            {/* Solid Ink-Black Action Button */}
            {isStudio ? (
              <button
                type="button"
                onClick={() => setIsUploadOpen(true)}
                className="hidden sm:inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 text-xs font-semibold tracking-wide transition-all shadow-xs cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 text-[var(--saffron)]" />
                <span>UPLOAD IMAGE</span>
              </button>
            ) : (
              <Link
                href="/studio"
                className="hidden sm:inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[var(--ink)] text-[var(--bg)] hover:opacity-90 text-xs font-semibold tracking-wide transition-all shadow-xs"
              >
                <span>LAUNCH STUDIO</span>
                <ArrowRight className="w-3.5 h-3.5 text-[var(--saffron)]" />
              </Link>
            )}

            {/* Mobile Hamburger Toggle */}
            <button
              type="button"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg border border-[var(--border)] text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-2)] transition-colors"
              aria-label="Toggle navigation menu"
            >
              {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Slide-down Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden border-b border-[var(--border)] bg-[var(--bg)] px-6 py-4 space-y-3 shadow-lg animate-page-reveal">
            <nav className="flex flex-col space-y-2">
              {NAV_LINKS.map((link) => {
                const isActive =
                  link.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(link.href);
                return (
                  <Link
                    key={link.name}
                    href={link.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`py-2 px-3 rounded-lg text-xs font-semibold tracking-[0.12em] transition-colors ${
                      isActive
                        ? "bg-[var(--bg-3)] text-[var(--saffron)] font-bold"
                        : "text-[var(--muted)] hover:text-[var(--ink)] hover:bg-[var(--bg-2)]"
                    }`}
                  >
                    {link.name}
                  </Link>
                );
              })}
            </nav>

            <div className="pt-3 border-t border-[var(--border)] flex items-center justify-between">
              <span className="text-[11px] font-mono text-[var(--muted)]">
                Node: {gpuName}
              </span>
              <Link
                href="/studio"
                onClick={() => setIsMobileMenuOpen(false)}
                className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--saffron)]"
              >
                Open Studio →
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Upload Modal Dialog */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
      />
    </>
  );
}
