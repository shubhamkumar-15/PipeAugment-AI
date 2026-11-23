"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import { Menu, X, ArrowRight, Sparkles } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { ApiStatusBadge } from "./ApiStatusBadge";
import { LayeredButton } from "@/components/ui/LayeredButton";
import { AnimatedButton } from "@/components/ui/AnimatedButton";

const NAV_LINKS = [
  { name: "OVERVIEW", href: "/" },
  { name: "STUDIO", href: "/studio" },
  { name: "LIBRARY", href: "/library" },
  { name: "BATCH", href: "/batch" },
  { name: "DATASETS", href: "/datasets" },
  { name: "EVALUATION", href: "/evaluation" },
  { name: "SETTINGS", href: "/settings" },
];

export interface TopNavShellProps {
  children: React.ReactNode;
  pageTitle?: string;
  fullWidth?: boolean;
}

export function TopNavShell(props: TopNavShellProps) {
  const { children, fullWidth = false } = props;
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const isStudioPage = pathname === "/studio";

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--ink-2)] flex flex-col transition-colors selection:bg-[var(--saffron)] selection:text-white relative">
      {/* Sticky Liquid-Glass Top-Nav Bar */}
      <header className="sticky top-0 z-40 h-16 w-full bg-[var(--bg)]/80 backdrop-blur-[20px] backdrop-saturate-[180%] border-b border-[var(--border)] transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">
          {/* LEFT: Bold Bebas Neue Saffron Wordmark */}
          <Link href="/" className="flex items-center gap-2.5 shrink-0 group">
            <div className="w-8 h-8 rounded-lg bg-[var(--ink)] text-[var(--bg)] flex items-center justify-center transition-transform group-hover:scale-105 shadow-sm">
              <svg
                className="w-4 h-4 text-[var(--saffron)]"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
              >
                <circle cx="12" cy="12" r="8" strokeDasharray="3 3" />
                <circle cx="12" cy="12" r="3" fill="currentColor" />
                <line x1="12" y1="2" x2="12" y2="4" strokeLinecap="round" />
                <line x1="12" y1="20" x2="12" y2="22" strokeLinecap="round" />
                <line x1="2" y1="12" x2="4" y2="12" strokeLinecap="round" />
                <line x1="20" y1="12" x2="22" y2="12" strokeLinecap="round" />
              </svg>
            </div>
            <span className="font-display text-2xl tracking-wider uppercase text-[var(--ink)] leading-none">
              PIPEAUGMENT <span className="text-[var(--saffron)]">AI</span>
            </span>
          </Link>

          {/* CENTER: Frosted Glass Nav Rail with Spring-Gliding Active Pill */}
          <nav className="hidden lg:flex items-center gap-1 p-1 rounded-full bg-[var(--bg-2)]/80 backdrop-blur-md border border-[var(--border)] shadow-xs relative">
            {NAV_LINKS.map((link) => {
              const isActive =
                link.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(link.href);
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`relative px-3.5 py-1.5 rounded-full text-[11px] font-semibold tracking-[0.14em] uppercase transition-colors z-10 select-none ${
                    isActive
                      ? "text-[var(--saffron)] font-bold"
                      : "text-[var(--muted)] hover:text-[var(--ink)]"
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="nav-glass-pill"
                      className="absolute inset-0 rounded-full bg-[var(--bg)] border border-[var(--border)] shadow-sm -z-10"
                      transition={{ type: "spring", stiffness: 420, damping: 32 }}
                    />
                  )}
                  {link.name}
                </Link>
              );
            })}
          </nav>

          {/* RIGHT: Status Beacon, Theme Toggle & Solid Ink CTA */}
          <div className="flex items-center gap-3 sm:gap-4">
            {/* Live Backend Beacon with Hover Tooltip (Isolated polling) */}
            <ApiStatusBadge />

            {/* Sun / Moon Theme Toggle */}
            <ThemeToggle />

            {/* Solid Ink-Black Action Button: OPEN STUDIO → */}
            <Link
              href="/studio"
              className="hidden sm:inline-block"
            >
              <LayeredButton
                variant="ink"
                size="sm"
                rightIcon={<ArrowRight className="w-3.5 h-3.5 text-[var(--saffron)]" />}
              >
                OPEN STUDIO
              </LayeredButton>
            </Link>

            {/* Mobile Hamburger Toggle */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="lg:hidden p-2 rounded-lg border border-[var(--border)] text-[var(--ink)] hover:bg-[var(--bg-2)] transition-colors cursor-pointer"
              aria-label="Toggle navigation menu"
            >
              {isMobileMenuOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {isMobileMenuOpen && (
          <div className="lg:hidden border-b border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur-md px-6 py-4 space-y-3">
            <div className="flex flex-col space-y-2">
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
                    className={`py-2 text-xs font-semibold tracking-[0.14em] uppercase transition-colors ${
                      isActive
                        ? "text-[var(--saffron)] font-bold"
                        : "text-[var(--muted)] hover:text-[var(--ink)]"
                    }`}
                  >
                    {link.name}
                  </Link>
                );
              })}
            </div>
            <div className="pt-3 border-t border-[var(--border)]">
              <Link
                href="/studio"
                onClick={() => setIsMobileMenuOpen(false)}
                className="w-full block"
              >
                <LayeredButton
                  variant="ink"
                  className="w-full justify-center"
                  rightIcon={<ArrowRight className="w-3.5 h-3.5 text-[var(--saffron)]" />}
                >
                  OPEN STUDIO
                </LayeredButton>
              </Link>
            </div>
          </div>
        )}
      </header>

      {/* Main Container */}
      <main
        className={`flex-1 w-full ${
          fullWidth
            ? "max-w-none px-0"
            : "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-10"
        }`}
      >
        {children}
      </main>

      {/* Floating Bottom-Right Saffron Pill CTA (on non-Studio pages) */}
      {!isStudioPage && (
        <div className="fixed bottom-6 right-6 z-40">
          <Link href="/studio">
            <AnimatedButton
              variant="saffron"
              size="sm"
              className="rounded-full shadow-lg shadow-[var(--saffron)]/30"
              leftIcon={<Sparkles className="w-3.5 h-3.5" />}
            >
              LAUNCH STUDIO →
            </AnimatedButton>
          </Link>
        </div>
      )}

      {/* Clean #0a0f1e Obsidian Footer */}
      <footer className="w-full bg-[#0a0f1e] text-white py-12 border-t border-white/10 mt-16 relative overflow-hidden">
        {/* Subtle top-right warm saffron radial glow */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-[radial-gradient(ellipse_at_top_right,rgba(240,78,0,0.15),transparent_70%)] pointer-events-none" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 space-y-8">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-8 border-b border-white/10">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[var(--saffron)]" />
                <span className="font-display text-2xl tracking-wider uppercase text-white">
                  PIPEAUGMENT <span className="text-[var(--saffron)]">AI</span>
                </span>
              </div>
              <p className="text-xs text-[#8ea0be] max-w-md font-sans leading-relaxed">
                Condition-controlled, corrosion-preserving visual inspection augmentation engine.
                Built for pipeline non-destructive testing (NDT) and robust defect detector evaluation.
              </p>
            </div>

            {/* Navigation route links */}
            <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-xs font-mono text-[#8ea0be]">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.name}
                  href={link.href}
                  className="hover:text-[var(--saffron)] transition-colors uppercase tracking-wider"
                >
                  {link.name}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono text-[#6c7d99]">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--green-live)]" />
              <span>Inspection Node Active</span>
            </div>
            <div>
              ControlNet Canny SD 1.5 Architecture · Dual-Environment Boundary
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Export as default AppShell wrapper as well for drop-in compatibility
export { TopNavShell as AppShell };
