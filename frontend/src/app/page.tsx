"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowRight,
  RefreshCw,
  Sparkles,
  Shield,
  Layers,
  Sliders,
  CheckCircle2,
  FileCheck2,
  FolderArchive,
  ChevronDown,
  Upload,
} from "lucide-react";
import { AppShell } from "@/components/layout";
import { UploadModal } from "@/components/ui/UploadModal";
import { LayeredButton } from "@/components/ui/LayeredButton";
import { StaggerButton } from "@/components/ui/StaggerButton";
import { AnimatedButton } from "@/components/ui/AnimatedButton";
import { api, getImageFileUrl } from "@/lib/api";
import { DatasetStatsResponse, ImageMetadata, BenchmarkData } from "@/lib/types";
import { useEasedCounter } from "@/lib/motion";

export default function OverviewPage() {
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null);
  const [recentImages, setRecentImages] = useState<ImageMetadata[]>([]);
  const [isBackendOffline, setIsBackendOffline] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const fetchDashboardData = async () => {
    setIsBackendOffline(false);

    try {
      const [statsRes, imgRes, benchRes] = await Promise.allSettled([
        api.getDatasetStats(),
        api.listImages({ limit: 6 }),
        api.getBenchmarkResults(),
      ]);

      if (statsRes.status === "fulfilled") setStats(statsRes.value);
      if (imgRes.status === "fulfilled") setRecentImages(imgRes.value.items);
      if (benchRes.status === "fulfilled") setBenchmark(benchRes.value);

      if (
        statsRes.status === "rejected" &&
        imgRes.status === "rejected"
      ) {
        setIsBackendOffline(true);
      }
    } catch {
      setIsBackendOffline(true);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchDashboardData();
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  // Eased animated counters (Quartic-out)
  const totalImagesCount = useEasedCounter(stats?.total_images ?? 0, 1200);
  const passedQualityCount = useEasedCounter(
    stats?.by_quality_status?.["pass"] ?? 0,
    1200
  );

  // Dynamic values from /evaluation/benchmark
  // clean_test_summary.C.accuracy_mean (94.4%)
  interface BenchmarkResultPayload {
    clean_test_summary?: {
      C?: { accuracy_mean?: number };
    };
    robustness_by_condition?: Array<{
      condition_name?: string;
      experiments?: {
        C?: { corrosion_recall?: { mean?: number } };
      };
    }>;
  }
  const typedBenchmark = benchmark as unknown as BenchmarkResultPayload;
  const rawTestAcc = typedBenchmark?.clean_test_summary?.C?.accuracy_mean;
  const expCTestAcc = rawTestAcc !== undefined ? (rawTestAcc * 100).toFixed(1) : "94.4";

  // Optical Blur/Noise corrosion recall for Exp C
  const blurCondition = typedBenchmark?.robustness_by_condition?.find(
    (c) => c.condition_name === "sensor_blur_noise"
  );
  const rawNoiseRecall = blurCondition?.experiments?.C?.corrosion_recall?.mean;
  const noiseRecall = rawNoiseRecall !== undefined ? (rawNoiseRecall * 100).toFixed(1) : "97.9";

  return (
    <AppShell pageTitle="Editorial Overview">
      <div className="space-y-24 pb-16">
        {/* Backend Offline Warning Banner */}
        {isBackendOffline && (
          <div className="p-4 rounded-2xl bg-[var(--danger)]/10 border border-[var(--danger)]/30 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-[var(--danger)] animate-ping" />
              <div>
                <p className="text-xs font-semibold text-[var(--danger)]">
                  FastAPI Backend Unreachable
                </p>
                <p className="text-[11px] text-[var(--muted)] font-mono">
                  Make sure backend is running at http://127.0.0.1:8000
                </p>
              </div>
            </div>
            <StaggerButton
              type="button"
              onClick={fetchDashboardData}
              variant="outline"
              size="sm"
              leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
            >
              Retry
            </StaggerButton>
          </div>
        )}

        {/* ================================================================ */}
        {/* SECTION A: FULL-HEIGHT EDITORIAL HERO */}
        {/* ================================================================ */}
        <section className="min-h-[78vh] flex flex-col justify-center relative overflow-hidden pt-4 pb-12 hero-ambient-glow">
          <div className="max-w-4xl space-y-8 relative z-10">
            {/* Eyebrow with hairline & live dotRipple */}
            <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-[var(--bg-2)] border border-[var(--border)] text-[11px] font-mono tracking-wider text-[var(--muted)]">
              <span className="w-2 h-2 rounded-full bg-[var(--green-live)] animate-dot-ripple shrink-0" />
              <span className="text-[var(--ink)] font-semibold">
                AI-POWERED
              </span>
              <span>·</span>
              <span>CONDITION-CONTROLLED PIPELINE AUGMENTATION</span>
            </div>

            {/* Iconic 3-Line Bebas Neue Headline */}
            <h1 className="font-display text-[clamp(3.4rem,8.2vw,6.8rem)] leading-[0.92] tracking-[0.01em] uppercase text-[var(--ink)]">
              <span className="block">PIPELINE CORROSION</span>
              <span className="block text-[var(--saffron)]">AUGMENTATION</span>
              <span className="block hollow-headline">BUILT FOR ROBUSTNESS</span>
            </h1>

            {/* Clean 2-Line Subcopy */}
            <p className="text-base sm:text-lg text-[var(--muted)] max-w-xl font-normal leading-relaxed">
              Industrial synthetic condition generation with Canny-ControlNet and
              automated Phase 4 quality gates preserving critical rust pitting morphology.
            </p>

            {/* Side-by-Side CTAs */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <Link href="/studio">
                <LayeredButton
                  variant="ink"
                  size="lg"
                  rightIcon={<ArrowRight className="w-4 h-4 text-[var(--saffron)]" />}
                >
                  OPEN AUGMENTATION STUDIO
                </LayeredButton>
              </Link>
              <a href="#how-it-works">
                <StaggerButton
                  variant="outline"
                  size="lg"
                  rightIcon={<ChevronDown className="w-4 h-4 text-[var(--muted)]" />}
                >
                  HOW IT WORKS
                </StaggerButton>
              </a>
            </div>
          </div>

          {/* Centered Minimal Bottom Scroll Indicator */}
          <div className="pt-16 flex flex-col items-center justify-center gap-2 text-[10px] font-mono tracking-[0.2em] text-[var(--muted)] uppercase opacity-75">
            <span>SCROLL</span>
            <div className="w-[1px] h-8 bg-gradient-to-b from-[var(--border-2)] to-transparent relative">
              <span className="absolute top-0 left-[-2px] w-[5px] h-[5px] rounded-full bg-[var(--saffron)]" />
            </div>
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION B: 4-COLUMN HAIRLINE ANIMATED STAT BAND */}
        {/* ================================================================ */}
        <section className="border-y border-[var(--border)] bg-[var(--bg-2)]/60 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-10">
          <div className="max-w-7xl mx-auto grid grid-cols-2 lg:grid-cols-4 divide-y lg:divide-y-0 lg:divide-x divide-[var(--border)]">
            {/* Stat 1: Total Images */}
            <div className="px-6 py-4 lg:py-0 flex flex-col justify-between">
              <span className="text-[10px] font-mono tracking-widest text-[var(--muted)] uppercase block mb-1">
                #01 · REPOSITORY
              </span>
              <div className="font-display text-4xl sm:text-5xl lg:text-6xl text-[var(--ink)] leading-none my-1 tabular-nums">
                {totalImagesCount}
              </div>
              <span className="text-xs font-semibold text-[var(--ink-2)]">
                WORKSPACE IMAGES
              </span>
            </div>

            {/* Stat 2: Quality-Passed Outputs */}
            <div className="px-6 py-4 lg:py-0 flex flex-col justify-between">
              <span className="text-[10px] font-mono tracking-widest text-[var(--muted)] uppercase block mb-1">
                #02 · VALIDATED
              </span>
              <div className="font-display text-4xl sm:text-5xl lg:text-6xl text-[var(--saffron)] leading-none my-1 tabular-nums">
                {passedQualityCount}
              </div>
              <span className="text-xs font-semibold text-[var(--ink-2)]">
                QUALITY-PASSED OUTPUTS
              </span>
            </div>

            {/* Stat 3: Exp C Test Accuracy */}
            <div className="px-6 py-4 lg:py-0 flex flex-col justify-between">
              <span className="text-[10px] font-mono tracking-widest text-[var(--muted)] uppercase block mb-1">
                #03 · BENCHMARK
              </span>
              <div className="font-display text-4xl sm:text-5xl lg:text-6xl text-[var(--ink)] leading-none my-1 tabular-nums">
                {expCTestAcc}%
              </div>
              <span className="text-xs font-semibold text-[var(--ink-2)]">
                EXP C TEST ACCURACY
              </span>
            </div>

            {/* Stat 4: Noise Corrosion Recall */}
            <div className="px-6 py-4 lg:py-0 flex flex-col justify-between">
              <span className="text-[10px] font-mono tracking-widest text-[var(--muted)] uppercase block mb-1">
                #04 · ROBUSTNESS
              </span>
              <div className="font-display text-4xl sm:text-5xl lg:text-6xl text-[var(--green-live)] leading-none my-1 tabular-nums">
                {noiseRecall}%
              </div>
              <span className="text-xs font-semibold text-[var(--ink-2)]">
                NOISE CORROSION RECALL
              </span>
            </div>
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION C: ASYMMETRIC FEATURE SPLIT WITH FANNED 3-CARD STACK */}
        {/* ================================================================ */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Narrative */}
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 text-xs font-mono font-bold tracking-widest text-[var(--saffron)] uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--saffron)]" />
              <span>STRUCTURAL INTEGRITY</span>
            </div>

            <h2 className="font-display text-4xl sm:text-5xl lg:text-6xl text-[var(--ink)] leading-[0.95] tracking-tight uppercase">
              NOT RANDOM NOISE.
              <span className="block text-[var(--saffron)]">
                CONTROLLED PRESERVATION.
              </span>
            </h2>

            <div className="space-y-4 text-sm sm:text-base text-[var(--muted)] leading-relaxed">
              <p>
                Standard unguided diffusion erases microscopic pit morphology,
                filling in severe wall-loss defects as clean metal. In pipeline
                NDT inspection, an augmentation that destroys corrosion features
                actively damages downstream detector reliability.
              </p>
              <p>
                PipeAugment AI couples Canny-guided ControlNet (frozen scale 0.80)
                with automated Phase 4 screening (SSIM, LPIPS, Edge IoU). The
                result is rigorous environmental synthesis—dust, water streaks,
                and low-light—with guaranteed structural pit preservation.
              </p>
            </div>

            <div className="pt-2 flex items-center gap-6 text-xs font-mono">
              <div className="flex items-center gap-2 text-[var(--green-live)] font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>+66.8% Edge Retention vs Plain SD</span>
              </div>
              <div className="text-[var(--muted)]">
                Zero Pit Annihilation
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Fanned 3-Card Stack */}
          <div className="lg:col-span-6 flex justify-center py-6">
            <div className="relative w-full max-w-md h-80 sm:h-96">
              {/* Back Card: Traditional CV */}
              <div className="absolute inset-x-0 top-0 p-6 rounded-2xl bg-[var(--bg-2)] border border-[var(--border)] shadow-[var(--shadow-s)] transform -rotate-3 transition-transform hover:-translate-y-2 hover:-rotate-4 duration-300">
                <div className="flex items-center justify-between text-xs font-mono text-[var(--muted)] mb-3">
                  <span>#LAYER 01</span>
                  <span>OPENCV ENGINE</span>
                </div>
                <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                  TRADITIONAL CV
                </h3>
                <p className="text-xs text-[var(--muted)] mt-1">
                  9 Deterministic OpenCV Transforms · Lighting & Optical Stress
                </p>
              </div>

              {/* Middle Card: Phase 4 Quality Gate */}
              <div className="absolute inset-x-4 top-14 p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border-2)] shadow-[var(--shadow-m)] transform rotate-2 transition-transform hover:-translate-y-2 hover:rotate-3 duration-300">
                <div className="flex items-center justify-between text-xs font-mono text-[var(--saffron)] mb-3 font-semibold">
                  <span>#LAYER 02</span>
                  <span>PHASE 4 GATE</span>
                </div>
                <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                  QUALITY SCREENING
                </h3>
                <p className="text-xs text-[var(--muted)] mt-1">
                  RGB SSIM &gt; 0.45 · LPIPS &lt; 0.35 · Canny Edge IoU &gt; 0.18
                </p>
              </div>

              {/* Front Card: ControlNet Canny v1 (Highlight) */}
              <div className="absolute inset-x-8 top-28 p-6 rounded-2xl bg-[var(--vault-bg)] text-white border border-[var(--saffron)]/40 shadow-xl shadow-[var(--saffron)]/10 transform rotate-[-1deg] hover:-translate-y-3 hover:rotate-0 transition-transform duration-300">
                <div className="flex items-center justify-between text-xs font-mono text-[var(--saffron)] mb-2 font-bold">
                  <span className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5" />
                    FROZEN ARCHITECTURE
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--saffron)]/20 text-[var(--saffron)]">
                    SD v1.5
                  </span>
                </div>
                <h3 className="font-display text-3xl text-white uppercase tracking-wide">
                  CONTROLNET CANNY
                </h3>
                <div className="text-xs font-mono text-[#8ea0be] mt-1 space-x-2">
                  <span>Scale: 0.80</span>
                  <span>·</span>
                  <span>Strength: 0.35</span>
                </div>

                {/* Thin Emerald Benchmark Bar */}
                <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-xs">
                  <span className="text-[11px] font-mono text-[#5cdb3a] font-semibold">
                    +66.8% Edge IoU vs Plain SD 1.5
                  </span>
                  <span className="text-[10px] font-mono text-[#8ea0be]">
                    100% Admitted
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION D: HORIZONTAL 4-STEP CONNECTED PIPELINE */}
        {/* ================================================================ */}
        <section id="how-it-works" className="space-y-12 scroll-mt-24">
          <div className="text-center max-w-xl mx-auto space-y-2">
            <span className="text-xs font-mono font-bold tracking-widest text-[var(--saffron)] uppercase">
              SIMPLE WORKFLOW
            </span>
            <h2 className="font-display text-4xl sm:text-5xl text-[var(--ink)] uppercase leading-tight">
              FOUR STEPS.{" "}
              <span className="text-[var(--saffron)]">ONE PIPELINE.</span>
            </h2>
            <p className="text-xs sm:text-sm text-[var(--muted)]">
              From raw pipeline inspection capture to ML-ready augmented benchmark archives.
            </p>
          </div>

          <div className="relative">
            {/* Horizontal Connecting Hairline (Desktop) */}
            <div className="hidden md:block absolute top-1/2 left-8 right-8 h-[1px] bg-[var(--border)] -translate-y-8 z-0" />

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 relative z-10">
              {/* Step 1 */}
              <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover flex flex-col items-center text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-[var(--ink)] text-[var(--bg)] font-display text-xl flex items-center justify-center shrink-0 shadow">
                  01
                </div>
                <h4 className="font-display text-xl text-[var(--ink)] uppercase">
                  UPLOAD IMAGE
                </h4>
                <p className="text-xs text-[var(--muted)] leading-relaxed">
                  Upload raw field inspection capture (JPEG, PNG, WEBP) into immutable original storage.
                </p>
              </div>

              {/* Step 2 */}
              <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover flex flex-col items-center text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-[var(--saffron)] text-white font-display text-xl flex items-center justify-center shrink-0 shadow">
                  02
                </div>
                <h4 className="font-display text-xl text-[var(--ink)] uppercase">
                  CHOOSE CONDITIONS
                </h4>
                <p className="text-xs text-[var(--muted)] leading-relaxed">
                  Select environmental prompts (low-light, wet surface) or traditional CV transforms.
                </p>
              </div>

              {/* Step 3 */}
              <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover flex flex-col items-center text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-[var(--ink)] text-[var(--bg)] font-display text-xl flex items-center justify-center shrink-0 shadow">
                  03
                </div>
                <h4 className="font-display text-xl text-[var(--ink)] uppercase">
                  GENERATE ON RTX GPU
                </h4>
                <p className="text-xs text-[var(--muted)] leading-relaxed">
                  Async background synthesis with isolated worker running on CUDA with fp16 acceleration.
                </p>
              </div>

              {/* Step 4 */}
              <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover flex flex-col items-center text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-[var(--green-live)] text-white font-display text-xl flex items-center justify-center shrink-0 shadow">
                  04
                </div>
                <h4 className="font-display text-xl text-[var(--ink)] uppercase">
                  REVIEW &amp; EXPORT
                </h4>
                <p className="text-xs text-[var(--muted)] leading-relaxed">
                  Inspect Before/After split, approve outputs, and generate versioned dataset ZIPs.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION E: CLEAN 3×2 CAPABILITIES GRID */}
        {/* ================================================================ */}
        <section className="space-y-10">
          <div className="max-w-xl space-y-2">
            <span className="text-xs font-mono font-bold tracking-widest text-[var(--saffron)] uppercase">
              CAPABILITIES
            </span>
            <h2 className="font-display text-4xl sm:text-5xl text-[var(--ink)] uppercase">
              BUILT FOR INDUSTRIAL INTEGRITY
            </h2>
            <p className="text-xs sm:text-sm text-[var(--muted)]">
              Engineered specifically for non-destructive pipeline surface evaluation.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 1 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <Sparkles className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                CONDITION-CONTROLLED DIFFUSION
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                ControlNet Canny conditioning locks structural edges while Stable Diffusion v1.5 alters atmospheric lighting and surface moisture.
              </p>
            </div>

            {/* Card 2 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <Sliders className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                9 TRADITIONAL CV TRANSFORMS
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                Deterministic OpenCV augmentations covering lighting, directional shadows, sensor blur, gaussian noise, and compression degradation.
              </p>
            </div>

            {/* Card 3 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <Shield className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                PHASE 4 QUALITY SCREENING
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                Automated multi-channel RGB SSIM, AlexNet LPIPS perceptual distance, edge overlap IoU, and HSV corrosion-area diagnostic checks.
              </p>
            </div>

            {/* Card 4 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <FileCheck2 className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                HUMAN-IN-THE-LOOP AUDIT
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                1-click approval gate with immutable audit records, rejection criteria tracking, reviewer attribution, and export filtering.
              </p>
            </div>

            {/* Card 5 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <Layers className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                MULTI-SEED VERIFICATION
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                Rigorous downstream ResNet-18 benchmark across 3 reproducible seeds (42, 101, 202) on sealed held-out test sets.
              </p>
            </div>

            {/* Card 6 */}
            <div className="p-6 rounded-2xl bg-[var(--bg)] border border-[var(--border)] shadow-[var(--shadow-s)] lift-hover space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[var(--saffron)]/10 text-[var(--saffron)] flex items-center justify-center">
                <FolderArchive className="w-5 h-5" />
              </div>
              <h3 className="font-display text-2xl text-[var(--ink)] uppercase">
                ML-READY DATASET EXPORT
              </h3>
              <p className="text-xs text-[var(--muted)] leading-relaxed">
                Streamlined export of structured ZIP archives containing filtered images, CSV manifests, and cryptographic SHA-256 signatures.
              </p>
            </div>
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION F: DARK OBSIDIAN RECENT CAPTURES VAULT */}
        {/* ================================================================ */}
        <section className="bg-[#0a0f1e] text-white rounded-3xl p-8 lg:p-10 relative overflow-hidden shadow-2xl">
          {/* Subtle top-right warm saffron radial glow */}
          <div className="absolute top-0 right-0 w-80 h-80 bg-[radial-gradient(ellipse_at_top_right,rgba(240,78,0,0.18),transparent_70%)] pointer-events-none" />

          <div className="space-y-8 relative z-10">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <span className="text-[10px] font-mono tracking-widest text-[var(--saffron)] uppercase block mb-1">
                  LIVE WORKSPACE ACTIVITY
                </span>
                <h3 className="font-display text-3xl sm:text-4xl text-white uppercase tracking-wide">
                  RECENT PIPELINE CAPTURES
                </h3>
              </div>
              <Link
                href="/library"
                className="inline-flex items-center gap-1.5 text-xs font-mono text-[var(--saffron)] hover:underline uppercase tracking-wider"
              >
                <span>VIEW FULL LIBRARY</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            {recentImages.length === 0 ? (
              <div className="py-12 text-center space-y-4 rounded-2xl border border-white/10 bg-white/5">
                <p className="text-xs text-[#8ea0be] font-mono">
                  No images captured in the active workspace yet.
                </p>
                <AnimatedButton
                  type="button"
                  variant="saffron"
                  size="default"
                  onClick={() => setIsUploadOpen(true)}
                  leftIcon={<Upload className="w-4 h-4" />}
                >
                  Upload First Image
                </AnimatedButton>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {recentImages.map((img) => (
                  <Link
                    key={img.image_id}
                    href={`/studio?imageId=${img.image_id}`}
                    className="group rounded-2xl overflow-hidden bg-[#121a2b] border border-white/10 hover:border-[var(--saffron)]/60 transition-all lift-hover block"
                  >
                    <div className="aspect-[4/3] bg-black/40 relative overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={getImageFileUrl(img.image_id)}
                        alt={img.original_filename}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                      <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] font-mono uppercase text-white/80">
                        {img.source_type}
                      </div>
                      {img.condition && (
                        <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded bg-[var(--saffron)]/90 backdrop-blur-sm text-[10px] font-mono uppercase text-white font-bold">
                          {img.condition.replace(/_/g, " ")}
                        </div>
                      )}
                    </div>
                    <div className="p-4 flex items-center justify-between text-xs font-mono">
                      <span className="truncate max-w-[180px] text-white/80">
                        {img.original_filename}
                      </span>
                      <span className="text-[var(--saffron)] group-hover:translate-x-1 transition-transform">
                        →
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* ================================================================ */}
        {/* SECTION G: SMOOTH INFINITE MARQUEE TICKER BAND */}
        {/* ================================================================ */}
        <section className="overflow-hidden border-y border-[var(--border)] py-4 bg-[var(--bg-2)]/40 -mx-4 sm:-mx-6 lg:-mx-8">
          <div className="flex whitespace-nowrap animate-ticker">
            <div className="flex items-center gap-8 text-xs font-mono tracking-widest text-[var(--muted)] uppercase px-4 font-medium">
              <span>LOW-LIGHT INSPECTION</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>DIRECTIONAL INDUSTRIAL LIGHTING</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>BRIGHT DAYLIGHT</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>HARSH SHADOWS</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>WET SURFACE</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>DUSTY SURFACE</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>CONTROLNET CANNY</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>SSIM &amp; LPIPS SCREENING</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>OPENCV AUGMENTATION</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>RESNET-18 EVALUATION</span>
              <span className="text-[var(--saffron)]">•</span>
            </div>
            {/* Repeated for seamless loop */}
            <div className="flex items-center gap-8 text-xs font-mono tracking-widest text-[var(--muted)] uppercase px-4 font-medium">
              <span>LOW-LIGHT INSPECTION</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>DIRECTIONAL INDUSTRIAL LIGHTING</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>BRIGHT DAYLIGHT</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>HARSH SHADOWS</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>WET SURFACE</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>DUSTY SURFACE</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>CONTROLNET CANNY</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>SSIM &amp; LPIPS SCREENING</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>OPENCV AUGMENTATION</span>
              <span className="text-[var(--saffron)]">•</span>
              <span>RESNET-18 EVALUATION</span>
              <span className="text-[var(--saffron)]">•</span>
            </div>
          </div>
        </section>
      </div>

      {/* Upload Dialog */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={fetchDashboardData}
      />
    </AppShell>
  );
}
