"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { BarChart3, CheckCircle2 } from "lucide-react";
import { AppShell } from "@/components/layout";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { api } from "@/lib/api";

export default function EvaluationPage() {
  const [, setHasResults] = useState<boolean>(false);

  useEffect(() => {
    let ignore = false;
    api.getBenchmarkResults().then(() => {
      if (!ignore) setHasResults(true);
    }).catch(() => {});
    return () => { ignore = true; };
  }, []);

  return (
    <AppShell pageTitle="Model Evaluation">
      <div className="max-w-4xl mx-auto py-12 text-center space-y-6">
        <div className="w-16 h-16 rounded-2xl bg-surface-base border border-border-accent flex items-center justify-center text-saffron mx-auto shadow-lg shadow-saffron/10">
          <BarChart3 className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <div className="inline-block">
            <Badge variant="saffron" dot>
              PHASE 5B MILESTONE
            </Badge>
          </div>
          <h1 className="font-heading font-bold text-3xl sm:text-4xl text-text-primary tracking-tight">
            Downstream Model Benchmark & Stress Suites
          </h1>
          <p className="text-sm text-text-muted max-w-xl mx-auto leading-relaxed">
            Multi-seed performance verification comparing Baseline A, Experiment B, and Experiment C across clean held-out tests and 6 deterministic stress suites.
          </p>
        </div>

        {/* Phase 7 Benchmark Summary Banner */}
        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle max-w-xl mx-auto text-left space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-heading font-semibold text-sm text-text-primary">
              Phase 7 Multi-Seed Verification:
            </h3>
            <span className="text-[11px] font-mono text-emerald flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> Benchmark Sealed
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle flex justify-between items-center">
              <div>
                <span className="font-semibold text-text-primary block">Experiment C (PipeAugment AI)</span>
                <span className="text-[10px] text-text-muted">Clean Test F1: 0.9667 ± 0.0000</span>
              </div>
              <Badge variant="success">Winner</Badge>
            </div>

            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle flex justify-between items-center">
              <div>
                <span className="font-medium text-text-primary block">Experiment B (Traditional Only)</span>
                <span className="text-[10px] text-text-muted">Clean Test F1: 0.9495 ± 0.0242</span>
              </div>
              <Badge variant="amber">Baseline</Badge>
            </div>

            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle flex justify-between items-center">
              <div>
                <span className="font-medium text-text-primary block">Baseline A (Originals Only)</span>
                <span className="text-[10px] text-text-muted">Clean Test F1: 0.9328 ± 0.0242</span>
              </div>
              <Badge variant="neutral">Control</Badge>
            </div>
          </div>

          <p className="text-xs text-text-muted pt-1">
            Phase 5B will display the full interactive stress-suite charts, confusion matrices, and seed breakdown tables.
          </p>
        </div>

        <div className="pt-4 flex justify-center gap-3">
          <Link href="/">
            <Button variant="primary">Return to Overview</Button>
          </Link>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
          >
            <Button variant="secondary">API Documentation</Button>
          </a>
        </div>
      </div>
    </AppShell>
  );
}
