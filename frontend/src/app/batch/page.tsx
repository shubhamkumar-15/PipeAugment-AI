"use client";

import React from "react";
import Link from "next/link";
import { Cpu, Layers } from "lucide-react";
import { AppShell } from "@/components/layout";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

export default function BatchJobsPage() {
  return (
    <AppShell pageTitle="Batch Jobs">
      <div className="max-w-4xl mx-auto py-12 text-center space-y-6">
        <div className="w-16 h-16 rounded-2xl bg-surface-base border border-border-accent flex items-center justify-center text-saffron mx-auto shadow-lg shadow-saffron/10">
          <Cpu className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <div className="inline-block">
            <Badge variant="saffron" dot>
              PHASE 5B MILESTONE
            </Badge>
          </div>
          <h1 className="font-heading font-bold text-3xl sm:text-4xl text-text-primary tracking-tight">
            Batch Augmentation Jobs
          </h1>
          <p className="text-sm text-text-muted max-w-xl mx-auto leading-relaxed">
            Multi-parent bulk condition synthesis queue, worker pool concurrency monitoring, and batch quality screening.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle max-w-lg mx-auto text-left space-y-3">
          <h3 className="font-heading font-semibold text-sm text-text-primary">
            Scheduled for Phase 5B:
          </h3>
          <ul className="text-xs text-text-muted space-y-2 list-disc list-inside">
            <li>Bulk selection from Image Library across multiple parent images</li>
            <li>Matrix condition scheduling (6 conditions × N parents)</li>
            <li>Parallel background worker execution and retry controls</li>
            <li>Real-time telemetry and export staging</li>
          </ul>
        </div>

        <div className="pt-4 flex justify-center gap-3">
          <Link href="/studio">
            <Button variant="primary" leftIcon={<Layers className="w-4 h-4" />}>
              Open Studio (Single-Parent Mode)
            </Button>
          </Link>
          <Link href="/library">
            <Button variant="secondary">Browse Library</Button>
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
