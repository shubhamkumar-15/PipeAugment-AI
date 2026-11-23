"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Database } from "lucide-react";
import { AppShell } from "@/components/layout";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { api } from "@/lib/api";
import { DatasetStatsResponse } from "@/lib/types";

export default function DatasetsPage() {
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);

  useEffect(() => {
    let ignore = false;
    api.getDatasetStats().then((data) => {
      if (!ignore) setStats(data);
    }).catch(() => {});
    return () => { ignore = true; };
  }, []);

  return (
    <AppShell pageTitle="Datasets">
      <div className="max-w-4xl mx-auto py-12 text-center space-y-6">
        <div className="w-16 h-16 rounded-2xl bg-surface-base border border-border-accent flex items-center justify-center text-saffron mx-auto shadow-lg shadow-saffron/10">
          <Database className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <div className="inline-block">
            <Badge variant="saffron" dot>
              PHASE 5B MILESTONE
            </Badge>
          </div>
          <h1 className="font-heading font-bold text-3xl sm:text-4xl text-text-primary tracking-tight">
            Dataset Manifests & Exports
          </h1>
          <p className="text-sm text-text-muted max-w-xl mx-auto leading-relaxed">
            Generate versioned ZIP archives and CSV training manifests with quality filtering for downstream detector training.
          </p>
        </div>

        {/* Current Dataset Stats Banner */}
        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle max-w-xl mx-auto text-left space-y-4">
          <h3 className="font-heading font-semibold text-sm text-text-primary">
            Current Repository Status:
          </h3>
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <div className="font-bold font-heading text-xl text-text-primary">
                {stats?.total_images ?? 0}
              </div>
              <div className="text-[10px] font-mono text-text-muted">Total Images</div>
            </div>
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <div className="font-bold font-heading text-xl text-saffron">
                {stats?.total_originals ?? 0}
              </div>
              <div className="text-[10px] font-mono text-text-muted">Originals</div>
            </div>
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <div className="font-bold font-heading text-xl text-amber">
                {stats?.total_generated ?? 0}
              </div>
              <div className="text-[10px] font-mono text-text-muted">Synthetic</div>
            </div>
          </div>

          <div className="pt-2 text-xs text-text-muted">
            Phase 5B will introduce custom export filtering (Quality Passed only, Human Approved only) and one-click ZIP package generation.
          </div>
        </div>

        <div className="pt-4 flex justify-center gap-3">
          <Link href="/library">
            <Button variant="primary">
              Explore Image Library
            </Button>
          </Link>
          <Link href="/">
            <Button variant="secondary">Back to Overview</Button>
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
