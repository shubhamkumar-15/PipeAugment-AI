"use client";

import React, { useState, useEffect } from "react";
import { AppShell } from "@/components/layout";
import { Badge } from "@/components/ui/Badge";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { api } from "@/lib/api";
import { SystemInfo } from "@/lib/types";

export default function SettingsPage() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    let ignore = false;
    api.getSystemInfo().then((data) => {
      if (!ignore) setSystemInfo(data);
    }).catch(() => {});
    return () => { ignore = true; };
  }, []);

  return (
    <AppShell pageTitle="Settings">
      <div className="max-w-4xl mx-auto py-8 space-y-8">
        <div className="flex items-center justify-between pb-6 border-b border-border-subtle/50">
          <div className="space-y-1">
            <h1 className="font-heading font-bold text-2xl sm:text-3xl text-text-primary tracking-tight">
              System Settings & Environment
            </h1>
            <p className="text-xs sm:text-sm text-text-muted">
              Compute hardware telemetry, dual-environment boundaries, and user interface configuration
            </p>
          </div>
          <ThemeToggle />
        </div>

        {/* Section: UI Theme & Appearance */}
        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-heading font-semibold text-base text-text-primary">
                Appearance & Theme
              </h3>
              <p className="text-xs text-text-muted">
                Switch between Cyber-Saffron Dark Obsidian and Industrial Slate Light Mode
              </p>
            </div>
            <ThemeToggle />
          </div>
        </div>

        {/* Section: Compute Node Telemetry */}
        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-heading font-semibold text-base text-text-primary">
                Compute Node & Hardware
              </h3>
              <p className="text-xs text-text-muted">
                FastAPI backend hardware inspection via nvidia-smi
              </p>
            </div>
            <Badge variant="saffron" dot>
              {systemInfo?.hardware?.gpu?.gpu_detected ? "CUDA Available" : "CPU Fallback"}
            </Badge>
          </div>

          <div className="divide-y divide-border-subtle/50 rounded-xl bg-surface-base/60 border border-border-subtle overflow-hidden text-xs">
            <div className="p-3 flex justify-between">
              <span className="text-text-muted">GPU Device</span>
              <span className="font-mono text-text-primary font-semibold">
                {systemInfo?.hardware?.gpu?.model_name || "CPU Pipeline"}
              </span>
            </div>
            <div className="p-3 flex justify-between">
              <span className="text-text-muted">Dedicated VRAM</span>
              <span className="font-mono text-text-primary">
                {systemInfo?.hardware?.gpu?.total_memory_mb
                  ? `${systemInfo.hardware.gpu.total_memory_mb} MB`
                  : "N/A"}
              </span>
            </div>
            <div className="p-3 flex justify-between">
              <span className="text-text-muted">NVIDIA Driver</span>
              <span className="font-mono text-text-primary">
                {systemInfo?.hardware?.gpu?.driver_version || "System"}
              </span>
            </div>
            <div className="p-3 flex justify-between">
              <span className="text-text-muted">AI Worker Subprocess</span>
              <span className="font-mono text-text-primary truncate max-w-sm">
                {systemInfo?.ai_worker?.python_executable || ".venv-ai/Scripts/python.exe"}
              </span>
            </div>
            <div className="p-3 flex justify-between">
              <span className="text-text-muted">Max Upload Limit</span>
              <span className="font-mono text-text-primary">
                {systemInfo?.upload_limits?.max_upload_bytes
                  ? `${systemInfo.upload_limits.max_upload_bytes / (1024 * 1024)} MB`
                  : "10 MB"}
              </span>
            </div>
          </div>
        </div>

        {/* Section: Frozen Generative Configuration */}
        <div className="p-6 rounded-2xl bg-surface-low border border-border-subtle space-y-4">
          <div>
            <h3 className="font-heading font-semibold text-base text-text-primary">
              Frozen Winning Generative Architecture
            </h3>
            <p className="text-xs text-text-muted">
              Frozen parameters from Phase 3 & 4 research milestone
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase block">Base Diffusion</span>
              <span className="font-bold text-text-primary">SD v1.5</span>
            </div>
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase block">Conditioning</span>
              <span className="font-bold text-saffron">Canny v11p</span>
            </div>
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase block">Control Scale</span>
              <span className="font-bold text-text-primary">0.80</span>
            </div>
            <div className="p-3 rounded-xl bg-surface-base/60 border border-border-subtle">
              <span className="text-[10px] text-text-muted uppercase block">Inference Steps</span>
              <span className="font-bold text-text-primary">30 Steps</span>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
