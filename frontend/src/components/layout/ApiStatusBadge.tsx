"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { SystemInfo } from "@/lib/types";

export function ApiStatusBadge() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    let isMounted = true;
    let timerId: NodeJS.Timeout | null = null;

    const fetchStatus = async () => {
      // Don't poll when the document is hidden/backgrounded
      if (document.hidden) return;

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

    fetchStatus();
    timerId = setInterval(fetchStatus, 15000);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        fetchStatus();
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      isMounted = false;
      if (timerId) clearInterval(timerId);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  const gpu = systemInfo?.hardware?.gpu;
  const gpuLabel = gpu?.gpu_detected
    ? `${gpu.model_name || "NVIDIA GPU"} (${
        gpu.total_memory_mb ? `${gpu.total_memory_mb} MB VRAM` : "Active"
      })`
    : "CPU Pipeline Mode";

  return (
    <div
      className="relative group flex items-center gap-2 px-2.5 py-1 rounded-full bg-[var(--bg-2)] border border-[var(--border)] text-[11px] font-mono cursor-default"
      title={gpuLabel}
    >
      <span
        className={`w-2 h-2 rounded-full ${
          isHealthy === true
            ? "bg-[var(--green-live)] animate-dot-ripple"
            : isHealthy === false
            ? "bg-[var(--danger)]"
            : "bg-[var(--muted)]"
        }`}
      />
      <span className="hidden sm:inline text-[var(--muted)] font-medium">
        {isHealthy === true
          ? "API LIVE"
          : isHealthy === false
          ? "API OFFLINE"
          : "CONNECTING"}
      </span>

      {/* Tooltip on Hover */}
      <div className="absolute top-full right-0 mt-2 hidden group-hover:block z-50 whitespace-nowrap p-2.5 rounded-xl bg-[#0a0f1e] text-white text-[10px] font-mono shadow-xl border border-white/10 pointer-events-none">
        <div className="text-[var(--saffron)] font-bold mb-0.5">
          Compute Node:
        </div>
        <div>{gpuLabel}</div>
      </div>
    </div>
  );
}
