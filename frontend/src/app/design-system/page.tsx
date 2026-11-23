"use client";

import React, { useState } from "react";
import { AppShell } from "@/components/layout";
import {
  Button,
  Badge,
  Chip,
  SegmentedControl,
  Slider,
  Input,
  Dialog,
  Drawer,
  Skeleton,
  EmptyState,
  SpotlightCard,
  RevealText,
  Popover,
  PopoverTrigger,
  PopoverContent,
  ScrollAreaPro,
  DataTable,
  ColumnDef,
} from "@/components/ui";
import {
  Sparkles,
  Upload,
  Search,
  Layers,
  Check,
  Filter,
  Database,
  Image as ImageIcon,
  Clock,
  HardDrive,
} from "lucide-react";

interface SampleImageRecord {
  id: string;
  filename: string;
  format: string;
  dimensions: string;
  size: string;
  uploadedAt: string;
}

const sampleImageData: SampleImageRecord[] = [
  {
    id: "img_001",
    filename: "pipeline_inspection_01.png",
    format: "PNG",
    dimensions: "1920 × 1080",
    size: "2.4 MB",
    uploadedAt: "2026-11-22 14:30",
  },
  {
    id: "img_002",
    filename: "weld_joint_section_b.jpg",
    format: "JPEG",
    dimensions: "2048 × 1536",
    size: "3.1 MB",
    uploadedAt: "2026-11-22 14:35",
  },
  {
    id: "img_003",
    filename: "flange_surface_macro.webp",
    format: "WEBP",
    dimensions: "1280 × 720",
    size: "840 KB",
    uploadedAt: "2026-11-22 15:10",
  },
];

const imageColumns: ColumnDef<SampleImageRecord>[] = [
  {
    accessorKey: "filename",
    header: "File Name",
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        <ImageIcon className="w-4 h-4 text-saffron shrink-0" />
        <span className="font-mono text-xs text-white truncate max-w-[180px]">
          {row.getValue("filename")}
        </span>
      </div>
    ),
  },
  {
    accessorKey: "format",
    header: "Format",
    cell: ({ row }) => (
      <Badge variant="outline" className="text-[10px]">
        {row.getValue("format")}
      </Badge>
    ),
  },
  {
    accessorKey: "dimensions",
    header: "Dimensions",
    cell: ({ row }) => (
      <span className="font-mono text-xs text-text-muted">
        {row.getValue("dimensions")}
      </span>
    ),
  },
  {
    accessorKey: "size",
    header: "Size",
    cell: ({ row }) => (
      <span className="font-mono text-xs text-text-body">
        {row.getValue("size")}
      </span>
    ),
  },
  {
    accessorKey: "uploadedAt",
    header: "Uploaded",
    cell: ({ row }) => (
      <span className="text-xs text-text-muted flex items-center gap-1">
        <Clock className="w-3 h-3 text-text-muted" />
        {row.getValue("uploadedAt")}
      </span>
    ),
  },
];

export default function DesignSystemPage() {
  const [sliderVal, setSliderVal] = useState(35);
  const [method, setMethod] = useState<"traditional" | "generative">("generative");
  const [varCount, setVarCount] = useState<number>(4);
  const [chipSelected, setChipSelected] = useState<string>("wet");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedFormatFilter, setSelectedFormatFilter] = useState<string>("all");

  return (
    <AppShell pageTitle="Design System">
      <div className="space-y-12 pb-16">
        {/* Header Masthead */}
        <div className="border-b border-border-subtle/50 pb-6">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 rounded-full bg-saffron animate-pulse" />
            <span className="font-mono text-xs uppercase tracking-wider text-saffron">
              Design System Showcase · Cyber-Saffron Dark
            </span>
          </div>
          <RevealText className="font-heading font-semibold text-3xl sm:text-4xl text-white tracking-tight">
            Component Style Guide & Token Registry
          </RevealText>
          <p className="text-text-muted text-sm sm:text-base mt-2 max-w-3xl leading-relaxed">
            Standardized foundational components and design tokens built for PipeAugment AI.
            Follows the 85-10-5 visual hierarchy: 85% Neutral Obsidian Base, 10% Structured Elevation, and 5% Saffron Accent.
          </p>
        </div>

        {/* 1. Color Palette Tokens */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white flex items-center gap-2">
            <span>1. Core Surface & Brand Tokens</span>
            <span className="text-xs font-mono text-text-muted font-normal">
              (Tailwind v4 @theme values)
            </span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
            <div className="p-3 rounded-lg bg-canvas border border-border-subtle">
              <div className="h-10 rounded bg-canvas border border-border-subtle mb-2" />
              <div className="font-mono text-xs text-white">canvas</div>
              <div className="font-mono text-[10px] text-text-muted">#080C14</div>
            </div>
            <div className="p-3 rounded-lg bg-surface-low border border-border-subtle">
              <div className="h-10 rounded bg-surface-low border border-border-subtle mb-2" />
              <div className="font-mono text-xs text-white">surface-low</div>
              <div className="font-mono text-[10px] text-text-muted">#0D1520</div>
            </div>
            <div className="p-3 rounded-lg bg-surface-base border border-border-subtle">
              <div className="h-10 rounded bg-surface-base border border-border-subtle mb-2" />
              <div className="font-mono text-xs text-white">surface-base</div>
              <div className="font-mono text-[10px] text-text-muted">#132038</div>
            </div>
            <div className="p-3 rounded-lg bg-surface-high border border-border-subtle">
              <div className="h-10 rounded bg-surface-high border border-border-subtle mb-2" />
              <div className="font-mono text-xs text-white">surface-high</div>
              <div className="font-mono text-[10px] text-text-muted">#1A2845</div>
            </div>
            <div className="p-3 rounded-lg bg-surface-base border border-border-subtle">
              <div className="h-10 rounded bg-saffron mb-2 shadow-[0_0_12px_rgba(255,90,0,0.5)]" />
              <div className="font-mono text-xs text-saffron font-bold">saffron</div>
              <div className="font-mono text-[10px] text-text-muted">#FF5A00</div>
            </div>
            <div className="p-3 rounded-lg bg-surface-base border border-border-subtle">
              <div className="h-10 rounded bg-amber mb-2 shadow-[0_0_12px_rgba(255,155,0,0.4)]" />
              <div className="font-mono text-xs text-amber font-bold">amber</div>
              <div className="font-mono text-[10px] text-text-muted">#FF9B00</div>
            </div>
          </div>
        </section>

        {/* 2. Buttons */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white">
            2. Button Hierarchy & States
          </h2>
          <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-6">
            <div>
              <span className="text-xs font-mono uppercase text-text-muted block mb-3">
                Variants
              </span>
              <div className="flex flex-wrap items-center gap-3">
                <Button variant="primary" leftIcon={<Sparkles className="w-4 h-4" />}>
                  Primary Saffron
                </Button>
                <Button variant="secondary" leftIcon={<Upload className="w-4 h-4" />}>
                  Secondary Low
                </Button>
                <Button variant="outline">
                  Outline Neutral
                </Button>
                <Button variant="ghost">
                  Ghost Tertiary
                </Button>
                <Button variant="danger">
                  Danger Action
                </Button>
                <Button variant="primary" isLoading>
                  Generating...
                </Button>
                <Button variant="secondary" disabled>
                  Disabled
                </Button>
              </div>
            </div>

            <div>
              <span className="text-xs font-mono uppercase text-text-muted block mb-3">
                Sizes
              </span>
              <div className="flex flex-wrap items-center gap-3">
                <Button size="sm" variant="secondary">Small (sm)</Button>
                <Button size="md" variant="secondary">Medium (md)</Button>
                <Button size="lg" variant="primary" leftIcon={<Sparkles className="w-5 h-5" />}>
                  Large (lg)
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Badges, Chips & Segmented Controls */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white">
            3. Indicators, Badges & Interactive Chips
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-4">
              <span className="text-xs font-mono uppercase text-text-muted block">
                Status Badges
              </span>
              <div className="flex flex-wrap gap-2">
                <Badge variant="saffron" dot pulse>Processing</Badge>
                <Badge variant="success" dot>Ready</Badge>
                <Badge variant="amber" dot>Queued</Badge>
                <Badge variant="danger">Error</Badge>
                <Badge variant="neutral">Offline</Badge>
                <Badge variant="outline">Coming soon</Badge>
              </div>
            </div>

            <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-4">
              <span className="text-xs font-mono uppercase text-text-muted block">
                Filter Chips
              </span>
              <div className="flex flex-wrap gap-2">
                <Chip label="Dry Corrosion" active={chipSelected === "dry"} onClick={() => setChipSelected("dry")} />
                <Chip label="Subsea Wet" active={chipSelected === "wet"} onClick={() => setChipSelected("wet")} />
                <Chip label="Flange Seam" active={chipSelected === "flange"} onClick={() => setChipSelected("flange")} />
                <Chip label="Micro-Pitting" active={chipSelected === "micro"} onClick={() => setChipSelected("micro")} />
              </div>
            </div>
          </div>

          <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-4">
            <span className="text-xs font-mono uppercase text-text-muted block">
              Segmented Controls
            </span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <span className="text-xs text-text-muted block mb-2 font-medium">
                  Augmentation Technique
                </span>
                <SegmentedControl
                  value={method}
                  onChange={(v) => setMethod(v as "traditional" | "generative")}
                  options={[
                    { value: "traditional", label: "OpenCV Filters", icon: <Layers className="w-3.5 h-3.5" /> },
                    { value: "generative", label: "Generative AI", icon: <Sparkles className="w-3.5 h-3.5" /> },
                  ]}
                />
              </div>
              <div>
                <span className="text-xs text-text-muted block mb-2 font-medium">
                  Variation Count
                </span>
                <SegmentedControl
                  value={String(varCount)}
                  onChange={(v) => setVarCount(Number(v))}
                  options={[
                    { value: "1", label: "1x" },
                    { value: "2", label: "2x" },
                    { value: "4", label: "4x" },
                    { value: "8", label: "8x" },
                  ]}
                />
              </div>
            </div>
          </div>
        </section>

        {/* 4. Sliders & Form Inputs */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white">
            4. Sliders & Text Inputs
          </h2>
          <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Slider
                label="Simulation Severity"
                helperText="Controls physical corrosion perturbation depth"
                value={sliderVal}
                onChange={setSliderVal}
                min={0}
                max={100}
                unit="%"
              />
              <Slider
                label="Atmospheric Moisture"
                value={60}
                onChange={() => {}}
                min={0}
                max={100}
                unit="%"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input
                label="Dataset Manifest Target"
                placeholder="manifests/original.json"
                leftIcon={<Search className="w-4 h-4" />}
                helperText="Relative to project root"
              />
              <Input
                label="Filter Pipeline Tag"
                placeholder="e.g. section-4b"
                leftIcon={<Layers className="w-4 h-4" />}
              />
            </div>
          </div>
        </section>

        {/* 5. Dialog & Drawer */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white">
            5. Modals & Side Drawers
          </h2>
          <div className="p-6 rounded-xl bg-surface-low border border-border-subtle flex flex-wrap gap-4 items-center">
            <Button variant="secondary" onClick={() => setDialogOpen(true)}>
              Open Confirmation Dialog
            </Button>
            <Button variant="secondary" onClick={() => setDrawerOpen(true)}>
              Open Metadata Inspector Drawer
            </Button>

            <Dialog
              isOpen={dialogOpen}
              onClose={() => setDialogOpen(false)}
              title="Confirm Pipeline Run"
              description="Are you sure you want to execute batch augmentation on the selected dataset?"
            >
              <div className="space-y-4">
                <div className="p-3 rounded-lg bg-surface-high border border-border-subtle text-xs text-text-muted space-y-1">
                  <div className="flex justify-between">
                    <span>Selected Images:</span>
                    <span className="text-white font-mono">0 captures</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Target Partition:</span>
                    <span className="text-white font-mono">datasets/augmented/</span>
                  </div>
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="ghost" size="sm" onClick={() => setDialogOpen(false)}>Cancel</Button>
                  <Button variant="primary" size="sm" onClick={() => setDialogOpen(false)}>Proceed</Button>
                </div>
              </div>
            </Dialog>

            <Drawer
              isOpen={drawerOpen}
              onClose={() => setDrawerOpen(false)}
              title="Image Inspection Manifest"
              description="Metadata for selected pipeline capture"
            >
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-surface-high border border-border-subtle space-y-3 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-text-muted">Filename:</span>
                    <span className="text-white">pipeline_inspection_01.png</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Dimensions:</span>
                    <span className="text-white">1920 × 1080 px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">File Size:</span>
                    <span className="text-saffron font-semibold">2.4 MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Format:</span>
                    <span className="text-white">PNG / RGB</span>
                  </div>
                </div>

                <Button className="w-full" variant="primary" onClick={() => setDrawerOpen(false)}>
                  Close Inspector
                </Button>
              </div>
            </Drawer>
          </div>
        </section>

        {/* 6. Loading Skeletons & Empty State */}
        <section className="space-y-4">
          <h2 className="font-heading font-semibold text-xl text-white">
            6. Loading Skeletons & Empty States
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-4">
              <span className="text-xs font-mono uppercase text-text-muted block">
                Loading Shimmer Skeletons
              </span>
              <div className="flex items-center gap-3">
                <Skeleton className="w-12 h-12 rounded-full" />
                <div className="space-y-2 flex-1">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
              <Skeleton className="h-32 w-full rounded-xl" />
            </div>

            <EmptyState
              title="Start with a pipeline image"
              description="Upload an inspection image to prepare condition-controlled variations with physical corrosion preservation."
              actionLabel="Choose Image"
              onAction={() => alert("Upload action triggered")}
              hint="JPEG, PNG or WEBP · Up to 10 MB"
            />
          </div>
        </section>

        {/* 7. ScrollX UI Integrated Components (Phase B) */}
        <section className="space-y-6 pt-4 border-t border-border-subtle/50">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full bg-saffron" />
              <span className="font-mono text-xs uppercase tracking-wider text-saffron">
                Phase B Integration
              </span>
            </div>
            <h2 className="font-heading font-semibold text-2xl text-white">
              7. ScrollX UI Components (Cyber-Saffron Adapted)
            </h2>
            <p className="text-text-muted text-sm mt-1 max-w-2xl">
              Adapted from ScrollX UI registry, stripped of demo flourishes, and restyled with Cyber-Saffron tokens.
              Strictly adheres to real data values and accessibility guidelines.
            </p>
          </div>

          {/* Component A: Spotlight Card (Allowed on Overview stat cards & Dataset cards) */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-text-muted block">
              A. Spotlight Cards (Overview Stat Cards & Dataset Cards)
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Stat Card 1 */}
              <SpotlightCard>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
                    Total Uploads
                  </span>
                  <HardDrive className="w-4 h-4 text-saffron" />
                </div>
                <div className="font-heading font-semibold text-3xl text-white tracking-tight">
                  0
                </div>
                <div className="text-xs text-text-muted mt-2 flex items-center gap-1.5">
                  <Badge variant="success" dot>Ready</Badge>
                  <span>Backend storage connected</span>
                </div>
              </SpotlightCard>

              {/* Stat Card 2 - Dataset Card */}
              <SpotlightCard>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
                    Dataset Partition
                  </span>
                  <Database className="w-4 h-4 text-amber" />
                </div>
                <div className="font-heading font-semibold text-xl text-white tracking-tight">
                  original/
                </div>
                <div className="text-xs text-text-muted mt-2 flex items-center justify-between">
                  <span>0 files indexed</span>
                  <Badge variant="outline">manifests/original</Badge>
                </div>
              </SpotlightCard>

              {/* Stat Card 3 - Future Feature */}
              <SpotlightCard>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono uppercase tracking-wider text-text-muted">
                    Batch Pipeline
                  </span>
                  <Sparkles className="w-4 h-4 text-text-muted" />
                </div>
                <div className="font-heading font-semibold text-xl text-text-muted tracking-tight">
                  Generative Synthesis
                </div>
                <div className="text-xs text-text-muted mt-2 flex items-center gap-1.5">
                  <Badge variant="outline">Coming soon</Badge>
                  <span>GPU worker queue</span>
                </div>
              </SpotlightCard>
            </div>
          </div>

          {/* Component B: Popover */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-text-muted block">
              B. Accessible Popover
            </span>
            <div className="p-6 rounded-xl bg-surface-low border border-border-subtle flex items-center gap-4">
              <Popover>
                <PopoverTrigger asChild>
                  <Button variant="secondary" leftIcon={<Filter className="w-4 h-4 text-saffron" />}>
                    Filter By Format ({selectedFormatFilter.toUpperCase()})
                  </Button>
                </PopoverTrigger>
                <PopoverContent align="start" className="space-y-3">
                  <div className="font-heading text-xs font-semibold text-white uppercase tracking-wider">
                    Select Format
                  </div>
                  <div className="space-y-1.5">
                    {["all", "png", "jpeg", "webp"].map((fmt) => (
                      <button
                        key={fmt}
                        onClick={() => setSelectedFormatFilter(fmt)}
                        className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                          selectedFormatFilter === fmt
                            ? "bg-saffron/15 text-saffron border border-saffron/30"
                            : "text-text-muted hover:text-white hover:bg-surface-high"
                        }`}
                      >
                        <span>.{fmt}</span>
                        {selectedFormatFilter === fmt && <Check className="w-3.5 h-3.5" />}
                      </button>
                    ))}
                  </div>
                </PopoverContent>
              </Popover>
              <span className="text-xs text-text-muted">
                Spring scale + fade with reduced-motion fallback. No cursor-shake effects.
              </span>
            </div>
          </div>

          {/* Component C: ScrollArea Pro */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-text-muted block">
              C. ScrollArea Pro (Saffron Progress Indicator & No Scroll-Jacking)
            </span>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Vertical Scroll Area */}
              <div className="p-4 rounded-xl bg-surface-low border border-border-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono uppercase text-text-muted">
                    Vertical Event Audit Stream
                  </span>
                  <Badge variant="outline" className="text-[10px]">showProgress=&quot;vertical&quot;</Badge>
                </div>
                <ScrollAreaPro
                  className="h-44 rounded-lg bg-surface-base/50 p-3 border border-border-subtle/50"
                  showProgress="vertical"
                >
                  <div className="space-y-2 font-mono text-xs text-text-muted pr-4">
                    <div className="text-text-body">[16:04:12] FastAPI Backend online (0.1.0)</div>
                    <div>[16:04:14] Storage directory verified: datasets/original/</div>
                    <div>[16:04:18] Metadata directory verified: datasets/metadata/</div>
                    <div>[16:04:22] Max upload bounded at 10.0 MB</div>
                    <div>[16:04:30] Pillow validator active (JPEG, PNG, WEBP)</div>
                    <div>[16:04:38] SHA-256 deduplication hashing ready</div>
                    <div>[16:04:45] Automated test suite: 22 passed</div>
                    <div>[16:05:00] Next.js 16 frontend bridge connected</div>
                    <div>[16:05:10] Standby for image upload operations...</div>
                  </div>
                </ScrollAreaPro>
              </div>

              {/* Horizontal Scroll Area */}
              <div className="p-4 rounded-xl bg-surface-low border border-border-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono uppercase text-text-muted">
                    Horizontal Thumbnail Preview Tray
                  </span>
                  <Badge variant="outline" className="text-[10px]">showProgress=&quot;horizontal&quot;</Badge>
                </div>
                <ScrollAreaPro
                  className="w-full rounded-lg bg-surface-base/50 p-3 border border-border-subtle/50 whitespace-nowrap"
                  showProgress="horizontal"
                >
                  <div className="flex gap-3 pb-2">
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                      <div
                        key={i}
                        className="w-28 h-24 rounded-lg bg-surface-high border border-border-subtle shrink-0 flex flex-col items-center justify-center gap-1.5 p-2 text-center"
                      >
                        <ImageIcon className="w-5 h-5 text-saffron/70" />
                        <span className="font-mono text-[10px] text-text-muted">
                          Slot {i}
                        </span>
                        <span className="text-[9px] text-text-muted/60">
                          Empty
                        </span>
                      </div>
                    ))}
                  </div>
                </ScrollAreaPro>
              </div>
            </div>
          </div>

          {/* Component D: Table & DataTable */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-text-muted block">
              D. Data Table (TanStack Table + Cyber-Saffron Styling)
            </span>
            <div className="p-6 rounded-xl bg-surface-low border border-border-subtle space-y-4">
              <DataTable
                columns={imageColumns}
                data={sampleImageData}
                searchable
                searchPlaceholder="Filter images by filename..."
                searchColumnKey="filename"
                pagination
                pageSize={5}
                emptyMessage="No inspection images recorded in metadata."
              />
            </div>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
