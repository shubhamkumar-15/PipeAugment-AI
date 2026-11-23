// MIT License - Adapted from ScrollX UI (Copyright (c) Aditya Kishore)
// Restyled for PipeAugment AI with Cyber-Saffron tokens and no scroll-jacking

'use client';

import * as React from 'react';
import * as ScrollAreaPrimitive from '@radix-ui/react-scroll-area';
import {
  motion,
  AnimatePresence,
  useMotionValue,
  useTransform,
} from 'motion/react';
import { cn } from '@/lib/utils';

export interface ScrollAreaProProps
  extends React.ComponentProps<typeof ScrollAreaPrimitive.Root> {
  className?: string;
  children: React.ReactNode;
  autoHide?: boolean;
  showProgress?: 'horizontal' | 'vertical';
  onScrollChange?: (progress: { x: number; y: number }) => void;
}

export interface ScrollBarProProps
  extends React.ComponentProps<typeof ScrollAreaPrimitive.ScrollAreaScrollbar> {
  className?: string;
  show?: boolean;
  autoHide?: boolean;
  isScrolling?: boolean;
  orientation?: 'vertical' | 'horizontal';
}

function ScrollBarPro({
  className,
  orientation = 'vertical',
  show = true,
  autoHide = false,
  isScrolling = false,
  ...props
}: ScrollBarProProps) {
  return (
    <AnimatePresence>
      {(!autoHide || show) && (
        <motion.div
          initial={{ opacity: autoHide ? 0 : 1 }}
          animate={{ opacity: show ? 1 : 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          <ScrollAreaPrimitive.ScrollAreaScrollbar
            data-slot='scroll-area-scrollbar'
            orientation={orientation}
            className={cn(
              'flex touch-none select-none transition-colors duration-150',
              orientation === 'vertical' &&
                'h-full w-2.5 border-l border-border-subtle/20 p-0.5',
              orientation === 'horizontal' &&
                'h-2.5 flex-col border-t border-border-subtle/20 p-0.5',
              className,
            )}
            {...props}
          >
            <ScrollAreaPrimitive.ScrollAreaThumb
              data-slot='scroll-area-thumb'
              className={cn(
                'relative flex-1 rounded-full transition-colors duration-200',
                isScrolling
                  ? 'bg-saffron shadow-[0_0_8px_rgba(255,90,0,0.5)]'
                  : 'bg-white/20 hover:bg-white/40',
              )}
            />
          </ScrollAreaPrimitive.ScrollAreaScrollbar>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function ScrollAreaPro({
  className,
  children,
  autoHide = false,
  showProgress,
  onScrollChange,
  ...props
}: ScrollAreaProProps) {
  const [isScrolling, setIsScrolling] = React.useState<boolean>(false);
  const [showScrollbars, setShowScrollbars] = React.useState<boolean>(!autoHide);
  const viewportRef = React.useRef<HTMLDivElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const scrollTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  const scrollX = useMotionValue(0);
  const scrollY = useMotionValue(0);
  const progressBarScaleX = useTransform(scrollX, [0, 100], [0, 1]);
  const progressBarScaleY = useTransform(scrollY, [0, 100], [0, 1]);

  const handleScroll = React.useCallback((): void => {
    if (!viewportRef.current) return;
    const element = viewportRef.current;
    const maxScrollX = element.scrollWidth - element.clientWidth;
    const maxScrollY = element.scrollHeight - element.clientHeight;
    const progressX = maxScrollX > 0 ? (element.scrollLeft / maxScrollX) * 100 : 0;
    const progressY = maxScrollY > 0 ? (element.scrollTop / maxScrollY) * 100 : 0;

    scrollX.set(progressX);
    scrollY.set(progressY);
    onScrollChange?.({ x: progressX, y: progressY });

    if (autoHide) {
      setIsScrolling(true);
      setShowScrollbars(true);
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
      scrollTimeoutRef.current = setTimeout(() => {
        setIsScrolling(false);
        setShowScrollbars(false);
      }, 1200);
    }
  }, [autoHide, onScrollChange, scrollX, scrollY]);

  React.useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    element.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      element.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    };
  }, [handleScroll]);

  return (
    <ScrollAreaPrimitive.Root
      ref={containerRef}
      data-slot='scroll-area'
      className={cn('relative overflow-hidden', className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        ref={viewportRef}
        data-slot='scroll-area-viewport'
        className='size-full rounded-[inherit] overflow-auto'
      >
        {children}
      </ScrollAreaPrimitive.Viewport>

      <ScrollBarPro
        orientation='vertical'
        show={showScrollbars}
        autoHide={autoHide}
        isScrolling={isScrolling}
      />
      <ScrollBarPro
        orientation='horizontal'
        show={showScrollbars}
        autoHide={autoHide}
        isScrolling={isScrolling}
      />
      <ScrollAreaPrimitive.Corner />

      {/* Optional Progress Indicator - Styled in Cyber-Saffron */}
      <AnimatePresence>
        {showProgress === 'horizontal' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className='absolute bottom-0 left-0 right-0 h-1 bg-surface-high/60 pointer-events-none'
          >
            <motion.div
              className='h-full bg-gradient-to-r from-saffron to-amber origin-left shadow-[0_0_8px_rgba(255,90,0,0.5)]'
              style={{ scaleX: progressBarScaleX }}
            />
          </motion.div>
        )}
        {showProgress === 'vertical' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className='absolute top-0 bottom-0 right-0 w-1 bg-surface-high/60 pointer-events-none'
          >
            <motion.div
              className='w-full bg-gradient-to-b from-saffron to-amber origin-top shadow-[0_0_8px_rgba(255,90,0,0.5)]'
              style={{ scaleY: progressBarScaleY }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </ScrollAreaPrimitive.Root>
  );
}

export { ScrollAreaPro, ScrollBarPro };
