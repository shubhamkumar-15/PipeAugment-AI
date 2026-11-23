// MIT License - Adapted from ScrollX UI (Copyright (c) Aditya Kishore)
// Restyled for PipeAugment AI with Cyber-Saffron design tokens

'use client';

import * as React from 'react';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import { motion, useReducedMotion } from 'motion/react';
import { cn } from '@/lib/utils';

function Popover({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot='popover' {...props} />;
}

function PopoverTrigger({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot='popover-trigger' {...props} />;
}

function PopoverContent({
  className,
  align = 'center',
  sideOffset = 6,
  children,
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Content>) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        data-slot='popover-content'
        align={align}
        sideOffset={sideOffset}
        asChild
        {...props}
      >
        <motion.div
          initial={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.95, y: shouldReduceMotion ? 0 : 6 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: shouldReduceMotion ? 1 : 0.95, y: shouldReduceMotion ? 0 : 4 }}
          transition={{
            duration: 0.18,
            ease: [0.16, 1, 0.3, 1],
          }}
          className={cn(
            'z-50 w-72 rounded-xl border border-border-subtle bg-surface-base p-4 text-text-primary shadow-[0_12px_36px_rgba(0,0,0,0.6)] backdrop-blur-md outline-none focus:outline-none',
            className,
          )}
        >
          {children}
        </motion.div>
      </PopoverPrimitive.Content>
    </PopoverPrimitive.Portal>
  );
}

function PopoverAnchor({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Anchor>) {
  return <PopoverPrimitive.Anchor data-slot='popover-anchor' {...props} />;
}

export { Popover, PopoverTrigger, PopoverContent, PopoverAnchor };
