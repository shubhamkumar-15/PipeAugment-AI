// MIT License - Adapted from ScrollX UI (Copyright (c) Aditya Kishore)
// Restyled for PipeAugment AI with Cyber-Saffron design tokens

'use client';

import React from 'react';
import { motion, useMotionTemplate, useMotionValue } from 'motion/react';
import { Card } from '@/components/ui/Card';

export interface SpotlightCardProps extends React.ComponentProps<typeof Card> {
  spotlightColor?: string; // RGB values, e.g. "255, 90, 0"
  contentClassName?: string;
  children: React.ReactNode;
}

export function SpotlightCard({
  spotlightColor = '255, 90, 0',
  children,
  className,
  contentClassName = '',
  style,
  ...props
}: SpotlightCardProps) {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const backgroundImage = useMotionTemplate`radial-gradient(320px circle at ${mouseX}px ${mouseY}px, rgba(${spotlightColor}, 0.12), transparent 75%)`;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const { left, top } = e.currentTarget.getBoundingClientRect();
    mouseX.set(e.clientX - left);
    mouseY.set(e.clientY - top);
  };

  return (
    <Card
      className={`group relative overflow-hidden bg-surface-low border border-border-subtle hover:border-border-accent/50 rounded-xl transition-colors duration-200 ${
        className ?? ''
      }`}
      style={
        {
          '--spotlight-color': spotlightColor,
          ...style,
        } as React.CSSProperties
      }
      onMouseMove={handleMouseMove}
      {...props}
    >
      <motion.div
        className='pointer-events-none absolute -inset-px rounded-xl opacity-0 transition duration-300 group-hover:opacity-100'
        style={{ backgroundImage }}
      />
      <div className={`relative z-10 w-full p-5 ${contentClassName}`}>
        {children}
      </div>
    </Card>
  );
}
