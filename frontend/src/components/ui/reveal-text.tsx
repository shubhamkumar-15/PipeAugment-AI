// MIT License - Adapted from ScrollX UI (Copyright (c) Aditya Kishore)
// Restyled for PipeAugment AI with Cyber-Saffron tokens and prefers-reduced-motion compliance

'use client';

import React, { useRef, useEffect } from 'react';
import { motion, useAnimation, useInView, useReducedMotion } from 'motion/react';
import { cn } from '@/lib/utils';

type Direction = 'up' | 'down' | 'left' | 'right';
type Mode = 'manual' | 'auto';

export interface RevealTextProps {
  children: React.ReactNode;
  className?: string;
  boxClassName?: string;
  delay?: number;
  duration?: number;
  direction?: Direction;
  mode?: Mode;
  stagger?: number;
  once?: boolean;
}

const baseBoxStyles = 'absolute inset-0 z-10 bg-surface-base border-r-2 border-saffron';

export const RevealText: React.FC<RevealTextProps> = ({
  children,
  className = '',
  boxClassName = '',
  delay = 0,
  duration = 0.6,
  direction = 'down',
  mode = 'manual',
  stagger = 0.08,
  once = true,
}) => {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once });
  const controls = useAnimation();
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    if (shouldReduceMotion) return;
    if (inView) {
      controls.set('initial');
      controls.start('animate');
    } else if (!once) {
      controls.start('initial');
    }
  }, [inView, controls, once, shouldReduceMotion]);

  // When prefers-reduced-motion is enabled, bypass animation entirely
  if (shouldReduceMotion) {
    return (
      <span ref={ref} className={cn('inline-block', className)}>
        {children}
      </span>
    );
  }

  const getAnimationValues = () => {
    switch (direction) {
      case 'up':
        return {
          initial: { scaleY: 1, originY: 0 },
          animate: { scaleY: 0 },
        };
      case 'down':
        return {
          initial: { scaleY: 1, originY: 1 },
          animate: { scaleY: 0 },
        };
      case 'left':
        return {
          initial: { scaleX: 1, originX: 0 },
          animate: { scaleX: 0 },
        };
      case 'right':
        return {
          initial: { scaleX: 1, originX: 1 },
          animate: { scaleX: 0 },
        };
    }
  };

  const animationValues = getAnimationValues();

  const renderWord = (word: string, i: number) => (
    <span key={i} className='relative inline-block overflow-hidden mr-2'>
      <motion.span
        variants={{
          initial: animationValues.initial,
          animate: animationValues.animate,
        }}
        initial='initial'
        animate={controls}
        transition={{
          delay: delay + i * stagger,
          duration,
          ease: [0.76, 0, 0.24, 1],
        }}
        className={cn(baseBoxStyles, boxClassName)}
      />

      <motion.span
        variants={{
          initial: { opacity: 0 },
          animate: { opacity: 1 },
        }}
        initial='initial'
        animate={controls}
        transition={{
          delay: delay + i * stagger + duration * 0.4,
          duration: duration * 0.4,
        }}
        className={className}
      >
        {word}
      </motion.span>
    </span>
  );

  if (mode === 'auto' && typeof children === 'string') {
    const words = children.split(' ');
    return (
      <span ref={ref} className='inline-block'>
        {words.map(renderWord)}
      </span>
    );
  }

  return (
    <span ref={ref} className='relative inline-block overflow-hidden'>
      <motion.span
        variants={{
          initial: animationValues.initial,
          animate: animationValues.animate,
        }}
        initial='initial'
        animate={controls}
        transition={{
          delay,
          duration,
          ease: [0.76, 0, 0.24, 1],
        }}
        className={cn(baseBoxStyles, boxClassName)}
      />

      <motion.span
        variants={{
          initial: { opacity: 0 },
          animate: { opacity: 1 },
        }}
        initial='initial'
        animate={controls}
        transition={{
          delay: delay + duration * 0.4,
          duration: duration * 0.4,
        }}
        className={className}
      >
        {children}
      </motion.span>
    </span>
  );
};
