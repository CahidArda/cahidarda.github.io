/**
 * Shared design system for the robots-vs-AI diagrams. Defined ONCE and reused in every
 * island so the post reads as authored: two fixed hues (robots = steel blue, AI = warm
 * orange), the same bordered figure frame, the same reduced-motion and hover-preview
 * hooks. The reader learns the cast in the first chart and reads every later one instantly.
 */
import { useEffect, useState } from 'react';

/** The two waves. One colour each, everywhere. */
export const ROBOT = 'var(--color-ra-robot)';
export const AI = 'var(--color-ra-ai)';
export const WARN = 'var(--color-ra-warn)';

/** Respects `prefers-reduced-motion`: widgets fall back to instant state / static posters. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);
  return reduced;
}

/**
 * Tab/segment preview-on-hover. The committed selection lives in `selected` (set on
 * click); this returns the value to actually render (`active`, follows the hovered
 * control and falls back to the selection on leave) plus a `bind(value)` to spread onto
 * each control. Clicking commits; hovering only previews.
 */
export function useHoverPreview<T>(selected: T) {
  const [hover, setHover] = useState<T | null>(null);
  const active = hover === null ? selected : hover;
  const bind = (value: T) => ({
    onMouseEnter: () => setHover(value),
    onMouseLeave: () => setHover(null),
  });
  return [active, bind] as const;
}

/* ── The bordered figure frame every diagram shares ── */
export function Widget({
  title,
  kicker,
  children,
  id,
}: {
  title: string;
  kicker?: string;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <div id={id} className="fx not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        {kicker && <span className="font-mono text-[0.65rem] text-muted">{kicker}</span>}
      </div>
      <div className="p-4 sm:p-5">{children}</div>
    </div>
  );
}

/* A small square legend swatch + label, used under most charts. */
export function LegendDot({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[0.7rem] text-ink-soft">
      <span
        aria-hidden
        style={{ width: 10, height: 10, background: color, display: 'inline-block' }}
      />
      {children}
    </span>
  );
}

/* A simple worker silhouette (head + shoulders), sized to a box. `struck` renders it as
   a removed/greyed figure (dashed, faded); otherwise it is filled solid in `color`. */
export function Worker({
  x,
  y,
  s = 1,
  color,
  struck = false,
  partial = 1,
}: {
  x: number;
  y: number;
  s?: number;
  color: string;
  struck?: boolean;
  partial?: number;
}) {
  // head r=4, body a rounded trapezoid; drawn in a ~20x26 footprint scaled by s.
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`} style={{ opacity: struck ? 0.32 : partial }}>
      <circle
        cx={10}
        cy={6}
        r={4.2}
        style={{
          fill: struck ? 'none' : color,
          stroke: struck ? 'var(--color-muted)' : color,
          strokeWidth: struck ? 1.4 : 0,
          strokeDasharray: struck ? '2 2' : undefined,
        }}
      />
      <path
        d="M2 26 C2 16 5 13 10 13 C15 13 18 16 18 26 Z"
        style={{
          fill: struck ? 'none' : color,
          stroke: struck ? 'var(--color-muted)' : color,
          strokeWidth: struck ? 1.4 : 0,
          strokeDasharray: struck ? '2 2' : undefined,
        }}
      />
    </g>
  );
}
