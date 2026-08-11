/**
 * Shared design system for the accountability diagrams. One colour per entity,
 * learned once: you (steel blue), the bot (violet), your tools (teal), and red
 * reserved exclusively for the liability flow. Same bordered frame and hooks as
 * the other topics so the post reads as authored.
 */
import { useEffect, useRef, useState } from 'react';

export const YOU = 'var(--color-acct-you)';
export const BOT = 'var(--color-acct-bot)';
export const TOOL = 'var(--color-acct-tool)';
export const LIAB = 'var(--color-acct-liab)';

/** Respects `prefers-reduced-motion`: widgets fall back to a static poster. */
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
 * Freezes a self-progressing animation while scrolled out of view so an
 * off-screen figure never reflows the reader's position. Attach the ref to the
 * widget frame and AND `inView` into every timer's pause gate.
 */
export function useInView<T extends Element = HTMLDivElement>() {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(true);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const obs = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), {
      threshold: 0,
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return [ref, inView] as const;
}

/* ── geometry helpers: anchor connectors to box EDGES, never centres ── */
export type Box = { cx: number; cy: number; w: number; h: number };
export const rightOf = (n: Box) => ({ x: n.cx + n.w / 2, y: n.cy });
export const leftOf = (n: Box) => ({ x: n.cx - n.w / 2, y: n.cy });
export const topOf = (n: Box) => ({ x: n.cx, y: n.cy - n.h / 2 });
export const bottomOf = (n: Box) => ({ x: n.cx, y: n.cy + n.h / 2 });

/* ── The bordered figure frame every diagram shares ── */
export function Widget({
  title,
  kicker,
  children,
  id,
  rootRef,
}: {
  title: string;
  kicker?: string;
  children: React.ReactNode;
  id?: string;
  rootRef?: React.Ref<HTMLDivElement>;
}) {
  return (
    <div ref={rootRef} id={id} className="fx not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        {kicker && <span className="font-mono text-[0.65rem] text-muted">{kicker}</span>}
      </div>
      <div className="p-4 sm:p-5">{children}</div>
    </div>
  );
}

/* ── arrow markers: one neutral, one for the liability flow ── */
export function ArrowDefs() {
  return (
    <defs>
      <marker
        id="acct-arrow"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0 L10 5 L0 10 z" style={{ fill: 'var(--color-ink-soft)' }} />
      </marker>
      <marker
        id="acct-arrow-liab"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0 L10 5 L0 10 z" style={{ fill: LIAB }} />
      </marker>
    </defs>
  );
}

export function Edge({
  from,
  to,
  on,
  dim,
}: {
  from: { x: number; y: number };
  to: { x: number; y: number };
  on?: boolean;
  dim?: boolean;
}) {
  return (
    <line
      x1={from.x}
      y1={from.y}
      x2={to.x}
      y2={to.y}
      markerEnd={on ? 'url(#acct-arrow)' : undefined}
      style={{
        stroke: on ? 'var(--color-ink-soft)' : 'var(--color-line-strong)',
        strokeWidth: on ? 2 : 1,
        strokeDasharray: on ? undefined : '3 3',
        opacity: dim ? 0.35 : 1,
        transition: 'stroke 250ms, opacity 250ms',
      }}
    />
  );
}

export type NState = 'off' | 'wait' | 'active';

/** A rect + title (+ subtitle) node with off / wait / active states. */
export function SvgNode({
  n,
  title,
  sub,
  color,
  state,
  titleSize = 12,
}: {
  n: Box;
  title: string;
  sub?: string;
  color: string;
  state: NState;
  titleSize?: number;
}) {
  const stroke = state === 'off' ? 'var(--color-line-strong)' : color;
  const fill = state === 'active' ? color : 'var(--color-paper)';
  const titleFill =
    state === 'active'
      ? 'var(--color-paper)'
      : state === 'off'
        ? 'var(--color-muted)'
        : 'var(--color-ink)';
  const subFill = state === 'active' ? 'var(--color-paper)' : 'var(--color-muted)';
  return (
    <g style={{ opacity: state === 'off' ? 0.55 : 1, transition: 'opacity 250ms' }}>
      <rect
        x={n.cx - n.w / 2}
        y={n.cy - n.h / 2}
        width={n.w}
        height={n.h}
        style={{
          fill,
          stroke,
          strokeWidth: state === 'off' ? 1.5 : 2,
          transition: 'fill 200ms, stroke 200ms',
        }}
      />
      <text
        x={n.cx}
        y={sub ? n.cy - 4 : n.cy + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{
          fill: titleFill,
          fontFamily: 'var(--font-mono)',
          fontSize: titleSize,
          fontWeight: 600,
        }}
      >
        {title}
      </text>
      {sub && (
        <text
          x={n.cx}
          y={n.cy + 11}
          textAnchor="middle"
          dominantBaseline="middle"
          style={{ fill: subFill, fontFamily: 'var(--font-mono)', fontSize: 8.5, opacity: 0.85 }}
        >
          {sub}
        </text>
      )}
    </g>
  );
}
