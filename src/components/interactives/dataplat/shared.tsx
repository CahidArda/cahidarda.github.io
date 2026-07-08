/**
 * Shared design system for the data-platform diagrams. Defined ONCE and reused in every
 * island so the series reads as authored: the same colour per component, the same bordered
 * figure frame, the same node/edge primitives. The reader learns the cast in the
 * architecture diagram and reads every later figure instantly.
 */
import { useEffect, useRef, useState } from 'react';

export interface Component {
  id: string;
  /** Product name shown as the node title. */
  name: string;
  /** What it is, one line, shown as the node subtitle. */
  role: string;
  /** Theme-aware CSS custom property (global.css). */
  color: string;
}

/** The platform cast: source, log, processor, and the three storage shapes. */
export const COMPONENTS: Component[] = [
  {
    id: 'producer',
    name: 'Producers',
    role: 'synthetic events',
    color: 'var(--color-dp-producer)',
  },
  { id: 'log', name: 'Redpanda', role: 'event log', color: 'var(--color-dp-log)' },
  { id: 'stream', name: 'Spark', role: 'stream processor', color: 'var(--color-dp-stream)' },
  { id: 'row', name: 'Postgres', role: 'row · OLTP', color: 'var(--color-dp-row)' },
  { id: 'col', name: 'ClickHouse', role: 'column · OLAP', color: 'var(--color-dp-col)' },
  { id: 'lake', name: 'Iceberg', role: 'lakehouse · history', color: 'var(--color-dp-lake)' },
];

export const COMPONENT_BY_ID: Record<string, Component> = Object.fromEntries(
  COMPONENTS.map((c) => [c.id, c]),
);

export type NState = 'off' | 'wait' | 'active'; // grayed · in the path (border only) · active (filled)

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
 * Freezes a widget's self-progressing animation while it is scrolled out of view, so an
 * off-screen figure can never reflow and shove the reader's position (mobile Safari has no
 * reliable scroll anchoring). Attach the returned ref to the widget's outer frame (via
 * `Widget`'s `rootRef`) and AND `inView` into each timer's pause gate. Defaults to `true`
 * so the widget animates immediately on hydration (it is on screen then, being `client:visible`).
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

/**
 * Tab preview-on-hover. A tabbed widget keeps its committed selection in `selected`
 * (set by onClick); this returns the value to actually render (`active`) which
 * follows the hovered tab while hovering and falls back to the selection on leave,
 * plus a `bind(value)` to spread onto each tab button. Clicking still commits;
 * hovering only previews. Keep `aria-pressed`/selected styling on `selected` so the
 * chosen tab stays marked while another is previewed.
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

/** Smoothly glides a point toward `target` (rAF lerp) in the caller's SVG user space. */
export function useGlide(target: { x: number; y: number }, reduced: boolean) {
  const [pos, setPos] = useState(target);
  const posRef = useRef(target);
  const targetRef = useRef(target);
  targetRef.current = target;
  useEffect(() => {
    if (reduced) {
      posRef.current = targetRef.current;
      setPos(targetRef.current);
      return;
    }
    let raf = 0;
    const tick = () => {
      const p = posRef.current;
      const t = targetRef.current;
      const nx = p.x + (t.x - p.x) * 0.18;
      const ny = p.y + (t.y - p.y) * 0.18;
      posRef.current = { x: nx, y: ny };
      setPos({ x: nx, y: ny });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduced]);
  return pos;
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
    <div
      ref={rootRef}
      id={id}
      className="fx not-prose border border-line-strong bg-paper-raised"
    >
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        {kicker && <span className="font-mono text-[0.65rem] text-muted">{kicker}</span>}
      </div>
      <div className="p-4 sm:p-5">{children}</div>
    </div>
  );
}

/* ── arrow marker def, shared by every diagram ── */
export function ArrowDefs() {
  return (
    <defs>
      <marker
        id="dp-arrow"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0 L10 5 L0 10 z" style={{ fill: 'var(--color-accent)' }} />
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
      markerEnd={on ? 'url(#dp-arrow)' : undefined}
      style={{
        stroke: on ? 'var(--color-accent)' : 'var(--color-line-strong)',
        strokeWidth: on ? 2 : 1,
        strokeDasharray: on ? undefined : '3 3',
        opacity: dim ? 0.3 : 1,
        transition: 'stroke 250ms, opacity 250ms',
      }}
    />
  );
}

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
    <g style={{ opacity: state === 'off' ? 0.5 : 1, transition: 'opacity 250ms' }}>
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
        y={sub ? n.cy - 3 : n.cy + 1}
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
          y={n.cy + 12}
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
