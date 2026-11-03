/**
 * Shared design system for the Tera fund-crisis diagrams. Defined ONCE and reused in every
 * island so the post reads as authored: the same colour per entity, the same bordered figure
 * frame, the same node/edge primitives. The reader learns the cast in the loop diagram and
 * reads every later figure instantly.
 *
 * The cast is the four things that chase each other round the loop:
 *   flow  = investor money moving in or out
 *   price = the share price of the thin-float stock the fund holds
 *   nav   = the fund's reported net asset value / return
 *   warn  = the reversal: default, freeze, liquidation
 */
import { useEffect, useRef, useState } from 'react';

export const FLOW = 'var(--color-tr-flow)';
export const PRICE = 'var(--color-tr-price)';
export const NAV = 'var(--color-tr-nav)';
export const WARN = 'var(--color-tr-warn)';

export type NState = 'off' | 'wait' | 'active'; // grayed · in the path (border only) · filled

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
 * Freezes a widget's self-progressing animation while it is scrolled out of view, so an
 * off-screen figure can never reflow and shove the reader's position. Attach the returned
 * ref to the widget frame via `Widget`'s `rootRef` and AND `inView` into each timer's gate.
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
 * Tab preview-on-hover. The committed selection stays in `selected` (set by onClick); this
 * returns the value to actually render, which follows the hovered tab and falls back to the
 * selection on leave, plus a `bind(value)` to spread onto each tab button.
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

/* ── geometry: anchor connectors to box EDGES, never centres ── */
export type Box = { cx: number; cy: number; w: number; h: number };
export const rightOf = (n: Box) => ({ x: n.cx + n.w / 2, y: n.cy });
export const leftOf = (n: Box) => ({ x: n.cx - n.w / 2, y: n.cy });
export const topOf = (n: Box) => ({ x: n.cx, y: n.cy - n.h / 2 });
export const bottomOf = (n: Box) => ({ x: n.cx, y: n.cy + n.h / 2 });

/* ── the bordered figure frame every diagram shares ── */
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

/* ── arrow markers. Two colours so a reversing loop can recolour its heads. ── */
export function ArrowDefs() {
  return (
    <defs>
      <marker
        id="tr-arrow"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0 L10 5 L0 10 z" style={{ fill: 'var(--color-accent)' }} />
      </marker>
      <marker
        id="tr-arrow-warn"
        viewBox="0 0 10 10"
        refX="9"
        refY="5"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0 L10 5 L0 10 z" style={{ fill: WARN }} />
      </marker>
    </defs>
  );
}

export function Edge({
  from,
  to,
  on,
  dim,
  warn,
}: {
  from: { x: number; y: number };
  to: { x: number; y: number };
  on?: boolean;
  dim?: boolean;
  warn?: boolean;
}) {
  const live = warn ? WARN : 'var(--color-accent)';
  return (
    <line
      x1={from.x}
      y1={from.y}
      x2={to.x}
      y2={to.y}
      markerEnd={on ? (warn ? 'url(#tr-arrow-warn)' : 'url(#tr-arrow)') : undefined}
      style={{
        stroke: on ? live : 'var(--color-line-strong)',
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

/** Formats a lira amount compactly for axis and chip labels. */
export function tl(n: number): string {
  if (Math.abs(n) >= 1e9) return `₺${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}mr`;
  if (Math.abs(n) >= 1e6) return `₺${(n / 1e6).toFixed(0)}mn`;
  return `₺${Math.round(n)}`;
}
