/**
 * Shared design system for "The Logic of Crowds" (the game-theory series).
 * Every widget on every page is built from these pieces so the series reads as one
 * authored object: the same specimen frame, the same big serif readouts, the same
 * square controls, the same colour per role.
 *
 * Colour roles (theme-aware, defined in global.css):
 *   GT.a   first player / shop A / top route       (ink-blue)
 *   GT.b   second player / shop B / bottom route   (brick-red)
 *   GT.opt the social optimum, "what's best for everyone" (green)
 *   GT.eq  the equilibrium, "what selfish play produces"   (site accent)
 *   GT.warn, GT.alt extra roles (amber, violet)
 * In SVG use style={{ fill: GT.a }} (never fill="var(..)"). On <canvas> read the BASE
 * tokens with cssVar('--gt-a') (never the @theme alias).
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';

export const GT = {
  a: 'var(--color-gt-a)',
  b: 'var(--color-gt-b)',
  opt: 'var(--color-gt-opt)',
  eq: 'var(--color-accent)',
  warn: 'var(--color-gt-warn)',
  alt: 'var(--color-gt-alt)',
  aSoft: 'var(--color-gt-a-soft)',
  bSoft: 'var(--color-gt-b-soft)',
  optSoft: 'var(--color-gt-opt-soft)',
  ink: 'var(--color-ink)',
  inkSoft: 'var(--color-ink-soft)',
  muted: 'var(--color-muted)',
  line: 'var(--color-line)',
  lineStrong: 'var(--color-line-strong)',
  paper: 'var(--color-paper)',
  paperRaised: 'var(--color-paper-raised)',
  mono: 'var(--font-mono)',
  display: 'var(--font-display)',
} as const;

/** Resolve a BASE css custom property for canvas drawing (e.g. cssVar('--gt-a')). */
export function cssVar(name: string): string {
  if (typeof window === 'undefined') return '#888';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

/** Re-render when the site theme flips (canvas widgets must redraw with new colours). */
export function useThemeVersion(): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => setV((x) => x + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);
  return v;
}

/* ───────────────────────────── hooks ───────────────────────────── */

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

/** Pause self-running animations while off screen. Attach ref to the Specimen root. */
export function useInView<T extends Element = HTMLDivElement>() {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(true);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const obs = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return [ref, inView] as const;
}

/** Hover a tab to preview it, click to commit. Render with `active`. */
export function useHoverPreview<T>(selected: T) {
  const [hover, setHover] = useState<T | null>(null);
  const active = hover === null ? selected : hover;
  const bind = (value: T) => ({
    onMouseEnter: () => setHover(value),
    onMouseLeave: () => setHover(null),
  });
  return [active, bind] as const;
}

/** Calls cb(dtSeconds) every animation frame while `running`. dt is clamped to 0.05s. */
export function useFrame(cb: (dt: number) => void, running: boolean) {
  const cbRef = useRef(cb);
  cbRef.current = cb;
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      cbRef.current(dt);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);
}

/** Smoothly tweens a number toward `target` (for big readouts that tick). */
export function useTween(target: number, reduced = false, speed = 10): number {
  const [v, setV] = useState(target);
  const vRef = useRef(target);
  const tRef = useRef(target);
  tRef.current = target;
  useEffect(() => {
    if (reduced) {
      vRef.current = target;
      setV(target);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const d = tRef.current - vRef.current;
      if (Math.abs(d) < 1e-3) {
        vRef.current = tRef.current;
        setV(tRef.current);
        return;
      }
      vRef.current += d * Math.min(1, dt * speed);
      setV(vRef.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, reduced, speed]);
  return v;
}

/** Deterministic PRNG so "random" visuals are identical on every load. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Convert a pointer event to SVG user-space coordinates of the given <svg>. */
export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const m = svg.getScreenCTM();
  if (!m) return { x: 0, y: 0 };
  const p = pt.matrixTransform(m.inverse());
  return { x: p.x, y: p.y };
}

/* ─────────────────────────── the frame ─────────────────────────── */

/**
 * The specimen card every widget lives in. Header: a mono catalogue code and a name
 * (e.g. "GT-04" · "a new road"), optional kicker on the right. Body: a dot-grid stage.
 * Footer: a one-line hint on the left (what to do), optional readout on the right.
 */
export function Specimen({
  code,
  title,
  kicker,
  hint,
  footer,
  children,
  rootRef,
  stage = true,
}: {
  code: string;
  title: string;
  kicker?: ReactNode;
  hint?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  rootRef?: React.Ref<HTMLDivElement>;
  stage?: boolean;
}) {
  return (
    <div ref={rootRef} className="fx gt-anim border border-line-strong bg-paper-raised">
      <div className="flex items-baseline justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="flex min-w-0 items-baseline gap-3">
          <span className="font-mono text-[0.65rem] tracking-[0.12em] text-accent">{code}</span>
          <span className="label truncate">{title}</span>
        </span>
        {kicker && (
          <span className="shrink-0 font-mono text-[0.65rem] text-muted">{kicker}</span>
        )}
      </div>
      <div className={`p-4 sm:p-5 ${stage ? 'gt-stage' : ''}`}>{children}</div>
      {(hint || footer) && (
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-line px-4 py-2">
          <span className="font-mono text-[0.68rem] text-muted">{hint}</span>
          {footer && <span className="font-mono text-[0.68rem] text-ink-soft">{footer}</span>}
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────── readouts ─────────────────────────── */

/**
 * A big serif number with a mono label: the "killer readout". Use 2-3 side by side,
 * e.g. "selfish 80 min" vs "optimal 65 min".
 */
export function BigStat({
  label,
  value,
  unit,
  color = GT.ink,
  sub,
  size = 'md',
  align = 'left',
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  color?: string;
  sub?: ReactNode;
  size?: 'sm' | 'md' | 'lg';
  align?: 'left' | 'center' | 'right';
}) {
  const fs = size === 'lg' ? 'text-5xl sm:text-6xl' : size === 'sm' ? 'text-2xl' : 'text-4xl';
  return (
    <div
      className={`min-w-0 ${align === 'center' ? 'text-center' : align === 'right' ? 'text-right' : 'text-left'}`}
    >
      <div className="label" style={{ color }}>
        {label}
      </div>
      <div className={`font-display leading-none tabular-nums ${fs}`} style={{ color }}>
        {value}
        {unit && <span className="ml-1 font-mono text-sm text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-1 font-mono text-[0.68rem] text-muted">{sub}</div>}
    </div>
  );
}

/** Coloured square + label, for legends. */
export function Chip({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[0.68rem] text-ink-soft">
      <span className="inline-block h-2.5 w-2.5" style={{ background: color }} />
      {children}
    </span>
  );
}

/* ─────────────────────────── controls ─────────────────────────── */

/** Segmented control (mono, square). Supports hover-preview via `bind`. */
export function Seg<T extends string>({
  options,
  value,
  onChange,
  bind,
  label,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  bind?: (v: T) => { onMouseEnter: () => void; onMouseLeave: () => void };
  label?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {label && <span className="label mr-1">{label}</span>}
      <div className="inline-flex flex-wrap border border-line-strong">
        {options.map((o, i) => {
          const on = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              aria-pressed={on}
              onClick={() => onChange(o.value)}
              {...(bind ? bind(o.value) : {})}
              className={`px-2.5 py-1 font-mono text-[0.72rem] transition-colors ${
                i > 0 ? 'border-l border-line-strong' : ''
              } ${on ? 'bg-ink text-paper' : 'text-ink-soft hover:bg-paper'}`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Labelled slider with a live value. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v: number) => String(v),
  color,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => ReactNode;
  color?: string;
}) {
  return (
    <label className="block min-w-0">
      <span className="flex items-baseline justify-between gap-2">
        <span className="label" style={color ? { color } : undefined}>
          {label}
        </span>
        <span className="font-mono text-xs tabular-nums text-ink">{format(value)}</span>
      </span>
      <input
        type="range"
        className="gt-range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

/** Square mono button. `primary` = filled ink. */
export function Btn({
  children,
  onClick,
  primary,
  disabled,
  title,
}: {
  children: ReactNode;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`border px-3 py-1 font-mono text-[0.72rem] transition-colors disabled:opacity-40 ${
        primary
          ? 'border-ink bg-ink text-paper hover:bg-accent hover:border-accent'
          : 'border-line-strong text-ink-soft hover:border-ink hover:text-ink'
      }`}
    >
      {children}
    </button>
  );
}

/** One-line caption under a figure stage, for the "what you are seeing" sentence. */
export function Caption({ children }: { children: ReactNode }) {
  return <p className="mt-3 text-[0.85rem] leading-snug text-ink-soft">{children}</p>;
}

/* ─────────────────────────── svg bits ─────────────────────────── */

/** Arrowhead markers: id `gt-arrow-<key>` for each colour role. */
export function SvgArrows() {
  const entries: [string, string][] = [
    ['ink', GT.ink],
    ['muted', GT.muted],
    ['a', GT.a],
    ['b', GT.b],
    ['opt', GT.opt],
    ['eq', GT.eq],
  ];
  return (
    <defs>
      {entries.map(([k, c]) => (
        <marker
          key={k}
          id={`gt-arrow-${k}`}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M0 0 L10 5 L0 10 z" style={{ fill: c }} />
        </marker>
      ))}
    </defs>
  );
}

/** Mono SVG label (default 10px, muted). */
export function T({
  x,
  y,
  children,
  anchor = 'middle',
  size = 10,
  color = GT.muted,
  weight = 400,
  font = 'mono',
  baseline = 'middle',
}: {
  x: number;
  y: number;
  children: ReactNode;
  anchor?: 'start' | 'middle' | 'end';
  size?: number;
  color?: string;
  weight?: number;
  font?: 'mono' | 'display' | 'sans';
  baseline?: 'middle' | 'auto' | 'hanging';
}) {
  const ff = font === 'mono' ? GT.mono : font === 'display' ? GT.display : 'var(--font-sans)';
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline={baseline}
      style={{ fill: color, fontFamily: ff, fontSize: size, fontWeight: weight }}
    >
      {children}
    </text>
  );
}
