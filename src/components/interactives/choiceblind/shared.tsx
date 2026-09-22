/**
 * Shared design system for the choice-blindness post. Defined ONCE and reused in every
 * island so the post reads as authored: the same colour per outcome (defended the swap /
 * caught it), the same bordered figure frame, the same bar-with-CI primitive.
 * Data all comes from results/summary.md of the choice-blindness-llms experiment.
 */
import { useEffect, useRef, useState } from 'react';

/* ── The cast: one fixed colour per outcome (global.css, theme-aware) ── */
export const CONFAB = 'var(--color-cb-confab)'; // defended the swapped answer
export const DETECT = 'var(--color-cb-detect)'; // caught the swap
export const ACCENT = 'var(--color-accent)';

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

/** Steps a phase counter 0..steps, looping. Jumps straight to `steps` under reduced motion. */
export function useTick(steps: number, ms: number, run = true): number {
  const reduced = useReducedMotion();
  const [t, setT] = useState(0);
  useEffect(() => {
    if (reduced || !run) {
      setT(steps);
      return;
    }
    setT(0);
    const id = setInterval(() => setT((x) => (x >= steps ? 0 : x + 1)), ms);
    return () => clearInterval(id);
  }, [steps, ms, run, reduced]);
  return t;
}

/* ── The bordered figure frame every diagram shares (matches fugu / dataplat) ── */

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

/* ── Outcome legend chip ── */

export function OutcomeChip({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[0.66rem]">
      <span className="inline-block h-2.5 w-2.5" style={{ background: color }} />
      <span style={{ color }}>{label}</span>
    </span>
  );
}

/* ── A horizontal rate bar with a Wilson 95% CI whisker. One coordinate system: the
      track is a % scale 0..100, the bar fills to `rate`, the whisker spans [lo, hi].
      Pure HTML/%, so it reflows on mobile and never drifts (skill §4.1). ── */

export interface RateRow {
  label: string;
  sub?: string;
  n: number;
  rate: number; // 0..1
  ci: [number, number]; // 0..1
  color: string;
  emphasize?: boolean;
}

export function RateBars({ rows }: { rows: RateRow[] }) {
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map((r) => {
        const pct = r.rate * 100;
        const lo = r.ci[0] * 100;
        const hi = r.ci[1] * 100;
        return (
          <div key={r.label} className="flex items-center gap-3">
            <span className="w-[8.5rem] shrink-0 leading-tight">
              <span
                className="block font-mono text-[0.72rem]"
                style={{
                  color: r.emphasize ? r.color : 'var(--color-ink)',
                  fontWeight: r.emphasize ? 700 : 400,
                }}
              >
                {r.label}
              </span>
              {r.sub && <span className="block font-mono text-[0.6rem] text-muted">{r.sub}</span>}
            </span>

            <div className="relative h-6 flex-1 border border-line bg-paper">
              {/* the point-estimate bar */}
              <div
                className="absolute inset-y-0 left-0 transition-[width] duration-500 ease-out"
                style={{
                  width: `${Math.max(pct, 0.5)}%`,
                  background: r.color,
                  opacity: r.emphasize ? 1 : 0.82,
                }}
              />
              {/* the Wilson 95% CI whisker, drawn over the bar */}
              <div
                className="absolute top-1/2 h-[2px] -translate-y-1/2"
                style={{
                  left: `${lo}%`,
                  width: `${Math.max(hi - lo, 0.4)}%`,
                  background: 'var(--color-ink)',
                  opacity: 0.55,
                }}
              />
              <div
                className="absolute top-1/2 h-2.5 w-[2px] -translate-y-1/2"
                style={{ left: `${lo}%`, background: 'var(--color-ink)', opacity: 0.55 }}
              />
              <div
                className="absolute top-1/2 h-2.5 w-[2px] -translate-y-1/2"
                style={{ left: `${hi}%`, background: 'var(--color-ink)', opacity: 0.55 }}
              />
            </div>

            <span className="w-10 shrink-0 text-right font-mono text-[0.72rem] font-semibold">
              {pct.toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
