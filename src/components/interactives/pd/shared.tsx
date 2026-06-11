/**
 * Shared design system for the Prisoner's-Dilemma widgets. Defined ONCE and reused in
 * every island so the post reads as authored, not assembled: the same two colours, the
 * same square glyph, and the same horizontal move-history tape appear everywhere.
 */
import { useEffect, useRef, useState } from 'react';

export type Move = 'C' | 'D';

export const MOVE_LABEL: Record<Move, string> = { C: 'Cooperate', D: 'Defect' };
/** CSS custom properties - theme-aware, defined in global.css. */
export const MOVE_COLOR: Record<Move, string> = { C: 'var(--color-c)', D: 'var(--color-d)' };
export const MOVE_COLOR_SOFT: Record<Move, string> = {
  C: 'var(--color-c-soft)',
  D: 'var(--color-d-soft)',
};

/** Respects `prefers-reduced-motion`; widgets fall back to instant state changes. */
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

/* ── The square glyph: one fixed look per move, color + letter (colour-blind safe) ── */

export function Glyph({
  move,
  size = 28,
  className = '',
  title,
}: {
  move: Move;
  size?: number;
  className?: string;
  title?: string;
}) {
  return (
    <span
      role="img"
      aria-label={title ?? MOVE_LABEL[move]}
      title={title ?? MOVE_LABEL[move]}
      className={`inline-flex items-center justify-center font-mono font-semibold text-paper select-none ${className}`}
      style={{
        width: size,
        height: size,
        background: MOVE_COLOR[move],
        fontSize: size * 0.5,
        lineHeight: 1,
      }}
    >
      {move}
    </span>
  );
}

/* ── The move-history tape: same component, every widget ── */

export function MoveTape({
  moves,
  label,
  highlightLast = false,
  cell = 22,
  emptyHint = '—',
}: {
  moves: Move[];
  label?: string;
  highlightLast?: boolean;
  cell?: number;
  emptyHint?: string;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      {label && (
        <span className="label shrink-0" style={{ width: '4.5rem' }}>
          {label}
        </span>
      )}
      <div
        className="flex min-w-0 flex-wrap gap-1"
        aria-label={`${label ?? 'moves'}: ${moves.join(' ') || 'none'}`}
      >
        {moves.length === 0 && <span className="text-muted text-sm">{emptyHint}</span>}
        {moves.map((m, i) => {
          const isLast = highlightLast && i === moves.length - 1;
          return (
            <span
              key={i}
              className={`inline-flex items-center justify-center font-mono font-semibold text-paper ${isLast ? 'pd-pop' : ''}`}
              style={{
                width: cell,
                height: cell,
                background: MOVE_COLOR[m],
                fontSize: cell * 0.48,
                outline: isLast ? '2px solid var(--color-accent)' : 'none',
                outlineOffset: 1,
              }}
              title={`Round ${i + 1}: ${MOVE_LABEL[m]}`}
            >
              {m}
            </span>
          );
        })}
      </div>
    </div>
  );
}

/* ── The two choice buttons - identical affordance wherever the reader picks a move ── */

export function ChoiceButton({
  move,
  onClick,
  disabled = false,
  selected = false,
  sub,
}: {
  move: Move;
  onClick: () => void;
  disabled?: boolean;
  selected?: boolean;
  sub?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      className="group flex flex-1 flex-col items-center gap-1.5 border-2 px-4 py-3 transition-all hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
      style={{
        borderColor: selected ? MOVE_COLOR[move] : 'var(--color-line-strong)',
        background: selected ? MOVE_COLOR_SOFT[move] : 'transparent',
      }}
    >
      <span className="flex items-center gap-2">
        <Glyph move={move} size={26} />
        <span className="font-display text-lg" style={{ color: 'var(--color-ink)' }}>
          {MOVE_LABEL[move]}
        </span>
      </span>
      {sub && (
        <span className="label normal-case" style={{ letterSpacing: '0.04em' }}>
          {sub}
        </span>
      )}
    </button>
  );
}

/* ── Layout chrome - the bordered frame + caption every widget shares ── */

export function Widget({
  title,
  children,
  id,
}: {
  title: string;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <div id={id} className="pd not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        <Legend />
      </div>
      <div className="p-4 sm:p-5">{children}</div>
    </div>
  );
}

export function Legend() {
  return (
    <span className="flex items-center gap-3 text-xs text-muted">
      <span className="flex items-center gap-1.5">
        <Glyph move="C" size={14} /> Cooperate
      </span>
      <span className="flex items-center gap-1.5">
        <Glyph move="D" size={14} /> Defect
      </span>
    </span>
  );
}

/** Animated count - ticks toward `value`. Honours reduced motion (jumps instantly). */
export function Ticker({ value, className = '' }: { value: number; className?: string }) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(value);
  const raf = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (reduced) {
      setShown(value);
      return;
    }
    const from = shown;
    const to = value;
    if (from === to) return;
    const start = performance.now();
    const dur = 260;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(Math.round(from + (to - from) * eased));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, reduced]);
  return <span className={className}>{shown}</span>;
}
