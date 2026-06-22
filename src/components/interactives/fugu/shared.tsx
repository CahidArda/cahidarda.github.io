/**
 * Shared design system for the Sakana Fugu diagrams. Defined ONCE and reused in every
 * island so the post reads as authored: the same colour per worker agent, the same
 * orchestrator glyph, the same bordered figure frame. The reader learns the cast in the
 * first diagram and reads every later one instantly.
 */
import { useEffect, useRef, useState } from 'react';

export interface Agent {
  id: string;
  name: string;
  short: string;
  provider: string;
  /** CSS custom property, theme-aware (global.css). */
  color: string;
  /** What this worker is repeatedly observed to be best at, per the technical report. */
  strength: string;
}

/** The frontier worker pool Fugu / Fugu-Ultra orchestrate (technical report, §3.2.3, §4.2). */
export const AGENTS: Agent[] = [
  {
    id: 'gpt',
    name: 'GPT-5.5',
    short: 'GPT',
    provider: 'OpenAI',
    color: 'var(--color-gpt)',
    strength: 'math, planning, agentic building',
  },
  {
    id: 'claude',
    name: 'Claude Opus 4.8',
    short: 'Opus',
    provider: 'Anthropic',
    color: 'var(--color-claude)',
    strength: 'software engineering, debugging, security',
  },
  {
    id: 'gemini',
    name: 'Gemini 3.1 Pro',
    short: 'Gemini',
    provider: 'Google',
    color: 'var(--color-gemini)',
    strength: 'science, niche factual recall',
  },
];

export const AGENT_BY_ID: Record<string, Agent> = Object.fromEntries(AGENTS.map((a) => [a.id, a]));

export const FUGU_COLOR = 'var(--color-fugu)';
export const OPEN_COLOR = 'var(--color-open)';

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

/** Drives a stepped animation; pauses when reduced-motion is on (jumps to `steps`). */
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

/** Smoothly glides a point toward `target` (rAF lerp). Snaps instantly under reduced motion.
 *  Coordinates are in the caller's SVG user space, so the dot stays pixel-aligned at any scale. */
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

/* ── Agent chip: the worker's name on its fixed colour, used in every figure ── */

export function AgentChip({
  agent,
  size = 'md',
  active = true,
  onClick,
}: {
  agent: Agent;
  size?: 'sm' | 'md';
  active?: boolean;
  onClick?: () => void;
}) {
  const pad = size === 'sm' ? 'px-1.5 py-0.5 text-[0.66rem]' : 'px-2 py-1 text-xs';
  const Tag = onClick ? 'button' : 'span';
  return (
    <Tag
      onClick={onClick}
      type={onClick ? 'button' : undefined}
      className={`inline-flex items-center gap-1.5 border-2 font-mono font-medium transition-all ${pad}`}
      style={{
        borderColor: agent.color,
        background: active ? agent.color : 'transparent',
        color: active ? 'var(--color-paper)' : agent.color,
        opacity: active ? 1 : 0.55,
      }}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: active ? 'var(--color-paper)' : agent.color }}
      />
      {agent.name}
    </Tag>
  );
}

/** The Fugu orchestrator badge — square glyph, accent colour, used as the hub. */
export function FuguBadge({ label = 'Fugu', sub }: { label?: string; sub?: string }) {
  return (
    <span
      className="inline-flex flex-col items-center justify-center border-2 px-3 py-1.5 text-center font-mono font-semibold text-paper"
      style={{ borderColor: FUGU_COLOR, background: FUGU_COLOR }}
    >
      <span className="text-sm tracking-wide">{label}</span>
      {sub && <span className="text-[0.6rem] font-normal opacity-90">{sub}</span>}
    </span>
  );
}

/* ── Inline citation: a small superscript link to a reference id in the post ── */

export function Cite({ href, n }: { href: string; n: number }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="ml-0.5 align-super font-mono text-[0.6rem] no-underline"
      style={{ color: 'var(--color-accent)' }}
    >
      [{n}]
    </a>
  );
}
