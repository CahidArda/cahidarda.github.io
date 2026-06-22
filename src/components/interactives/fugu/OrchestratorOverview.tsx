import { useEffect, useState } from 'react';
import { AGENTS, Widget, useGlide, useReducedMotion } from './shared';

type Mode = 'fugu' | 'ultra';

// One SVG coordinate system (viewBox units) shared by every node and connector, so
// nothing can drift out of alignment. Each node is an axis-aligned box; lines anchor to
// box edges, never centres.
const VB = { w: 400, h: 240 };
const NODE = {
  you: { cx: 48, cy: 120, w: 76, h: 44 },
  fugu: { cx: 178, cy: 120, w: 104, h: 54 },
  gpt: { cx: 344, cy: 52, w: 100, h: 44 },
  claude: { cx: 344, cy: 120, w: 100, h: 44 },
  gemini: { cx: 344, cy: 188, w: 100, h: 44 },
} as const;
const right = (n: { cx: number; w: number; cy: number }) => ({ x: n.cx + n.w / 2, y: n.cy });
const left = (n: { cx: number; w: number; cy: number }) => ({ x: n.cx - n.w / 2, y: n.cy });

const PHASES = 5;
const CAPTION: Record<Mode, string[]> = {
  fugu: [
    'You send one request to a single endpoint.',
    'Fugu reads the query and routes it — no answer is written, just a choice.',
    'The single best-suited frontier worker is dispatched.',
    'That worker solves the task with frontier-level skill.',
    'One answer comes back. You never managed a team.',
  ],
  ultra: [
    'You send one request to a single endpoint.',
    'Fugu-Ultra designs a workflow across several workers.',
    'It dispatches a team, each with a tailored subtask.',
    'Their outputs are combined and verified.',
    'One synthesized answer comes back — a multi-agent system, as one model.',
  ],
};

export default function OrchestratorOverview() {
  const reduced = useReducedMotion();
  const [mode, setMode] = useState<Mode>('fugu');
  const [phase, setPhase] = useState(reduced ? 2 : 0);
  const selected = mode === 'fugu' ? ['claude'] : ['gpt', 'claude', 'gemini'];

  useEffect(() => {
    if (reduced) {
      setPhase(2);
      return;
    }
    setPhase(0);
    const id = setInterval(() => setPhase((p) => (p + 1) % PHASES), 1400);
    return () => clearInterval(id);
  }, [mode, reduced]);

  // The travelling token's target node, by phase.
  const tokenNode =
    phase === 0
      ? NODE.you
      : phase === 1
        ? NODE.fugu
        : phase === 2
          ? NODE[selected[0] as keyof typeof NODE]
          : phase === 3
            ? NODE.fugu
            : NODE.you;
  const token = useGlide({ x: tokenNode.cx, y: tokenNode.cy }, reduced);

  const fuguOn = phase >= 1 && phase <= 3;
  const queryLineOn = phase === 1 || phase === 4;
  const workerLineOn = (id: string) => selected.includes(id) && (phase === 2 || phase === 3);

  return (
    <Widget title="How Fugu works" kicker="one interface · many minds">
      <div className="mb-3 flex flex-wrap gap-2">
        {(['fugu', 'ultra'] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            aria-pressed={mode === m}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: mode === m ? 'var(--color-accent)' : 'var(--color-line)',
              color: mode === m ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {m === 'fugu' ? 'Fugu — one worker' : 'Fugu-Ultra — a team'}
          </button>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 300 }}
        role="img"
        aria-label="Fugu routes one request to frontier workers and returns one answer"
      >
        <defs>
          <marker
            id="fx-arrow"
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

        {/* connectors (anchored to box edges) */}
        <Edge from={right(NODE.you)} to={left(NODE.fugu)} on={queryLineOn} />
        {AGENTS.map((ag) => (
          <Edge
            key={ag.id}
            from={right(NODE.fugu)}
            to={left(NODE[ag.id as keyof typeof NODE])}
            on={workerLineOn(ag.id)}
            dim={!selected.includes(ag.id)}
          />
        ))}

        {/* nodes */}
        <SvgNode
          n={NODE.you}
          title="You"
          sub="one API call"
          color="var(--color-line-strong)"
          active={phase === 0 || phase === 4}
        />
        <SvgNode
          n={NODE.fugu}
          title={mode === 'ultra' ? 'Fugu-Ultra' : 'Fugu'}
          sub="orchestrator"
          color="var(--color-fugu)"
          active={fuguOn}
          filledWhenActive
        />
        {AGENTS.map((ag) => (
          <SvgNode
            key={ag.id}
            n={NODE[ag.id as keyof typeof NODE]}
            title={ag.name}
            sub={ag.provider}
            color={ag.color}
            active={selected.includes(ag.id) && (phase === 2 || phase === 3)}
            dim={!selected.includes(ag.id)}
            filledWhenActive
          />
        ))}

        {/* travelling token */}
        {!reduced && (
          <circle cx={token.x} cy={token.y} r={6} style={{ fill: 'var(--color-accent)' }}></circle>
        )}
      </svg>

      <p className="mt-2 min-h-[2.4rem] text-sm text-ink-soft">
        <span className="font-mono text-xs text-muted">{phase + 1}/5 · </span>
        {CAPTION[mode][phase]}
      </p>
    </Widget>
  );
}

function Edge({
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
      markerEnd={on ? 'url(#fx-arrow)' : undefined}
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

function SvgNode({
  n,
  title,
  sub,
  color,
  active,
  dim,
  filledWhenActive,
}: {
  n: { cx: number; cy: number; w: number; h: number };
  title: string;
  sub?: string;
  color: string;
  active?: boolean;
  dim?: boolean;
  filledWhenActive?: boolean;
}) {
  const filled = active && filledWhenActive;
  return (
    <g style={{ opacity: dim ? 0.45 : 1, transition: 'opacity 250ms' }}>
      <rect
        x={n.cx - n.w / 2}
        y={n.cy - n.h / 2}
        width={n.w}
        height={n.h}
        style={{
          fill: filled ? color : 'var(--color-paper)',
          stroke: color,
          strokeWidth: active ? 2.5 : 1.5,
          transition: 'fill 200ms, stroke-width 200ms',
        }}
      />
      <text
        x={n.cx}
        y={sub ? n.cy - 3 : n.cy + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{
          fill: filled ? 'var(--color-paper)' : 'var(--color-ink)',
          fontFamily: 'var(--font-mono)',
          fontSize: 13,
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
          style={{
            fill: filled ? 'var(--color-paper)' : 'var(--color-muted)',
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            opacity: 0.85,
          }}
        >
          {sub}
        </text>
      )}
    </g>
  );
}
