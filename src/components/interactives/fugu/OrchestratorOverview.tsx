import { useEffect, useState } from 'react';
import { AGENTS, Widget, useGlide, useReducedMotion } from './shared';

type Mode = 'fugu' | 'ultra';
type NState = 'off' | 'wait' | 'active'; // grayed · in the chain (border only) · active (filled)

// One SVG coordinate system (viewBox units) shared by every node and connector, so
// nothing can drift out of alignment. Lines anchor to box edges, never centres.
const VB = { w: 400, h: 240 };
const NODE = {
  you: { cx: 46, cy: 120, w: 76, h: 44 },
  fugu: { cx: 176, cy: 120, w: 108, h: 54 },
  gpt: { cx: 340, cy: 52, w: 112, h: 46 },
  claude: { cx: 340, cy: 120, w: 112, h: 46 },
  gemini: { cx: 340, cy: 188, w: 112, h: 46 },
} as const;
const WORKER_TITLE: Record<string, string> = {
  gpt: 'GPT-5.5',
  claude: 'Opus 4.8',
  gemini: 'Gemini 3.1',
};
const rightOf = (n: { cx: number; w: number; cy: number }) => ({ x: n.cx + n.w / 2, y: n.cy });
const leftOf = (n: { cx: number; w: number; cy: number }) => ({ x: n.cx - n.w / 2, y: n.cy });

const PHASES = 5;
const CAPTION: Record<Mode, string[]> = {
  fugu: [
    'You send one request to a single endpoint.',
    'Fugu reads the query and picks the best worker — no answer written, just a choice.',
    'The chosen frontier worker gets the task and solves it.',
    'Its result is handed back to Fugu.',
    'One answer comes back to you. You never managed a team.',
  ],
  ultra: [
    'You send one request to a single endpoint.',
    'Fugu-Ultra reads the query and designs a workflow across several workers.',
    'It dispatches the team; each tackles its own subtask.',
    'Their outputs come back to be combined and verified.',
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

  // Node state: a node is grayed until the request "reaches" it, then it joins the
  // chain (coloured border), and fills solid only while it is the active step.
  const reached = (reach: number) => phase >= reach;
  const youState: NState = phase === 0 || phase === 4 ? 'active' : 'wait';
  const fuguState: NState = phase === 1 || phase === 3 ? 'active' : reached(1) ? 'wait' : 'off';
  const workerState = (id: string): NState => {
    if (!selected.includes(id)) return 'off';
    if (phase === 2) return 'active'; // filled only while the token is AT the worker
    return reached(2) ? 'wait' : 'off';
  };

  // Token target node by phase.
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

  // Edges carry an arrow at their `to` end. On the return trip (phase 3: worker→Fugu,
  // phase 4: Fugu→You) we swap endpoints so the arrow flips to point back.
  const queryEdge =
    phase === 4
      ? { a: leftOf(NODE.fugu), b: rightOf(NODE.you), on: true }
      : { a: rightOf(NODE.you), b: leftOf(NODE.fugu), on: phase === 1 };
  const workerEdge = (id: string) => {
    const node = NODE[id as keyof typeof NODE];
    const sel = selected.includes(id);
    if (sel && phase === 3) return { a: leftOf(node), b: rightOf(NODE.fugu), on: true, dim: false };
    return { a: rightOf(NODE.fugu), b: leftOf(node), on: sel && phase === 2, dim: !sel };
  };

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

        {/* connectors (anchored to box edges, arrow points along the active direction) */}
        <Edge from={queryEdge.a} to={queryEdge.b} on={queryEdge.on} />
        {AGENTS.map((ag) => {
          const e = workerEdge(ag.id);
          return <Edge key={ag.id} from={e.a} to={e.b} on={e.on} dim={e.dim} />;
        })}

        {/* travelling token — drawn before the nodes so it passes BEHIND the boxes */}
        {!reduced && (
          <circle cx={token.x} cy={token.y} r={6} style={{ fill: 'var(--color-accent)' }} />
        )}

        {/* nodes */}
        <SvgNode
          n={NODE.you}
          title="You"
          sub="one API call"
          color="var(--color-ink-soft)"
          state={youState}
        />
        <SvgNode
          n={NODE.fugu}
          title={mode === 'ultra' ? 'Fugu-Ultra' : 'Fugu'}
          sub="orchestrator"
          color="var(--color-fugu)"
          state={fuguState}
          titleSize={13}
        />
        {AGENTS.map((ag) => (
          <SvgNode
            key={ag.id}
            n={NODE[ag.id as keyof typeof NODE]}
            title={WORKER_TITLE[ag.id]}
            sub={ag.provider}
            color={ag.color}
            state={workerState(ag.id)}
            titleSize={12}
          />
        ))}
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
  state,
  titleSize = 13,
}: {
  n: { cx: number; cy: number; w: number; h: number };
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
          style={{ fill: subFill, fontFamily: 'var(--font-mono)', fontSize: 9, opacity: 0.85 }}
        >
          {sub}
        </text>
      )}
    </g>
  );
}
