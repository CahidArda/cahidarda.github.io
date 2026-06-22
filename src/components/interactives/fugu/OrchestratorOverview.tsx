import { useEffect, useState } from 'react';
import { AGENTS, FuguBadge, Widget, useReducedMotion, type Agent } from './shared';

type Mode = 'fugu' | 'ultra';

// Node anchor points in %, shared by the connector SVG and the HTML chips.
const POS = {
  user: { x: 8, y: 50 },
  fugu: { x: 44, y: 50 },
  gpt: { x: 84, y: 16 },
  claude: { x: 84, y: 50 },
  gemini: { x: 84, y: 82 },
} as const;

const PHASES = 5;
const CAPTION: Record<Mode, string[]> = {
  fugu: [
    'You send one request to a single endpoint.',
    'Fugu reads the query and routes it — no text generated, just a decision.',
    'The single best-suited frontier worker is dispatched.',
    'That worker solves the task with frontier-level skill.',
    'One answer returns. You never managed a team.',
  ],
  ultra: [
    'You send one request to a single endpoint.',
    'Fugu-Ultra designs a workflow over several workers.',
    'It dispatches a team, each with a tailored subtask.',
    'Their outputs are combined and verified.',
    'One synthesized answer returns — a multi-agent system, as one model.',
  ],
};

export default function OrchestratorOverview() {
  const reduced = useReducedMotion();
  const [mode, setMode] = useState<Mode>('fugu');
  const [phase, setPhase] = useState(reduced ? 4 : 0);

  const selected: string[] = mode === 'fugu' ? ['claude'] : ['gpt', 'gemini', 'claude'];

  useEffect(() => {
    if (reduced) {
      setPhase(4);
      return;
    }
    setPhase(0);
    const id = setInterval(() => setPhase((p) => (p + 1) % PHASES), 1300);
    return () => clearInterval(id);
  }, [mode, reduced]);

  // Token position by phase.
  const tokenAt =
    phase <= 0
      ? POS.user
      : phase === 1
        ? POS.fugu
        : phase === 2
          ? POS[selected[0] as keyof typeof POS]
          : phase === 3
            ? POS.fugu
            : POS.user;

  const workerActive = (id: string) => selected.includes(id) && phase >= 2 && phase <= 3;
  const fuguActive = phase === 1 || phase === 3;

  return (
    <Widget title="How Fugu works" kicker="one interface · many minds">
      <div className="mb-3 flex gap-2">
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

      <div className="relative w-full" style={{ aspectRatio: '16 / 9', minHeight: 220 }}>
        {/* Connectors */}
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <Line a={POS.user} b={POS.fugu} active={phase === 1 || phase === 4} />
          {AGENTS.map((ag) => (
            <Line
              key={ag.id}
              a={POS.fugu}
              b={POS[ag.id as keyof typeof POS]}
              active={workerActive(ag.id)}
              dim={!selected.includes(ag.id)}
            />
          ))}
        </svg>

        {/* Travelling token */}
        {!reduced && (
          <span
            aria-hidden
            className="absolute z-10 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full"
            style={{
              left: `${tokenAt.x}%`,
              top: `${tokenAt.y}%`,
              background: 'var(--color-accent)',
              transition: 'left 500ms ease, top 500ms ease',
              boxShadow: '0 0 0 4px color-mix(in srgb, var(--color-accent) 25%, transparent)',
            }}
          />
        )}

        {/* Nodes */}
        <Node pos={POS.user} label="You" sub="one API call" />
        <div
          className="absolute z-10 -translate-x-1/2 -translate-y-1/2 transition-transform"
          style={{
            left: `${POS.fugu.x}%`,
            top: `${POS.fugu.y}%`,
            transform: `translate(-50%,-50%) scale(${fuguActive ? 1.08 : 1})`,
          }}
        >
          <FuguBadge label={mode === 'ultra' ? 'Fugu-Ultra' : 'Fugu'} sub="orchestrator" />
        </div>
        {AGENTS.map((ag) => (
          <WorkerNode
            key={ag.id}
            pos={POS[ag.id as keyof typeof POS]}
            agent={ag}
            active={workerActive(ag.id)}
            dim={!selected.includes(ag.id)}
          />
        ))}
      </div>

      <p className="mt-3 min-h-[2.4rem] text-sm text-ink-soft">
        <span className="font-mono text-xs text-muted">{phase + 1}/5 · </span>
        {CAPTION[mode][phase]}
      </p>
    </Widget>
  );
}

function Line({
  a,
  b,
  active,
  dim,
}: {
  a: { x: number; y: number };
  b: { x: number; y: number };
  active?: boolean;
  dim?: boolean;
}) {
  return (
    <line
      x1={a.x}
      y1={a.y}
      x2={b.x}
      y2={b.y}
      stroke={active ? 'var(--color-accent)' : 'var(--color-line-strong)'}
      strokeWidth={active ? 0.9 : 0.45}
      strokeDasharray={active ? undefined : '1.5 1.5'}
      opacity={dim ? 0.3 : 1}
      vectorEffect="non-scaling-stroke"
      style={{ transition: 'stroke 250ms' }}
    />
  );
}

function Node({ pos, label, sub }: { pos: { x: number; y: number }; label: string; sub: string }) {
  return (
    <div
      className="absolute z-10 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center border-2 border-line-strong bg-paper px-2.5 py-1.5 text-center"
      style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
    >
      <span className="font-mono text-xs font-semibold">{label}</span>
      <span className="text-[0.6rem] text-muted">{sub}</span>
    </div>
  );
}

function WorkerNode({
  pos,
  agent,
  active,
  dim,
}: {
  pos: { x: number; y: number };
  agent: Agent;
  active?: boolean;
  dim?: boolean;
}) {
  return (
    <div
      className="absolute z-10 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center border-2 px-2 py-1 text-center transition-all"
      style={{
        left: `${pos.x}%`,
        top: `${pos.y}%`,
        borderColor: agent.color,
        background: active ? agent.color : 'var(--color-paper)',
        color: active ? 'var(--color-paper)' : 'var(--color-ink)',
        opacity: dim ? 0.5 : 1,
        transform: `translate(-50%,-50%) scale(${active ? 1.06 : 1})`,
      }}
    >
      <span className="font-mono text-[0.72rem] font-semibold leading-tight">{agent.name}</span>
      <span className="text-[0.55rem] leading-tight opacity-80">{agent.provider}</span>
    </div>
  );
}
