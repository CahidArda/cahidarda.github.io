import { useMemo, useState } from 'react';
import { AGENT_BY_ID, FUGU_COLOR, Widget, useTick } from './shared';

interface Step {
  subtask: string;
  agent: string; // worker id, or 'fugu' for the orchestrator-as-worker (recursion)
  access: number[]; // indices of prior steps whose outputs are visible
}
interface Topology {
  id: string;
  label: string;
  blurb: string;
  steps: Step[];
}

// Topologies the Conductor can describe in natural language, with examples drawn from
// the Conductor paper (Fig. 2) and the Fugu technical report (§4.4).
const TOPOS: Topology[] = [
  {
    id: 'chain',
    label: 'Sequential chain',
    blurb:
      'Plan, then implement. The Conductor’s own Figure 2 example: count “complete subarrays”.',
    steps: [
      {
        subtask: 'Develop an efficient algorithm to count complete subarrays',
        agent: 'gemini',
        access: [],
      },
      { subtask: 'Implement the algorithm in Python', agent: 'gpt', access: [0] },
    ],
  },
  {
    id: 'bestofn',
    label: 'Best-of-N',
    blurb: 'Several independent attempts, then one worker reads them all and returns the best.',
    steps: [
      { subtask: 'Solve the problem independently (attempt A)', agent: 'gpt', access: [] },
      { subtask: 'Solve the problem independently (attempt B)', agent: 'gemini', access: [] },
      { subtask: 'Solve the problem independently (attempt C)', agent: 'claude', access: [] },
      {
        subtask: 'Read all attempts; return the strongest answer',
        agent: 'gpt',
        access: [0, 1, 2],
      },
    ],
  },
  {
    id: 'tree',
    label: 'Debate + aggregation tree',
    blurb:
      'Two independent attempts at the leaves; a domain-matched aggregator at the root resolves them (Calabi–Yau invariant, HLE).',
    steps: [
      { subtask: 'Attempt: derive the invariant', agent: 'gemini', access: [] },
      { subtask: 'Attempt: derive the invariant', agent: 'claude', access: [] },
      {
        subtask: 'Resolve the disagreement by re-deriving the spectral numbers; synthesize',
        agent: 'gpt',
        access: [0, 1],
      },
    ],
  },
  {
    id: 'recursive',
    label: 'Recursive',
    blurb:
      'The Conductor names itself as a worker — a sub-workflow inside the workflow, a new test-time-scaling axis.',
    steps: [
      { subtask: 'Build a first-pass chosen-plaintext attack', agent: 'claude', access: [] },
      {
        subtask: '↻ Fugu spawns a sub-workflow to derive the differential constant',
        agent: 'fugu',
        access: [0],
      },
      { subtask: 'Assemble and run the final attack', agent: 'gpt', access: [0, 1] },
    ],
  },
];

const colorOf = (id: string) =>
  id === 'fugu' ? FUGU_COLOR : (AGENT_BY_ID[id]?.color ?? 'var(--color-line-strong)');
const nameOf = (id: string) => (id === 'fugu' ? 'Fugu' : (AGENT_BY_ID[id]?.name ?? id));

export default function ConductorWorkflow() {
  const [topoId, setTopoId] = useState('tree');
  const topo = TOPOS.find((t) => t.id === topoId)!;
  const exec = useTick(topo.steps.length, 1100); // execution cursor

  // Layout: column = dependency depth, row = position within depth.
  const layout = useMemo(() => {
    const depth = topo.steps.map(() => 0);
    for (let i = 0; i < topo.steps.length; i++) {
      depth[i] = topo.steps[i].access.length
        ? 1 + Math.max(...topo.steps[i].access.map((j) => depth[j]))
        : 0;
    }
    const maxDepth = Math.max(...depth);
    const byDepth: number[][] = Array.from({ length: maxDepth + 1 }, () => []);
    depth.forEach((d, i) => byDepth[d].push(i));
    const pos = topo.steps.map((_, i) => {
      const d = depth[i];
      const col = byDepth[d];
      const row = col.indexOf(i);
      const x = maxDepth === 0 ? 50 : 12 + (d / maxDepth) * 76;
      const y = ((row + 1) / (col.length + 1)) * 100;
      return { x, y };
    });
    return { pos, depth };
  }, [topo]);

  return (
    <Widget title="Fugu-Ultra — the Conductor’s workflow" kicker="three lists become a topology">
      <p className="mb-3 text-sm text-ink-soft">
        Fugu-Ultra emits a whole agentic workflow as natural language: a list of <em>subtasks</em>,
        the <em>worker</em> for each, and an <em>access list</em> saying which earlier outputs each
        worker may see. Those three lists <strong>are</strong> the communication topology.
      </p>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {TOPOS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTopoId(t.id)}
            aria-pressed={t.id === topoId}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: t.id === topoId ? 'var(--color-accent)' : 'var(--color-line)',
              color: t.id === topoId ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.05fr_1fr]">
        {/* Topology graph */}
        <div
          className="relative border border-line bg-paper"
          style={{ aspectRatio: '4 / 3', minHeight: 200 }}
        >
          <svg
            className="absolute inset-0 h-full w-full"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {topo.steps.map((s, i) =>
              s.access.map((j) => {
                const a = layout.pos[j];
                const b = layout.pos[i];
                const active = exec >= i;
                return (
                  <line
                    key={`${i}-${j}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={active ? 'var(--color-accent)' : 'var(--color-line-strong)'}
                    strokeWidth={active ? 0.9 : 0.5}
                    vectorEffect="non-scaling-stroke"
                    style={{ transition: 'stroke 250ms' }}
                  />
                );
              }),
            )}
          </svg>
          {topo.steps.map((s, i) => {
            const p = layout.pos[i];
            const on = exec >= i;
            const isFinal = i === topo.steps.length - 1;
            return (
              <div
                key={i}
                className="absolute z-10 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center border-2 px-2 py-1 text-center transition-all"
                style={{
                  left: `${p.x}%`,
                  top: `${p.y}%`,
                  borderColor: colorOf(s.agent),
                  background: on ? colorOf(s.agent) : 'var(--color-paper)',
                  color: on ? 'var(--color-paper)' : 'var(--color-ink)',
                  transform: `translate(-50%,-50%) scale(${on ? 1 : 0.95})`,
                  opacity: on ? 1 : 0.6,
                }}
                title={s.subtask}
              >
                <span className="font-mono text-[0.7rem] font-semibold leading-tight">
                  {nameOf(s.agent)}
                </span>
                <span className="text-[0.54rem] leading-tight opacity-85">
                  step {i}
                  {isFinal ? ' · output' : ''}
                </span>
              </div>
            );
          })}
        </div>

        {/* The three lists */}
        <div className="border border-line bg-paper p-3 font-mono text-[0.68rem] leading-relaxed">
          <div className="mb-1 text-muted"># the Conductor’s output</div>
          <div>
            <span className="text-muted">model_id&nbsp;&nbsp;= [</span>
            {topo.steps.map((s, i) => (
              <span key={i} style={{ color: colorOf(s.agent), fontWeight: exec >= i ? 700 : 400 }}>
                {nameOf(s.agent).split(' ')[0]}
                {i < topo.steps.length - 1 ? ', ' : ''}
              </span>
            ))}
            <span className="text-muted">]</span>
          </div>
          <div className="mt-1.5 text-muted">
            access&nbsp;&nbsp;&nbsp;= [{topo.steps.map((s) => `[${s.access.join(',')}]`).join(', ')}
            ]
          </div>
          <div className="mt-2 flex flex-col gap-1">
            {topo.steps.map((s, i) => (
              <div
                key={i}
                className="border-l-2 pl-2 transition-opacity"
                style={{ borderColor: colorOf(s.agent), opacity: exec >= i ? 1 : 0.4 }}
              >
                <span className="text-muted">{i}:</span>{' '}
                <span className="text-ink-soft">{s.subtask}</span>
                {s.access.length > 0 && (
                  <span className="text-muted">
                    {' '}
                    ⟵ sees step{s.access.length > 1 ? 's' : ''} {s.access.join(', ')}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="mt-3 text-sm text-ink-soft">{topo.blurb}</p>
    </Widget>
  );
}
