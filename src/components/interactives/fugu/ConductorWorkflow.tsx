import { useMemo, useState } from 'react';
import { AGENT_BY_ID, FUGU_COLOR, Widget, useTick } from './shared';

interface Step {
  subtask: string;
  agent: string; // worker id, or 'fugu' for the orchestrator-as-worker (recursion)
  access: number[]; // indices of prior steps whose outputs this worker may see
}
interface Topology {
  id: string;
  label: string;
  blurb: string;
  steps: Step[];
}

// Topologies the Conductor can describe in natural language. Examples drawn from the
// Conductor paper (Fig. 2) and the Fugu technical report.
const TOPOS: Topology[] = [
  {
    id: 'chain',
    label: 'Sequential chain',
    blurb:
      'Plan, then implement — the Conductor’s own Figure 2 example (count “complete subarrays”).',
    steps: [
      {
        subtask: 'Design an efficient algorithm to count complete subarrays',
        agent: 'gemini',
        access: [],
      },
      { subtask: 'Implement that algorithm in Python', agent: 'gpt', access: [0] },
    ],
  },
  {
    id: 'bestofn',
    label: 'Best-of-N',
    blurb: 'Three independent attempts, then one worker reads them all and returns the strongest.',
    steps: [
      { subtask: 'Solve independently (attempt A)', agent: 'gpt', access: [] },
      { subtask: 'Solve independently (attempt B)', agent: 'gemini', access: [] },
      { subtask: 'Solve independently (attempt C)', agent: 'claude', access: [] },
      {
        subtask: 'Read all three attempts; return the strongest answer',
        agent: 'gpt',
        access: [0, 1, 2],
      },
    ],
  },
  {
    id: 'tree',
    label: 'Debate tree',
    blurb:
      'Two workers attempt the problem independently; a domain-matched aggregator resolves them (a Calabi–Yau invariant from Humanity’s Last Exam).',
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
      'The Conductor names itself as a worker — a sub-workflow nested inside the workflow, a new test-time-scaling axis.',
    steps: [
      { subtask: 'Build a first-pass chosen-plaintext attack', agent: 'claude', access: [] },
      {
        subtask: 'Spawn a sub-workflow to derive the differential constant',
        agent: 'fugu',
        access: [0],
      },
      { subtask: 'Assemble and run the final attack', agent: 'gpt', access: [0, 1] },
    ],
  },
];

const colorOf = (id: string) =>
  id === 'fugu' ? FUGU_COLOR : (AGENT_BY_ID[id]?.color ?? 'var(--color-line-strong)');
const shortOf = (id: string) => (id === 'fugu' ? 'Fugu' : (AGENT_BY_ID[id]?.short ?? id));
const nameOf = (id: string) => (id === 'fugu' ? 'Fugu' : (AGENT_BY_ID[id]?.name ?? id));

const NW = 80; // node width (svg units)
const NH = 40; // node height
const VBW = 360;

export default function ConductorWorkflow() {
  const [topoId, setTopoId] = useState('tree');
  const topo = TOPOS.find((t) => t.id === topoId)!;
  const exec = useTick(topo.steps.length, 1200); // execution cursor

  // Layout: x by dependency depth, y by row within that depth-column.
  const { pos, vbh } = useMemo(() => {
    const n = topo.steps.length;
    const depth = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      depth[i] = topo.steps[i].access.length
        ? 1 + Math.max(...topo.steps[i].access.map((j) => depth[j]))
        : 0;
    }
    const maxDepth = Math.max(...depth);
    const cols: number[][] = Array.from({ length: maxDepth + 1 }, () => []);
    depth.forEach((d, i) => cols[d].push(i));
    const maxCount = Math.max(...cols.map((c) => c.length));
    const rowGap = NH + 22;
    const vbh = Math.max(150, maxCount * NH + (maxCount - 1) * 22 + 36);
    const pad = NW / 2 + 14;
    const colGap = maxDepth > 0 ? (VBW - 2 * pad) / maxDepth : 0;
    const pos = topo.steps.map((_, i) => {
      const d = depth[i];
      const col = cols[d];
      const row = col.indexOf(i);
      const x = maxDepth > 0 ? pad + d * colGap : VBW / 2;
      const y = vbh / 2 + (row - (col.length - 1) / 2) * rowGap;
      return { x, y };
    });
    return { pos, vbh };
  }, [topo]);

  return (
    <Widget title="Fugu-Ultra — the Conductor’s workflow" kicker="three lists become a topology">
      <p className="mb-3 text-sm text-ink-soft">
        Fugu-Ultra emits a whole workflow as natural language: a list of <em>subtasks</em>, the{' '}
        <em>worker</em> for each, and an <em>access list</em> naming which earlier outputs that
        worker may read. Those access lists <strong>are</strong> the topology — arrows show who
        reads whom.
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

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        {/* Topology graph (one SVG coordinate system) */}
        <div className="border border-line bg-paper p-2">
          <svg
            viewBox={`0 0 ${VBW} ${vbh}`}
            className="w-full"
            role="img"
            aria-label={`${topo.label} topology`}
          >
            <defs>
              <marker
                id="cw-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6.5"
                markerHeight="6.5"
                orient="auto-start-reverse"
              >
                <path d="M0 0 L10 5 L0 10 z" style={{ fill: 'var(--color-accent)' }} />
              </marker>
            </defs>

            {topo.steps.map((s, i) =>
              s.access.map((j) => {
                const a = pos[j];
                const b = pos[i];
                const on = exec >= i;
                return (
                  <line
                    key={`${i}-${j}`}
                    x1={a.x + NW / 2}
                    y1={a.y}
                    x2={b.x - NW / 2}
                    y2={b.y}
                    markerEnd={on ? 'url(#cw-arrow)' : undefined}
                    style={{
                      stroke: on ? 'var(--color-accent)' : 'var(--color-line-strong)',
                      strokeWidth: on ? 2 : 1.2,
                      strokeDasharray: on ? undefined : '3 3',
                      transition: 'stroke 250ms',
                    }}
                  />
                );
              }),
            )}

            {topo.steps.map((s, i) => {
              const p = pos[i];
              const on = exec >= i;
              const isFinal = i === topo.steps.length - 1;
              const c = colorOf(s.agent);
              return (
                <g key={i}>
                  <rect
                    x={p.x - NW / 2}
                    y={p.y - NH / 2}
                    width={NW}
                    height={NH}
                    style={{
                      fill: on ? c : 'var(--color-paper)',
                      stroke: c,
                      strokeWidth: on ? 2.5 : 1.5,
                      transition: 'fill 200ms',
                    }}
                  />
                  <text
                    x={p.x}
                    y={p.y - 4}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    style={{
                      fill: on ? 'var(--color-paper)' : 'var(--color-ink)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 13,
                      fontWeight: 600,
                    }}
                  >
                    {shortOf(s.agent)}
                  </text>
                  <text
                    x={p.x}
                    y={p.y + 10}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    style={{
                      fill: on ? 'var(--color-paper)' : 'var(--color-muted)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 8.5,
                      opacity: 0.9,
                    }}
                  >
                    step {i}
                    {isFinal ? ' · out' : ''}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* The execution trace, one row per step */}
        <div className="flex flex-col gap-1.5">
          {topo.steps.map((s, i) => {
            const on = exec >= i;
            const c = colorOf(s.agent);
            return (
              <div
                key={i}
                className="border-l-2 py-1 pl-2.5 transition-opacity"
                style={{ borderColor: c, opacity: on ? 1 : 0.4 }}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[0.7rem] font-semibold" style={{ color: c }}>
                    {nameOf(s.agent)}
                  </span>
                  <span className="font-mono text-[0.6rem] text-muted">
                    step {i}
                    {s.access.length > 0 ? ` · reads step ${s.access.join(', ')}` : ' · no inputs'}
                  </span>
                </div>
                <p className="text-[0.72rem] leading-snug text-ink-soft">{s.subtask}</p>
              </div>
            );
          })}
        </div>
      </div>

      <p className="mt-3 text-sm text-ink-soft">{topo.blurb}</p>
    </Widget>
  );
}
