import { useState } from 'react';
import { ROBOT, Worker, useHoverPreview, Widget } from './shared';

// Makes the abstract coefficient physical. One robot added -> ~5.6 fewer workers in the
// local labour market (Acemoglu-Restrepo 2020). The two stat tiles toggle between the
// average local market and the most-exposed markets.

type Scope = 'avg' | 'exposed';
const SCOPES: Record<Scope, { label: string; emp: string; wage: string }> = {
  avg: { label: 'Average local market', emp: '-0.20pp', wage: '-0.42%' },
  exposed: { label: 'Most-exposed markets', emp: '-0.39pp', wage: '-0.77%' },
};

const VB = { w: 380, h: 132 };

export default function RobotJobs() {
  const [sel, setSel] = useState<Scope>('avg');
  const [scope, bindScope] = useHoverPreview(sel);
  const s = SCOPES[scope];

  // 6 worker slots; 5.6 are "removed" (the 6th is 60% removed / 40% remaining).
  const slots = Array.from({ length: 6 }, (_, i) => i);

  return (
    <Widget title="One robot, 5.6 workers" kicker="Acemoglu-Restrepo (2020), US 1990-2007">
      <p className="mb-4 text-sm text-ink-soft">
        The most rigorous natural experiment we have for a machine directly substituting for labour.
        Across US commuting zones, each new industrial robot coincided with about{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>5.6 fewer workers</span> in its local labour
        market.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="One robot on the left, an arrow, then roughly 5.6 worker figures shown as removed."
      >
        {/* robot glyph */}
        <g transform="translate(6 40)">
          <rect
            x={4}
            y={16}
            width={44}
            height={40}
            rx={4}
            style={{ fill: ROBOT, opacity: 0.16, stroke: ROBOT, strokeWidth: 2 }}
          />
          <rect
            x={16}
            y={2}
            width={20}
            height={16}
            rx={3}
            style={{ fill: ROBOT, opacity: 0.16, stroke: ROBOT, strokeWidth: 2 }}
          />
          <circle cx={22} cy={10} r={2.4} style={{ fill: ROBOT }} />
          <circle cx={30} cy={10} r={2.4} style={{ fill: ROBOT }} />
          <line x1={26} y1={2} x2={26} y2={-6} style={{ stroke: ROBOT, strokeWidth: 2 }} />
          <circle cx={26} cy={-8} r={2.6} style={{ fill: ROBOT }} />
          <circle cx={16} cy={30} r={3} style={{ fill: ROBOT }} />
          <circle cx={36} cy={30} r={3} style={{ fill: ROBOT }} />
          <text
            x={26}
            y={70}
            textAnchor="middle"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
          >
            +1 robot
          </text>
        </g>

        {/* arrow */}
        <g>
          <line
            x1={70}
            y1={56}
            x2={104}
            y2={56}
            style={{ stroke: 'var(--color-line-strong)', strokeWidth: 2 }}
          />
          <path d="M104 56 l-7 -4 v8 z" style={{ fill: 'var(--color-line-strong)' }} />
        </g>

        {/* 5.6 removed workers */}
        {slots.map((i) => (
          <Worker
            key={i}
            x={116 + i * 42}
            y={36}
            s={1.35}
            color={ROBOT}
            struck
            partial={i === 5 ? 0.6 : 1}
          />
        ))}
        <text
          x={116 + 2.5 * 42 + 8}
          y={118}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
        >
          ~5.6 fewer workers
        </text>
      </svg>

      <div className="mt-4 flex flex-wrap gap-2">
        {(Object.keys(SCOPES) as Scope[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setSel(k)}
            aria-pressed={scope === k}
            {...bindScope(k)}
            className="border px-2.5 py-1 font-mono text-[0.68rem] tracking-wide transition-colors"
            style={{
              borderColor: scope === k ? 'var(--color-accent)' : 'var(--color-line)',
              color: scope === k ? 'var(--color-accent)' : 'var(--color-muted)',
            }}
          >
            {SCOPES[k].label}
          </button>
        ))}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <div className="border border-line bg-paper p-3">
          <div className="font-mono text-[0.62rem] uppercase tracking-wide text-muted">
            employment-to-pop
          </div>
          <div className="font-mono text-xl font-semibold" style={{ color: ROBOT }}>
            {s.emp}
          </div>
          <div className="font-mono text-[0.6rem] text-muted">per robot / 1,000 workers</div>
        </div>
        <div className="border border-line bg-paper p-3">
          <div className="font-mono text-[0.62rem] uppercase tracking-wide text-muted">wages</div>
          <div className="font-mono text-xl font-semibold" style={{ color: ROBOT }}>
            {s.wage}
          </div>
          <div className="font-mono text-[0.6rem] text-muted">per robot / 1,000 workers</div>
        </div>
      </div>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Concentrated by geography: Detroit was the single most-exposed commuting zone; parts of
        Texas hit 5-10 robots per 1,000 workers. Aggregate US employment moved only a small share.
      </p>
    </Widget>
  );
}
