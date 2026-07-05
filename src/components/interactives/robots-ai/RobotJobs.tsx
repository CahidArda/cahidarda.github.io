import { useState } from 'react';
import { ROBOT, Worker, useHoverPreview, Widget } from './shared';

// Makes the abstract coefficient physical. Within a local commuting zone, one added robot
// coincided with about six fewer workers (Acemoglu-Restrepo 2020, published JPE). The tiles
// toggle between the local commuting-zone effect (~6 workers, -0.39pp/-0.77%) and the smaller
// aggregate/national effect after inter-region spillovers (~3.3 workers, -0.20pp/-0.42%).

type Scope = 'local' | 'aggregate';
const SCOPES: Record<Scope, { label: string; emp: string; wage: string; per: string }> = {
  local: {
    label: 'Local commuting zone',
    emp: '-0.39pp',
    wage: '-0.77%',
    per: '≈6 workers / robot',
  },
  aggregate: {
    label: 'Aggregate, US',
    emp: '-0.20pp',
    wage: '-0.42%',
    per: '≈3.3 workers / robot',
  },
};

const VB = { w: 380, h: 132 };

export default function RobotJobs() {
  const [sel, setSel] = useState<Scope>('local');
  const [scope, bindScope] = useHoverPreview(sel);
  const s = SCOPES[scope];

  // The visual depicts the headline local figure: about six removed workers per robot.
  const slots = Array.from({ length: 6 }, (_, i) => i);

  return (
    <Widget title="One robot, six workers" kicker="Acemoglu-Restrepo (2020), US 1990-2007">
      <p className="mb-4 text-sm text-ink-soft">
        The most rigorous natural experiment we have for a machine directly substituting for labour.
        Within a local commuting zone, each new industrial robot coincided with about{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>six fewer workers</span> (the smaller
        national figure, after spillovers between regions, is about 3.3).
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="One robot on the left, an arrow, then six worker figures shown as removed."
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

        {/* six removed workers (local commuting-zone effect) */}
        {slots.map((i) => (
          <Worker key={i} x={116 + i * 42} y={36} s={1.35} color={ROBOT} struck />
        ))}
        <text
          x={116 + 2.5 * 42 + 8}
          y={118}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
        >
          ~6 fewer workers (local)
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
            employment-to-population ratio
          </div>
          <div className="font-mono text-xl font-semibold" style={{ color: ROBOT }}>
            {s.emp}
          </div>
          <div className="font-mono text-[0.6rem] text-muted">{s.per}</div>
        </div>
        <div className="border border-line bg-paper p-3">
          <div className="font-mono text-[0.62rem] uppercase tracking-wide text-muted">wages</div>
          <div className="font-mono text-xl font-semibold" style={{ color: ROBOT }}>
            {s.wage}
          </div>
          <div className="font-mono text-[0.6rem] text-muted">per robot / 1,000 workers</div>
        </div>
      </div>

      <p className="mt-2 font-mono text-[0.6rem] text-muted">
        pp = percentage points (the employment-to-population ratio is itself a percentage, so its
        change is in points, not percent). Both figures are per robot added per 1,000 workers.
      </p>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Concentrated by geography: Detroit was the single most-exposed commuting zone, with the
        other most-exposed zones clustered in the industrial Midwest. Aggregate US employment moved
        only a small share.
      </p>
    </Widget>
  );
}
