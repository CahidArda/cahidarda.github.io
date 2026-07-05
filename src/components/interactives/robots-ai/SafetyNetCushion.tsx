import { ROBOT, Widget } from './shared';

// IMF Staff Discussion Note SDN/2024/002 (Brollo, Dabla-Norris et al., 2024), Figure 1. Over US
// commuting zones in 2000-07, the wage decline from robot adoption was "about two-thirds smaller"
// in states with more generous unemployment insurance than in stingier ones. It was a binary
// high/low-UI comparison, not a continuous scatter, and it moved wages, not employment. No
// invented numbers here: the two bars only encode that real ~2/3 ratio; the axis carries no scale.

const VB = { w: 340, h: 214 };
const Y0 = 46; // zero baseline (robots arrive)
const LOW_LEN = 138; // low-UI wage decline (relative height only)
const HIGH_LEN = LOW_LEN / 3; // "about two-thirds smaller"

export default function SafetyNetCushion() {
  return (
    <Widget title="The cushion that worked" kicker="IMF SDN 2024/002 · US commuting zones, 2000-07">
      <p className="mb-3 text-sm text-ink-soft">
        Same robot shock, two kinds of state. Where unemployment insurance was more generous, the
        wage decline from robots was{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>about two-thirds smaller</span>.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two bars showing the wage decline from robots: large in low-UI states, about two-thirds smaller in high-UI states."
      >
        {/* zero baseline */}
        <line
          x1={30}
          y1={Y0}
          x2={VB.w - 14}
          y2={Y0}
          style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        <text
          x={30}
          y={Y0 - 5}
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          before robots (no change)
        </text>

        {/* low-UI bar: full decline */}
        <rect x={70} y={Y0} width={74} height={LOW_LEN} style={{ fill: ROBOT }} />
        {/* high-UI bar: about a third of it */}
        <rect x={210} y={Y0} width={74} height={HIGH_LEN} style={{ fill: ROBOT, opacity: 0.42 }} />

        {/* dashed guide from the low-UI depth across to show the gap */}
        <line
          x1={144}
          y1={Y0 + LOW_LEN}
          x2={284}
          y2={Y0 + LOW_LEN}
          style={{ stroke: 'var(--color-muted)', strokeWidth: 1, strokeDasharray: '3 3' }}
        />
        <text
          x={247}
          y={Y0 + HIGH_LEN + 16}
          textAnchor="middle"
          style={{ fill: ROBOT, fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 700 }}
        >
          ≈ two-thirds
        </text>
        <text
          x={247}
          y={Y0 + HIGH_LEN + 27}
          textAnchor="middle"
          style={{ fill: ROBOT, fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 700 }}
        >
          smaller
        </text>

        {/* bar labels */}
        {[
          { cx: 107, main: 'low-UI states', sub: 'stingier benefits' },
          { cx: 247, main: 'high-UI states', sub: 'more generous' },
        ].map((b) => (
          <g key={b.main}>
            <text
              x={b.cx}
              y={VB.h - 20}
              textAnchor="middle"
              style={{
                fill: 'var(--color-ink)',
                fontFamily: 'var(--font-mono)',
                fontSize: 9,
                fontWeight: 600,
              }}
            >
              {b.main}
            </text>
            <text
              x={b.cx}
              y={VB.h - 9}
              textAnchor="middle"
              style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
            >
              {b.sub}
            </text>
          </g>
        ))}

        {/* y-axis label (qualitative, no scale) */}
        <text
          transform={`translate(12 ${Y0 + LOW_LEN / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          wage decline from robots ↓
        </text>
      </svg>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        The cushion showed up in wages, not employment, and was strongest for workers without a
        college degree. Bar heights encode only the reported ~two-thirds ratio, not absolute
        figures.
      </p>
    </Widget>
  );
}
