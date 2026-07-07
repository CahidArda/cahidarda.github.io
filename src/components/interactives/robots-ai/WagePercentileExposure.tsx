import { useState } from 'react';
import { AI, ROBOT, LegendDot, Widget } from './shared';

// The WAGE cut of Labaj, Oles & Prochazka (2025), Fig. 4. Redrawn (traced by eye) from the
// paper's smoothed cubic curves, so the SHAPE is faithful but no exact score is claimed: the
// y-axis carries no numbers on purpose. Robot exposure is an inverted-U, "lower at both ends
// of the distribution... primarily in the middle part"; AI exposure "increases monotonically
// as we move to the right across the earning distribution." Wage percentile on x is the real
// axis of the figure.

// Standardized exposure score, traced from Fig. 4 at 5-percentile steps.
const ROBOT_PTS: [number, number][] = [
  [0, -0.28],
  [5, -0.27],
  [10, -0.24],
  [15, -0.2],
  [20, -0.16],
  [25, -0.06],
  [30, 0.01],
  [35, 0.07],
  [40, 0.11],
  [45, 0.14],
  [50, 0.155],
  [55, 0.16],
  [60, 0.15],
  [65, 0.12],
  [70, 0.08],
  [75, 0.03],
  [80, -0.03],
  [85, -0.08],
  [90, -0.11],
  [95, -0.14],
  [100, -0.15],
];
const AI_PTS: [number, number][] = [
  [0, -0.52],
  [5, -0.44],
  [10, -0.4],
  [15, -0.37],
  [20, -0.33],
  [25, -0.28],
  [30, -0.22],
  [35, -0.16],
  [40, -0.1],
  [45, -0.05],
  [50, 0.0],
  [55, 0.06],
  [60, 0.11],
  [65, 0.17],
  [70, 0.22],
  [75, 0.26],
  [80, 0.3],
  [85, 0.36],
  [90, 0.42],
  [95, 0.45],
  [100, 0.46],
];

const VB = { w: 380, h: 250 };
const PAD = { l: 30, r: 14, t: 18, b: 46 };
const YMAX = 0.5;
const YMIN = -0.6;

export default function WagePercentileExposure() {
  const [hoverP, setHoverP] = useState<number | null>(null);

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (p: number) => PAD.l + (p / 100) * w;
  const y = (v: number) => PAD.t + ((YMAX - v) / (YMAX - YMIN)) * h;

  const line = (pts: [number, number][]) => pts.map(([p, v]) => `${x(p)},${y(v)}`).join(' ');

  return (
    <Widget title="The robot hump, by wage" kicker="Labaj et al. (2025), Fig. 4 · redrawn">
      <p className="mb-3 text-sm text-ink-soft">
        The same study, cut by wage instead of education. Robot exposure{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>peaks in the middle</span> of the wage
        distribution and fades at both ends: low-wage work is often too unstructured to automate
        cheaply, high-wage work too non-routine. AI exposure{' '}
        <span style={{ color: AI, fontWeight: 600 }}>climbs</span> the whole way up.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two curves over the wage percentile: robot exposure is an inverted-U peaking in the middle, AI exposure rises monotonically from left to right."
      >
        {/* zero baseline (the only meaningful reference on a standardized score) */}
        <line
          x1={PAD.l}
          y1={y(0)}
          x2={VB.w - PAD.r}
          y2={y(0)}
          style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        <text
          x={PAD.l - 4}
          y={y(0) + 3}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          0
        </text>

        {/* the two curves */}
        <polyline
          points={line(ROBOT_PTS)}
          fill="none"
          style={{ stroke: ROBOT, strokeWidth: 2.5 }}
        />
        <polyline points={line(AI_PTS)} fill="none" style={{ stroke: AI, strokeWidth: 2.5 }} />

        {/* hover guide */}
        {hoverP !== null && (
          <line
            x1={x(hoverP)}
            y1={PAD.t}
            x2={x(hoverP)}
            y2={PAD.t + h}
            style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
          />
        )}

        {/* x ticks (the real axis of the figure) */}
        {[0, 20, 40, 60, 80, 100].map((p) => (
          <text
            key={p}
            x={x(p)}
            y={VB.h - 28}
            textAnchor="middle"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
          >
            {p}
          </text>
        ))}

        {/* full-width hover capture, snapping to 5-percentile steps */}
        {ROBOT_PTS.map(([p]) => (
          <rect
            key={p}
            x={x(p) - w / 40}
            y={PAD.t}
            width={w / 20}
            height={h}
            style={{ fill: 'transparent', cursor: 'crosshair' }}
            onMouseEnter={() => setHoverP(p)}
            onMouseLeave={() => setHoverP(null)}
          />
        ))}

        {/* axis labels */}
        <text
          transform={`translate(11 ${PAD.t + h / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          exposure score
        </text>
        <text
          x={PAD.l + w / 2}
          y={VB.h - 12}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          wage percentile
        </text>
      </svg>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <span className="font-mono text-[0.72rem] text-ink">
          {hoverP === null
            ? 'robots bulge in the middle; AI rises to the top'
            : `wage percentile ${hoverP}`}
        </span>
        <div className="flex gap-3">
          <LegendDot color={ROBOT}>robot exposure</LegendDot>
          <LegendDot color={AI}>AI exposure</LegendDot>
        </div>
      </div>
    </Widget>
  );
}
