import { useState } from 'react';
import { LegendDot, WARN, ROBOT, Widget } from './shared';

// The scarring literature, drawn. Earnings of displaced workers relative to non-displaced
// peers, by years since displacement. Both curves start near -30% and never fully recover.
// Endpoint values are from the studies; the shape between the reported points is
// interpolated (illustrative of the reported path, not year-by-year data).

interface Pt {
  t: number;
  v: number;
}
// von Wachter, Song & Manchester (2009): ~-30% at displacement, still ~-20% at 15-20 years.
// (Jacobson-LaLonde-Sullivan 1993 is the foundational study but only observed ~5 years out,
// where it found ~-25%; the 15-20 year horizon is von Wachter et al.'s.)
const VW: Pt[] = [
  { t: 0, v: -30 },
  { t: 2, v: -27 },
  { t: 5, v: -24 },
  { t: 10, v: -22 },
  { t: 15, v: -20 },
  { t: 20, v: -20 },
];
// Couch-Placzek (2010): ~-33% immediate, still ~-15% at 6 years (shorter horizon).
const CP: Pt[] = [
  { t: 0, v: -33 },
  { t: 1, v: -28 },
  { t: 2, v: -23 },
  { t: 4, v: -18 },
  { t: 6, v: -15 },
];

// Only these years are figures the studies actually report; every other point on the line
// is interpolated to show the path, so only these get dots and feed the readout.
const VW_REPORTED = new Set([0, 15, 20]);
const CP_REPORTED = new Set([0, 6]);
const REP_VW = VW.filter((p) => VW_REPORTED.has(p.t));
const REP_CP = CP.filter((p) => CP_REPORTED.has(p.t));

const VB = { w: 380, h: 230 };
const PAD = { l: 46, r: 14, t: 14, b: 40 };
const YMIN = -36;

export default function EarningsScar() {
  const [hoverT, setHoverT] = useState<number | null>(null);

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (t: number) => PAD.l + (t / 20) * w;
  const y = (v: number) => PAD.t + (v / YMIN) * h; // v is negative, YMIN negative -> 0..1

  const line = (pts: Pt[]) => pts.map((p) => `${x(p.t)},${y(p.v)}`).join(' ');

  // snap the readout to the nearest REPORTED figure (never an interpolated point)
  const readT = hoverT ?? 20;
  const jlsAt = REP_VW.reduce((a, b) => (Math.abs(b.t - readT) < Math.abs(a.t - readT) ? b : a));
  const cpAt =
    readT <= 6
      ? REP_CP.reduce((a, b) => (Math.abs(b.t - readT) < Math.abs(a.t - readT) ? b : a))
      : null;

  return (
    <Widget title="The scar does not heal" kicker="displaced workers vs. non-displaced peers">
      <p className="mb-3 text-sm text-ink-soft">
        Displacement is not a temporary dip. Earnings drop by roughly a third at once and, for many,
        settle permanently below where they would have been. Hover across the years to read the gap.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two curves of earnings loss over 20 years since displacement, both starting near minus 30 percent and plateauing below the starting peers."
      >
        {/* y gridlines */}
        {[0, -10, -20, -30].map((v) => (
          <g key={v}>
            <line
              x1={PAD.l}
              y1={y(v)}
              x2={VB.w - PAD.r}
              y2={y(v)}
              style={{
                stroke: 'var(--color-line)',
                strokeWidth: 1,
                strokeDasharray: v === 0 ? undefined : '2 3',
              }}
            />
            <text
              x={PAD.l - 6}
              y={y(v) + 3}
              textAnchor="end"
              style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
            >
              {v}%
            </text>
          </g>
        ))}
        {/* zero baseline label */}
        <text
          x={VB.w - PAD.r}
          y={y(0) - 4}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          non-displaced peers
        </text>

        {/* the two paths */}
        <polyline points={line(CP)} fill="none" style={{ stroke: WARN, strokeWidth: 2.5 }} />
        <polyline points={line(VW)} fill="none" style={{ stroke: ROBOT, strokeWidth: 2.5 }} />

        {/* dots ONLY on the years each study actually reports */}
        {REP_VW.map((p) => (
          <circle
            key={p.t}
            cx={x(p.t)}
            cy={y(p.v)}
            r={3.6}
            style={{ fill: ROBOT, stroke: 'var(--color-paper)', strokeWidth: 1 }}
          />
        ))}
        {REP_CP.map((p) => (
          <circle
            key={p.t}
            cx={x(p.t)}
            cy={y(p.v)}
            r={3.6}
            style={{ fill: WARN, stroke: 'var(--color-paper)', strokeWidth: 1 }}
          />
        ))}

        {/* hover guide */}
        {hoverT !== null && (
          <line
            x1={x(readT)}
            y1={PAD.t}
            x2={x(readT)}
            y2={PAD.t + h}
            style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
          />
        )}

        {/* x ticks + hit areas */}
        {[0, 5, 10, 15, 20].map((t) => (
          <text
            key={t}
            x={x(t)}
            y={VB.h - 22}
            textAnchor="middle"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
          >
            {t}
          </text>
        ))}
        <text
          x={PAD.l + w / 2}
          y={VB.h - 8}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          years since displacement
        </text>
        <text
          transform={`translate(12 ${PAD.t + h / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          earnings vs. peers
        </text>
        {/* full-width hover capture */}
        {Array.from({ length: 21 }, (_, t) => (
          <rect
            key={t}
            x={x(t) - w / 40}
            y={PAD.t}
            width={w / 20}
            height={h}
            style={{ fill: 'transparent', cursor: 'crosshair' }}
            onMouseEnter={() => setHoverT(t)}
            onMouseLeave={() => setHoverT(null)}
          />
        ))}
      </svg>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <div className="flex flex-col gap-0.5 font-mono text-sm">
          <span className="text-[0.7rem] text-muted">nearest reported figure</span>
          <span style={{ color: ROBOT }}>
            von Wachter et al. <span className="font-semibold">{jlsAt.v}%</span> at year {jlsAt.t}
          </span>
          {cpAt && (
            <span style={{ color: WARN }}>
              Couch-Placzek <span className="font-semibold">{cpAt.v}%</span> at year {cpAt.t}
            </span>
          )}
        </div>
        <div className="flex flex-col items-start gap-1">
          <LegendDot color={ROBOT}>von Wachter, Song & Manchester (2009)</LegendDot>
          <LegendDot color={WARN}>Couch-Placzek (2010), to year 6</LegendDot>
        </div>
      </div>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Dots mark each study's reported figures; the line between them is interpolated to show the
        path. This literature is mass-layoff-wide, not robot-specific.
      </p>
    </Widget>
  );
}
