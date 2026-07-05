import { ROBOT, Widget } from './shared';

// The policy punchline: US regions with more generous unemployment benefits (2000-2007)
// saw SMALLER automation-driven wage losses (IMF, via Bipartisan Policy Center). The
// points are illustrative of that reported relationship, not the underlying regional data.
// Y is inverted so "smaller loss" reads as higher up.

// Deterministic illustrative points (benefit generosity, wage loss %). Hand-placed around
// a downward trend so SSR and client render identically (no runtime randomness).
const POINTS: [number, number][] = [
  [8, -3.1],
  [15, -3.4],
  [19, -2.4],
  [24, -2.9],
  [31, -2.1],
  [37, -2.4],
  [43, -1.7],
  [49, -2.0],
  [55, -1.3],
  [61, -1.6],
  [68, -1.1],
  [74, -1.4],
  [81, -0.8],
  [88, -1.0],
  [93, -0.6],
];

const VB = { w: 380, h: 230 };
const PAD = { l: 36, r: 14, t: 16, b: 42 };
const YMIN = -3.8;

export default function SafetyNetCushion() {
  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (g: number) => PAD.l + (g / 100) * w;
  const y = (v: number) => PAD.t + (v / YMIN) * h;

  // simple least-squares fit for the trend line
  const n = POINTS.length;
  const sx = POINTS.reduce((a, [g]) => a + g, 0);
  const sy = POINTS.reduce((a, [, v]) => a + v, 0);
  const sxx = POINTS.reduce((a, [g]) => a + g * g, 0);
  const sxy = POINTS.reduce((a, [g, v]) => a + g * v, 0);
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const intercept = (sy - slope * sx) / n;
  const fit = (g: number) => intercept + slope * g;

  return (
    <Widget
      title="The cushion that worked"
      kicker="IMF, via Bipartisan Policy Center · illustrative"
    >
      <p className="mb-3 text-sm text-ink-soft">
        The counterintuitive result from the policy literature: regions with more generous
        unemployment benefits absorbed automation with{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>smaller</span> wage losses. The safety net
        cushioned displacement more reliably than retraining did.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="A scatter of regions: more generous unemployment benefits on the x-axis correspond to smaller automation wage losses, with a downward-then-upward trend line."
      >
        {/* y gridlines */}
        {[0, -1, -2, -3].map((v) => (
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

        {/* fit line */}
        <line
          x1={x(4)}
          y1={y(fit(4))}
          x2={x(96)}
          y2={y(fit(96))}
          style={{ stroke: ROBOT, strokeWidth: 2, strokeDasharray: '5 4', opacity: 0.85 }}
        />

        {/* points */}
        {POINTS.map(([g, v], i) => (
          <circle
            key={i}
            cx={x(g)}
            cy={y(v)}
            r={3.4}
            style={{ fill: ROBOT, opacity: 0.7, stroke: 'var(--color-paper)', strokeWidth: 1 }}
          />
        ))}

        {/* axis labels */}
        <text
          x={PAD.l}
          y={VB.h - 20}
          textAnchor="start"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          less generous benefits
        </text>
        <text
          x={VB.w - PAD.r}
          y={VB.h - 20}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          more generous →
        </text>
        <text
          x={PAD.l + w / 2}
          y={VB.h - 6}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          unemployment-benefit generosity by region
        </text>
        <text
          transform={`translate(11 ${PAD.t + h / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          automation wage loss
        </text>
      </svg>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Points are illustrative of the reported relationship, not the underlying regional data.
        Trade Adjustment Assistance retraining helped only when bundled with job-search and
        placement support, and its earnings benefit dissipated over time.
      </p>
    </Widget>
  );
}
