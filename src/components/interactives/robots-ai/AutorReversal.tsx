import { useState } from 'react';
import { AI, ROBOT, LegendDot, Widget, useHoverPreview } from './shared';

// Autor (2024): AI might REVERSE the wage polarization that robots and earlier IT produced.
// Past automation hollowed out the middle of the skill distribution (the classic U-shaped
// "polarization" curve: growth at the low and high ends, a sag in the middle). Autor's
// hypothesis is that AI, by extending expert judgment to less-credentialed workers, could
// rebuild the middle instead. Both curves are illustrative of the directional argument, not
// data: the point is the shape (hollow middle vs. filled middle), not the exact heights.

interface Band {
  key: string;
  short: string;
  label: string;
  past: number; // relative employment/wage growth index, 0-100
  ai: number;
}

const BANDS: Band[] = [
  { key: 'low', short: 'Low', label: 'Low-wage work', past: 78, ai: 52 },
  { key: 'lmid', short: 'Low-mid', label: 'Lower-middle', past: 46, ai: 64 },
  { key: 'mid', short: 'Middle', label: 'Middle-skill', past: 20, ai: 82 },
  { key: 'umid', short: 'Up-mid', label: 'Upper-middle', past: 55, ai: 70 },
  { key: 'high', short: 'High', label: 'High-wage / expert', past: 88, ai: 60 },
];

type Mode = 'past' | 'ai';
const MODES: Record<Mode, { title: string; color: string; note: string }> = {
  past: {
    title: 'What happened (robots + IT, 1980-2015)',
    color: ROBOT,
    note: 'Automation hollowed out the middle: employment grew at the low-wage and high-wage ends and sagged in the middle. This is the polarization Autor and others documented.',
  },
  ai: {
    title: "Autor's AI hypothesis",
    color: AI,
    note: 'AI could rebuild the middle by extending expert judgment to less-credentialed workers, instead of hollowing it out. A hypothesis, not a finding: the direction of AI exposure is clearer than its effect on wages.',
  },
};

const VB = { w: 380, h: 250 };
const PAD = { l: 44, r: 14, t: 18, b: 46 };

export default function AutorReversal() {
  const [sel, setSel] = useState<Mode>('ai');
  const [mode, bindMode] = useHoverPreview(sel);

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (BANDS.length - 1)) * w;
  const y = (v: number) => PAD.t + h - (v / 100) * h;

  const pastPts = BANDS.map((d, i) => `${x(i)},${y(d.past)}`).join(' ');
  const aiPts = BANDS.map((d, i) => `${x(i)},${y(d.ai)}`).join(' ');

  return (
    <Widget
      title="The middle, hollowed then rebuilt?"
      kicker="illustrative of Autor (2024), not data"
    >
      <p className="mb-3 text-sm text-ink-soft">
        Robots and earlier IT polarized the labour market: they grew the top and bottom of the skill
        ladder and hollowed out the middle. Autor's hypothesis is that AI could run the other way,
        rebuilding the middle by putting expert judgment within reach of less-credentialed workers.
        Toggle the two scenarios.
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two curves of employment growth across the skill ladder: past automation sags in the middle, Autor's AI hypothesis peaks in the middle."
      >
        {/* y gridlines */}
        {[0, 25, 50, 75, 100].map((v) => (
          <line
            key={v}
            x1={PAD.l}
            y1={y(v)}
            x2={VB.w - PAD.r}
            y2={y(v)}
            style={{ stroke: 'var(--color-line)', strokeWidth: 1, strokeDasharray: '2 3' }}
          />
        ))}

        {/* both curves; the inactive one is dimmed */}
        <polyline
          points={pastPts}
          fill="none"
          style={{ stroke: ROBOT, strokeWidth: 2.5, opacity: mode === 'past' ? 1 : 0.28 }}
        />
        <polyline
          points={aiPts}
          fill="none"
          style={{ stroke: AI, strokeWidth: 2.5, opacity: mode === 'ai' ? 1 : 0.28 }}
        />

        {/* dots on the active curve */}
        {BANDS.map((d, i) => (
          <circle
            key={d.key}
            cx={x(i)}
            cy={y(mode === 'past' ? d.past : d.ai)}
            r={3.4}
            style={{ fill: MODES[mode].color, stroke: 'var(--color-paper)', strokeWidth: 1 }}
          />
        ))}

        {/* x labels */}
        {BANDS.map((d, i) => (
          <text
            key={d.key}
            x={x(i)}
            y={VB.h - 28}
            textAnchor="middle"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
          >
            {d.short}
          </text>
        ))}

        {/* axis labels */}
        <text
          transform={`translate(12 ${PAD.t + h / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          employment growth
        </text>
        <text
          x={PAD.l}
          y={VB.h - 12}
          textAnchor="start"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          lower skill / wage
        </text>
        <text
          x={VB.w - PAD.r}
          y={VB.h - 12}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          higher skill / wage →
        </text>
      </svg>

      <div className="mt-4 flex flex-wrap gap-2">
        {(Object.keys(MODES) as Mode[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setSel(k)}
            aria-pressed={mode === k}
            {...bindMode(k)}
            className="border px-2.5 py-1 font-mono text-[0.68rem] tracking-wide transition-colors"
            style={{
              borderColor: mode === k ? MODES[k].color : 'var(--color-line)',
              color: mode === k ? MODES[k].color : 'var(--color-muted)',
            }}
          >
            {MODES[k].title}
          </button>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <p className="max-w-md text-sm text-ink-soft">{MODES[mode].note}</p>
        <div className="flex flex-col items-start gap-1">
          <LegendDot color={ROBOT}>past automation</LegendDot>
          <LegendDot color={AI}>Autor's AI hypothesis</LegendDot>
        </div>
      </div>
    </Widget>
  );
}
