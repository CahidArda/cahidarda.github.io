import { useState } from 'react';
import { AI, ROBOT, Widget } from './shared';

// The signature figure, the EDUCATION cut of Labaj, Oles & Prochazka (2025): robot exposure
// is strongest among high-school dropouts and "declines monotonically with education", while
// AI exposure runs the other way, lightest on the least-educated and heaviest on college
// graduates. The curve heights are illustrative of those reported DIRECTIONS, not data, so no
// numbers are shown. Each line is labelled in its own colour, on the chart, so the inversion
// reads in one look: the word "robots" sits where robots peak (left), "AI" where AI peaks.

interface Band {
  key: string;
  label: string;
  short: string;
  robot: number; // relative shape only, never displayed
  ai: number;
}

// Values are perfectly linear (equal steps) so each line renders dead straight and the dots
// sit on it; the two cross exactly at "Some college". Heights are illustrative direction only.
const BANDS: Band[] = [
  { key: 'nohs', label: 'No HS diploma', short: '<HS', robot: 96, ai: 8 },
  { key: 'hs', label: 'High-school grad', short: 'HS', robot: 74, ai: 30 },
  { key: 'some', label: 'Some college', short: 'Some', robot: 52, ai: 52 },
  { key: 'ba', label: "Bachelor's", short: 'BA', robot: 30, ai: 74 },
  { key: 'adv', label: 'Advanced degree', short: 'Adv', robot: 8, ai: 96 },
];

const VB = { w: 380, h: 250 };
const PAD = { l: 32, r: 14, t: 22, b: 46 };

export default function InvertedLadder() {
  const [sel, setSel] = useState<number | null>(null);
  // No band is highlighted until one is actually hovered.
  const active = sel;

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (BANDS.length - 1)) * w;
  const y = (v: number) => PAD.t + h - (v / 100) * h;

  const robotPts = BANDS.map((d, i) => `${x(i)},${y(d.robot)}`).join(' ');
  const aiPts = BANDS.map((d, i) => `${x(i)},${y(d.ai)}`).join(' ');

  return (
    <Widget title="The inverted ladder" kicker="by education · direction from Labaj et al. (2025)">
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two curves over the education ladder: robot exposure falls from left to right, AI exposure rises, and they cross in the middle."
      >
        {/* faint gridlines for structure (no numeric scale: the index is illustrative) */}
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

        {/* the two curves */}
        <polyline points={robotPts} fill="none" style={{ stroke: ROBOT, strokeWidth: 2.5 }} />
        <polyline points={aiPts} fill="none" style={{ stroke: AI, strokeWidth: 2.5 }} />

        {/* direct colour-coded labels, each sitting where its line peaks */}
        <text
          x={x(0) + 6}
          y={y(100) - 7}
          textAnchor="start"
          style={{ fill: ROBOT, fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700 }}
        >
          robots ↓
        </text>
        <text
          x={x(4) - 6}
          y={y(100) - 7}
          textAnchor="end"
          style={{ fill: AI, fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700 }}
        >
          ↑ AI
        </text>

        {/* data dots, highlighted on the active band */}
        {BANDS.map((d, i) => (
          <g key={d.key}>
            <circle
              cx={x(i)}
              cy={y(d.robot)}
              r={i === active ? 4.5 : 3}
              style={{ fill: ROBOT, stroke: 'var(--color-paper)', strokeWidth: 1 }}
            />
            <circle
              cx={x(i)}
              cy={y(d.ai)}
              r={i === active ? 4.5 : 3}
              style={{ fill: AI, stroke: 'var(--color-paper)', strokeWidth: 1 }}
            />
          </g>
        ))}

        {/* active band guide + hover hit areas + x labels */}
        {BANDS.map((d, i) => {
          const colW = w / (BANDS.length - 1);
          return (
            <g
              key={d.key}
              onMouseEnter={() => setSel(i)}
              onMouseLeave={() => setSel(null)}
              onClick={() => setSel(i)}
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={x(i) - colW / 2}
                y={PAD.t}
                width={colW}
                height={h}
                style={{ fill: 'transparent' }}
              />
              {sel !== null && i === active && (
                <line
                  x1={x(i)}
                  y1={PAD.t}
                  x2={x(i)}
                  y2={PAD.t + h}
                  style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
                />
              )}
              <text
                x={x(i)}
                y={VB.h - 28}
                textAnchor="middle"
                style={{
                  fill: i === active ? 'var(--color-ink)' : 'var(--color-muted)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 8.5,
                  fontWeight: i === active ? 600 : 400,
                }}
              >
                {d.short}
              </text>
            </g>
          );
        })}

        {/* axis labels */}
        <text
          transform={`translate(11 ${PAD.t + h / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          exposure →
        </text>
        <text
          x={PAD.l}
          y={VB.h - 12}
          textAnchor="start"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          least educated
        </text>
        <text
          x={VB.w - PAD.r}
          y={VB.h - 12}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          most educated →
        </text>
      </svg>

      <p className="mt-3 text-sm text-ink-soft">
        Relative exposure to automation across the education ladder. Robots fall{' '}
        <span style={{ color: ROBOT, fontWeight: 600 }}>down</span> it, hitting high-school dropouts
        hardest; AI climbs <span style={{ color: AI, fontWeight: 600 }}>up</span> it, heaviest on
        college graduates.
      </p>
    </Widget>
  );
}
