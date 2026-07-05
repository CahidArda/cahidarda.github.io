import { useState } from 'react';
import { AI, ROBOT, LegendDot, Widget } from './shared';

// The signature figure. Two exposure curves over the same education / income axis:
// robots slope DOWN the ladder (worst for the least-educated), AI slopes UP it (rises
// with skill and income). They cross in the middle. The index values are illustrative
// of the directions the studies report, not exact figures: the robot curve traces
// Acemoglu-Restrepo (2020) and the UK study's "declines monotonically with education";
// the AI curve traces the same UK study's "increases monotonically across income
// percentiles" plus Autor (2024).

interface Band {
  key: string;
  label: string;
  short: string;
  robot: number; // relative exposure index, 0-100
  ai: number;
}

const BANDS: Band[] = [
  { key: 'nohs', label: 'No HS diploma', short: '<HS', robot: 100, ai: 16 },
  { key: 'hs', label: 'High-school grad', short: 'HS', robot: 76, ai: 33 },
  { key: 'some', label: 'Some college', short: 'Some', robot: 54, ai: 54 },
  { key: 'ba', label: "Bachelor's", short: 'BA', robot: 30, ai: 82 },
  { key: 'adv', label: 'Advanced degree', short: 'Adv', robot: 11, ai: 100 },
];

const VB = { w: 380, h: 250 };
const PAD = { l: 34, r: 14, t: 18, b: 46 };

export default function InvertedLadder({ variant = 'full' }: { variant?: 'full' | 'knowledge' }) {
  // Default readout = the crossover band ("some college"), where the two waves meet.
  const [sel, setSel] = useState<number | null>(null);
  const active = sel ?? 2;
  const b = BANDS[active];

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (BANDS.length - 1)) * w;
  const y = (v: number) => PAD.t + h - (v / 100) * h;

  const robotPts = BANDS.map((d, i) => `${x(i)},${y(d.robot)}`).join(' ');
  const aiPts = BANDS.map((d, i) => `${x(i)},${y(d.ai)}`).join(' ');
  // The lines cross exactly at band index 2 (both = 54), so the crossover marker sits there.
  const cross = { x: x(2), y: y(54) };

  const focusBA = variant === 'knowledge'; // dim the low end, spotlight the credentialed end

  return (
    <Widget
      title={variant === 'knowledge' ? 'The same ladder, from the top' : 'The inverted ladder'}
      kicker="exposure index · illustrative of the reported directions"
    >
      <p className="mb-3 text-sm text-ink-soft">
        {variant === 'knowledge' ? (
          <>
            The right half is where most software, design, legal and analyst work sits. It is the
            low point of the robot curve and the high point of the AI curve. Hover a band to read
            both.
          </>
        ) : (
          <>
            Relative exposure to automation across the education / income ladder. Robots fall{' '}
            <span style={{ color: ROBOT, fontWeight: 600 }}>down</span> it; AI climbs{' '}
            <span style={{ color: AI, fontWeight: 600 }}>up</span> it. Hover a band to read both,
            and notice where they cross.
          </>
        )}
      </p>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="Two curves over the education ladder: robot exposure falls from left to right, AI exposure rises, and they cross in the middle."
      >
        {/* y gridlines */}
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line
              x1={PAD.l}
              y1={y(v)}
              x2={VB.w - PAD.r}
              y2={y(v)}
              style={{ stroke: 'var(--color-line)', strokeWidth: 1, strokeDasharray: '2 3' }}
            />
            <text
              x={PAD.l - 6}
              y={y(v) + 3}
              textAnchor="end"
              style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
            >
              {v}
            </text>
          </g>
        ))}

        {/* knowledge-worker spotlight band (section-3 variant) */}
        {focusBA && (
          <rect
            x={x(2.5)}
            y={PAD.t}
            width={x(4) - x(2.5) + PAD.r}
            height={h}
            style={{ fill: AI, opacity: 0.08 }}
          />
        )}

        {/* the two curves */}
        <polyline
          points={robotPts}
          fill="none"
          style={{ stroke: ROBOT, strokeWidth: 2.5, opacity: focusBA ? 0.5 : 1 }}
        />
        <polyline points={aiPts} fill="none" style={{ stroke: AI, strokeWidth: 2.5 }} />

        {/* crossover marker */}
        <circle
          cx={cross.x}
          cy={cross.y}
          r={4}
          style={{ fill: 'var(--color-paper)', stroke: 'var(--color-ink)', strokeWidth: 1.5 }}
        />
        {sel === null && !focusBA && (
          <text
            x={cross.x}
            y={cross.y - 10}
            textAnchor="middle"
            style={{
              fill: 'var(--color-ink)',
              fontFamily: 'var(--font-mono)',
              fontSize: 8.5,
              fontWeight: 600,
            }}
          >
            they cross here
          </text>
        )}

        {/* data dots, highlighted on the active band */}
        {BANDS.map((d, i) => (
          <g key={d.key}>
            <circle
              cx={x(i)}
              cy={y(d.robot)}
              r={i === active ? 4.5 : 3}
              style={{
                fill: ROBOT,
                opacity: focusBA && i < 3 ? 0.4 : 1,
                stroke: 'var(--color-paper)',
                strokeWidth: 1,
              }}
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
              {i === active && (
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
          x={PAD.l}
          y={VB.h - 12}
          textAnchor="start"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          least educated / lower income
        </text>
        <text
          x={VB.w - PAD.r}
          y={VB.h - 12}
          textAnchor="end"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}
        >
          most educated / higher income →
        </text>
      </svg>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <div className="flex flex-col gap-0.5">
          <span className="font-mono text-[0.7rem] text-muted">{b.label}</span>
          <div className="flex gap-4">
            <span className="font-mono text-sm" style={{ color: ROBOT }}>
              robots <span className="font-semibold">{b.robot}</span>
            </span>
            <span className="font-mono text-sm" style={{ color: AI }}>
              AI <span className="font-semibold">{b.ai}</span>
            </span>
          </div>
        </div>
        <div className="flex gap-3">
          <LegendDot color={ROBOT}>robot exposure</LegendDot>
          <LegendDot color={AI}>AI exposure</LegendDot>
        </div>
      </div>
    </Widget>
  );
}
