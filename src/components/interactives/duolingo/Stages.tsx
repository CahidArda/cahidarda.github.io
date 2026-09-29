/**
 * The four stages as a staircase: foundation, bridge, immersion, and output (dashed, because
 * it is still unsolved). Static.
 */
import { ACCENT, Caption, MONO, Widget } from './shared';

const STEPS = [
  { title: '1. Foundation', sub: ['basics,', 'core grammar'] },
  { title: '2. Bridge', sub: ['loanwords you', 'already know'] },
  { title: '3. Immersion', sub: ['videos you’d', 'watch anyway'] },
  { title: '4. Output', sub: ['actually', 'speaking', '(unsolved)'] },
];
const W = 128;
const BOTTOM = 232;

export default function Stages() {
  return (
    <Widget title="Stages" kicker="step 4 is still open">
      <svg
        viewBox="0 0 560 240"
        className="block h-auto w-full"
        role="img"
        aria-label="Four ascending steps: 1 Foundation, basics and core grammar. 2 Bridge, loanwords you already know. 3 Immersion, videos you'd watch anyway. 4 Output, actually speaking, still unsolved."
      >
        {STEPS.map((s, i) => {
          const x = 10 + i * (W + 8);
          const h = 62 + i * 46;
          const y = BOTTOM - h;
          const open = i === STEPS.length - 1;
          const cx = x + W / 2;
          return (
            <g key={s.title}>
              <rect
                x={x}
                y={y}
                width={W}
                height={h}
                style={{
                  fill: 'var(--color-paper)',
                  stroke: open ? 'var(--color-line-strong)' : i === 2 ? ACCENT : 'var(--color-ink)',
                  strokeWidth: open ? 1.5 : 2,
                  strokeDasharray: open ? '5 4' : undefined,
                }}
              />
              <text
                x={cx}
                y={y + 22}
                textAnchor="middle"
                style={{
                  fill: open ? 'var(--color-muted)' : 'var(--color-ink)',
                  fontFamily: MONO,
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                {s.title}
              </text>
              {s.sub.map((t, j) => (
                <text
                  key={t}
                  x={cx}
                  y={y + 40 + j * 14}
                  textAnchor="middle"
                  style={{ fill: 'var(--color-muted)', fontFamily: MONO, fontSize: 10.5 }}
                >
                  {t}
                </text>
              ))}
            </g>
          );
        })}
      </svg>
      <Caption>Duolingo-style drills earn their place in step 1. The mistake is staying there for 120 days.</Caption>
    </Widget>
  );
}
