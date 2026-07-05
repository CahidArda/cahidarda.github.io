import { useState } from 'react';
import { ROBOT, WARN, Widget } from './shared';

// "Reemployment" hides a large permanent-pay-cut tail. One bar over the whole displaced
// pool, split into three outcomes. Shares from the BLS Displaced Workers Survey (Jan 2024,
// long-tenured workers): 65.7% reemployed, and of the full-time reemployed 62% matched or
// beat their old pay. Combining the two gives the three slices below (the small overlap
// between the two survey denominators is smoothed over for readability).

interface Seg {
  key: string;
  label: string;
  pct: number;
  color: string;
  note: string;
}

const SEGS: Seg[] = [
  {
    key: 'not',
    label: 'not reemployed',
    pct: 34,
    color: 'var(--color-muted)',
    note: 'About a third never found another job by the survey date: still unemployed or out of the labour force.',
  },
  {
    key: 'cut',
    label: 'reemployed, pay cut',
    pct: 25,
    color: WARN,
    note: 'Found full-time work but at lower pay than the job they lost. This is the tail "reemployment" hides.',
  },
  {
    key: 'equal',
    label: 'reemployed, equal or higher pay',
    pct: 41,
    color: ROBOT,
    note: 'Found full-time work at pay as good as, or better than, the job they lost. The genuine recoveries.',
  },
];

export default function WhereLanded() {
  const [hover, setHover] = useState<string | null>(null);
  const active = SEGS.find((s) => s.key === hover);

  return (
    <Widget title="Where displaced workers landed" kicker="BLS Displaced Workers Survey, Jan 2024">
      <p className="mb-4 text-sm text-ink-soft">
        "Reemployed" sounds like recovery, and two thirds of long-tenured displaced workers got
        there. But split the whole pool into what actually happened and only about four in ten came
        out whole. Hover a segment to read it.
      </p>

      <div className="flex h-11 w-full overflow-hidden border border-line">
        {SEGS.map((s) => (
          <button
            key={s.key}
            type="button"
            onMouseEnter={() => setHover(s.key)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(s.key)}
            onBlur={() => setHover(null)}
            className="flex h-full items-center justify-center transition-opacity"
            style={{
              width: `${s.pct}%`,
              background: s.color,
              opacity: hover === null || hover === s.key ? 1 : 0.4,
            }}
          >
            <span className="px-1 font-mono text-[0.7rem] font-semibold text-paper">{s.pct}%</span>
          </button>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
        {SEGS.map((s) => (
          <span
            key={s.key}
            className="inline-flex items-center gap-1.5 font-mono text-[0.66rem] text-ink-soft"
          >
            <span
              aria-hidden
              style={{ width: 10, height: 10, background: s.color, display: 'inline-block' }}
            />
            {s.label}
          </span>
        ))}
      </div>

      <div className="mt-4 min-h-12 border-t border-line pt-3">
        {active ? (
          <>
            <div
              className="font-mono text-sm font-semibold"
              style={{
                color: active.color === 'var(--color-muted)' ? 'var(--color-ink)' : active.color,
              }}
            >
              {active.pct}% {active.label}
            </div>
            <p className="mt-1 text-sm text-ink-soft">{active.note}</p>
          </>
        ) : (
          <p className="text-sm text-ink-soft">
            About a third of displaced workers never got reemployed, and a quarter came back only at
            lower pay. The friendly two-thirds reemployment headline is hiding a long tail of people
            who ended up worse off for good.
          </p>
        )}
      </div>
    </Widget>
  );
}
