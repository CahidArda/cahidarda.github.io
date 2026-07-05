import { useState } from 'react';
import { ROBOT, WARN, Widget } from './shared';

// "Reemployment" hides a large permanent-pay-cut tail. Two stacked bars with DIFFERENT
// denominators, kept explicit: the whole displaced pool, then the reemployed subset.
// BLS Displaced Workers Survey (January 2024), long-tenured workers.

interface Seg {
  key: string;
  label: string;
  pct: number;
  color: string;
  denom: string;
  note: string;
}

const POOL: Seg[] = [
  {
    key: 'reemp',
    label: 'reemployed',
    pct: 65.7,
    color: ROBOT,
    denom: 'of all long-tenured displaced workers',
    note: '65.7% found another job by the survey date.',
  },
  {
    key: 'not',
    label: 'not reemployed',
    pct: 34.3,
    color: 'var(--color-muted)',
    denom: 'of all long-tenured displaced workers',
    note: 'The remaining ~34% were unemployed or had left the labour force.',
  },
];

const REEMP: Seg[] = [
  {
    key: 'equal',
    label: 'equal or higher pay',
    pct: 62,
    color: ROBOT,
    denom: 'of full-time reemployed workers',
    note: '62% earned as much as, or more than, the job they lost.',
  },
  {
    key: 'cut',
    label: 'took a pay cut',
    pct: 38,
    color: WARN,
    denom: 'of full-time reemployed workers',
    note: '38% landed a permanent pay cut. This is the tail "reemployment" hides.',
  },
];

function Bar({
  segs,
  hover,
  setHover,
}: {
  segs: Seg[];
  hover: string | null;
  setHover: (k: string | null) => void;
}) {
  return (
    <div className="flex h-11 w-full overflow-hidden border border-line">
      {segs.map((s) => (
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
  );
}

export default function WhereLanded() {
  const [hover, setHover] = useState<string | null>(null);
  const all = [...POOL, ...REEMP];
  const active = all.find((s) => s.key === hover);

  return (
    <Widget title="Where displaced workers landed" kicker="BLS Displaced Workers Survey, Jan 2024">
      <p className="mb-4 text-sm text-ink-soft">
        "Reemployed" sounds like recovery. Two thirds got there. But the denominators matter, and
        the two bars below use different ones. Hover a segment to see which.
      </p>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[0.68rem] text-ink-soft">the displaced pool (100%)</span>
        </div>
        <Bar segs={POOL} hover={hover} setHover={setHover} />
        <div className="mb-2 flex justify-between font-mono text-[0.62rem] text-muted">
          <span>reemployed</span>
          <span>not reemployed</span>
        </div>

        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[0.68rem] text-ink-soft">
            of the full-time reemployed (100%)
          </span>
        </div>
        <Bar segs={REEMP} hover={hover} setHover={setHover} />
        <div className="flex justify-between font-mono text-[0.62rem] text-muted">
          <span>equal or higher</span>
          <span style={{ color: WARN }}>pay cut</span>
        </div>
      </div>

      <div className="mt-4 min-h-[3rem] border-t border-line pt-3">
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
            <div className="font-mono text-[0.66rem] text-muted">{active.denom}</div>
            <p className="mt-1 text-sm text-ink-soft">{active.note}</p>
          </>
        ) : (
          <p className="text-sm text-ink-soft">
            Combine the bars: about a third of the displaced never got reemployed, and of those who
            did, nearly 4 in 10 took a permanent pay cut. The headline number is the friendliest
            slice.
          </p>
        )}
      </div>
    </Widget>
  );
}
