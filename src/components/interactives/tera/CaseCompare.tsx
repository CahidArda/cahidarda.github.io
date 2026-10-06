import { Fragment, useState } from 'react';
import { FLOW, PRICE, useHoverPreview, WARN, Widget } from './shared';

// Three earlier collapses, each compared with Turkey 2026 on the three things that actually
// differ: which direction the failure ran, what set the price, and who absorbed the loss.
// A table rather than three paragraphs, because the point is the comparison.

interface Case {
  key: string;
  name: string;
  when: string;
  color: string;
  rows: [string, string][]; // [that case, Turkey 2026]
  note: string;
  href: string;
}

const ROW_LABELS = ['Direction of failure', 'What set the price', 'Who absorbed the loss'];

const CASES: Case[] = [
  {
    key: 'woodford',
    name: 'Woodford',
    when: 'UK, 2019',
    color: FLOW,
    rows: [
      ['On the way out: selling liquid assets to meet redemptions', 'Inflows are what the research points at'],
      ['Unquoted holdings, stale prices', 'Listed holdings with very little free float'],
      ['Investors who stayed, left with the illiquid residue', 'Unit holders, through the liquidation'],
    ],
    note: 'The comparison most coverage reached for. The mechanism is the opposite way round: Woodford’s problem was redemptions, not inflows, and his marks were stale rather than pushed.',
    href: 'https://www.fca.org.uk/news/press-releases/fca-fines-over-woodford-equity-income-fund',
  },
  {
    key: 'archegos',
    name: 'Archegos',
    when: 'US, 2021',
    color: PRICE,
    rows: [
      ['On the way in, then a forced unwind', 'Inflows, then redemptions'],
      ['Its own buying, via derivatives, in thin names', 'Its own buying, in cash equities with low free float'],
      ['The investment banks that financed the trades', 'Retail unit holders'],
    ],
    note: 'The closest match on mechanism: a concentrated buyer setting the price in names it dominated. The difference is who was exposed. Archegos was a family office with bank counterparties; these were funds sold to the public.',
    href: 'https://www.sec.gov/newsroom/press-releases/2022-70',
  },
  {
    key: 'bankerler',
    name: 'Bankerler krizi',
    when: 'Turkey, 1982',
    color: WARN,
    rows: [
      ['Inflow stopped by a rule change, then illiquidity', 'Position caps in August, defaults in September'],
      ['Not a price loop: a spread on certificates of deposit', 'A price loop is what the research describes'],
      ['Depositors, then the state via public banks', 'Unit holders, with İş Bankası and Ziraat liquidating'],
    ],
    note: 'Not the same mechanism, but the same shape of ending: a regulation cut the inflow, the structure failed weeks later, and the state absorbed the resolution through public banks.',
    href: 'https://documents1.worldbank.org/curated/en/245901468760494639/pdf/multi-page.pdf',
  },
];

export default function CaseCompare() {
  const [sel, setSel] = useState(1);
  const [active, bindActive] = useHoverPreview(sel);
  const c = CASES[active];

  return (
    <Widget title="Three earlier collapses" kicker="compared with Turkey 2026">
      <div className="mb-4 flex flex-wrap gap-2">
        {CASES.map((x, i) => (
          <button
            key={x.key}
            type="button"
            onClick={() => setSel(i)}
            aria-pressed={i === active}
            {...bindActive(i)}
            className="border-2 px-2.5 py-1 text-left font-mono text-[0.7rem] transition-colors"
            style={{
              borderColor: i === active ? x.color : 'var(--color-line)',
              color: i === active ? x.color : 'var(--color-ink-soft)',
            }}
          >
            {x.name}
            <span className="ml-1.5 text-[0.6rem] opacity-70">{x.when}</span>
          </button>
        ))}
      </div>

      {/* One grid for every cell, so a long value cannot widen a single row (skill §10). */}
      <div
        className="grid gap-px border border-line-strong"
        style={{
          gridTemplateColumns: 'minmax(5.5rem, 0.9fr) minmax(0, 1.3fr) minmax(0, 1.3fr)',
          background: 'var(--color-line)',
        }}
      >
        <div className="bg-paper-raised px-2 py-1.5" />
        <div className="bg-paper-raised px-2 py-1.5 font-mono text-[0.66rem] font-semibold" style={{ color: c.color }}>
          {c.name}
        </div>
        <div className="bg-paper-raised px-2 py-1.5 font-mono text-[0.66rem] font-semibold text-ink">
          Turkey 2026
        </div>

        {c.rows.map(([a, b], i) => (
          <Fragment key={ROW_LABELS[i]}>
            <div
              className="bg-paper-raised px-2 py-2 font-mono text-[0.62rem] leading-snug text-muted"
            >
              {ROW_LABELS[i]}
            </div>
            <div
              className="bg-paper-raised px-2 py-2 font-mono text-[0.64rem] leading-snug text-ink-soft"
            >
              {a}
            </div>
            <div
              className="bg-paper-raised px-2 py-2 font-mono text-[0.64rem] leading-snug text-ink-soft"
            >
              {b}
            </div>
          </Fragment>
        ))}
      </div>

      <p className="mt-3 min-h-[4rem] text-sm text-ink-soft">{c.note}</p>
      <a
        href={c.href}
        target="_blank"
        rel="noopener noreferrer"
        className="font-mono text-[0.68rem] text-muted underline underline-offset-2 hover:text-ink"
      >
        source →
      </a>
    </Widget>
  );
}
