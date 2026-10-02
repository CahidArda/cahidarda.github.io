import { useState } from 'react';
import { FLOW, PRICE, useHoverPreview, WARN, Widget } from './shared';

// What TLY actually held. The point of this figure is not that the fund was concentrated
// (plenty of funds are) but WHERE the concentration pointed: at companies chaired by the
// fund manager's own founder, and at a stock whose IPO the group had underwritten and
// whose largest allocatee was this same fund.
//
// Weights are TLY's disclosed portfolio distribution as of 31 Aug 2026 (fon.org.tr).

type Link = 'chair' | 'underwrote' | 'none';

const LINK: Record<Link, { label: string; color: string }> = {
  chair: { label: 'Tera founder is board chair', color: WARN },
  underwrote: { label: 'IPO underwritten by Tera; TLY was largest allocatee', color: PRICE },
  none: { label: 'no disclosed Tera relationship', color: FLOW },
};

interface Holding {
  ticker: string;
  name: string;
  pct: number;
  link: Link;
  note: string;
}

const HOLDINGS: Holding[] = [
  {
    ticker: 'DSTKF',
    name: 'Destek Finans Faktoring',
    pct: 24.8,
    link: 'underwrote',
    note: 'Floated Feb 2025 at ₺46.98 by Tera Yatırım Menkul Değerler. TLY took 15.42% of the offering, the largest single allocation. Its owner later told prosecutors the rise to ₺1.3tn was an abnormal figure.',
  },
  {
    ticker: 'OZATD',
    name: 'Özata Denizcilik',
    pct: 19.8,
    link: 'none',
    note: 'No Tera board relationship is disclosed in the sources reviewed here.',
  },
  {
    ticker: 'TEHOL',
    name: 'Tera Yatırım Teknoloji Holding',
    pct: 14.1,
    link: 'chair',
    note: 'A Tera group holding company. Emre Tezmen, founder and main shareholder of Tera, is its board chair.',
  },
  {
    ticker: 'TRHOL',
    name: 'Tera Finansal Yatırımlar Holding',
    pct: 8.28,
    link: 'chair',
    note: 'A second Tera group holding company, also chaired by Tezmen.',
  },
  {
    ticker: 'PEKGY',
    name: 'Peker GYO',
    pct: 6.29,
    link: 'chair',
    note: 'A real estate investment trust. Tera’s corporate page lists Tezmen as its board chair alongside the Tera entities.',
  },
];

const OTHER = +(100 - HOLDINGS.reduce((s, h) => s + h.pct, 0)).toFixed(2);
const CHAIR = HOLDINGS.filter((h) => h.link === 'chair').reduce((s, h) => s + h.pct, 0);
const RELATED = CHAIR + HOLDINGS.filter((h) => h.link === 'underwrote').reduce((s, h) => s + h.pct, 0);

export default function Concentration() {
  const [sel, setSel] = useState(0);
  const [active, bindActive] = useHoverPreview(sel);
  const h = HOLDINGS[active];

  return (
    <Widget title="Where the money pointed" kicker="TLY portfolio, 31 Aug 2026">
      {/* One stacked bar, so the weights are comparable at a glance and the related-party
          share reads as a contiguous block rather than as five separate numbers. */}
      <div className="flex h-9 w-full overflow-hidden border border-line-strong">
        {HOLDINGS.map((x, i) => (
          <button
            key={x.ticker}
            type="button"
            onClick={() => setSel(i)}
            aria-pressed={i === active}
            aria-label={`${x.name}, ${x.pct}% of the fund`}
            {...bindActive(i)}
            className="h-full transition-opacity"
            style={{
              width: `${x.pct}%`,
              background: LINK[x.link].color,
              opacity: i === active ? 1 : 0.55,
              borderRight: '1px solid var(--color-paper-raised)',
            }}
          />
        ))}
        <span
          aria-hidden
          className="h-full"
          style={{ width: `${OTHER}%`, background: 'var(--color-line-strong)', opacity: 0.5 }}
        />
      </div>

      <div className="mt-1 flex justify-between font-mono text-[0.62rem] text-muted">
        <span>0%</span>
        <span>everything else {OTHER}%</span>
        <span>100%</span>
      </div>

      {/* The two numbers that matter, stated rather than left to be inferred from the bar. */}
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="border border-line px-3 py-2">
          <div className="font-mono text-xl font-semibold" style={{ color: WARN }}>
            {CHAIR.toFixed(1)}%
          </div>
          <div className="font-mono text-[0.66rem] leading-tight text-ink-soft">
            in companies chaired by Tera’s founder
          </div>
        </div>
        <div className="border border-line px-3 py-2">
          <div className="font-mono text-xl font-semibold" style={{ color: PRICE }}>
            {RELATED.toFixed(1)}%
          </div>
          <div className="font-mono text-[0.66rem] leading-tight text-ink-soft">
            in names the group chaired or underwrote
          </div>
        </div>
      </div>

      <ol className="mt-4 flex flex-col" style={{ margin: '1rem 0 0', padding: 0 }}>
        {HOLDINGS.map((x, i) => (
          <li key={x.ticker}>
            <button
              type="button"
              onClick={() => setSel(i)}
              aria-pressed={i === active}
              {...bindActive(i)}
              className="flex w-full items-center gap-2.5 py-1 text-left transition-opacity"
              style={{ opacity: i === active ? 1 : 0.62 }}
            >
              <span
                aria-hidden
                className="shrink-0"
                style={{ width: 10, height: 10, background: LINK[x.link].color }}
              />
              <span className="w-[4.2rem] shrink-0 font-mono text-[0.72rem] font-semibold text-ink">
                {x.ticker}
              </span>
              <span className="w-[3.1rem] shrink-0 text-right font-mono text-[0.72rem] text-ink-soft">
                {x.pct}%
              </span>
              <span className="truncate font-mono text-[0.68rem] text-muted">{x.name}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="mt-3 border-t border-line pt-3">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            style={{ width: 10, height: 10, background: LINK[h.link].color, display: 'inline-block' }}
          />
          <span className="font-mono text-sm font-semibold text-ink">
            {h.ticker} · {h.pct}%
          </span>
        </div>
        <div className="mt-1 font-mono text-[0.66rem] text-muted">{LINK[h.link].label}</div>
        <p className="mt-2 min-h-[4.5rem] text-sm text-ink-soft">{h.note}</p>
      </div>
    </Widget>
  );
}
