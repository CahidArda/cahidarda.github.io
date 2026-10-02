import { useState } from 'react';
import { FLOW, useHoverPreview, WARN, Widget } from './shared';

// What points which way. The honest shape of this question is a balance, not an argument, so
// the figure is two columns of evidence rather than a conclusion. Each item is a thing in the
// public record; the detail says what it is and what it does not establish.

type Side = 'own' | 'deliberate';

interface Item {
  side: Side;
  label: string;
  detail: string;
  href?: string;
}

const ITEMS: Item[] = [
  {
    side: 'own',
    label: 'No one coordinated 102,616 people',
    detail:
      'TLY had 102,616 investors by the end of August 2026. They bought a fund that was near the top of a public return table, through an app, which is what the platform exists to let them do.',
    href: 'https://fon.org.tr/fon/TLY',
  },
  {
    side: 'own',
    label: 'The loop needs no author',
    detail:
      'Self-inflating returns are a property of concentrated funds in thin markets, not of anyone’s intent. Nothing in the mechanism requires a decision to inflate anything.',
  },
  {
    side: 'own',
    label: 'It runs in ordinary funds too',
    detail:
      'van der Beck, Bouchaud and Villamaina find the same effect at lower amplitude across ordinary American and European funds, including plain ETFs, where no misconduct is alleged.',
    href: 'https://arxiv.org/pdf/2405.12768',
  },
  {
    side: 'deliberate',
    label: 'The group underwrote the IPO its own fund bought',
    detail:
      'Destek Faktoring floated in January 2025 through Tera Yatırım Menkul Değerler, and the largest single buyer at flotation was TLY, a Tera fund, at 15.42% of the offering.',
    href: 'https://halkarz.com/destek-finans-faktoring-a-s/',
  },
  {
    side: 'deliberate',
    label: '29% of the fund in one person’s companies',
    detail:
      'Three of the fund’s holdings, 28.7% of it, are companies chaired by Emre Tezmen, Tera’s founder and main shareholder. Two are the group’s own listed holding companies.',
    href: 'https://terafinansalyatirimlar.com/sayfa-yonetim-kurulu-22',
  },
  {
    side: 'deliberate',
    label: 'The holdings were chosen for low free float',
    detail:
      'A crowd does not select for free float. Concentrating in companies with few freely tradable shares is the condition the mechanism needs, and it is a portfolio decision.',
  },
  {
    side: 'deliberate',
    label: 'The regulator wrote rules at this exact structure',
    detail:
      'SPK stopped counting a group’s own fund holdings as free float on 4 June 2026, then capped fund positions in low-float companies on 28 August. Both describe the arrangement in Tera’s disclosures.',
    href: 'https://www.bloomberght.com/spkdan-serbest-fonlara-sert-fren-3786791',
  },
];

const SIDE: Record<Side, { label: string; color: string; note: string }> = {
  own: { label: 'could run on its own', color: FLOW, note: 'no intent required' },
  deliberate: { label: 'looks like decisions', color: WARN, note: 'intent not established' },
};

export default function EvidenceBalance() {
  const [sel, setSel] = useState(3);
  const [active, bindActive] = useHoverPreview(sel);
  const it = ITEMS[active];

  const column = (side: Side) => (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 border-b border-line pb-1.5">
        <span
          aria-hidden
          style={{ width: 10, height: 10, background: SIDE[side].color, display: 'inline-block' }}
        />
        <span className="font-mono text-[0.68rem] font-semibold text-ink">
          {SIDE[side].label}
        </span>
      </div>
      {ITEMS.map((x, i) =>
        x.side !== side ? null : (
          <button
            key={x.label}
            type="button"
            onClick={() => setSel(i)}
            aria-pressed={i === active}
            {...bindActive(i)}
            className="border-l-2 py-1 pl-2 text-left font-mono text-[0.7rem] leading-snug transition-opacity"
            style={{
              borderColor: SIDE[side].color,
              opacity: i === active ? 1 : 0.55,
              color: i === active ? 'var(--color-ink)' : 'var(--color-ink-soft)',
            }}
          >
            {x.label}
          </button>
        ),
      )}
      <span className="font-mono text-[0.6rem] text-muted">{SIDE[side].note}</span>
    </div>
  );

  return (
    <Widget title="What points which way" kicker="click an item">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
        {column('own')}
        {column('deliberate')}
      </div>

      <div className="mt-4 border-t border-line pt-3">
        <div className="flex items-start gap-2">
          <span
            aria-hidden
            className="mt-1 shrink-0"
            style={{ width: 10, height: 10, background: SIDE[it.side].color }}
          />
          <span className="font-mono text-[0.76rem] font-semibold leading-snug text-ink">
            {it.label}
          </span>
        </div>
        <p className="mt-2 min-h-[4.5rem] text-sm text-ink-soft">{it.detail}</p>
        {it.href && (
          <a
            href={it.href}
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-[0.68rem] text-muted underline underline-offset-2 hover:text-ink"
          >
            source →
          </a>
        )}
      </div>

      <p className="mt-3 border-t border-line pt-2 font-mono text-[0.66rem] leading-relaxed text-muted">
        Nothing on the right establishes intent, which is what a court has to find. Nothing on the
        left rules it out.
      </p>
    </Widget>
  );
}
