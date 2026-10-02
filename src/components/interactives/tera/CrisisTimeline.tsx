import { useState } from 'react';
import { FLOW, NAV, PRICE, useHoverPreview, WARN, Widget } from './shared';

// The timeline, coloured by Kindleberger stage rather than by month, so the shape of the
// episode is visible before any individual date is read. Vertical, so it reflows on mobile
// instead of scrolling sideways.

type Stage = 'structural' | 'displacement' | 'euphoria' | 'distress' | 'revulsion';

const STAGE: Record<Stage, { label: string; color: string }> = {
  structural: { label: 'structural conditions', color: 'var(--color-muted)' },
  displacement: { label: 'displacement', color: FLOW },
  euphoria: { label: 'euphoria', color: PRICE },
  distress: { label: 'distress', color: NAV },
  revulsion: { label: 'revulsion', color: WARN },
};

interface Node {
  date: string;
  title: string;
  stage: Stage;
  detail: string;
  href?: string;
}

const NODES: Node[] = [
  {
    date: 'Mar 2021',
    title: 'TLY is founded',
    stage: 'structural',
    detail:
      'Tera Portföy Birinci Serbest Fon launches as a hedge fund (serbest fon), a vehicle legally restricted to qualified investors. At this point almost nobody is in it.',
  },
  {
    date: '29–31 Jan 2025',
    title: 'Destek Faktoring IPO',
    stage: 'displacement',
    detail:
      'Destek Finans Faktoring floats 25% of the company at ₺46.98 a share, raising about ₺3.9bn. The underwriter is Tera Yatırım Menkul Değerler. The largest single allocatee is TLY, Tera’s own fund, at 15.42%. This is the instrument the loop will run on.',
    href: 'https://halkarz.com/destek-finans-faktoring-a-s/',
  },
  {
    date: 'Through 2025',
    title: 'TEFAS puts hedge funds in the bank app',
    stage: 'displacement',
    detail:
      'Serbest fonlar become reachable through TEFAS, the platform inside every Turkish bank app. A product designed for a narrow professional audience acquires a retail distribution channel. The qualified-investor threshold is still ₺1m, a bar that years of inflation had quietly lowered in real terms.',
  },
  {
    date: '18 Dec 2025',
    title: 'SPK raises the qualified-investor bar to ₺10m',
    stage: 'distress',
    detail:
      'The threshold for qualified-investor status goes from ₺1m to ₺10m of financial assets. The regulator is, in effect, conceding that the old bar had stopped doing its job. The money already inside is not affected.',
    href: 'https://www.finansopia.com/ekonomi/spkdan-fonlarda-manipulasyon-duzenlemesi/',
  },
  {
    date: 'Through 2026',
    title: 'The returns become unignorable',
    stage: 'euphoria',
    detail:
      'TLY posts a one-year return of 669%, a three-year return of 71,787% and a five-year return of 762,829%. By late August it holds roughly ₺244bn for 102,616 investors, with 24.8% of the portfolio in Destek Faktoring and another 22.4% in two listed Tera group holdings.',
    href: 'https://fon.org.tr/fon/TLY',
  },
  {
    date: '4 Jun 2026',
    title: 'Fund-held shares stop counting as free float',
    stage: 'distress',
    detail:
      'SPK decision 34/1044 rules that shares held through funds controlled by a company’s dominant shareholder no longer count toward its free float, with MKK calculating daily. The regulator has named the mechanism.',
    href: 'https://www.paramedya.com/devami/142075/spkdan-fiili-dolasim-hamlesi-patronun-fonundaki-hisse-artik-serbest-sayilmayacak/',
  },
  {
    date: '28 Aug 2026',
    title: 'Position caps tied to free float',
    stage: 'distress',
    detail:
      'Hedge funds are capped at 8% of a company’s circulating shares where free float is under 25%, scaling down to 2% above 75%. Positions over 5% of a fund’s value cannot together exceed 20% of it. The inflow side of the loop is now illegal. Everything after this is the loop running backwards.',
    href: 'https://www.bloomberght.com/spkdan-serbest-fonlara-sert-fren-3786791',
  },
  {
    date: '15 Sep 2026',
    title: 'Pusula defaults',
    stage: 'revulsion',
    detail:
      'Pusula Portföy announces it cannot meet redemption payments on some funds. Eighteen days after the position caps.',
  },
  {
    date: '16 Sep 2026',
    title: 'Tera defaults; BIST 100 falls 5.5%',
    stage: 'revulsion',
    detail:
      'Tera Portföy declares default on its money market fund and an equity fund, and changes exit terms on six hedge funds. The index drops 5.54% in a day.',
  },
  {
    date: '17 Sep 2026',
    title: 'SPK freezes 131 funds',
    stage: 'revulsion',
    detail:
      'Trading is suspended in 131 funds across seven managers: Tera, Pusula, Hedef, Atlas, A1, Pardus and Bulls. İş Bankası is appointed liquidator for Tera’s funds, Ziraat for the rest, with assets held at Takasbank.',
    href: 'https://www.aa.com.tr/tr/ekonomi/spk-tasfiye-edilen-yatirim-fonlarinin-tasfiye-surecine-iliskin-usul-ve-esaslari-belirledi/4060721',
  },
  {
    date: '23 Sep 2026',
    title: '455,758 investors',
    stage: 'revulsion',
    detail:
      'SPK publishes the number of people holding units in the frozen funds. Not a professional clientele.',
    href: 'https://www.diken.com.tr/tasfiye-edilen-131-fon-455-bin-758-kisi-yatirim-yapmis/',
  },
  {
    date: '25–26 Sep 2026',
    title: 'Arrests reach the board',
    stage: 'revulsion',
    detail:
      'Erkan Kilimci, a deputy governor of the central bank from 2016 to 2018 and a Tera board member, is arrested along with ten others. Prosecutors are running the case through the financial-crimes and money-laundering bureau.',
    href: 'https://t24.com.tr/gundem/fon-krizi-sorusturmasinda-yeni-gelisme-eski-merkez-bankasi-baskan-yardimcisi-erkan-kilimci-dahil-11-kisi-tutuklandi,1349763',
  },
  {
    date: '30 Sep 2026',
    title: 'Five companies to TMSF',
    stage: 'revulsion',
    detail:
      'BDDK transfers Tera, Destek and Hedef investment banks plus Destek Finans Faktoring and Tera Finans Faktoring to the deposit insurance fund. The three banks are 0.22% of banking sector assets, which is the containment argument in one number.',
    href: 'https://www.finansingundemi.com/haber/tera-destek-ve-hedef-yatirim-bankalari-tmsfye-devredildi/1910039',
  },
  {
    date: '1 Oct 2026',
    title: 'Interim payments, capped at ₺1m',
    stage: 'revulsion',
    detail:
      'SPK authorises advances against final liquidation proceeds: up to ₺1m per investor against net investment as calculated by MKK, money market funds first. Anyone below the cap is made whole on paper; anyone above waits for the liquidation.',
    href: 'https://www.hurriyet.com.tr/gundem/spkdan-tasfiye-edilen-fonlar-icin-karar-ara-odemeler-yapilacak-43325364',
  },
];

export default function CrisisTimeline() {
  const [sel, setSel] = useState(NODES.length - 1);
  const [active, bindActive] = useHoverPreview(sel);
  const n = NODES[active];

  return (
    <Widget title="Five years, coloured by stage" kicker="click a date to read it">
      <p className="mb-4 text-sm text-ink-soft">
        Kindleberger’s model says a mania has a shape: conditions, a displacement that starts it,
        euphoria, distress as the conditions turn, then revulsion. The dates below are coloured by
        which stage they belong to rather than by what kind of event they were.
      </p>

      <div className="flex flex-wrap gap-x-3 gap-y-1.5">
        {(Object.keys(STAGE) as Stage[]).map((k) => (
          <span
            key={k}
            className="inline-flex items-center gap-1.5 font-mono text-[0.64rem] text-ink-soft"
          >
            <span
              aria-hidden
              style={{ width: 10, height: 10, background: STAGE[k].color, display: 'inline-block' }}
            />
            {STAGE[k].label}
          </span>
        ))}
      </div>

      <ol className="relative mt-4 flex flex-col gap-0" style={{ margin: '1rem 0 0', padding: 0 }}>
        <span
          aria-hidden
          className="absolute bottom-2 left-[5px] top-2 w-px"
          style={{ background: 'var(--color-line-strong)' }}
        />
        {NODES.map((node, i) => (
          <li key={node.title} className="relative">
            <button
              type="button"
              onClick={() => setSel(i)}
              aria-pressed={i === active}
              className="flex w-full items-start gap-3 py-1.5 text-left transition-opacity"
              style={{ opacity: i === active ? 1 : 0.6 }}
              {...bindActive(i)}
            >
              <span
                aria-hidden
                className="relative z-10 mt-1 shrink-0 rounded-full"
                style={{
                  width: 11,
                  height: 11,
                  background: i === active ? STAGE[node.stage].color : 'var(--color-paper)',
                  border: `2px solid ${STAGE[node.stage].color}`,
                }}
              />
              <span className="w-[5.5rem] shrink-0 font-mono text-[0.66rem] leading-tight text-muted">
                {node.date}
              </span>
              <span
                className="font-mono text-[0.76rem] leading-tight"
                style={{
                  color: i === active ? 'var(--color-ink)' : 'var(--color-ink-soft)',
                  fontWeight: i === active ? 600 : 400,
                }}
              >
                {node.title}
              </span>
            </button>
          </li>
        ))}
      </ol>

      <div className="mt-4 border-t border-line pt-3">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            style={{
              width: 10,
              height: 10,
              background: STAGE[n.stage].color,
              display: 'inline-block',
            }}
          />
          <span className="font-mono text-sm font-semibold text-ink">
            {n.date} · {n.title}
          </span>
        </div>
        <p className="mt-2 min-h-[5rem] text-sm text-ink-soft">{n.detail}</p>
        {n.href && (
          <a
            href={n.href}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-block font-mono text-[0.68rem] text-muted underline underline-offset-2 hover:text-ink"
          >
            source →
          </a>
        )}
      </div>
    </Widget>
  );
}
