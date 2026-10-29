/**
 * The twelve months in three calendars that share one Mesopotamian root: the Ottoman (Rumi)
 * names, Turkish today, and the Levantine Arabic names still used for Gregorian months in
 * Iraq, Syria, Jordan, Lebanon and Palestine. Each name is tinted by where the word comes
 * from. One CSS grid for every cell so columns stay one width across rows (skill 10).
 */
import { ARAM, LATIN, Legend, TURK, Widget } from './shared';

type Src = 'aram' | 'latin' | 'turk';
const COLOR: Record<Src, string> = { aram: ARAM, latin: LATIN, turk: TURK };

interface Name {
  n: string;
  s: Src;
  g?: string; // gloss
}

interface Row {
  en: string;
  ott: Name;
  tr: Name;
  lev: Name;
}

const A = (n: string, g?: string): Name => ({ n, s: 'aram', g });
const L = (n: string): Name => ({ n, s: 'latin' });
const T = (n: string, g: string): Name => ({ n, s: 'turk', g });

const ROWS: Row[] = [
  { en: 'Jan', ott: A('Kânûn-ı Sânî', '2nd Kânûn'), tr: T('Ocak', 'stove'), lev: A('Kānūn al-Thānī') },
  { en: 'Feb', ott: A('Şubat'), tr: A('Şubat'), lev: A('Shubāṭ') },
  { en: 'Mar', ott: L('Mart'), tr: L('Mart'), lev: A('Ādhār') },
  { en: 'Apr', ott: A('Nisan'), tr: A('Nisan'), lev: A('Naysān') },
  { en: 'May', ott: L('Mayıs'), tr: L('Mayıs'), lev: A('Ayyār') },
  { en: 'Jun', ott: A('Haziran'), tr: A('Haziran'), lev: A('Ḥazīrān') },
  { en: 'Jul', ott: A('Temmuz', 'god Tammuz'), tr: A('Temmuz'), lev: A('Tammūz') },
  { en: 'Aug', ott: L('Ağustos'), tr: L('Ağustos'), lev: A('Āb') },
  { en: 'Sep', ott: A('Eylül'), tr: A('Eylül'), lev: A('Aylūl') },
  { en: 'Oct', ott: A('Teşrin-i Evvel', '1st Teşrin'), tr: T('Ekim', 'sowing'), lev: A('Tishrīn al-Awwal') },
  { en: 'Nov', ott: A('Teşrin-i Sânî', '2nd Teşrin'), tr: T('Kasım', 'divider'), lev: A('Tishrīn al-Thānī') },
  { en: 'Dec', ott: A('Kânûn-ı Evvel', '1st Kânûn'), tr: T('Aralık', 'gap'), lev: A('Kānūn al-Awwal') },
];

function Cell({ x, strong }: { x: Name; strong?: boolean }) {
  return (
    <div
      className="py-1.5 pl-2 pr-1 leading-tight"
      style={{ borderLeft: `3px solid ${COLOR[x.s]}` }}
    >
      <div className={strong ? 'font-semibold text-ink' : 'text-ink'}>{x.n}</div>
      {x.g && <div className="font-mono text-[0.62rem] text-muted">{x.g}</div>}
    </div>
  );
}

const HEAD = ['', 'Ottoman', 'Turkish', 'Levant'];

export default function MonthNames() {
  return (
    <Widget title="Twelve months, three calendars" kicker="1945 renames in bold">
      <div
        className="grid gap-x-2 text-[0.8rem]"
        style={{ gridTemplateColumns: '2.2rem repeat(3, minmax(min-content, 1fr))' }}
      >
        {HEAD.map((h, i) => (
          <div key={i} className="label border-b border-line pb-1.5 text-[0.62rem]">
            {h}
          </div>
        ))}
        {ROWS.map((r) => {
          const renamed = r.tr.s === 'turk';
          const bg = renamed ? 'color-mix(in srgb, var(--color-tk-turk) 12%, transparent)' : undefined;
          return (
            <div key={r.en} className="contents">
              <div className="flex items-center font-mono text-[0.7rem] text-muted" style={{ background: bg }}>
                {r.en}
              </div>
              <div style={{ background: bg }}>
                <Cell x={r.ott} />
              </div>
              <div style={{ background: bg }}>
                <Cell x={r.tr} strong={renamed} />
              </div>
              <div style={{ background: bg }}>
                <Cell x={r.lev} />
              </div>
            </div>
          );
        })}
      </div>
      <Legend
        items={[
          { color: ARAM, label: 'Babylonian, via Aramaic' },
          { color: LATIN, label: 'Latin' },
          { color: TURK, label: 'renamed in 1945' },
        ]}
      />
    </Widget>
  );
}
