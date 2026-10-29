/**
 * The named days of the Anatolian/Ottoman folk year on one vertical year bar (Jan at the
 * top). The bar is coloured by the two halves (Hizir / Kasim); spans (Erbain, Hamsin,
 * Eyyam-ı bahur) sit in a side column; single days are dots. Labels are spread so they
 * never collide, with a leader line back to the date. One SVG coordinate system.
 */
import { ACCENT, HIZIR, KASIM, Label, Legend, MONTHS, TURK, Widget, doy } from './shared';

const VB = { w: 400, h: 560 };
const TOP = 20;
const PX_PER_DAY = 1.425;
const BAR = { x: 70, w: 16 };
const BAND = { x: 92, w: 9 };
const LABEL_X = 120;
const MIN_GAP = 34;

const y = (d: number) => TOP + d * PX_PER_DAY;

interface Ev {
  title: string;
  sub: string;
  target: number; // day the leader points at
  dots?: number[]; // single days marked on the bar
}

const EVENTS: Ev[] = [
  { title: 'Erbain · "forty"', sub: '22 Dec to 30 Jan · the deep cold', target: doy(1, 14) },
  { title: 'Hamsin · "fifty"', sub: '31 Jan to 21 Mar · winter loosens', target: doy(2, 20) },
  {
    title: 'Cemre · "ember"',
    sub: 'air 20 Feb · water 27 Feb · earth 6 Mar',
    target: doy(2, 27),
    dots: [doy(2, 20), doy(2, 27), doy(3, 6)],
  },
  { title: 'Nevruz · "new day"', sub: '21 Mar · equinox, Persian new year', target: doy(3, 21), dots: [doy(3, 21)] },
  { title: 'Hıdırellez', sub: '6 May · summer half begins', target: doy(5, 6), dots: [doy(5, 6)] },
  { title: 'Eyyam-ı bahur', sub: '1 to 8 Aug · the hottest days', target: doy(8, 4) },
  { title: 'Kasım · "divider"', sub: '8 Nov · winter half begins', target: doy(11, 8), dots: [doy(11, 8)] },
  { title: 'Erbain begins', sub: '22 Dec · the longest night', target: doy(12, 22), dots: [doy(12, 22)] },
];

// spread labels top-down so consecutive blocks keep MIN_GAP
function layout(evs: Ev[]) {
  const out: number[] = [];
  evs.forEach((e, i) => {
    const want = y(e.target);
    out.push(i === 0 ? want : Math.max(want, out[i - 1] + MIN_GAP));
  });
  return out;
}

const SPANS = [
  { from: doy(12, 22), to: 365, color: KASIM }, // Erbain, December part
  { from: 0, to: doy(1, 31), color: KASIM }, // Erbain, January part
  { from: doy(1, 31), to: doy(3, 22), color: KASIM, light: true }, // Hamsin
  { from: doy(8, 1), to: doy(8, 9), color: TURK }, // Eyyam-ı bahur
];

export default function FolkYear() {
  const ly = layout(EVENTS);
  const HIDIR = doy(5, 6);
  const KAS = doy(11, 8);
  return (
    <Widget title="The folk year" kicker="named days and spans">
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="A vertical year from January to December. Erbain, 22 December to 30 January. Hamsin, 31 January to 21 March. Three cemre, 20 February, 27 February, 6 March. Nevruz, 21 March. Hidirellez, 6 May. Eyyam-i bahur, 1 to 8 August. Kasim, 8 November."
      >
        {/* the year bar, coloured by the two halves */}
        <rect x={BAR.x} y={y(0)} width={BAR.w} height={y(HIDIR) - y(0)} style={{ fill: KASIM, opacity: 0.8 }} />
        <rect x={BAR.x} y={y(HIDIR)} width={BAR.w} height={y(KAS) - y(HIDIR)} style={{ fill: HIZIR, opacity: 0.8 }} />
        <rect x={BAR.x} y={y(KAS)} width={BAR.w} height={y(365) - y(KAS)} style={{ fill: KASIM, opacity: 0.8 }} />

        {/* month ticks + names */}
        {MONTHS.map((m, i) => {
          const d0 = doy(i + 1, 1);
          return (
            <g key={m}>
              <line
                x1={BAR.x - 4}
                y1={y(d0)}
                x2={BAR.x + BAR.w}
                y2={y(d0)}
                style={{ stroke: 'var(--color-paper-raised)', strokeWidth: 1.5 }}
              />
              <Label x={BAR.x - 8} y={y(d0 + 15)} anchor="end" size={10}>
                {m}
              </Label>
            </g>
          );
        })}

        {/* spans in the side column */}
        {SPANS.map((s, i) => (
          <rect
            key={i}
            x={BAND.x}
            y={y(s.from)}
            width={BAND.w}
            height={y(s.to) - y(s.from)}
            style={{ fill: s.color, opacity: s.light ? 0.45 : 1 }}
          />
        ))}

        {/* leaders + labels + day dots */}
        {EVENTS.map((e, i) => (
          <g key={e.title}>
            <polyline
              points={`${BAND.x + BAND.w + 3},${y(e.target)} ${LABEL_X - 4},${ly[i]}`}
              style={{ fill: 'none', stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
            />
            <Label x={LABEL_X} y={ly[i] - 7} anchor="start" size={12} bold muted={false}>
              {e.title}
            </Label>
            <Label x={LABEL_X} y={ly[i] + 9} anchor="start" size={10.5}>
              {e.sub}
            </Label>
            {e.dots?.map((d) => (
              <circle
                key={d}
                cx={BAR.x + BAR.w / 2}
                cy={y(d)}
                r={3.6}
                style={{ fill: 'var(--color-paper)', stroke: ACCENT, strokeWidth: 2 }}
              />
            ))}
          </g>
        ))}
      </svg>
      <Legend
        items={[
          { color: KASIM, label: 'Kasım half / cold spans' },
          { color: HIZIR, label: 'Hızır half' },
          { color: TURK, label: 'heat span' },
        ]}
      />
    </Widget>
  );
}
