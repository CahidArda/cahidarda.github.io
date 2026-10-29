/**
 * The Ottoman folk year as one ring: Rûz-ı Hızır (6 May to 7 Nov, 186 days) and Rûz-ı Kasım
 * (8 Nov to 5 May, 179 days). Jan 1 at the top, clockwise. The two hinge days are called out
 * with the saint whose feast they sit on. One SVG coordinate system (skill 4.1).
 */
import { ACCENT, HIZIR, KASIM, Label, MONTHS, MONO, Widget, doy } from './shared';

const VB = { w: 400, h: 300 };
const C = { x: 200, y: 150 };
const R_OUT = 104;
const R_IN = 76;

const ang = (d: number) => (d / 365) * Math.PI * 2 - Math.PI / 2;
const pt = (d: number, r: number) => ({
  x: C.x + r * Math.cos(ang(d)),
  y: C.y + r * Math.sin(ang(d)),
});

function arc(d0: number, d1: number) {
  const large = d1 - d0 > 365 / 2 ? 1 : 0;
  const a = pt(d0, R_OUT);
  const b = pt(d1, R_OUT);
  const c = pt(d1, R_IN);
  const e = pt(d0, R_IN);
  return `M${a.x} ${a.y} A${R_OUT} ${R_OUT} 0 ${large} 1 ${b.x} ${b.y} L${c.x} ${c.y} A${R_IN} ${R_IN} 0 ${large} 0 ${e.x} ${e.y} Z`;
}

const HIDIR = doy(5, 6); // 125
const KAS = doy(11, 8); // 311

export default function YearSplit() {
  const monthStarts = MONTHS.map((_, i) => doy(i + 1, 1));
  const hA = pt(HIDIR, R_OUT + 2);
  const kA = pt(KAS, R_OUT + 2);
  return (
    <Widget title="One year, two seasons" kicker="Ottoman folk calendar">
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        role="img"
        aria-label="A ring of the year split into the days of Hizir, 6 May to 7 November, 186 days, and the days of Kasim, 8 November to 5 May, 179 days."
      >
        <path d={arc(HIDIR, KAS)} style={{ fill: HIZIR, opacity: 0.85 }} />
        <path d={arc(KAS, HIDIR + 365)} style={{ fill: KASIM, opacity: 0.85 }} />

        {/* month boundaries + short names inside the ring */}
        {monthStarts.map((d, i) => {
          const a = pt(d, R_IN);
          const b = pt(d, R_OUT);
          const lab = pt(d + 15, R_IN - 13);
          return (
            <g key={i}>
              <line
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                style={{ stroke: 'var(--color-paper-raised)', strokeWidth: 1.5 }}
              />
              <Label x={lab.x} y={lab.y} size={8.5}>
                {MONTHS[i]}
              </Label>
            </g>
          );
        })}

        {/* centre totals */}
        <Label x={C.x} y={C.y - 10} size={11} bold color={HIZIR}>
          Hızır 186 d
        </Label>
        <Label x={C.x} y={C.y + 10} size={11} bold color={KASIM}>
          Kasım 179 d
        </Label>

        {/* hinge: Hıdırellez (lower right) */}
        <circle
          cx={hA.x}
          cy={hA.y}
          r={4.5}
          style={{ fill: 'var(--color-paper)', stroke: ACCENT, strokeWidth: 2 }}
        />
        <polyline
          points={`${hA.x + 4},${hA.y + 3} ${308},${232} ${318},${232}`}
          style={{ fill: 'none', stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        <Label x={322} y={226} anchor="start" size={12} bold muted={false}>
          Hıdırellez
        </Label>
        <Label x={322} y={242} anchor="start" size={10}>
          6 May
        </Label>
        <Label x={322} y={256} anchor="start" size={10}>
          St George
        </Label>

        {/* hinge: Kasım (upper left) */}
        <circle
          cx={kA.x}
          cy={kA.y}
          r={4.5}
          style={{ fill: 'var(--color-paper)', stroke: ACCENT, strokeWidth: 2 }}
        />
        <polyline
          points={`${kA.x - 4},${kA.y - 3} ${92},${62} ${82},${62}`}
          style={{ fill: 'none', stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        <Label x={78} y={46} anchor="end" size={12} bold muted={false}>
          Kasım
        </Label>
        <Label x={78} y={62} anchor="end" size={10}>
          8 Nov
        </Label>
        <Label x={78} y={76} anchor="end" size={10}>
          St Demetrius
        </Label>

        <text
          x={C.x}
          y={VB.h - 8}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: MONO, fontSize: 9.5 }}
        >
          Jan 1 at the top · clockwise
        </text>
      </svg>
    </Widget>
  );
}
