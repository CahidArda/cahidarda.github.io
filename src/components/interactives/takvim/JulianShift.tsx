/**
 * Why the saints' days land on 6 May and 8 November: the feasts are fixed on the Julian
 * calendar, which now runs 13 days behind the Gregorian one. Below, the gap by century.
 * One SVG coordinate system; arrows anchored to box edges (skill 4.1, 4.2).
 */
import { ArrowDefs } from '../dataplat/shared';
import { ACCENT, HIZIR, KASIM, Label, Widget } from './shared';

const J = { cx: 82, w: 124 }; // Julian column
const G = { cx: 300, w: 150 }; // Gregorian column
const H = 38;

const ROWS = [
  { y: 62, j: '23 Apr', js: 'St George', g: '6 May', gs: 'Hıdırellez', color: HIZIR },
  { y: 112, j: '26 Oct', js: 'St Demetrius', g: '8 Nov', gs: 'Kasım', color: KASIM },
];

const GAPS = [
  { era: '1700s', d: 11 },
  { era: '1800s', d: 12 },
  { era: '1900-2099', d: 13, now: true },
  { era: '2100s', d: 14 },
];

function Box({
  cx,
  y,
  w,
  title,
  sub,
  color,
}: {
  cx: number;
  y: number;
  w: number;
  title: string;
  sub: string;
  color: string;
}) {
  return (
    <g>
      <rect
        x={cx - w / 2}
        y={y - H / 2}
        width={w}
        height={H}
        style={{ fill: 'var(--color-paper)', stroke: color, strokeWidth: 2 }}
      />
      <Label x={cx} y={y - 7} size={12} bold muted={false}>
        {title}
      </Label>
      <Label x={cx} y={y + 9} size={10}>
        {sub}
      </Label>
    </g>
  );
}

export default function JulianShift() {
  const segW = 360 / GAPS.length;
  return (
    <Widget title="Same feast, two calendars" kicker="Julian + 13 days = Gregorian">
      <svg
        viewBox="0 0 400 210"
        className="w-full"
        role="img"
        aria-label="23 April Julian, St George, is 6 May Gregorian, Hidirellez. 26 October Julian, St Demetrius, is 8 November Gregorian, Kasim. The gap was 11 days in the 1700s, 12 in the 1800s, 13 from 1900 to 2099, and becomes 14 in 2100."
      >
        <ArrowDefs />
        <Label x={J.cx} y={22} size={10}>
          CHURCH · JULIAN
        </Label>
        <Label x={G.cx} y={22} size={10}>
          CIVIL · GREGORIAN
        </Label>

        {ROWS.map((r) => (
          <g key={r.g}>
            <line
              x1={J.cx + J.w / 2}
              y1={r.y}
              x2={G.cx - G.w / 2}
              y2={r.y}
              markerEnd="url(#dp-arrow)"
              style={{ stroke: ACCENT, strokeWidth: 2 }}
            />
            <Label x={(J.cx + J.w / 2 + G.cx - G.w / 2) / 2} y={r.y - 9} size={10} color={ACCENT}>
              +13
            </Label>
            <Box
              cx={J.cx}
              y={r.y}
              w={J.w}
              title={r.j}
              sub={r.js}
              color="var(--color-line-strong)"
            />
            <Box cx={G.cx} y={r.y} w={G.w} title={r.g} sub={r.gs} color={r.color} />
          </g>
        ))}

        <Label x={200} y={157} size={10}>
          THE GAP, IN DAYS
        </Label>
        {GAPS.map((g, i) => {
          const x = 20 + i * segW;
          return (
            <g key={g.era}>
              <rect
                x={x}
                y={168}
                width={segW - 4}
                height={32}
                style={{
                  fill: g.now ? ACCENT : 'var(--color-paper)',
                  stroke: g.now ? ACCENT : 'var(--color-line-strong)',
                  strokeWidth: 1.5,
                }}
              />
              <Label
                x={x + (segW - 4) / 2}
                y={178}
                size={9.5}
                color={g.now ? 'var(--color-paper)' : undefined}
              >
                {g.era}
              </Label>
              <Label
                x={x + (segW - 4) / 2}
                y={192}
                size={12}
                bold
                color={g.now ? 'var(--color-paper)' : 'var(--color-ink)'}
              >
                {g.d}
              </Label>
            </g>
          );
        })}
      </svg>
    </Widget>
  );
}
