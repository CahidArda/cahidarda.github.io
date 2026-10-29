/**
 * Shared primitives for the Ottoman-calendar figures. Reuses the data-platform frame so the
 * post reads as house style. One fixed colour per idea, learned in the first figure and reused
 * after: Hizir (summer half) green, Kasim (winter half) blue, and one colour per month-name
 * source. All figures are static (no hooks), so they render at build time with no client JS.
 */
export { Widget } from '../dataplat/shared';

export const HIZIR = 'var(--color-tk-hizir)';
export const KASIM = 'var(--color-tk-kasim)';
export const ARAM = 'var(--color-tk-aram)';
export const LATIN = 'var(--color-tk-latin)';
export const TURK = 'var(--color-tk-turk)';
export const ACCENT = 'var(--color-accent)';
export const MONO = 'var(--font-mono)';

export const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];
const LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** Day of a common (non-leap) year, 0-based: doy(1, 1) = 0, doy(12, 31) = 364. */
export function doy(month: number, day: number): number {
  let d = day - 1;
  for (let m = 0; m < month - 1; m++) d += LEN[m];
  return d;
}

/** Small mono label inside an SVG figure. */
export function Label({
  x,
  y,
  children,
  anchor = 'middle',
  muted = true,
  size = 11,
  bold = false,
  color,
}: {
  x: number;
  y: number;
  children: React.ReactNode;
  anchor?: 'start' | 'middle' | 'end';
  muted?: boolean;
  size?: number;
  bold?: boolean;
  color?: string;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="middle"
      style={{
        fill: color ?? (muted ? 'var(--color-muted)' : 'var(--color-ink)'),
        fontFamily: MONO,
        fontSize: size,
        fontWeight: bold ? 700 : 400,
      }}
    >
      {children}
    </text>
  );
}

/** Colour-swatch legend row, HTML (wraps on phones). */
export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[0.7rem] text-muted">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5" style={{ background: it.color }} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}
