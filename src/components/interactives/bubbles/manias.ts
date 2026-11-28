import { BOOM } from './data';

/** Months from the peak shown on the chart. */
export const M0 = -36;
export const M1 = 48;

export type Mania = {
  id: string;
  label: string;
  peakLabel: string;
  color: string;
  today?: boolean;
  todayLabel?: string;
  /** m = months from the series' own peak, v = price as % of the peak */
  points: { m: number; v: number }[];
};

/** Linear interpolation on the same points the line is drawn from; null outside the data. */
export function valueAt(s: Mania, m: number): number | null {
  const p = s.points;
  if (!p.length || m < p[0].m || m > p[p.length - 1].m) return null;
  for (let i = 1; i < p.length; i++) {
    if (m <= p[i].m) {
      const a = p[i - 1];
      const b = p[i];
      const u = b.m === a.m ? 0 : (m - a.m) / (b.m - a.m);
      return a.v + (b.v - a.v) * u;
    }
  }
  return p[p.length - 1].v;
}

/** Build a series from dated values: rescales to % of the max and to months from its date. */
function fromDated(rows: [string, number][]) {
  const toM = (d: string) => {
    const [y, mo] = d.split('-').map(Number);
    return y * 12 + (mo - 1);
  };
  let peak = rows[0];
  for (const r of rows) if (r[1] > peak[1]) peak = r;
  const pm = toM(peak[0]);
  return rows
    .map(([d, v]) => ({ m: toM(d) - pm, v: (v / peak[1]) * 100 }))
    .filter((p) => p.m >= M0 && p.m <= M1);
}

// DATA: filled from sourced series (see the article's source notes).
export const MANIAS: Mania[] = [
  { id: 'rail', label: 'British railways', peakLabel: 'peak 1845', ...BOOM.rail, points: fromDated([]) },
  { id: 'radio', label: 'RCA (radio)', peakLabel: 'peak 1929', ...BOOM.radio, points: fromDated([]) },
  { id: 'fiber', label: 'Cisco', peakLabel: 'peak 2000', ...BOOM.fiber, points: fromDated([]) },
  { id: 'ai', label: 'Nvidia', peakLabel: 'peak Oct 2026', ...BOOM.ai, today: true, points: fromDated([]) },
];
