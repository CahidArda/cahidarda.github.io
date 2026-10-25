/**
 * The time post's precision ladder: the series' TIME rungs with plainer chart labels, and the
 * pure layout the plate uses (copied from PrecisionLadder's x/y), so a script can check that
 * every rung sits at its value and no two labels collide.
 */
import type { LadderConfig } from '../PrecisionLadder';
import { TIME, TIME_TICKS, fmtTime } from '../ladderData';

/** Same values as TIME_LADDER in PrecisionLadder.tsx (kept here so node can run the sweep). */
const TIME_LADDER = { rungs: TIME, ticks: TIME_TICKS, lo: -14, hi: 4, fmt: fmtTime, per: 'per day', start: 3 };

const SHORT: Record<string, string> = {
  'sundial, clepsydra': 'sundial, water clock',
  verge: 'verge clock',
  pendulum: 'pendulum clock',
  H4: 'Harrison H4',
  Shortt: 'Shortt clock',
  quartz: 'quartz clock',
  caesium: 'caesium clock',
  'NIST-F2': 'NIST-F2',
  optical: 'optical clock',
};

/** Vertical nudges (px) so neighbouring labels clear each other; found by the sweep. */
const DY: Record<number, number> = { 0: -12, 1: 6, 3: -7, 4: 3, 6: -3 };

export const CLOCK_LADDER: LadderConfig = {
  ...TIME_LADDER,
  rungs: TIME_LADDER.rungs.map((r) => ({ ...r, short: SHORT[r.short] ?? r.short })),
  unit: 'seconds gained or lost per day',
  dy: (i, narrow) => (DY[i] ?? 0) * (narrow ? 0.8 : 1),
  // the Shortt clock (1925) sits just below-left of quartz (1935): both labels go left,
  // stacked, so neither covers the other's mark
  side: (i) => (i === 4 ? 'left' : undefined),
};

/** Same maths as PrecisionLadder (keep in sync). */
export function ladderLayout(cfg: LadderConfig, W: number) {
  const yrs = cfg.years ?? { y0: -1500, yb: 1500, y1: 2040, split: 0.3 };
  const narrow = W < 520;
  const H = narrow ? 340 : 400;
  const m = { l: narrow ? 46 : 58, r: narrow ? 14 : 22, t: 22, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const x = (yr: number) =>
    yr <= yrs.yb
      ? m.l + ((yr - yrs.y0) / (yrs.yb - yrs.y0)) * yrs.split * iw
      : m.l + yrs.split * iw + ((yr - yrs.yb) / (yrs.y1 - yrs.yb)) * (1 - yrs.split) * iw;
  const y = (v: number) => m.t + ((Math.log10(v) - cfg.lo) / (cfg.hi - cfg.lo)) * ih;
  const size = narrow ? 11.5 : 12.5;
  const rungs = cfg.rungs.map((r, i) => {
    const px = x(r.year);
    const py = y(r.value);
    const forced = cfg.side?.(i);
    const right = forced ? forced === 'right' : px < m.l + iw * 0.8;
    const dy = cfg.dy ? cfg.dy(i, narrow) : 0;
    const w = r.short.length * size * 0.5;
    const tx = px + (right ? 12 : -12);
    const x0 = right ? tx : tx - w;
    return { r, px, py, box: { x0, x1: x0 + w, y0: py + dy - size * 0.55, y1: py + dy + size * 0.45 } };
  });
  return { W, H, m, iw, ih, narrow, x, y, rungs };
}
