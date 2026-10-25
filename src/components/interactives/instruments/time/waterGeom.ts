/**
 * Episode 2, the water clock: physics and drawn coordinates for WaterClockPlate.tsx, swept by
 * time.sweep.ts.
 *
 * Physics (Torricelli): water leaves a small hole at speed √(2gh), so the level h in a vessel of
 * cross-section A(h) falls as  dh/dt = −k √h / A(h).  Units: depth 0..1 (full = 1), time in
 * hours, and k chosen so each vessel empties in exactly NIGHT hours. The "reading" is what an
 * EVENLY spaced scale says: elapsed = NIGHT · (1 − h), i.e. assuming the level falls at a
 * steady rate. The gap between that and the true elapsed time is the clock's error.
 *  - 'cyl'    straight walls: h = (1 − t/T)², the level falls fast when full, slowly near empty.
 *  - 'slope'  walls flaring upward, top diameter 46 cm over bottom 26 cm (the Karnak vessel's
 *             outside, McMaster AEA): the wide top drains slowly, so the fall is closer to even.
 *  - 'head'   Ctesibius: a header tank kept brim-full by an overflow feeds a straight receiving
 *             vessel through the hole at a CONSTANT head, so the float rises at a constant rate.
 * `clog` (0..0.3) narrows the hole: every flow is (1 − clog) times slower and the clock lags.
 */

export const NIGHT = 12; // hours from sunset (18:00 at the equinox) to sunrise
export const SUNSET = 18;
export type Vessel = 'cyl' | 'slope' | 'head';
export const TOP_OVER_BOTTOM = 46 / 26;

/** Radius profile r(h), h in 0..1, bottom radius 1. */
export const radius = (v: Vessel, h: number) => (v === 'slope' ? 1 + (TOP_OVER_BOTTOM - 1) * h : 1);

/** Time to drain from depth h to 0 with k = 1:  ∫₀ʰ A(u)/√u du  (A = r²; π folded into k). */
function drainIntegral(v: Vessel, h: number) {
  if (v !== 'slope') return 2 * Math.sqrt(h);
  // r = 1 + c u → r² = 1 + 2cu + c²u²;  ∫ u^-1/2 (1 + 2cu + c²u²) du = 2√u + (4c/3)u^{3/2} + (2c²/5)u^{5/2}
  const c = TOP_OVER_BOTTOM - 1;
  return 2 * Math.sqrt(h) + ((4 * c) / 3) * h ** 1.5 + ((2 * c * c) / 5) * h ** 2.5;
}
/** k so the full vessel empties in NIGHT hours. */
const kOf = (v: Vessel) => drainIntegral(v, 1) / NIGHT;

/** Depth after t hours (0 ≤ t), with the hole narrowed by `clog`. */
export function level(v: Vessel, t: number, clog = 0): number {
  if (v === 'head') return Math.min(1, (t * (1 - clog)) / NIGHT); // the receiver fills evenly
  const k = kOf(v) * (1 - clog);
  const target = drainIntegral(v, 1) - k * t; // remaining integral
  if (target <= 0) return 0;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2;
    if (drainIntegral(v, m) < target) lo = m;
    else hi = m;
  }
  return (lo + hi) / 2;
}
/** Elapsed hours an evenly spaced scale shows for a depth. */
export const readElapsed = (v: Vessel, h: number) => (v === 'head' ? NIGHT * h : NIGHT * (1 - h));
/** Where the true hour marks sit (depth at k hours), for drawing. */
export const trueMarks = (v: Vessel, clog = 0) => Array.from({ length: NIGHT + 1 }, (_, k) => level(v, k, clog));

export function waterClock(v: Vessel, t: number, clog = 0) {
  const h = level(v, t, clog);
  const elapsed = readElapsed(v, h);
  const errSec = (elapsed - t) * 3600; // negative = the clock is behind
  return { v, t, clog, h, elapsed, errSec, read: SUNSET + elapsed, now: SUNSET + t, empty: v !== 'head' && h <= 1e-6 };
}
export type WaterClock = ReturnType<typeof waterClock>;

/** Largest |error| over the night (seconds), by sampling every minute. */
export function worstError(v: Vessel, clog = 0) {
  let worst = 0;
  for (let m = 0; m <= NIGHT * 60; m++) {
    const w = waterClock(v, m / 60, clog);
    worst = Math.max(worst, Math.abs(w.errSec));
  }
  return worst;
}

/* ───────────────────────────── screen layout ───────────────────────────── */

export function vesselFrame(W: number, v: Vessel) {
  const H = 300;
  const narrow = W < 400;
  // the vessel occupies the left part; the level-vs-time chart the right part
  const vw = narrow ? W * 0.46 : Math.min(W * 0.42, 210);
  const cx = 14 + vw / 2;
  const yTop = 52;
  const yBot = 236;
  const depth = yBot - yTop;
  const rMaxPx = vw / 2 - 18;
  const rb = rMaxPx / (v === 'slope' ? TOP_OVER_BOTTOM : 1.15);
  const rAt = (h: number) => rb * radius(v === 'head' ? 'cyl' : v, h) * (v === 'head' ? 1.0 : 1);
  const yAt = (h: number) => yBot - h * depth;
  // chart
  const c = { x0: 14 + vw + (narrow ? 24 : 34), x1: W - 12, y0: yTop, y1: yBot };
  const xc = (t: number) => c.x0 + (t / NIGHT) * (c.x1 - c.x0);
  const yc = (h: number) => c.y1 - h * (c.y1 - c.y0);
  return { W, H, narrow, cx, yTop, yBot, rb, rAt, yAt, c, xc, yc };
}

/** The chart's curve: level against time, sampled. */
export function chartCurve(v: Vessel, W: number, clog = 0) {
  const f = vesselFrame(W, v);
  return Array.from({ length: 97 }, (_, i) => {
    const t = (i / 96) * NIGHT;
    return { x: f.xc(t), y: f.yc(level(v, t, clog)) };
  });
}
