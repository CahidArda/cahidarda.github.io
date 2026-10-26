/**
 * Episode 3, the sine quadrant (rubʿ al-mujayyab, rubu tahtası) as a procedure: the numbers and
 * every drawn coordinate of QuadrantPlate.tsx, swept by time.sweep.ts.
 *
 * The board. Apex at the top-left corner, the sight edge along the TOP, the other straight edge
 * down the LEFT, the 90° rim between them; radius = 60 parts. A thread at angle θ from the left
 * edge (θ = 0 hangs straight down the left edge) crosses radius r at
 *     x = r sin θ   (read on the TOP scale, 0..60 to the right)
 *     y = r cos θ   (read on the LEFT scale, 0..60 downward).
 * So one thread does a sine on the top scale and a cosine on the left scale, and a bead at
 * distance r along the thread multiplies: x = r · sin θ. That is all the board needs.
 *
 * The muvakkit's working for the hour angle H, from  sin h = sin φ sin δ + cos φ cos δ cos H :
 *   1 sight the sun: the plumb thread falls at θ = h, the sun's altitude;
 *   2 read 60 sin h on the top scale;
 *   3 date terms, with the thread at the board's latitude θ = φ = 41° (fixed: "made for the 41st
 *     parallel"): a bead at r = 60|sin δ| gives the date term 60 sin φ |sin δ| on the TOP scale;
 *     a bead at r = 60 cos δ gives the divisor 60 cos φ cos δ on the LEFT scale;
 *   4 subtract the date term when δ > 0 (sun north of the equator), add it when δ < 0;
 *   5 divide: set the thread so the bead at r = divisor sits over the difference (its x equals
 *     it): sin θ = difference / divisor = cos H, so H = 90° − θ (when the difference is negative,
 *     the sun is below the "six o'clock" height: H = 90° + θ, θ from |difference|).
 * All in parts of 60. Time from noon = H / 15° per hour.
 *
 * The WORLD (true time, date) is set in step 1 and only there. The reader's measured altitude
 * is their own input and is NEVER changed by another control: when it does not match the sun,
 * or exceeds the day's noon height, it is flagged, not moved.
 *
 * Screen: y DOWN, viewBox units = CSS px of the measured width W.
 */
import { PHI, acosd, altitudeAt, cosd, declination, noonAltitude, sind } from './astro.ts';

export const H_MIN = 0;
export const H_MAX = 90;
export const H_STEP = 0.1;
/** The sight counts as "on the sun" within this many degrees (about half the sun's width). */
export const SIGHT_TOL = 0.25;
/** True-time slider range, morning only (the board gives the same H in the afternoon). */
export const T_MIN = 5;
export const T_MAX = 11.75;

export function quadrant(day: number, tTrue: number, hMeas: number) {
  const dec = declination(day);
  const hNoon = noonAltitude(dec);
  const hTrue = altitudeAt(tTrue, dec);
  const sunUp = hTrue > 0;
  const tooHigh = hMeas > hNoon + 1e-9;
  const onSun = sunUp && Math.abs(hMeas - hTrue) <= SIGHT_TOL;
  const sinh = sind(hMeas);
  const a = sind(PHI) * sind(dec); // signed date term
  const b = cosd(PHI) * cosd(dec); // divisor
  const diff = sinh - a;
  const valid = !tooHigh && hMeas >= 0 && Math.abs(diff) <= b + 1e-12;
  const cosH = valid ? Math.max(-1, Math.min(1, diff / b)) : NaN;
  const Hdeg = valid ? acosd(cosH) : NaN;
  const theta = valid ? 90 - acosd(Math.abs(cosH)) : NaN; // dividing thread angle, from the left edge
  const read = valid ? 12 - Hdeg / 15 : null; // morning
  const errSec = read === null ? null : (read - tTrue) * 3600;
  return { day, dec, hNoon, tTrue, hTrue, sunUp, hMeas, tooHigh, onSun, sinh, a, b, diff, valid, cosH, Hdeg, theta, read, errSec };
}
export type Quadrant = ReturnType<typeof quadrant>;

/* ───────────────────────────── screen layout ───────────────────────────── */

export type QLabel = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end'; role: 'ink' | 'soft' | 'red' | 'brass' | 'sky' };
export const qBox = (l: QLabel) => {
  const w = [...l.text].length * l.size * 0.6;
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'middle' ? l.x - w / 2 : l.x - w;
  return { x0, x1: x0 + w, y0: l.y - l.size * 0.62, y1: l.y + l.size * 0.38, name: l.id };
};

export function board(W: number) {
  const BAND = 20; // numbered border band on the top and left edges
  const side = Math.min(W, 470);
  const ox = (W - side) / 2;
  const AX = ox + 14 + BAND;
  const AY = 30 + BAND;
  const R = side - 14 - BAND - 46;
  const U = R / 60;
  const H = AY + R + 58;
  /** Point at radius r (parts) on a thread at θ (degrees from the left edge). */
  const at = (r: number, th: number) => ({ x: AX + r * U * sind(th), y: AY + r * U * cosd(th) });
  return { W, H, BAND, AX, AY, R, U, at, side, ox };
}
export type Board = ReturnType<typeof board>;

/** Everything drawn at step `i` (0..4) of the procedure. */
export function quadScene(q: Quadrant, step: number, W: number) {
  const g = board(W);
  const { AX, AY, U, at } = g;
  const p = (v: number) => v * 60; // parts
  const L: QLabel[] = [];
  const small = W < 400 ? 10 : 11;

  // the noon-height limit on the rim: the thread can go past it, but the sun cannot
  const noonMark = q.hNoon > 0 ? { a: at(60, q.hNoon), b: at(66, q.hNoon) } : null;
  // the sun's true direction (for the sighting step), only when the sun is up
  const sunDir = q.sunUp ? { a: at(60, q.hTrue), b: at(65, q.hTrue) } : null;
  // the plumb thread at the measured altitude
  const thread = { a: { x: AX, y: AY }, b: at(60, q.hMeas), bob: at(65, q.hMeas) };

  const sinX = AX + p(q.sinh) * U; // 60 sin h on the top scale
  const latEnd = at(60, PHI);
  const beadA = at(p(Math.abs(sind(q.dec))), PHI); // r = 60|sin δ|
  const beadB = at(p(cosd(q.dec)), PHI); // r = 60 cos δ
  const aX = AX + p(Math.abs(q.a)) * U;
  const bY = AY + p(q.b) * U;
  const shiftY = AY + 16; // the row the subtraction runs along
  const diffX = AX + p(Math.abs(q.diff)) * U;
  const divEnd = q.valid ? at(60, q.theta) : null;
  const divBead = q.valid ? at(p(q.b), q.theta) : null;
  const rimH = q.valid ? at(60, q.theta) : null;

  // labels
  if (step >= 2 && !q.tooHigh) {
    const text = `60 sin h = ${p(q.sinh).toFixed(1)}`;
    const w = text.length * small * 0.6;
    const x = Math.max(AX - g.BAND + w / 2, Math.min(AX + g.R - w / 2, sinX));
    L.push({ id: 'sin', x, y: AY - g.BAND - 7, text, size: small, anchor: 'middle', role: 'sky' });
  }
  if (step >= 3) {
    const lp = at(30, PHI - 6);
    L.push({ id: 'lat', x: lp.x, y: lp.y, text: 'φ 41°', size: small, anchor: 'middle', role: 'brass' });
  }
  const out = {
    g,
    noonMark,
    sunDir,
    thread,
    sinX,
    latEnd,
    beadA,
    beadB,
    aX,
    bY,
    shiftY,
    diffX,
    divEnd,
    divBead,
    rimH,
    labels: L,
  };
  return out;
}
export type QuadScene = ReturnType<typeof quadScene>;

/** Status line of the sighting step: honest about what is wrong. */
export function sightSay(q: Quadrant) {
  if (!q.sunUp) return 'The sun is down: nothing to sight.';
  if (q.tooHigh) return `The sun never gets this high today: its noon height is ${q.hNoon.toFixed(1)}°.`;
  if (q.onSun) return 'Sunlight passes through both sights: the thread gives the altitude.';
  const d = q.hMeas - q.hTrue;
  return `The edge points ${Math.abs(d).toFixed(1)}° ${d > 0 ? 'above' : 'below'} the sun: tilt it ${d > 0 ? 'down' : 'up'}.`;
}
