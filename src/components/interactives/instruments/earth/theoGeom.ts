/**
 * Episode 5: triangulation with the theodolite. One function (eyepiece) decides what the
 * reader sees through the telescope, whether it is "on", and feeds the status and equation.
 *
 * Horizontal circle readings run CLOCKWISE seen from above, in degrees, like a real
 * theodolite. A circle is never set to north: each station has its own arbitrary zero, so the
 * angle between two targets is the DIFFERENCE of two readings.
 * Vertical readings: elevation above level, degrees.
 * World (site.ts plan): A at (480, 0), B at (480, 200), C at (1180, 0), metres, north = +y.
 */
import { SITE, FAR_X } from './site.ts';

export const RAD = Math.PI / 180;
/** Bearing (clockwise from north, deg) from p to q in the plan. */
export const bearing = (p: { x: number; y: number }, q: { x: number; y: number }) =>
  ((Math.atan2(q.x - p.x, q.y - p.y) / RAD) % 360 + 360) % 360;

export const PA = { x: SITE.SHORE_X, y: 0 };
export const PB = { x: SITE.SHORE_X, y: SITE.B_Y };
export const PC = { x: FAR_X, y: 0 };
export const BASE = SITE.B_Y; // 200 m, chained along the quay

/** Each station's circle zero: the circle reading that points due north (arbitrary). */
export const ZERO = { A: 347.4, B: 52.15 }; // degrees
/** Circle reading that points at q from station s. */
export const readingTo = (s: 'A' | 'B', q: { x: number; y: number }) =>
  (((bearing(s === 'A' ? PA : PB, q) + ZERO[s]) % 360) + 360) % 360;

/** The true angles of the triangle (deg): at A between B and C, at B between A and C. */
export const ALPHA = (((readingTo('A', PC) - readingTo('A', PB)) % 360) + 360) % 360; // 90°
export const BETA = (((readingTo('B', PA) - readingTo('B', PC)) % 360) + 360) % 360; // 74.05°
export const widthFrom = (b: number, a: number, be: number) => (b * Math.sin(a * RAD) * Math.sin(be * RAD)) / Math.sin((a + be) * RAD);

/** Slider model: whole degrees on the circle + minutes on the tangent screw (0.1′ steps). */
export const MIN_STEP = 0.1;
export const ON_TOL = 0.06 / 60; // deg: within 0.06′ of the target counts as on (the 0.1′ step can always reach it)
export const toDeg = (d: number, m: number) => d + m / 60;
export function split(v: number) {
  const t = Math.round((((v % 360) + 360) % 360) * 600) / 600;
  let d = Math.floor(t + 1e-9);
  let m = Math.round((t - d) * 600) / 10;
  if (m >= 60) {
    m -= 60;
    d += 1;
  }
  return { d: d % 360, m };
}
/** The best slider setting for a target reading (what "skip" uses). */
export const snap = (v: number) => split(v);
/** Signed smallest difference a − b in degrees, in (−180, 180]. */
export const diff = (a: number, b: number) => {
  const x = (((a - b) % 360) + 540) % 360 - 180;
  return x === -180 ? 180 : x;
};
export function dm(v: number) {
  const s = split(v);
  return `${s.d}°${s.m.toFixed(1).padStart(4, '0')}′`;
}

/**
 * What the eyepiece shows. Field of view FOV degrees across. Targets are drawn at their
 * offsets (target − instrument) in the field; out of the field they become edge arrows.
 * `on` uses ON_TOL for both axes.
 */
export const FOV = 1.0; // deg
export type Target = { name: string; h: number; v: number; kind: 'pole' | 'tip' };
export function eyepiece(W: number, h: number, v: number, targets: Target[]) {
  const H = 186;
  const R = Math.min(70, W / 2 - 30);
  const cx = W / 2;
  const cy = 84;
  const k = (2 * R) / FOV; // px per degree
  const out = targets.map((t) => {
    const dx = diff(t.h, h); // + = target is clockwise (to the right)
    const dy = t.v - v; // + = target is higher
    const x = cx + dx * k;
    const y = cy - dy * k;
    const inside = Math.hypot(x - cx, y - cy) <= R - 4;
    const ang = Math.atan2(-(y - cy), x - cx);
    const edge = { x: cx + (R - 12) * Math.cos(ang), y: cy - (R - 12) * Math.sin(ang), ang };
    // a pole is a vertical line: only the horizontal offset matters; a finial is a point
    const on = Math.abs(dx) <= ON_TOL + 1e-12 && (t.kind === 'pole' || Math.abs(dy) <= ON_TOL + 1e-12);
    return { ...t, dx, dy, x, y, inside, edge, on };
  });
  return { W, H, R, cx, cy, k, targets: out };
}

/* ── the height of the minaret ── */

/** Stations on the flat precinct, 100 m and 50 m from the minaret's foot. Angles are taken
 *  from the ground mark (the instrument's own height is left out to keep the sums short). */
export const TOWER = { far: 100, near: 50, H: SITE.MINARET_H };
export const EL_FAR = Math.atan(TOWER.H / TOWER.far) / RAD; // 37.23°
export const EL_NEAR = Math.atan(TOWER.H / TOWER.near) / RAD; // 56.66°
export const heightFrom = (b: number, a: number, be: number) => (b * Math.sin(a * RAD) * Math.sin(be * RAD)) / Math.sin((be - a) * RAD);
export const V_MIN = -30;
export const V_MAX = 60; // scopeClearance (geometry.ts) allows up to about +63°
