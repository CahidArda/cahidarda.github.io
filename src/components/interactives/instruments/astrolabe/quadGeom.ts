/**
 * The almucantar (astrolabic) quadrant as a quarter of the 41° plate (episode 6). Pure; swept.
 * Quarter frame: apex = the pole (the plate's pin), plate units, +u = WEST (right), +v = up
 * (the upper meridian). The quarter is u ≥ 0, v ≥ 0, r ≤ RC: hour angles 0..90° west.
 * Morning and afternoon are mirror images, so |H| is used. The thread from the apex stands at
 * angle |H| from the meridian edge; the bead on it sits at the sun's distance from the pole,
 * r = tan((90° − δ)/2): exactly where the rete would have put the sun. So the bead lies on the
 * almucantar of the sun's altitude, and the thread's angle on the 90° arc is the hour angle.
 * Hours with |H| > 90° (before 6 or after 18) fall outside the quarter; the plate says so.
 */
import { PHI, hourAngle } from './astro.ts';
import { ALMS, RC, almucantarRuns, radiusOf, type UV } from './plateGeom.ts';

export const QUAD_MAX = 90;
export const THREAD_TOL = 0.3; // degrees of altitude: bead "on" the circle

export function quarterRuns(phi = PHI) {
  return ALMS.map((h) => ({
    h,
    runs: almucantarRuns(h, phi)
      .map((run) => run.filter((p) => p.u >= -1e-9 && p.v >= -1e-9))
      .filter((r) => r.length > 1),
  }));
}
export const bead = (dec: number, absH: number): UV => {
  const r = radiusOf(dec);
  const a = (absH * Math.PI) / 180;
  return { u: r * Math.sin(a), v: r * Math.cos(a) };
};
export const threadEnd = (absH: number): UV => {
  const a = (absH * Math.PI) / 180;
  return { u: RC * Math.sin(a), v: RC * Math.cos(a) };
};
/** The thread angle that puts the bead on altitude h (|H|), or null. */
export const threadFor = (h: number, dec: number, phi = PHI) => hourAngle(h, dec, phi);
export { RC };
