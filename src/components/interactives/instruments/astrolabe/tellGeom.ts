/**
 * The "tell the time" procedure as pure state: every number the plate prints and every
 * gate it opens comes from here (swept by astrolabe.sweep.ts).
 *   The world: the sun at its true place for (date, sun time) at 41° N.
 *   The reader: an alidade setting `ali`, a rete turn `lst`, a rule angle `ruleH`.
 * Snapping: a control within its tolerance snaps to the exact value (so the readout, the
 * drawing and the equation show one number), never a value the drawing would refuse.
 */
import { PHI, cosHour, eclDec, eclRA, hourAngle, sunAt, timeOf, wrap180, wrap360, eotMinutes, lonMinutes } from './astro.ts';
import { SIGHT_TOL, plateAltitude, solveRete, sunHourOnRete } from './plateGeom.ts';

export const ALI_MIN = 0;
export const ALI_MAX = 90;
export const ALI_STEP = 0.1;
export const RETE_STEP = 0.25;
export const RETE_TOL = 0.3; // degrees of altitude between the sun's point and the almucantar
export const RULE_STEP = 0.25;
export const RULE_TOL = 0.5; // degrees of hour angle between the rule and the sun's point

export const round1 = (x: number) => Math.round(x * 10) / 10;

export function world(doy: number, t: number, phi = PHI) {
  return sunAt(doy, t, phi);
}
/** Snap the alidade: within the sight tolerance it settles on the sun (to 0.1°). */
export const snapAli = (a: number, h: number) => (Math.abs(a - h) <= SIGHT_TOL ? round1(h) : a);
export const sighted = (a: number, h: number) => Math.abs(a - h) <= SIGHT_TOL;

/** Is the sun's point on the measured almucantar, on the right side? */
export function reteState(lam: number, lst: number, hMeas: number, side: 'am' | 'pm', phi = PHI) {
  const hPlate = plateAltitude(lam, lst, phi);
  const Hs = sunHourOnRete(lam, lst);
  const sideOk = side === 'am' ? Hs < 0 : Hs > 0;
  return { hPlate, Hs, sideOk, on: sideOk && Math.abs(hPlate - hMeas) <= RETE_TOL };
}
/** Snap the rete: within tolerance, turn it to the exact solution. */
export function snapRete(lam: number, lst: number, hMeas: number, side: 'am' | 'pm', phi = PHI) {
  const st = reteState(lam, lst, hMeas, side, phi);
  if (!st.on) return lst;
  const s = solveRete(lam, hMeas, side, phi);
  return s ? s.lst : lst;
}
export const snapRule = (ruleH: number, Hs: number) => (Math.abs(wrap180(ruleH - Hs)) <= RULE_TOL ? Hs : ruleH);

/** The solution the plate is computing, from the MEASURED altitude. */
export function solution(lam: number, hMeas: number, side: 'am' | 'pm', phi = PHI) {
  const dec = eclDec(lam);
  const c = cosHour(hMeas, dec, phi);
  const H0 = hourAngle(hMeas, dec, phi);
  const H = H0 === null ? null : side === 'am' ? -H0 : H0;
  return { dec, cosH: c, H, t: H === null ? null : timeOf(H) };
}
/** Clock time in Istanbul (UTC+3) from apparent sun time: t − EoT + 64 min. */
export const clockOf = (t: number, doy: number) => t + (-eotMinutes(doy) + lonMinutes) / 60;

/** Initial rete turn for the "turn the rete" step: the sun's point at the upper meridian
 *  minus 90° (well away from any solution), so the reader has to do the turning. */
export const reteStart = (lam: number) => wrap360(eclRA(lam) + 120);
