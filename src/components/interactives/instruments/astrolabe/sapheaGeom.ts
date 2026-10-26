/**
 * The universal plate (saphea) and the wrong-plate check (episode 5). Pure; swept.
 *
 * Saphea frame: the sky projected from the EAST point onto the meridian plane (the plane that
 * holds both poles). Unit disc; +y up = the NORTH celestial pole, +x right = the upper (south)
 * meridian direction... drawn so the pole is at the top. A point (δ, H) with
 *   X (west) = cos δ sin H, Y (south meridian) = cos δ cos H, Z (pole) = sin δ
 * lands at (x, y) = (Y, Z) / (1 + X). The western half of the sky (X ≥ 0) fills the disc.
 * Great circles through the east and west points (the horizon of EVERY latitude, the equator)
 * become straight diameters, which is why one grid serves every latitude: the horizon of
 * latitude φ is the diameter cos φ · x + sin φ · y = 0 (checked numerically in the sweep).
 */
import { PHI, cosd, halfDay, hourAngle, sind, timeOf, wrap180 } from './astro.ts';

export type XY = { x: number; y: number };
export const PLACES = [
  { id: 'ist', name: 'Istanbul', lat: 41.01 },
  { id: 'cai', name: 'Cairo', lat: 30.04 },
  { id: 'mec', name: 'Mecca', lat: 21.42 },
] as const;
export type PlaceId = (typeof PLACES)[number]['id'];

export function saphea(dec: number, H: number) {
  const X = cosd(dec) * sind(H);
  const Y = cosd(dec) * cosd(H);
  const Z = sind(dec);
  return { x: Y / (1 + X), y: Z / (1 + X), west: X >= -1e-12 };
}

/** Sample a curve, keep the parts inside the unit disc. */
function inside(f: (t: number) => XY, t0: number, t1: number, n: number) {
  const runs: XY[][] = [];
  let cur: XY[] | null = null;
  for (let i = 0; i <= n; i++) {
    const p = f(t0 + ((t1 - t0) * i) / n);
    if (Math.hypot(p.x, p.y) <= 1 + 1e-9) {
      if (!cur) runs.push((cur = []));
      cur.push(p);
    } else cur = null;
  }
  return runs.filter((r) => r.length > 1);
}

export const SAPHEA_DECS = [-20, -10, 0, 10, 20];
export function sapheaGrid() {
  const days = SAPHEA_DECS.map((d) => ({ d, runs: inside((H) => saphea(d, H), 0, 180, 180) }));
  const hours: { H: number; runs: XY[][] }[] = [];
  for (let H = 0; H <= 180; H += 15) hours.push({ H, runs: inside((d) => saphea(d, H), -89.9, 89.9, 180) });
  return { days, hours };
}
/** The horizon diameter of latitude φ: ends on the unit circle. */
export function horizonDiameter(phi: number) {
  // direction along the line cos φ x + sin φ y = 0
  const dx = sind(phi);
  const dy = -cosd(phi);
  return { a: { x: -dx, y: -dy }, b: { x: dx, y: dy } };
}
/** Sunset on the saphea: where the sun's day circle meets the horizon diameter (west half). */
export function sunsetOn(phi: number, dec: number) {
  const H0 = halfDay(dec, phi);
  return { H0, t: timeOf(H0), pt: saphea(dec, H0) };
}

/** What a plate made for `plateLat` says the time is, from an altitude measured at `trueLat`. */
export function plateReading(plateLat: number, h: number, dec: number, side: 'am' | 'pm') {
  const H0 = hourAngle(h, dec, plateLat);
  if (H0 === null) return null;
  const H = side === 'am' ? -H0 : H0;
  return { H, t: timeOf(H) };
}
export const placeLat = (id: PlaceId) => PLACES.find((p) => p.id === id)!.lat;
export { PHI, wrap180 };
