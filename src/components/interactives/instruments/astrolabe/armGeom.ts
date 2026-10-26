/**
 * Geometry of the armillary-sphere plate (episode 1). Pure; swept by astrolabe.sweep.ts.
 * The instrument is drawn from OUTSIDE, with the same projection code as the reference dome
 * (skyGeom.camera) but its own camera, so the brass model and the sky beside it are clearly two
 * pictures of one thing. Fixed rings: meridian (vertical, north-south) and horizon. The inner
 * sphere (equator ring with the hour scale, the ecliptic ring, the polar axis) is tilted so the
 * axis stands `axis` degrees above the north horizon, and turned by the sun's hour angle H.
 *   Frame: local (w = west, s = south, z = up); hour angle + west; screen y down.
 */
import { altitude, eclDec, eclRA, wrap180 } from './astro.ts';
import { camera, local, localAltAz, runs, type Pt, type Run } from './skyGeom.ts';

export const AXIS_MIN = 0;
export const AXIS_MAX = 90;
export const ARM_H_MIN = -150;
export const ARM_H_MAX = 150;
export const ARM_TOL = 0.3; // degrees of altitude: "the bead is on the measured height"

/** Height of the strip at the bottom kept for the status note. */
export const NOTE_BAND = 28;
export const armH = (W: number) => Math.round(Math.max(270, Math.min(380, W * 0.82)));

export function armScene(W: number, o: { axis: number; lam: number; H: number }) {
  const H = armH(W);
  const cam = camera(W, 16, -32, H, 0.92);
  const dec = eclDec(lam(o));
  const ra = eclRA(o.lam);
  const phi = o.axis;
  const meridian = runs(cam, (t) => localAltAz(t, 0), -90, 270, 240); // full vertical ring
  const horizon = runs(cam, (A) => localAltAz(0, A), -180, 180, 180);
  const equator = runs(cam, (h) => local(0, h, phi), -180, 180, 240);
  // the ecliptic ring: λ' at (δ(λ'), H = H_sun + α_sun − α(λ'))
  const ecliptic = runs(cam, (l) => local(eclDec(l), o.H + ra - eclRA(l), phi), 0, 360, 240);
  const hourCircle = runs(cam, (d) => local(d, o.H, phi), 0, dec, 40);
  const sunDay = runs(cam, (h) => local(dec, h, phi), -180, 180, 240);
  const poleN = cam.proj(local(90, 0, phi));
  const poleS = cam.proj(local(-90, 0, phi));
  const sun = cam.proj(local(dec, o.H, phi));
  const foot = cam.proj(local(0, o.H, phi));
  const ticks: { a: Pt; b: Pt; lab: Pt; text: string; front: boolean }[] = [];
  for (let k = -6; k <= 6; k++) {
    const Hk = k * 15;
    const p = cam.proj(local(0, Hk, phi));
    const q = (r: number) => {
      const v = local(0, Hk, phi);
      return cam.proj({ w: v.w * r, s: v.s * r, z: v.z * r });
    };
    const lab = q(1.13);
    ticks.push({ a: { x: p.x, y: p.y }, b: q(1.05), lab, text: `${12 + k}`, front: p.front });
  }
  // the stand ends above the status note (baseline H − 12, about 12 px tall)
  const stand = { top: cam.proj({ w: 0, s: 0, z: -1.08 }), foot: { x: cam.cx, y: H - NOTE_BAND } };
  const alt = altitude(dec, o.H, phi);
  return { W, H, cam, meridian, horizon, equator, ecliptic, hourCircle, sunDay, poleN, poleS, sun, foot, ticks, stand, dec, alt, R: cam.R, cx: cam.cx, cy: cam.cy };
}
const lam = (o: { lam: number }) => o.lam;

/** The sphere's hour reading at the sun (12 at the meridian, east = morning). */
export const armReading = (H: number) => 12 + wrap180(H) / 15;
export type { Run };
