/**
 * Plate 1 (Eratosthenes with a stick), pure numbers and drawn coordinates, so
 * `gnomonGeom.sweep.ts` can check them. Used by GnomonProcedure.tsx.
 *
 * World. Midsummer noon, as Eratosthenes set it up: the sun stands straight over Syene, so its
 * declination equals Syene's latitude. Alexandria and Syene at their modern latitudes (31.20° N,
 * 24.09° N). The sun's angular radius SD = 0.267° blurs every shadow tip (the penumbra). The
 * stick is G = 1 m tall. The sun's altitude at local solar time t comes from `sunAltAz` (the same
 * spherical formula as the noon-sight plate).
 *
 * Drawings. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W.
 * Step 1, ground seen from above: north up, east right, the stick's foot fixed, the shadow drawn
 * to one scale k px per metre. Step 2, a ruler along the noon line in centimetres, the penumbra
 * drawn to the same scale. Step 3, the stick and its shadow side on, to scale.
 */
import { sunAltAz, snapTime, hhmm, dm } from './noonProcGeom.ts';
import { viewH } from './latRefGeom.ts';

export { snapTime, hhmm, dm };

const RAD = Math.PI / 180;

export const ALEX = 31.2; // Alexandria, deg N
export const SYENE = 24.09; // Aswan (Syene), deg N
export const DEC = SYENE; // the sun overhead at Syene at noon (Eratosthenes' premise)
export const SD = 0.267; // the sun's angular radius, deg
export const G = 1; // stick height, m
export const DIST = 5000; // stadia, Syene to Alexandria (Cleomedes)
export const EARTH_KM = 40008; // meridional circumference, km
export const T_MIN = 9;
export const T_MAX = 15;
export const T_START = 9.5;
/** Ruler slider (cm along the noon line) and its step. */
export const S_MIN = 10;
export const S_MAX = 15;
export const S_STEP = 0.05;

/** The stadion lengths (Wikipedia, Earth's circumference): short 155 to 160 m (midpoint), Olympic, Italian. */
export const STADIA = [
  { id: 'short', name: 'short', m: 157.5 },
  { id: 'olympic', name: 'Olympic', m: 176.4 },
  { id: 'italian', name: 'Italian', m: 184.8 },
] as const;
export type StadionId = (typeof STADIA)[number]['id'];

/** The sun at Alexandria at local solar time t. */
export const sunAt = (t: number) => sunAltAz(ALEX, DEC, t);
/** Shadow length (m) of the stick for a sun altitude (deg). */
export const shadowLen = (alt: number) => G / Math.tan(alt * RAD);
/** The noon sun at Alexandria and its shadow: centre, and the penumbra's inner and outer tip. */
export const NOON_ALT = 90 - (ALEX - DEC);
export const TIP = shadowLen(NOON_ALT);
export const TIP_IN = shadowLen(NOON_ALT + SD);
export const TIP_OUT = shadowLen(NOON_ALT - SD);
/** θ from a measured shadow s (cm) on the 1 m stick. */
export const thetaOf = (sCm: number) => Math.atan(sCm / 100 / G) / RAD;
/** The mark is on the shadow's edge: anywhere inside the blur (to the ruler's step). */
export const onEdge = (sCm: number) => sCm >= TIP_IN * 100 - 1e-9 && sCm <= TIP_OUT * 100 + 1e-9;
export const THETA_LO = thetaOf(TIP_IN * 100);
export const THETA_HI = thetaOf(TIP_OUT * 100);

export function circumference(theta: number, stadionM: number) {
  const stadia = (360 / theta) * DIST;
  const km = (stadia * stadionM) / 1000;
  return { stadia, km, err: (km / EARTH_KM - 1) * 100 };
}

/* ───────────────────────────── step 1: the ground from above ───────────────────────────── */

const SAMPLES = Array.from({ length: 121 }, (_, i) => T_MIN + ((T_MAX - T_MIN) * i) / 120);
/** Shadow tip offset (m): east and north of the stick's foot. */
export function tipOffset(t: number) {
  const { alt, az } = sunAt(t);
  const L = shadowLen(alt);
  return { e: L * Math.sin(az * RAD), n: L * Math.cos(az * RAD), L, alt, az };
}

export function groundScene(t: number, W: number) {
  const H = viewH(W);
  const pad = 16;
  const offs = SAMPLES.map(tipOffset);
  const maxN = Math.max(...offs.map((o) => o.n));
  const maxE = Math.max(...offs.map((o) => Math.abs(o.e)));
  const foot = { x: W / 2, y: H - 58 };
  const top = 34;
  const k = Math.min((foot.y - top) / maxN, (W / 2 - pad) / maxE); // px per metre
  const at = (o: { e: number; n: number }) => ({ x: foot.x + o.e * k, y: foot.y - o.n * k });
  const now = tipOffset(t);
  const tip = at(now);
  // the penumbra's two tips along the same shadow direction
  const ux = (tip.x - foot.x) / (now.L * k);
  const uy = (tip.y - foot.y) / (now.L * k);
  const Lin = shadowLen(now.alt + SD);
  const Lout = shadowLen(now.alt - SD);
  const tipIn = { x: foot.x + ux * Lin * k, y: foot.y + uy * Lin * k };
  const tipOut = { x: foot.x + ux * Lout * k, y: foot.y + uy * Lout * k };
  const track = offs.map(at);
  const noon = t === 12;
  // ruler along the noon line, 10 cm ticks
  const ticks: { y: number; m: number }[] = [];
  for (let m = 0; m <= maxN + 1e-9; m += 0.1) ticks.push({ y: foot.y - m * k, m: Math.round(m * 10) / 10 });
  return { W, H, foot, k, tip, tipIn, tipOut, track, noon, now, ticks, top, pad };
}
export function groundSay(t: number) {
  const o = tipOffset(t);
  if (t === 12) return `noon: shadow ${(o.L * 100).toFixed(1)} cm, the shortest`;
  const d = 1 / 120;
  const rate = (tipOffset(t + d).L - tipOffset(t - d).L) / (2 * d); // m per hour
  return `${hhmm(t)}: shadow ${(o.L * 100).toFixed(0)} cm, ${rate < 0 ? 'shrinking' : 'growing'}`;
}

/* ───────────────────────────── step 2: the ruler at the tip ───────────────────────────── */

export function rulerScene(sCm: number, W: number) {
  const H = viewH(W);
  const x0 = 24;
  const x1 = W - 24;
  const kx = (x1 - x0) / (S_MAX - S_MIN); // px per cm
  const xOf = (cm: number) => x0 + (cm - S_MIN) * kx;
  const rulerY = H * 0.62;
  const shadowY0 = 40;
  const shadowY1 = rulerY - 10;
  const ticks: { x: number; cm: number; major: boolean; mid: boolean }[] = [];
  for (let mm = S_MIN * 10; mm <= S_MAX * 10; mm++)
    ticks.push({ x: xOf(mm / 10), cm: mm / 10, major: mm % 10 === 0, mid: mm % 5 === 0 });
  return {
    W, H, x0, x1, kx, xOf, rulerY, shadowY0, shadowY1, ticks,
    xIn: xOf(TIP_IN * 100), xTip: xOf(TIP * 100), xOut: xOf(TIP_OUT * 100), xMark: xOf(sCm),
  };
}
export function rulerSay(sCm: number) {
  if (onEdge(sCm)) return `mark at ${sCm.toFixed(2)} cm: on the blurred edge`;
  return sCm < TIP_IN * 100 ? `mark at ${sCm.toFixed(2)} cm: still in full shadow` : `mark at ${sCm.toFixed(2)} cm: in full sunlight`;
}

/* ───────────────────────────── step 3: the triangle, side on ───────────────────────────── */

export function triangleScene(sCm: number, W: number) {
  const H = viewH(W);
  const base = { x: W * 0.36, y: H - 50 };
  const k = (H - 50 - 44) / G; // px per metre
  const top = { x: base.x, y: base.y - G * k };
  const tip = { x: base.x + (sCm / 100) * k, y: base.y };
  const theta = thetaOf(sCm);
  // the sun's ray through the top of the stick to the tip, extended up
  const ext = 30 / Math.cos(theta * RAD);
  const rayFrom = { x: top.x - Math.sin(theta * RAD) * ext, y: top.y - Math.cos(theta * RAD) * ext };
  const arcR = Math.min(70, G * k * 0.32);
  return { W, H, base, top, tip, k, theta, rayFrom, arcR };
}
