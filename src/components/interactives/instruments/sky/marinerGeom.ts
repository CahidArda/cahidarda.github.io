/**
 * Plate 5 (the mariner's astrolabe), pure numbers and drawn coordinates, swept by
 * `marinerGeom.sweep.ts`. Used by MarinerAstrolabeProcedure.tsx.
 *
 * World. The same noon sun as Plates 3 and 4 (Istanbul, 25 October 2026, h = H_SUN), fixed.
 * The instrument: a heavy brass ring (178 mm across, like the Valentia astrolabe at Greenwich)
 * hanging from its shackle. Gravity makes its 0° line horizontal ONLY when it hangs still; on a
 * rolling ship it swings by ψ (degrees, + = the scale's zero end tipped up, which makes the
 * scale read LOW). The alidade turns on the centre; its two sight plates are ALI_GAP apart with
 * pinholes. The reader sets the alidade to `a` degrees on the scale; its direction in the world
 * is a + ψ. Sunlight through the upper pinhole lands on the lower plate at
 *     offset = ALI_GAP · tan((a + ψ) − h)   (mm, + = above the pinhole: the alidade is too steep),
 * and the reading is a. So when the spot sits on the pinhole, a = h − ψ: the swing goes straight
 * into the reading.
 *
 * Swing. ψ(phase) = A · sin(phase). Readings are taken at a fixed sequence of phases (seeded, so
 * the plate is reproducible); a single reading's spread is A/√2 (RMS of a sine).
 *
 * Drawings. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W.
 */
import { DEC, H_SUN, IST, latFrom } from './staffGeom.ts';
import { viewH } from './latRefGeom.ts';

export { DEC, H_SUN, IST, latFrom };

const RAD = Math.PI / 180;

export const RING_MM = 178;
export const ALI_GAP = 120; // mm between the alidade's sight plates
export const PIN_MM = 0.6; // pinhole diameter, mm ("a fyne sowing needle")
/** The spot counts as on the pinhole within this angle, deg (spot offset ≤ ~0.2 mm). */
export const ALI_TOL = 0.1;
export const A_MIN = 0;
export const A_MAX = 90;
export const A_STEP = 0.05;
/** Swing amplitude slider, degrees. */
export const SWING_MAX = 3;
export const SWING_STEP = 0.1;
export const N_READ = 8;

/** Seeded phases (mulberry32), degrees, for the readings at sea. */
function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(1538); // João de Castro's year
export const PHASES = Array.from({ length: N_READ }, () => rnd() * 360);

export const swingAt = (amp: number, i: number) => amp * Math.sin(PHASES[i] * RAD);
/** Spot offset on the lower plate, mm, for alidade setting a and swing ψ. */
export const spotMm = (a: number, psi: number) => ALI_GAP * Math.tan((a + psi - H_SUN) * RAD);
/** What a careful observer reads: the setting that puts the spot on the pinhole. */
export const readingAt = (psi: number) => H_SUN - psi;
export function readings(amp: number, n: number) {
  const r = Array.from({ length: n }, (_, i) => readingAt(swingAt(amp, i)));
  const mean = r.reduce((s, v) => s + v, 0) / Math.max(1, n);
  return { r, mean };
}

/* ───────────────────────────── the ring (steps 1-3) ───────────────────────────── */

export function ringScene(a: number, psi: number, W: number) {
  const H = viewH(W);
  const R = Math.min(W * 0.34, (H - 90) / 2);
  const cx = W / 2;
  const cy = 36 + 18 + R; // the shackle above
  const shackle = { x: cx, y: cy - R - 18 };
  // the ring's scale: 0° at the ring's horizontal (tipped by ψ), 90° at its top
  const scale = (deg: number) => {
    const s = deg + psi; // world screen angle of a scale mark, on the right half
    return { x: cx + R * Math.cos(s * RAD), y: cy - R * Math.sin(s * RAD) };
  };
  const ticks = Array.from({ length: 19 }, (_, i) => i * 5).map((d) => ({ d, p: scale(d), q: { x: cx + (R - (d % 10 === 0 ? 12 : 7)) * Math.cos((d + psi) * RAD), y: cy - (R - (d % 10 === 0 ? 12 : 7)) * Math.sin((d + psi) * RAD) } }));
  const world = a + psi; // alidade's direction in the world
  const ali = {
    a: { x: cx - R * 0.92 * Math.cos(world * RAD), y: cy + R * 0.92 * Math.sin(world * RAD) },
    b: { x: cx + R * 0.92 * Math.cos(world * RAD), y: cy - R * 0.92 * Math.sin(world * RAD) },
  };
  const spot = spotMm(a, psi);
  const on = Math.abs(a + psi - H_SUN) <= ALI_TOL;
  // the sun's rays from the upper right, fixed in the world
  const sunDir = H_SUN;
  const rays = [-0.5, 0, 0.5].map((o) => {
    const nx = Math.sin(sunDir * RAD);
    const ny = Math.cos(sunDir * RAD);
    const p = { x: cx + nx * o * R, y: cy + ny * o * R };
    const ux = Math.cos(sunDir * RAD);
    const uy = -Math.sin(sunDir * RAD);
    const t = Math.min(R * 1.6, (W - 4 - p.x) / ux, uy < 0 ? (4 - p.y) / uy : Infinity);
    const far = { x: p.x + ux * t, y: p.y + uy * t };
    return { a: far, b: p };
  });
  const level = { a: { x: cx - R - 14, y: cy }, b: { x: cx + R + 14, y: cy } };
  return { W, H, R, cx, cy, shackle, ticks, ali, spot, on, rays, level, world, psi };
}
export function ringSay(a: number, psi: number) {
  const err = a + psi - H_SUN;
  if (Math.abs(err) <= ALI_TOL) return 'light through both pinholes';
  const mm = Math.abs(spotMm(a, psi));
  if (mm > 6) return `no light: turn the alidade ${err > 0 ? 'down' : 'up'}`;
  return `spot ${mm.toFixed(1)} mm ${err > 0 ? 'above' : 'below'} the pinhole`;
}

/* ───────────────────────────── the lower plate, close up (step 2) ───────────────────────────── */

export function plateScene(a: number, psi: number, W: number) {
  const H = viewH(W);
  const cx = W / 2;
  const cy = H / 2 - 6;
  const k = Math.min(W, H) * 0.34 / 6; // px per mm: ±6 mm across
  const off = spotMm(a, psi);
  const spot = { x: cx, y: cy - off * k }; // + offset = above the pinhole (alidade too steep)
  return { W, H, cx, cy, k, spot, off, pinR: (PIN_MM / 2) * k };
}

/* ───────────────────────────── the scatter strip (step 4) ───────────────────────────── */

export function scatterScene(amp: number, n: number, W: number) {
  const H = viewH(W);
  const x0 = 34;
  const x1 = W - 20;
  const half = (SWING_MAX + 0.3) * 60;
  const k = (x1 - x0) / (2 * half);
  const xOf = (deg: number) => x0 + ((deg - H_SUN) * 60 + half) * k;
  const { r, mean } = readings(amp, n);
  const y = H * 0.48;
  const dots = r.map((v, i) => ({ x: xOf(v), y: y - 14 - (i % 4) * 12, v }));
  const ticks: { x: number; m: number }[] = [];
  for (let m = -180; m <= 180; m += 60) ticks.push({ x: xOf(H_SUN + m / 60), m });
  return { W, H, x0, x1, k, y, dots, ticks, mean, xMean: xOf(mean), xTrue: xOf(H_SUN), half };
}
