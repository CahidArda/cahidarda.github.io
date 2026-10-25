/**
 * Plates 3 and 4 (cross-staff and backstaff), pure numbers and drawn coordinates, swept by
 * `staffGeom.sweep.ts`. Used by CrossStaffProcedure.tsx and BackstaffProcedure.tsx.
 *
 * World. The noon sun over Istanbul on 25 October 2026, the same sight as the sextant plates:
 * the sun's centre at H_SUN = 90° − (φ − δ) (δ from the USNO formula via ./noonProcGeom.ts).
 * Refraction, dip and the sun's radius are a few arcminutes; these staffs read to about a sixth
 * of a degree, so the plates leave them out and say so.
 *
 * Cross-staff. A staff STAFF_LEN long, its end on the cheek; a transom of length L slides on it,
 * centred on the staff. At distance x from the eye its ends span θ = 2·atan(L / 2x). The staff is
 * aimed so the lower end sits on the horizon, so the upper end stands θ above it.
 *
 * Backstaff (Davis quadrant). Angles at the horizon vane O, measured from the sight line (O to the
 * sight vane, i.e. away from the horizon). The staff runs at a30 from the sight line, the shadow
 * vane at a30 + a60. The shadow of the vane's edge falls on the slit when a60 + a30 = h. Off by
 * Δ = (a60 + a30) − h, the shadow edge sits R60 · tan Δ from the slit on the horizon vane.
 *
 * Drawings. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W.
 */
import { DOY_PRESET, PLACES, sight } from './noonProcGeom.ts';
import { viewH, placer, arcPts, type Seg } from './latRefGeom.ts';

const RAD = Math.PI / 180;

export const IST = PLACES[0];
export const SIGHT = sight(DOY_PRESET, IST);
export const H_SUN = SIGHT.hTrue;
export const DEC = SIGHT.dec;
export const SD = SIGHT.sd;
export const latFrom = (h: number) => 90 - h + DEC;

/* ───────────────────────────── cross-staff ───────────────────────────── */

export const STAFF_LEN = 100; // cm
export const X_MIN = 12; // cm: nearer than this the transom hits your face
/** Nearest setting for a transom: its ends may span at most 90° (as real staffs were marked). */
export const xMinFor = (L: number) => Math.max(X_MIN, L / 2);
export const X_MAX = STAFF_LEN;
export const X_STEP = 0.1;
/** Transoms, after Ifland's 90/60/30/10° set, as lengths on a 100 cm staff. */
export const TRANSOMS = [
  { id: 'short', name: '10 cm', L: 10 },
  { id: 'mid', name: '30 cm', L: 30 },
  { id: 'long', name: '60 cm', L: 60 },
] as const;
export type TransomId = (typeof TRANSOMS)[number]['id'];
export const transomL = (id: TransomId) => TRANSOMS.find((t) => t.id === id)!.L;

export const csTheta = (L: number, x: number) => (2 * Math.atan(L / (2 * x))) / RAD;
export const csX = (L: number, theta: number) => L / (2 * Math.tan((theta / 2) * RAD));
/** Range of angles a transom can frame on the staff. */
export const csRange = (L: number) => ({ lo: csTheta(L, X_MAX), hi: csTheta(L, X_MIN) });
/** The transom fits today's sun on the staff. */
export const csFits = (L: number) => {
  const r = csRange(L);
  return H_SUN > r.lo && H_SUN < r.hi;
};
/** Ends on sun and horizon within this (deg): about what one can hold on a deck. */
export const CS_TOL = 0.1;
/** How much a 1 mm slide changes the reading, deg, at the right setting. */
export const csPerMm = (L: number) => {
  const x = csX(L, H_SUN);
  return csTheta(L, x - 0.05) - csTheta(L, x + 0.05);
};

export function csSky(L: number, x: number, W: number) {
  const H = viewH(W);
  const horizonY = H - 52;
  const top = 30;
  const span = 60; // degrees shown
  const k = (horizonY - top) / span;
  const cx = W * 0.5;
  const theta = csTheta(L, x);
  const sun = { x: cx, y: horizonY - H_SUN * k, r: Math.max(5, SD * k * 2.5) };
  const upper = { x: cx, y: horizonY - Math.min(theta, span + 4) * k };
  const gap = theta - H_SUN;
  const on = Math.abs(gap) <= CS_TOL;
  const ticks = [10, 20, 30, 40, 50].map((a) => ({ a, y: horizonY - a * k }));
  return { W, H, horizonY, top, k, cx, theta, sun, upper, gap, on, ticks, clipped: theta > span + 4 };
}
export function csSay(g: ReturnType<typeof csSky>) {
  if (g.on) return 'lower end on the horizon, upper end on the sun';
  return `upper end ${Math.abs(g.gap).toFixed(1)}° ${g.gap > 0 ? 'above' : 'below'} the sun`;
}
export function csSide(L: number, x: number, W: number) {
  const H = viewH(W);
  const pad = 22;
  const eyeY = H * 0.62;
  // one scale per transom: the highest point over the whole slider range must fit above the eye
  let top = 0;
  for (let xx = xMinFor(L); xx <= X_MAX; xx += 1) {
    const tl = csTheta(L, xx) / 2;
    top = Math.max(top, xx * Math.sin(tl * RAD) + (L / 2) * Math.cos(tl * RAD), STAFF_LEN * Math.sin(tl * RAD));
  }
  const k = Math.min((W - 2 * pad - 20) / STAFF_LEN, (eyeY - 30) / top); // px per cm
  const theta = csTheta(L, x);
  const tilt = theta / 2; // the staff points midway between horizon and sun
  const eye = { x: pad + 6, y: eyeY };
  const u = { x: Math.cos(tilt * RAD), y: -Math.sin(tilt * RAD) };
  const n = { x: -Math.sin(tilt * RAD), y: -Math.cos(tilt * RAD) }; // up, square to the staff
  const end = { x: eye.x + u.x * STAFF_LEN * k, y: eye.y + u.y * STAFF_LEN * k };
  const mid = { x: eye.x + u.x * x * k, y: eye.y + u.y * x * k };
  const a = { x: mid.x - n.x * (L / 2) * k, y: mid.y - n.y * (L / 2) * k }; // lower end
  const b = { x: mid.x + n.x * (L / 2) * k, y: mid.y + n.y * (L / 2) * k }; // upper end
  const ticks: { p: { x: number; y: number }; cm: number }[] = [];
  for (let cm = 10; cm <= STAFF_LEN; cm += 10) ticks.push({ p: { x: eye.x + u.x * cm * k, y: eye.y + u.y * cm * k }, cm });
  // labels, placed greedily off every line and the h arc (shared placer)
  const arcR = 34;
  const along = (cm: number, off: number) => ({ x: eye.x + u.x * cm * k + n.x * off, y: eye.y + u.y * cm * k + n.y * off });
  const segs: Seg[] = [
    { id: 'staff', a: eye, b: end },
    { id: 'transom', a, b },
    { id: 'lo', a: eye, b: a },
    { id: 'hi', a: eye, b },
  ];
  const pl = placer(W, H - 26, segs, [arcPts(eye.x, eye.y, arcR, 0, theta)], [], 3);
  const C = (id: string, text: string, size: number, pts: { x: number; y: number }[], anchor: 'start' | 'middle' | 'end' = 'middle') =>
    pts.map((q) => ({ id, text, size, anchor, x: q.x, y: q.y }));
  pl.put(C('eye', 'eye', 11, [{ x: eye.x - 8, y: eye.y + 18 }, { x: eye.x - 8, y: eye.y + 30 }, { x: eye.x - 8, y: eye.y - 14 }], 'start'));
  const hc: { x: number; y: number }[] = [];
  for (const r of [arcR - 12, arcR + 10, arcR + 18, arcR + 28, arcR + 40])
    for (const f of [0.5, 0.35, 0.65, 1.2, 1.4, 1.7, -0.3, -0.6])
      hc.push({ x: eye.x + r * Math.cos(theta * f * RAD), y: eye.y - r * Math.sin(theta * f * RAD) });
  for (const dx of [-4, 8, 20]) hc.push({ x: eye.x + dx, y: eye.y - arcR - 14 });
  pl.put(C('h', 'h', 14, hc));
  const xt = `x = ${x.toFixed(1)} cm`;
  const xc: { x: number; y: number }[] = [];
  for (const off of [-16, -26, 16, 26]) for (const f of [0.5, 0.65, 0.35, 0.8]) xc.push(along(x * f, off));
  // a short x: below the horizon line, under the transom's foot
  for (const dy of [18, 32, 46]) for (const dx of [0, 30, 60]) xc.push({ x: a.x + dx, y: eye.y + dy });
  pl.put(C('x', xt, 12, xc));
  for (const t of ticks) if (t.cm % 20 === 0) pl.put(C(`t${t.cm}`, `${t.cm}`, 10, [along(t.cm, 11), along(t.cm, -11)]));
  const labels = pl.placed.filter((l) => l.placed || !l.id.startsWith('t'));
  return { W, H, k, eye, end, mid, a, b, theta, tilt, ticks, u, n, pad, arcR, labels };
}

/* ───────────────────────────── backstaff ───────────────────────────── */

export const A60_STEP = 5;
export const A60_MIN = 10;
export const A60_MAX = 60;
export const A30_MIN = 0;
export const A30_MAX = 30;
export const A30_STEP = 1 / 30; // the big arc's transversals, 2′
export const R60 = 0.3; // m, small arc radius
export const R30 = 0.75; // m, big arc radius (about 2½ ft)
/** The shadow edge is blurred by the sun's disc: aligned when the blur's middle is within this, deg. */
export const BS_TOL = 0.05;
/** The preset must leave the rest within the big arc. */
export const bsPresetOk = (a60: number) => H_SUN - a60 >= A30_MIN && H_SUN - a60 <= A30_MAX;

/**
 * The horizon vane, close up. A shadow-vane angle α = a60 + a30 casts the vane's shadow edge
 * r60 · sin(α − h) / cos h from the slit (above it when α > h); the sun's disc blurs the edge over
 * ± r60 · sin(SD) / cos h. The reader aligns the MIDDLE of the blur (the sun's centre) with the
 * slit, so no semi-diameter correction is needed here.
 */
export const bsOffMm = (a60: number, a30: number) =>
  (R60 * 1000 * Math.sin((a60 + a30 - H_SUN) * RAD)) / Math.cos(H_SUN * RAD);
export const BS_BLUR_MM = (R60 * 1000 * Math.sin(SD * RAD)) / Math.cos(H_SUN * RAD);

export function bsVane(a60: number, a30: number, W: number) {
  const H = viewH(W);
  const d = a60 + a30 - H_SUN; // deg
  const offMm = bsOffMm(a60, a30);
  const vane = { x0: W / 2 - Math.min(110, W * 0.28), x1: W / 2 + Math.min(110, W * 0.28), y0: 40, y1: H - 56 };
  const cy = (vane.y0 + vane.y1) / 2;
  const kmm = (vane.y1 - vane.y0) / 2 / 6; // px per mm: ±6 mm fills the vane
  const slitY = cy;
  const edgeY = cy - offMm * kmm; // above the slit when the angle is too big
  const blur = BS_BLUR_MM * kmm;
  const on = Math.abs(d) <= BS_TOL;
  const off = Math.abs(offMm) > 6.5 ? (offMm > 0 ? 'above' : 'below') : null;
  return { W, H, d, kmm, cx: W / 2, cy, slitY, edgeY, blur, on, vane, offMm, off };
}
export function bsSay(g: ReturnType<typeof bsVane>) {
  if (g.on) return 'shadow edge on the slit, horizon in the slit';
  if (Math.abs(g.offMm) > 6.5) return `shadow off the vane, ${g.d > 0 ? 'above' : 'below'}`;
  return `shadow edge ${Math.abs(g.offMm).toFixed(1)} mm ${g.d > 0 ? 'above' : 'below'} the slit`;
}

/**
 * Side diagram (world fixed: the sight line to the horizon is level, the sun behind you at h).
 * O is the horizon vane, the arcs' common centre. Screen angles at O, 0° = towards the eye (right).
 * The sight vane sits on the 30° arc at 0°, the staff at +a30, the shadow vane on the 60° arc at
 * a30 + a60. The shadow ray runs from the shadow vane, away from the sun, towards O.
 */
export function bsSide(a60: number, a30: number, W: number) {
  const H = viewH(W);
  const O = { x: Math.max(70, W * 0.2), y: H * 0.66 };
  const k = Math.min((W - O.x - 40) / R30, (O.y - 30) / (R30 * Math.sin(H_SUN * RAD + 0.2)));
  const r30 = R30 * k;
  const r60 = R60 * k;
  const at = (r: number, deg: number) => ({ x: O.x + r * Math.cos(deg * RAD), y: O.y - r * Math.sin(deg * RAD) });
  const sightVane = at(r30, 0);
  const eye = { x: sightVane.x + 12, y: O.y };
  const staffEnd = at(r30 * 1.04, a30);
  const alpha = a30 + a60;
  const shadowVane = at(r60, alpha);
  // shadow ray: from the shadow vane, direction away from the sun, to where it crosses the vane line at O (x = O.x)
  const ux = -Math.cos(H_SUN * RAD);
  const uy = Math.sin(H_SUN * RAD); // screen
  const t = (O.x - shadowVane.x) / ux;
  const shadowHit = { x: O.x, y: shadowVane.y + uy * t };
  const sunIn = { x: shadowVane.x - ux * 46, y: shadowVane.y - uy * 46 };
  const horizonEnd = { x: 8, y: O.y };
  return { W, H, O, k, r30, r60, sightVane, eye, staffEnd, shadowVane, shadowHit, sunIn, horizonEnd, alpha, a30 };
}
