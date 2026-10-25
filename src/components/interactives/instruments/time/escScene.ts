/**
 * Screen layout of the deadbeat escapement for the 2D views (PendulumPlate's step view and the
 * 3D plate's poster): the front elevation and a magnified close-up of the pallet in contact.
 * Every coordinate comes from ./deadbeat.ts, so what is drawn is what the solver computed.
 * Screen: y DOWN, viewBox units = CSS px of the measured width W.
 */
import { AMP, PALLETS, PIVOT, R_ROOT, contactInfo, palletsAt, wheelOutline, type Pt } from './deadbeat.ts';
import { rodWorld } from './pendulumGeom.ts';

export const BOB_R = 0.3;

/** Anchor body: a hub at the pivot, two arms to the pallets (drawn only; never in contact). */
const add = (a: Pt, b: Pt): Pt => [a[0] + b[0], a[1] + b[1]];
const mul = (a: Pt, k: number): Pt => [a[0] * k, a[1] * k];
const centroid = (p: Pt[]): Pt => mul(p.reduce(add, [0, 0] as Pt), 1 / p.length);
export const ARM_TO = PALLETS.map((p) => {
  // the pallet's outer end (the part farthest from the wheel)
  const c = centroid([p[0], p[p.length - 1]]);
  return c;
});

const rotP = (p: Pt, c: Pt, a: number): Pt => {
  const s = Math.sin(a);
  const k = Math.cos(a);
  const x = p[0] - c[0];
  const y = p[1] - c[1];
  return [c[0] + x * k - y * s, c[1] + x * s + y * k];
};

export function escScene(w: number, a: number, L: number, W: number, opts: { zoom?: boolean; tilt?: number } = {}) {
  const narrow = W < 460;
  const zoomOn = opts.zoom ?? true;
  // world extent: wheel (r 1) to the anchor top, then the pendulum below the pivot
  const rod = rodWorld(L);
  const top = PIVOT[1] + 0.35;
  const bottom = PIVOT[1] - rod - BOB_R - 0.15;
  const mainW = zoomOn && !narrow ? W * 0.56 : W;
  const H = narrow && zoomOn ? 520 : 420;
  const mainH = narrow && zoomOn ? 300 : H - 30;
  const s = Math.min((mainW - 20) / 3.0, (mainH - 10) / (top - bottom));
  const ox = mainW / 2;
  const oy = 10 + top * s;
  const P = (p: Pt) => ({ x: ox + p[0] * s, y: oy - p[1] * s });
  const tilt = opts.tilt ?? 0;
  const pals = palletsAt(a);
  const arms = ARM_TO.map((t) => P(rotP(t, PIVOT, a)));
  // the pendulum hangs plumb from the pivot (world vertical), the case may be tilted: draw the
  // movement rotated by tilt about the pivot, the pendulum not
  const swing = a + tilt; // world angle of the pendulum
  const bob: Pt = [PIVOT[0] + rod * Math.sin(swing), PIVOT[1] - rod * Math.cos(swing)];
  const c = contactInfo(w, a);
  // close-up around the contact point
  const Z = 9;
  let zoom: null | { cx: number; cy: number; r: number; f: (p: Pt) => { x: number; y: number }; c: Pt } = null;
  if (zoomOn) {
    const r = narrow ? Math.min(W / 2 - 16, 100) : Math.min(W - mainW - 24, 150) / 2 + 20;
    const cx = narrow ? W / 2 : mainW + (W - mainW) / 2;
    const cy = narrow ? mainH + 24 + r : H / 2 - 6;
    const k = s * Z;
    const cp = c.point;
    zoom = { cx, cy, r, c: cp, f: (p: Pt) => ({ x: cx + (p[0] - cp[0]) * k, y: cy - (p[1] - cp[1]) * k }) };
  }
  return {
    W,
    H,
    s,
    P,
    narrow,
    wheel: wheelOutline(w),
    hubR: R_ROOT * 0.35,
    pals,
    arms,
    pivot: P(PIVOT),
    centre: P([0, 0]),
    bob: P(bob),
    bobR: BOB_R * s,
    arc: { r: (rod + BOB_R + 0.1) * s, a0: -AMP * 1.5 + tilt, a1: AMP * 1.5 + tilt },
    contact: c,
    zoom,
    Z,
  };
}
export type EscScene = ReturnType<typeof escScene>;
