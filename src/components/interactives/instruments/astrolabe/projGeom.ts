/**
 * Geometry of the projection plate (episode 2). Pure; swept by astrolabe.sweep.ts.
 * Two linked drawings of ONE construction:
 *   side: the celestial sphere cut along the meridian. Unit circle, centre (0,0); x to the
 *         SOUTH (the upper meridian, the plate's +v), y to the NORTH POLE. The equator plane is
 *         the line y = 0; the projection centre is the south pole (0, −1).
 *   plan: the plate seen face on (plate units; +v up). The equator plane's line in `side` IS the
 *         plate's vertical diameter, so a side-view crossing at x lands on the plan at v = x.
 * A point on the meridian at angle θ from the south horizon direction... we use the angle θ
 * from the +x axis (the equator, south side), counter-clockwise towards the pole: θ = δ on the
 * upper meridian, θ = 180° − δ on the lower one. Its ray from the south pole meets y = 0 at
 * x = cos θ / (1 + sin θ).
 */
import { cosd, sind, tand } from './astro.ts';
import { RC, RCAN, almucantar } from './plateGeom.ts';

export type XY = { x: number; y: number };
export const DEC_MIN = -23;
export const DEC_MAX = 80;
export const ALT_MIN = 0;
export const ALT_MAX = 80;

/** Where the ray from the south pole through the meridian point at angle θ crosses the equator plane. */
export const crossAt = (theta: number) => cosd(theta) / (1 + sind(theta));
/** Meridian angles of the two points of almucantar h at latitude φ (south side, north side). */
export const almThetas = (h: number, phi: number) => [phi + h - 90, phi + 90 - h] as const;

export function projH(W: number) {
  return Math.round(Math.max(300, Math.min(420, W * 0.95)));
}

export function projScene(W: number, o: { mode: 'star' | 'alm'; dec: number; h: number; phi: number }) {
  const H = projH(W);
  // two panels stacked vertically when narrow, side by side when wide
  const wide = W >= 520;
  const pad = 14;
  const sideBox = wide ? { x0: 0, y0: 0, w: W * 0.5, h: H - 30 } : { x0: 0, y0: 0, w: W, h: (H - 30) * 0.5 };
  const planBox = wide ? { x0: W * 0.5, y0: 0, w: W * 0.5, h: H - 30 } : { x0: 0, y0: (H - 30) * 0.5, w: W, h: (H - 30) * 0.5 };
  // side view: the sphere (radius 1) and the plate line out to ±RC
  const thetas0 = o.mode === 'star' ? [o.dec] : [...almThetas(o.h, o.phi)];
  const reach = Math.max(RC, ...thetas0.map((t) => Math.abs(crossAt(t))));
  const ks = Math.min((sideBox.w - 2 * pad) / (2 * reach), (sideBox.h - 2 * pad - 14) / 2.3);
  const sc = { x: sideBox.x0 + sideBox.w / 2, y: sideBox.y0 + pad + 14 + 1.1 * ks };
  const S = (x: number, y: number): XY => ({ x: sc.x + x * ks, y: sc.y - y * ks });
  // plan view: the plate (radius RC) — the plate's v axis drawn VERTICALLY (up = south meridian)
  const kp = Math.min((planBox.w - 2 * pad) / (2 * RC), (planBox.h - 2 * pad - 14) / (2 * RC));
  const pc = { x: planBox.x0 + planBox.w / 2, y: planBox.y0 + planBox.h / 2 + 7 };
  const Pl = (u: number, v: number): XY => ({ x: pc.x + u * kp, y: pc.y - v * kp });

  const south = S(0, -1);
  const pts: { theta: number; on: XY; cross: XY; x: number; planPt: XY }[] = [];
  const thetas = o.mode === 'star' ? [o.dec] : [...almThetas(o.h, o.phi)];
  for (const th of thetas) {
    const x = crossAt(th);
    pts.push({ theta: th, on: S(cosd(th), sind(th)), cross: S(x, 0), x, planPt: Pl(0, x) });
  }
  // the resulting circle on the plate
  let circle: { c: XY; r: number; cv: number; rr: number } | null = null;
  if (o.mode === 'star') {
    const r = tand((90 - o.dec) / 2);
    circle = { c: Pl(0, 0), r: r * kp, cv: 0, rr: r };
  } else {
    const a = almucantar(o.h, o.phi);
    circle = { c: Pl(0, a.cv), r: a.r * kp, cv: a.cv, rr: a.r };
  }
  // the almucantar's chord in the side view (the plane of the circle, edge-on)
  const chord = o.mode === 'alm' ? [pts[0].on, pts[1].on] : null;
  const horizonLine = (() => {
    // the horizon in the side view: the line through the centre perpendicular to the zenith
    const z = { x: cosd(o.phi), y: sind(o.phi) };
    return [S(-z.y * 1.08, z.x * 1.08), S(z.y * 1.08, -z.x * 1.08)];
  })();
  const zenith = S(cosd(o.phi), sind(o.phi));
  return {
    W,
    H,
    wide,
    sideBox,
    planBox,
    ks,
    kp,
    sc,
    pc,
    S,
    Pl,
    south,
    north: S(0, 1),
    plateLine: [S(-reach, 0), S(reach, 0)] as const,
    reach,
    pts,
    circle,
    chord,
    horizonLine,
    zenith,
    tropics: { cancer: RCAN * kp, cap: RC * kp, eq: 1 * kp },
  };
}
