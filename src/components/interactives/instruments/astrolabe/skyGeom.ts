/**
 * The STABLE REFERENCE drawing for every plate of the astrolabe post: the sky over an observer at
 * latitude φ, as a sphere seen from a little above and behind (north of) the observer, facing
 * south, so EAST IS ON THE LEFT, as on the astrolabe's face. Same size, place and camera in every
 * plate and every step; only the highlights change. Pure (no React); swept by astrolabe.sweep.ts.
 *
 * Screen: SVG units, y DOWN. Sphere centre (cx, cy), radius R. Local frame: x_w west, y_s south,
 * z up. Camera: facing azimuth "south turned γ towards west", raised κ above the horizon.
 * Points on the far half of the sphere (away from the camera) are drawn dashed and faint.
 */
import { EPS, PHI, altitude, cosd, eclDec, sind, wrap180 } from './astro.ts';

export type Pt = { x: number; y: number };
export type Run = { pts: Pt[]; front: boolean };
export type Box = { x0: number; y0: number; x1: number; y1: number };
export type Label = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end'; placed: boolean };

const KAPPA = 20; // camera elevation
const GAMMA = 28; // camera turned from due south towards the west

/** Local (west, south, up) unit vector for declination δ, hour angle H. */
export function local(dec: number, H: number, phi = PHI) {
  return {
    w: cosd(dec) * sind(H),
    s: sind(phi) * cosd(dec) * cosd(H) - cosd(phi) * sind(dec),
    z: sind(phi) * sind(dec) + cosd(phi) * cosd(dec) * cosd(H),
  };
}
/** Local vector of altitude h, azimuth A (from south, + west). */
export const localAltAz = (h: number, A: number) => ({ w: cosd(h) * sind(A), s: cosd(h) * cosd(A), z: sind(h) });

export function viewH(W: number) {
  return Math.round(Math.max(250, Math.min(360, W * 0.78)));
}

export function camera(W: number, kappa = KAPPA, gamma = GAMMA, Hfix?: number, scale = 1) {
  const H = Hfix ?? viewH(W);
  const R = Math.min(W * 0.36, H * 0.4) * scale;
  const cx = W / 2;
  const cy = H * 0.5;
  const KAPPA = kappa;
  // forward f (horizontal) and right r
  const f = { s: cosd(gamma), w: sind(gamma) };
  const r = { s: -sind(gamma), w: cosd(gamma) };
  const proj = (p: { w: number; s: number; z: number }) => {
    const fp = f.s * p.s + f.w * p.w;
    const rp = r.s * p.s + r.w * p.w;
    const up = cosd(KAPPA) * p.z + sind(KAPPA) * fp;
    const depth = -cosd(KAPPA) * fp + sind(KAPPA) * p.z; // > 0: on the camera's side
    return { x: cx + R * rp, y: cy - R * up, front: depth >= -1e-9 };
  };
  return { W, H, R, cx, cy, proj };
}

export function runs(cam: ReturnType<typeof camera>, f: (t: number) => { w: number; s: number; z: number }, t0: number, t1: number, n: number, keep?: (t: number) => boolean): Run[] {
  const out: Run[] = [];
  let cur: Run | null = null;
  for (let i = 0; i <= n; i++) {
    const t = t0 + ((t1 - t0) * i) / n;
    if (keep && !keep(t)) {
      cur = null;
      continue;
    }
    const p = cam.proj(f(t));
    if (!cur || cur.front !== p.front) {
      const prev: Pt | undefined = cur?.pts[cur.pts.length - 1];
      cur = { pts: prev ? [prev] : [], front: p.front };
      out.push(cur);
    }
    cur.pts.push({ x: p.x, y: p.y });
  }
  return out.filter((r) => r.pts.length > 1);
}

export type DomeOpts = {
  phi?: number;
  dec: number; // the sun's declination (its day circle)
  sunH: number | null; // the sun's hour angle now (null: no sun drawn)
  alm?: number | null; // an almucantar to draw (altitude circle)
  starDec?: number | null; // a star's day circle
  starH?: number | null;
  showHourArc?: boolean;
};

export function dome(W: number, o: DomeOpts) {
  const phi = o.phi ?? PHI;
  const cam = camera(W);
  const { R, cx, cy } = cam;
  const horizon = runs(cam, (A) => localAltAz(0, A), -180, 180, 180);
  const meridian = runs(cam, (h) => localAltAz(h, 0), 0, 180, 120); // S → zenith → N
  const equator = runs(cam, (H) => local(0, H, phi), -180, 180, 180, (H) => altitude(0, H, phi) >= -0.01);
  const day = runs(cam, (H) => local(o.dec, H, phi), -180, 180, 240);
  const dayUp = runs(cam, (H) => local(o.dec, H, phi), -180, 180, 240, (H) => altitude(o.dec, H, phi) >= 0);
  const pole = cam.proj({ w: 0, s: -cosd(phi), z: sind(phi) });
  const ctr = { x: cx, y: cy };
  const zen = cam.proj({ w: 0, s: 0, z: 1 });
  const cardinal = (A: number) => cam.proj(localAltAz(0, A));
  const N = cardinal(180);
  const S = cardinal(0);
  const E = cardinal(-90);
  const Wp = cardinal(90);
  let sun: (Pt & { front: boolean; h: number; A: number }) | null = null;
  let hArc: Run[] = [];
  let hFoot: Pt | null = null;
  let hourArc: Run[] = [];
  if (o.sunH !== null) {
    const v = local(o.dec, o.sunH, phi);
    const p = cam.proj(v);
    const h = altitude(o.dec, o.sunH, phi);
    const A = (Math.atan2(v.w, v.s) * 180) / Math.PI;
    sun = { x: p.x, y: p.y, front: p.front, h, A };
    if (h > 0) {
      hArc = runs(cam, (t) => localAltAz(t, A), 0, h, 40);
      const f = cam.proj(localAltAz(0, A));
      hFoot = { x: f.x, y: f.y };
    }
    if (o.showHourArc) {
      const Hs = wrap180(o.sunH);
      hourArc = runs(cam, (t) => local(0, t, phi), 0, Hs, 60);
    }
  }
  let alm: Run[] = [];
  if (o.alm !== null && o.alm !== undefined) {
    const a = o.alm;
    alm = runs(cam, (A) => localAltAz(a, A), -180, 180, 180);
  }
  let star: Run[] = [];
  let starPt: (Pt & { front: boolean }) | null = null;
  if (o.starDec !== null && o.starDec !== undefined) {
    const d = o.starDec;
    star = runs(cam, (H) => local(d, H, phi), -180, 180, 240);
    if (o.starH !== null && o.starH !== undefined) {
      const p = cam.proj(local(d, o.starH, phi));
      starPt = { x: p.x, y: p.y, front: p.front };
    }
  }

  // labels: fixed anchors, nudged outward from the sphere where needed
  const out = (p: Pt, d: number) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const L = Math.hypot(dx, dy) || 1;
    return { x: p.x + (dx / L) * d, y: p.y + (dy / L) * d };
  };
  const labels: Label[] = [];
  const put = (id: string, p: Pt, text: string, size = 12, anchor: Label['anchor'] = 'middle') =>
    labels.push({ id, x: p.x, y: p.y, text, size, anchor, placed: true });
  put('N', { x: N.x, y: N.y + 14 }, 'N', 12);
  put('S', { x: S.x, y: S.y - 9 }, 'S', 12);
  put('E', { x: E.x - 9, y: E.y + 4 }, 'E', 12, 'end');
  put('W', { x: Wp.x + 9, y: Wp.y + 4 }, 'W', 12, 'start');
  put('zenith', { x: zen.x, y: zen.y - 9 }, 'zenith', 11);
  const pl = out(pole, 0);
  put('pole', { x: pl.x + 8, y: pl.y + 3 }, 'pole', 11, 'start');
  put('you', { x: cx + 8, y: cy + 15 }, 'you', 11, 'start');
  if (sun && sun.h > -2) {
    const right = sun.x >= cx;
    put('sun', { x: sun.x + (right ? 10 : -10), y: sun.y - 8 }, 'sun', 12, right ? 'start' : 'end');
  }
  if (hArc.length && sun && hFoot) {
    const mid = hArc[0].pts[Math.floor(hArc[0].pts.length / 2)];
    const right = sun.x >= cx;
    put('h', { x: mid.x + (right ? 9 : -9), y: mid.y + 4 }, 'h', 13, right ? 'start' : 'end');
  }
  return { ...cam, horizon, meridian, equator, day, dayUp, pole, ctr, zen, N, S, E, Wp, sun, hArc, hFoot, hourArc, alm, star, starPt, labels };
}

/** The dome for the sun on a date at an hour angle (helper used by several plates). */
export const sunDome = (W: number, lam: number, H: number | null, extra: Partial<DomeOpts> = {}) =>
  dome(W, { dec: eclDec(lam), sunH: H, ...extra });

export function labelBox(l: Pick<Label, 'x' | 'y' | 'text' | 'size' | 'anchor'>): Box {
  const w = [...l.text].length * l.size * 0.56;
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'middle' ? l.x - w / 2 : l.x - w;
  return { x0, y0: l.y - l.size * 0.8, x1: x0 + w, y1: l.y + l.size * 0.25 };
}
export { EPS };
