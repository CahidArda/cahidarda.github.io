/**
 * The astrolabe's engraving, as pure geometry: one source for the 3D model's textures and
 * meshes, the SVG posters, every 2D plate in this folder, and the sweep scripts.
 *
 * Frame: PLATE UNITS, the equator circle has radius 1. Origin = the pole (the pin).
 *   +u = right = WEST, +v = up = towards the throne = the upper (south) meridian.
 *   Projection: stereographic from the SOUTH celestial pole onto the equator plane, so a point at
 *   declination δ and hour angle H (+ west) lands at r = tan((90° − δ)/2), (u, v) = r (sin H, cos H).
 *   Northern astrolabe: the tropic of Capricorn (δ = −ε) is the plate's edge.
 *   Angles of prims (`rot`) are degrees counter-clockwise on screen (maths convention, y up).
 * Front of the instrument: plate (tympan) for latitude φ, the rete (the sky) on top, the rule.
 * Back: altitude scale (upper quadrants), zodiac ring, calendar ring, the alidade.
 *   Back angle convention: the alidade at "setting" a has its sighting tip at screen angle
 *   ψ = 180° − a (a = 0: tip pointing left, horizontal; a = 90°: straight up). So the altitude
 *   scale reads a directly in the upper-left quadrant, and the zodiac ring puts ecliptic
 *   longitude λ at ψ = 180° − λ, the calendar ring each date at ψ = 180° − λ(date).
 *   The back is drawn as seen from behind (you look at the back face).
 */
import {
  EPS,
  STARS,
  SIGN_NAMES,
  acosd,
  altitude,
  asind,
  atan2d,
  cosd,
  doyOf,
  eclDec,
  eclRA,
  hourAngle,
  sind,
  sunLambda,
  tand,
  wrap180,
  wrap360,
} from './astro.ts';

export type Role = 'ink' | 'soft' | 'faint' | 'paper' | 'deep' | 'red' | 'brass' | 'sky';
export type UV = { u: number; v: number };
export type Prim =
  | { k: 'line'; pts: UV[]; role: Role; w: number; dash?: number[]; id?: string }
  | { k: 'circle'; c: UV; r: number; role: Role; w: number; dash?: number[]; fill?: Role; id?: string }
  | {
      k: 'text';
      at: UV;
      text: string;
      size: number;
      role: Role;
      anchor: 'start' | 'middle' | 'end';
      rot?: number;
      id?: string;
    };

/* ───────────────────────────── sizes (plate units) ───────────────────────────── */
export const RCAN = tand(45 - EPS / 2); // tropic of Cancer, 0.656
export const REQ = 1; // equator
export const RC = tand(45 + EPS / 2); // tropic of Capricorn = the plate's edge, 1.524
export const RM = RC * 1.2; // the mater's outer radius (limb ring RC..RM on the front)
export const LIMB_TICK_IN = RC * 1.015;

/* ───────────────────────────── the projection ───────────────────────────── */
export const radiusOf = (dec: number) => tand((90 - dec) / 2);
export const stereo = (dec: number, H: number): UV => {
  const r = radiusOf(dec);
  return { u: r * sind(H), v: r * cosd(H) };
};
/** Inverse: plate point → (δ, H). */
export function unstereo(p: UV) {
  const r = Math.hypot(p.u, p.v);
  return { dec: 90 - 2 * (Math.atan(r) * 180) / Math.PI, H: atan2d(p.u, p.v) };
}
/** (δ, H) of the point at altitude h, azimuth A (from south, + west), for latitude φ. */
export function decHOf(h: number, A: number, phi: number) {
  const sd = sind(phi) * sind(h) - cosd(phi) * cosd(h) * cosd(A);
  const dec = asind(sd);
  const H = atan2d(cosd(h) * sind(A), sind(phi) * cosd(h) * cosd(A) + cosd(phi) * sind(h));
  return { dec, H };
}
export const altAzToPlate = (h: number, A: number, phi: number) => {
  const { dec, H } = decHOf(h, A, phi);
  return stereo(dec, H);
};
/** Almucantar of altitude h: a circle centred on the meridian. */
export const almucantar = (h: number, phi: number) => ({
  cv: cosd(phi) / (sind(phi) + sind(h)),
  r: cosd(h) / (sind(phi) + sind(h)),
});
export const zenithOf = (phi: number): UV => ({ u: 0, v: tand((90 - phi) / 2) });

/* ───────────────────────────── clipping to the plate ───────────────────────────── */
/** Sample f(t) on [t0, t1]; keep runs inside radius `R`, ends refined by bisection. */
export function clipRuns(f: (t: number) => UV, t0: number, t1: number, n: number, R = RC) {
  const ins = (p: UV) => Math.hypot(p.u, p.v) <= R + 1e-12;
  const edge = (ta: number, tb: number) => {
    // ta inside, tb outside (or reverse): find the crossing
    let a = ta;
    let b = tb;
    const ia = ins(f(a));
    for (let i = 0; i < 40; i++) {
      const m = (a + b) / 2;
      if (ins(f(m)) === ia) a = m;
      else b = m;
    }
    return f(ia ? a : b);
  };
  const runs: UV[][] = [];
  let cur: UV[] | null = null;
  let prevT = t0;
  for (let i = 0; i <= n; i++) {
    const t = t0 + ((t1 - t0) * i) / n;
    const p = f(t);
    if (ins(p)) {
      if (!cur) {
        cur = [];
        if (i > 0) cur.push(edge(t, prevT));
        runs.push(cur);
      }
      cur.push(p);
    } else if (cur) {
      cur.push(edge(prevT, t));
      cur = null;
    }
    prevT = t;
  }
  return runs.filter((r) => r.length > 1);
}

/* ───────────────────────────── the plate (tympan) ───────────────────────────── */
export const ALMS = [0, 10, 20, 30, 40, 50, 60, 70, 80];
export const AZIS = [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150];

export function almucantarRuns(h: number, phi: number) {
  const { cv, r } = almucantar(h, phi);
  return clipRuns((t) => ({ u: r * cosd(t), v: cv + r * sind(t) }), -90, 270, 720);
}
export function azimuthRuns(A: number, phi: number) {
  return clipRuns((h) => altAzToPlate(h, A, phi), 0, 90, 360);
}
/** Where to write an almucantar's number: on its east (left) side, along an azimuth line. */
export function almLabelAt(h: number, phi: number): UV {
  const p = altAzToPlate(h, -75, phi);
  return { u: p.u, v: p.v };
}

const circ = (r: number, role: Role, w: number, id: string, dash?: number[]): Prim => ({
  k: 'circle',
  c: { u: 0, v: 0 },
  r,
  role,
  w,
  id,
  dash,
});

/** Everything engraved on the plate for latitude φ (no rete, no limb). */
export function platePrims(phi: number, opts: { labels?: boolean; azimuths?: boolean } = {}): Prim[] {
  const out: Prim[] = [];
  out.push(circ(RCAN, 'soft', 0.8, 'cancer'));
  out.push(circ(REQ, 'soft', 0.8, 'equator'));
  out.push(circ(RC, 'ink', 1.2, 'capricorn'));
  out.push({ k: 'line', pts: [{ u: 0, v: -RC }, { u: 0, v: RC }], role: 'faint', w: 0.7, id: 'meridian' });
  out.push({ k: 'line', pts: [{ u: -RC, v: 0 }, { u: RC, v: 0 }], role: 'faint', w: 0.7, id: 'ew' });
  if (opts.azimuths !== false)
    for (const A of AZIS)
      for (const run of azimuthRuns(A, phi))
        out.push({ k: 'line', pts: run, role: 'faint', w: 0.6, id: `az${A}` });
  for (const h of ALMS)
    for (const run of almucantarRuns(h, phi))
      out.push({ k: 'line', pts: run, role: h === 0 ? 'ink' : 'soft', w: h === 0 ? 1.4 : 0.75, id: `alm${h}` });
  const z = zenithOf(phi);
  out.push({ k: 'circle', c: z, r: 0.012, role: 'ink', w: 0.8, fill: 'ink', id: 'zenith' });
  if (opts.labels !== false) {
    for (const h of ALMS.filter((x) => x > 0 && x <= 60)) {
      const p = almLabelAt(h, phi);
      out.push({ k: 'text', at: p, text: `${h}`, size: 0.075, role: 'soft', anchor: 'middle', id: `lab${h}` });
    }
    const e = altAzToPlate(0, -90, phi);
    const w = altAzToPlate(0, 90, phi);
    out.push({ k: 'text', at: { u: e.u - 0.06, v: e.v + 0.06 }, text: 'E', size: 0.09, role: 'ink', anchor: 'end', id: 'labE' });
    out.push({ k: 'text', at: { u: w.u + 0.06, v: w.v + 0.06 }, text: 'W', size: 0.09, role: 'ink', anchor: 'start', id: 'labW' });
    out.push({ k: 'text', at: { u: 0.04, v: -RC + 0.08 }, text: `${Math.abs(phi).toFixed(phi % 1 ? 1 : 0)}°`, size: 0.08, role: 'ink', anchor: 'start', id: 'labPhi' });
  }
  return out;
}

/* ───────────────────────────── the limb (front) ───────────────────────────── */
/** Screen angle (maths, CCW) of hour angle H on the front: noon at the top, west to the right. */
export const limbAngle = (H: number) => 90 - H;
/** 24-hour scale, noon at the top, morning on the left (east). */
export function limbPrims(): Prim[] {
  const out: Prim[] = [];
  out.push(circ(RM, 'ink', 1.3, 'limbOut'));
  out.push(circ(RC * 1.0, 'ink', 1.0, 'limbIn'));
  out.push(circ(RC * 1.09, 'faint', 0.6, 'limbMid'));
  for (let H = -180; H < 180; H += 1) {
    const a = limbAngle(H);
    const major = H % 15 === 0;
    const mid = H % 5 === 0;
    const r1 = LIMB_TICK_IN;
    const r2 = major ? RC * 1.09 : mid ? RC * 1.06 : RC * 1.035;
    if (!mid && !major) continue; // 1° ticks are below what a hand-sized plate shows at this scale
    out.push({
      k: 'line',
      pts: [
        { u: r1 * cosd(a), v: r1 * sind(a) },
        { u: r2 * cosd(a), v: r2 * sind(a) },
      ],
      role: 'ink',
      w: major ? 0.9 : 0.6,
    });
    if (major) {
      const hr = ((12 + H / 15 + 11) % 12) + 1; // 1..12
      const rt = RC * 1.145;
      out.push({
        k: 'text',
        at: { u: rt * cosd(a), v: rt * sind(a) },
        text: `${hr}`,
        size: 0.085,
        role: 'ink',
        anchor: 'middle',
        rot: a - 90,
        id: `hr${H}`,
      });
    }
  }
  return out;
}
/** Sun time (hours) read at a limb angle. */
export const timeAtLimb = (H: number) => 12 + H / 15;

/* ───────────────────────────── the rete ───────────────────────────── */
/** Ecliptic circle on the plate when the rete is turned to sidereal time `lst`. */
export function eclipticCircle(lst: number) {
  const H = lst - 270; // direction of the Capricorn point (λ = 270°, α = 270°)
  return { c: { u: tand(EPS) * sind(H), v: tand(EPS) * cosd(H) }, r: 1 / cosd(EPS) };
}
export const eclipticPoint = (lam: number, lst: number) => stereo(eclDec(lam), lst - eclRA(lam));
export const starPoint = (s: { ra: number; dec: number }, lst: number) => stereo(s.dec, lst - s.ra);
export const ECL_BAND = 0.07; // the ecliptic ring's width (inward from the circle)

export function retePrims(lst: number, opts: { stars?: boolean; signs?: boolean } = {}): Prim[] {
  const out: Prim[] = [];
  const ec = eclipticCircle(lst);
  out.push({ k: 'circle', c: ec.c, r: ec.r, role: 'ink', w: 1.2, id: 'ecl' });
  out.push({ k: 'circle', c: ec.c, r: ec.r - ECL_BAND, role: 'ink', w: 0.8, id: 'eclIn' });
  // sign boundaries across the band (every 30° of λ), drawn radially to the ring's centre
  for (let lam = 0; lam < 360; lam += 30) {
    const p = eclipticPoint(lam, lst);
    const d = Math.hypot(p.u - ec.c.u, p.v - ec.c.v);
    const q = { u: ec.c.u + ((p.u - ec.c.u) * (d - ECL_BAND)) / d, v: ec.c.v + ((p.v - ec.c.v) * (d - ECL_BAND)) / d };
    out.push({ k: 'line', pts: [p, q], role: 'ink', w: 0.8 });
  }
  // the rete's frame: Capricorn rim and a bar along the solstitial line
  out.push({ k: 'circle', c: { u: 0, v: 0 }, r: RC * 0.985, role: 'ink', w: 1.0, id: 'reteRim' });
  const a = eclipticPoint(90, lst);
  const b = eclipticPoint(270, lst);
  out.push({ k: 'line', pts: [a, b], role: 'ink', w: 0.9, id: 'reteBar' });
  if (opts.stars !== false)
    for (const s of STARS) {
      const tip = starPoint(s, lst);
      out.push(...pointer(tip, s.id));
    }
  if (opts.signs) {
    for (let i = 0; i < 12; i++) {
      const lam = i * 30 + 15;
      const p = eclipticPoint(lam, lst);
      const d = Math.hypot(p.u - ec.c.u, p.v - ec.c.v);
      const q = { u: ec.c.u + ((p.u - ec.c.u) * (d - ECL_BAND / 2)) / d, v: ec.c.v + ((p.v - ec.c.v) * (d - ECL_BAND / 2)) / d };
      out.push({ k: 'text', at: q, text: SIGN_NAMES[i].slice(0, 3), size: 0.045, role: 'ink', anchor: 'middle', rot: atan2d(q.v - ec.c.v, q.u - ec.c.u) - 90 });
    }
  }
  return out;
}
/** A star pointer: a small flame whose tip is the star, its base towards the plate centre. */
export function pointerShape(tip: UV) {
  const r = Math.hypot(tip.u, tip.v) || 1;
  const du = -tip.u / r;
  const dv = -tip.v / r;
  const L = 0.11;
  const w = 0.03;
  const base = { u: tip.u + du * L, v: tip.v + dv * L };
  return [tip, { u: base.u - dv * w, v: base.v + du * w }, { u: base.u + dv * w, v: base.v - du * w }];
}
function pointer(tip: UV, id: string): Prim[] {
  const t = pointerShape(tip);
  return [{ k: 'line', pts: [...t, t[0]], role: 'ink', w: 0.9, id: `star-${id}` }];
}

/* ───────────────────────────── the back ───────────────────────────── */
export const BACK = {
  altOut: RM,
  altIn: RM * 0.9,
  zodOut: RM * 0.9,
  zodIn: RM * 0.79,
  calOut: RM * 0.79,
  calIn: RM * 0.68,
};
export const backPsi = (a: number) => 180 - a;
const at = (r: number, psi: number): UV => ({ u: r * cosd(psi), v: r * sind(psi) });
const MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

export function backPrims(): Prim[] {
  const out: Prim[] = [];
  const B = BACK;
  for (const r of [B.altOut, B.altIn, B.zodIn, B.calIn]) out.push(circ(r, 'ink', r === B.altOut ? 1.3 : 0.9, `b${r.toFixed(3)}`));
  out.push({ k: 'line', pts: [{ u: -B.calIn, v: 0 }, { u: B.calIn, v: 0 }], role: 'faint', w: 0.6 });
  out.push({ k: 'line', pts: [{ u: 0, v: -B.calIn }, { u: 0, v: B.calIn }], role: 'faint', w: 0.6 });
  // altitude scale, both upper quadrants (left reads a; right is its mirror)
  for (let a = 0; a <= 90; a += 1) {
    const major = a % 10 === 0;
    const mid = a % 5 === 0;
    for (const psi of [180 - a, a]) {
      const r2 = major ? B.altIn + (B.altOut - B.altIn) * 0.55 : mid ? B.altIn + (B.altOut - B.altIn) * 0.38 : B.altIn + (B.altOut - B.altIn) * 0.22;
      out.push({ k: 'line', pts: [at(B.altIn, psi), at(r2, psi)], role: 'ink', w: major ? 0.8 : 0.45 });
    }
    if (major && a > 0 && a < 90) {
      const rt = B.altIn + (B.altOut - B.altIn) * 0.78;
      out.push({ k: 'text', at: at(rt, 180 - a), text: `${a}`, size: 0.075, role: 'ink', anchor: 'middle', rot: 90 - a, id: `alt${a}` });
    }
  }
  // zodiac ring: 12 signs of 30°, ticks every 5° (1° on the band's inner edge)
  for (let lam = 0; lam < 360; lam += 1) {
    const psi = backPsi(lam);
    const major = lam % 30 === 0;
    const mid = lam % 5 === 0;
    const r2 = major ? B.zodOut : mid ? B.zodIn + (B.zodOut - B.zodIn) * 0.3 : B.zodIn + (B.zodOut - B.zodIn) * 0.15;
    out.push({ k: 'line', pts: [at(B.zodIn, psi), at(r2, psi)], role: 'ink', w: major ? 0.8 : 0.4 });
  }
  for (let i = 0; i < 12; i++) {
    const psi = backPsi(i * 30 + 15);
    const rt = B.zodIn + (B.zodOut - B.zodIn) * 0.62;
    out.push({ k: 'text', at: at(rt, psi), text: SIGN_NAMES[i], size: 0.06, role: 'soft', anchor: 'middle', rot: psi - 90, id: `sign${i}` });
  }
  // calendar ring: each date at the sun's longitude for that date (so the ring is uneven)
  for (let m = 0; m < 12; m++) {
    for (let d = 1; d <= MDAYS[m]; d++) {
      const doy = doyOf(m, d);
      const psi = backPsi(sunLambda(doy));
      const first = d === 1;
      const five = d % 5 === 0;
      if (!first && !five) continue;
      const r2 = first ? B.calIn : B.calOut - (B.calOut - B.calIn) * 0.25;
      out.push({ k: 'line', pts: [at(B.calOut, psi), at(r2, psi)], role: 'ink', w: first ? 0.8 : 0.45 });
    }
    const mid = backPsi(sunLambda(doyOf(m, 16)));
    const rt = B.calIn + (B.calOut - B.calIn) * 0.45;
    out.push({ k: 'text', at: at(rt, mid), text: MONTH[m], size: 0.06, role: 'soft', anchor: 'middle', rot: mid - 90, id: `mon${m}` });
  }
  return out;
}

/* ───────────────────────────── the alidade sight (back) ───────────────────────────── */
export const VANE_D = RM * 0.78; // distance of each vane from the pin
export const HOLE = 0.012; // pinhole radius (plate units)
/** Angle at which the light from the top hole still falls on the bottom hole. */
export const SIGHT_TOL = (Math.atan(HOLE / (2 * VANE_D)) * 180) / Math.PI + 0.12; // ≈ 0.36°, with the sun's disc
/**
 * The alidade at setting a (degrees above horizontal, tip up-left), the sun at altitude h in the
 * plane of the instrument (it hangs facing the sun edge-on). The ray through the top hole lands
 * on the bottom vane `miss` units from its hole (perpendicular to the alidade).
 */
export function sight(a: number, h: number) {
  const tip = at(VANE_D, backPsi(a));
  const tail = at(VANE_D, backPsi(a) + 180);
  const miss = 2 * VANE_D * tand(h - a);
  const on = Math.abs(h - a) <= SIGHT_TOL;
  // the light: parallel rays from the sun (upper left at altitude h), one through the top hole
  const dir = { u: cosd(h), v: -sind(h) }; // direction of travel
  // where that ray meets the line of the bottom vane (perpendicular to the alidade through tail)
  const n = { u: cosd(backPsi(a)), v: sind(backPsi(a)) }; // alidade axis (towards the tip)
  const t = -(2 * VANE_D) / (dir.u * n.u + dir.v * n.v); // tip + t·dir lies at distance −VANE_D along n
  const hit = { u: tip.u + dir.u * t, v: tip.v + dir.v * t };
  return { tip, tail, miss, on, dir, hit };
}

/* ───────────────────────────── solving the plate ───────────────────────────── */
/** The rete turn (LST) that puts the sun's point (λ) on almucantar h, east (morning) or west side. */
export function solveRete(lam: number, h: number, side: 'am' | 'pm', phi: number) {
  const dec = eclDec(lam);
  const H0 = hourAngle(h, dec, phi);
  if (H0 === null) return null;
  const H = side === 'am' ? -H0 : H0;
  return { H, lst: wrap360(H + eclRA(lam)), dec };
}
/** Altitude of the sun's point when the rete is at `lst` (what the plate shows under it). */
export const plateAltitude = (lam: number, lst: number, phi: number) =>
  altitude(eclDec(lam), wrap180(lst - eclRA(lam)), phi);
export const sunHourOnRete = (lam: number, lst: number) => wrap180(lst - eclRA(lam));
export { acosd };
