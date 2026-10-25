/**
 * Graham deadbeat escapement: exact 2D geometry and a contact solver, shared by the 3D model,
 * its SVG poster and the close-up of PendulumPlate.tsx, and swept by deadbeat.sweep.ts.
 *
 * Frame: escape-wheel centre at the origin, tip radius 1, y UP, angles in radians, maths
 * convention. The wheel turns CLOCKWISE (its angle `w` grows clockwise). The anchor (pallet
 * arbor) pivot P sits straight above at √2, so the two tangents from P touch the tip circle at
 * 135° and 45°, 90° apart: the pallets span 7½ teeth of the 30-tooth wheel (12° pitch). The
 * anchor angle `a` is counter-clockwise; it swings with the pendulum, a = AMP·sin(phase).
 *
 * Deadbeat: each pallet has a LOCKING face that is an arc centred on P, so while a tooth rests
 * on it the anchor can swing on without pushing the wheel back (no recoil, unlike the anchor
 * escapement). The other end is the slanted IMPULSE face: when the locking corner leaves the tip
 * circle, the tooth tip slides along it and pushes the pallet out (the impulse that keeps the
 * pendulum going), then drops off the far corner and the wheel runs free until a tooth lands on
 * the other pallet's locking face.
 *   entry pallet (left): teeth move TOWARD P; locking face = its outer arc, radius 1 + WP/2.
 *   exit pallet (right): teeth move AWAY from P; locking face = its inner arc, radius 1 − WP/2.
 * Per swing the wheel turns half a tooth (6°): WP of impulse + DROP of free run.
 *
 * Teeth: pointed and raked: the tip leads, the front face leans back from it, so only the tip
 * touches a locking face.
 *
 * Solver: kinematic. The pendulum is prescribed; each substep the anchor moves, then the wheel
 * (always pushed forward by the weight) advances until a tooth touches a pallet, found by
 * stepping and bisection to CONTACT_TOL. It never moves backward: if a pallet ever moved INTO a
 * tooth, that would be recoil, and the sweep asserts it never happens.
 */
export const N_TEETH = 30;
export const PITCH = (2 * Math.PI) / N_TEETH;
export const R_TIP = 1;
export const R_ROOT = 0.8;
export const SPAN = 7.5 * PITCH; // 90°
export const PIVOT_Y = R_TIP / Math.cos(SPAN / 2); // √2
export type Pt = [number, number];
export const PIVOT: Pt = [0, PIVOT_Y];

const D = Math.PI / 180;
/** Pallet width along the tooth path (wheel radians at the tip circle) and the drop. */
export const DROP = 1.0 * D;
export const WP = PITCH / 2 - DROP; // 5°
/** Depth of the pallet corners inside the tip circle when the anchor is at 0 (in tip radii):
 *  locking corner (shallow) and impulse corner (deep). Impulse angle ≈ DB − DA ≈ 2°. */
export const DA = 0.6 * D;
export const DB = 2.6 * D;
/** Anchor (and pendulum) amplitude: lock, impulse and a little supplementary arc. */
export const AMP = 3.6 * D;
/** How far outside the tip circle the pallet bodies reach. */
const OUT = 0.13;

const rot = (p: Pt, c: Pt, a: number): Pt => {
  const s = Math.sin(a);
  const k = Math.cos(a);
  const x = p[0] - c[0];
  const y = p[1] - c[1];
  return [c[0] + x * k - y * s, c[1] + x * s + y * k];
};
const fromPivot = (rho: number, psi: number): Pt => [PIVOT[0] + rho * Math.cos(psi), PIVOT[1] + rho * Math.sin(psi)];

/** On the circle of radius rho about P, the angle ψ (near psi0) where the point is at distance
 *  `r` from the wheel centre. Bisection on a bracket around psi0. */
function psiAtRadius(rho: number, r: number, psi0: number, side: 1 | -1): number {
  // moving ψ by +side brings the point toward the wheel centre near the pallet
  let lo = psi0 - 0.6;
  let hi = psi0 + 0.6;
  const f = (psi: number) => Math.hypot(...fromPivot(rho, psi)) - r;
  // f decreases as ψ moves in direction `side` near psi0
  if (side === -1) [lo, hi] = [hi, lo];
  for (let i = 0; i < 80; i++) {
    const m = (lo + hi) / 2;
    if (f(m) > 0) lo = m;
    else hi = m;
  }
  return (lo + hi) / 2;
}

/** Pallet polygon at anchor angle 0. `entry`: left pallet. */
function pallet(entry: boolean): Pt[] {
  const psi0 = entry ? (225 * Math.PI) / 180 : (315 * Math.PI) / 180;
  // CCW rotation about P moves the entry pallet INTO the wheel, the exit pallet OUT of it
  const side: 1 | -1 = entry ? 1 : -1;
  const rLock = entry ? 1 + WP / 2 : 1 - WP / 2;
  const rFar = entry ? 1 - WP / 2 : 1 + WP / 2;
  const n = 48;
  const arc = (rho: number, rFrom: number, rTo: number) => {
    const a0 = psiAtRadius(rho, rFrom, psi0, side);
    const a1 = psiAtRadius(rho, rTo, psi0, side);
    return Array.from({ length: n + 1 }, (_, i) => fromPivot(rho, a0 + ((a1 - a0) * i) / n));
  };
  // locking face: from outside the wheel down to the locking corner A (depth DA)
  const lock = arc(rLock, R_TIP + OUT, R_TIP - DA);
  // far face: from the impulse corner B (depth DB) back out
  const far = arc(rFar, R_TIP - DB, R_TIP + OUT);
  // A → B is the straight impulse face; the far face's outer end joins the lock face's outer end
  // along the arc about P outside the wheel (a straight chord is fine there: nothing touches it)
  return [...lock, ...far];
}
export const PALLETS: Pt[][] = [pallet(true), pallet(false)];
/** Corner indices: A (locking corner) and B (impulse corner) in each polygon. */
export const CORNER_A = 48;
export const CORNER_B = 49;

export function palletsAt(a: number): Pt[][] {
  return PALLETS.map((p) => p.map((q) => rot(q, PIVOT, a)));
}

/**
 * Tooth i at wheel angle w as [tip, shoulder, back root, front root]. The tip leads (clockwise);
 * the front face leans back from it (raked, so only the tip touches a locking face); the back
 * falls steeply from the tip to a shoulder, then slopes to the root. The steep back is what
 * lets the pallet's far corner swing back in after the drop without touching the tooth that
 * just left it (the sweep checks this).
 */
const FRONT = 0.12;
const BACK = 0.6;
const SH_R = 0.9; // shoulder radius
const SH_A = 0.05; // shoulder angle behind the tip, in pitches (0.6°)
export function toothPoly(i: number, w = 0): Pt[] {
  const th = Math.PI / 2 - i * PITCH - w;
  const P = (r: number, t: number): Pt => [r * Math.cos(t), r * Math.sin(t)];
  return [P(R_TIP, th), P(SH_R, th + SH_A * PITCH), P(R_ROOT, th + BACK * PITCH), P(R_ROOT, th + FRONT * PITCH)];
}
/** The tooth as two convex triangles (for the exact contact test). */
const toothTris = (t: Pt[]): Pt[][] => [
  [t[0], t[1], t[3]],
  [t[1], t[2], t[3]],
];
export function wheelOutline(w = 0): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < N_TEETH; i++) {
    const t = toothPoly(i, w);
    out.push(t[2], t[1], t[0], t[3]);
  }
  return out.reverse();
}

/* ───────────────────────────── contact ───────────────────────────── */
/*
 * Contact is computed EXACTLY against each pallet's true shape, not its drawn polygon (whose
 * chords would sag inside the arcs): in the anchor's own frame a pallet is the region
 *   ρ1 ≤ |X − P| ≤ ρ2   (between its two arcs about the pivot),
 *   on the far side of its impulse face AB (away from the wheel's free path), and
 *   within ±WIN of its centre direction ψ0 seen from P.
 * A tooth (triangle) overlaps it when a tooth vertex is inside, a pallet corner (A, B) is inside
 * the tooth, or a tooth edge crosses an arc (inside the window, beyond AB) or the face AB.
 */
type Region = { rho1: number; rho2: number; A: Pt; B: Pt; psi0: number; sideSign: number };
const WIN = 0.45;
function region(entry: boolean): Region {
  const poly = PALLETS[entry ? 0 : 1];
  const A = poly[CORNER_A];
  const B = poly[CORNER_B];
  const rA = Math.hypot(A[0] - PIVOT[0], A[1] - PIVOT[1]);
  const rB = Math.hypot(B[0] - PIVOT[0], B[1] - PIVOT[1]);
  const psi0 = entry ? (225 * Math.PI) / 180 : (315 * Math.PI) / 180;
  // a point well inside the pallet body: mid radius, at the outer (pivot-ward) end
  const midRho = (rA + rB) / 2;
  const probe = fromPivot(midRho, Math.atan2(A[1] + B[1] - 2 * PIVOT[1], A[0] + B[0] - 2 * PIVOT[0]) + (entry ? -0.08 : 0.08)); // outward: away from the wheel
  const sideSign = Math.sign(cross(A, B, probe));
  return { rho1: Math.min(rA, rB), rho2: Math.max(rA, rB), A, B, psi0, sideSign };
}
function cross(a: Pt, b: Pt, c: Pt) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
const angDiff = (a: number, b: number) => {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
};
function inRegion(p: Pt, g: Region, strict = true) {
  const rho = Math.hypot(p[0] - PIVOT[0], p[1] - PIVOT[1]);
  const e = strict ? 1e-12 : -1e-12;
  if (rho <= g.rho1 + e || rho >= g.rho2 - e) return false;
  if (Math.abs(angDiff(Math.atan2(p[1] - PIVOT[1], p[0] - PIVOT[0]), g.psi0)) > WIN) return false;
  return g.sideSign * cross(g.A, g.B, p) > e;
}
function inTri(p: Pt, t: Pt[]) {
  const c1 = cross(t[0], t[1], p);
  const c2 = cross(t[1], t[2], p);
  const c3 = cross(t[2], t[0], p);
  return (c1 > 1e-12 && c2 > 1e-12 && c3 > 1e-12) || (c1 < -1e-12 && c2 < -1e-12 && c3 < -1e-12);
}
function segSeg(a: Pt, b: Pt, c: Pt, d: Pt) {
  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);
  return d1 * d2 < -1e-24 && d3 * d4 < -1e-24;
}
/** Points where segment a→b crosses the circle of radius rho about P (strictly inside the segment). */
function segCircle(a: Pt, b: Pt, rho: number): Pt[] {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const fx = a[0] - PIVOT[0];
  const fy = a[1] - PIVOT[1];
  const A2 = dx * dx + dy * dy;
  const B2 = 2 * (fx * dx + fy * dy);
  const C2 = fx * fx + fy * fy - rho * rho;
  const disc = B2 * B2 - 4 * A2 * C2;
  if (disc <= 0) return [];
  const sq = Math.sqrt(disc);
  const out: Pt[] = [];
  for (const t of [(-B2 - sq) / (2 * A2), (-B2 + sq) / (2 * A2)]) if (t > 1e-9 && t < 1 - 1e-9) out.push([a[0] + t * dx, a[1] + t * dy]);
  return out;
}
function onArcPart(p: Pt, g: Region) {
  if (Math.abs(angDiff(Math.atan2(p[1] - PIVOT[1], p[0] - PIVOT[0]), g.psi0)) > WIN) return false;
  return g.sideSign * cross(g.A, g.B, p) > 1e-12;
}
export const REGIONS = [region(true), region(false)];
/** Tooth triangle (wheel frame) vs pallet k at anchor angle a. */
export function toothHitsPallet(tri: Pt[], k: number, a: number) {
  const g = REGIONS[k];
  const t = tri.map((q) => rot(q, PIVOT, -a)); // into the anchor's frame
  for (const p of t) if (inRegion(p, g)) return true;
  if (inTri(g.A, t) || inTri(g.B, t)) return true;
  for (let i = 0; i < 3; i++) {
    const p = t[i];
    const q = t[(i + 1) % 3];
    if (segSeg(p, q, g.A, g.B)) return true;
    for (const rho of [g.rho1, g.rho2]) for (const x of segCircle(p, q, rho)) if (onArcPart(x, g)) return true;
  }
  return false;
}
/** Teeth near the pallets (upper half of the wheel). */
function nearTeeth(w: number) {
  const out: number[] = [];
  for (let i = 0; i < N_TEETH; i++) {
    const th = Math.PI / 2 - i * PITCH - w;
    if (Math.sin(th) > 0.45) out.push(i);
  }
  return out;
}
/** Does any tooth overlap any pallet at (w, a)? Exact (see above). */
export function collides(w: number, a: number) {
  for (const i of nearTeeth(w)) {
    for (const t of toothTris(toothPoly(i, w))) if (toothHitsPallet(t, 0, a) || toothHitsPallet(t, 1, a)) return true;
  }
  return false;
}
/** Distance from a point to pallet k's true boundary (anchor frame point). */
function distToPallet(p: Pt, g: Region) {
  const rho = Math.hypot(p[0] - PIVOT[0], p[1] - PIVOT[1]);
  const psi = Math.atan2(p[1] - PIVOT[1], p[0] - PIVOT[0]);
  const cands: number[] = [segDist(p, g.A, g.B)];
  // nearest point on each arc, if it lies on the arc part
  for (const r of [g.rho1, g.rho2]) {
    const q = fromPivot(r, psi);
    if (onArcPart(q, g)) cands.push(Math.abs(rho - r));
  }
  return Math.min(...cands);
}
function segDist(p: Pt, a: Pt, b: Pt) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}
/** Smallest tooth–pallet distance (tip radii) over tooth vertices vs pallet faces and pallet
 *  corners vs tooth edges; and which pallet (0 entry, 1 exit). */
export function clearance(w: number, a: number) {
  let best = Infinity;
  let which = -1;
  for (const i of nearTeeth(w)) {
    const poly = toothPoly(i, w);
    for (let k = 0; k < 2; k++) {
      const g = REGIONS[k];
      const t = poly.map((q) => rot(q, PIVOT, -a));
      let d = Infinity;
      for (const p of t) d = Math.min(d, distToPallet(p, g));
      for (const c of [g.A, g.B]) for (let j = 0; j < t.length; j++) d = Math.min(d, segDist(c, t[j], t[(j + 1) % t.length]));
      if (d < best) {
        best = d;
        which = k;
      }
    }
  }
  return { gap: best, pallet: which };
}

/** The wheel stops this close to a pallet (tip radii; on a 40 mm wheel, under 0.1 µm). */
export const CONTACT_TOL = 2e-6;
const SCAN = 0.0015;
/** Advance the wheel from w (not colliding at anchor a) until first contact; at most maxRun. */
export function advance(w: number, a: number, maxRun = 0.2): number {
  let run = 0;
  while (run < maxRun) {
    const step = Math.min(SCAN, maxRun - run);
    if (collides(w + step, a)) {
      let lo = 0;
      let hi = step;
      while (hi - lo > CONTACT_TOL * 0.25) {
        const m = (lo + hi) / 2;
        if (collides(w + m, a)) hi = m;
        else lo = m;
      }
      return w + lo;
    }
    w += step;
    run += step;
  }
  return w;
}

/** Largest anchor move between two solver calls (radians). */
const MAX_DA = 0.05 * D;

/**
 * Advance the movement by dt seconds at pendulum period T: phase and wheel angle, substepped so
 * the anchor never moves more than MAX_DA at once. `tilt` (radians) is the case standing off
 * level: the pendulum still hangs plumb, so relative to the movement the anchor swings about
 * −tilt ("out of beat"). `amp` scales the swing. `recoil` counts substeps where the anchor moved
 * a pallet into a tooth (must stay 0 for a deadbeat).
 */
export function stepMovement(
  s: { phase: number; w: number },
  dt: number,
  T: number,
  o: { tilt?: number; amp?: number } = {},
) {
  const amp = AMP * (o.amp ?? 1);
  const tilt = o.tilt ?? 0;
  const dPhase = (2 * Math.PI * dt) / T;
  const n = Math.max(1, Math.min(400, Math.ceil((amp * Math.abs(dPhase)) / MAX_DA)));
  let { phase, w } = s;
  let recoil = 0;
  for (let i = 0; i < n; i++) {
    phase += dPhase / n;
    const a = amp * Math.sin(phase) - tilt;
    if (collides(w, a)) recoil++;
    w = advance(w, a);
  }
  return { phase, w, recoil };
}

/** Teeth advanced per pendulum period with the case tilted (0 when the clock stops). */
export function teethPerPeriod(tilt: number, periods = 4) {
  const T = 2;
  let s = { phase: Math.PI / 2, w: advance(0, AMP - tilt, 1) };
  // let it settle one period, then count
  for (let k = 0; k < 60; k++) s = stepMovement(s, T / 60, T, { tilt });
  const w0 = s.w;
  for (let k = 0; k < 60 * periods; k++) s = stepMovement(s, T / 60, T, { tilt });
  return (s.w - w0) / PITCH / periods;
}

/** A settled starting state: anchor at +AMP, wheel run on to its first lock. */
export function initialState() {
  const phase = Math.PI / 2;
  return { phase, w: advance(0, AMP * Math.sin(phase), 1) };
}

/**
 * What the escapement is doing at (w, a): which pallet holds the wheel and on which face.
 * 'lock'    a tooth tip rests on a locking arc (the wheel is dead still, the anchor slides on);
 * 'impulse' a tooth tip is on the slanted face (the wheel is pushing the pallet out);
 * the contact point is where tip and pallet meet (wheel frame).
 */
export function contactInfo(w: number, a: number) {
  const c = clearance(w, a);
  const g = REGIONS[c.pallet];
  let face: 'lock' | 'impulse' = 'impulse';
  let point: Pt = [0, 0];
  let tooth = -1;
  let best = Infinity;
  for (let i = 0; i < N_TEETH; i++) {
    const th = Math.PI / 2 - i * PITCH - w;
    if (Math.sin(th) < 0.45) continue;
    const tip = toothPoly(i, w)[0];
    const tl = rot(tip, PIVOT, -a);
    const d = distToPallet(tl, g);
    if (d < best) {
      best = d;
      tooth = i;
      point = tip;
      const rho = Math.hypot(tl[0] - PIVOT[0], tl[1] - PIVOT[1]);
      const lockRho = c.pallet === 0 ? g.rho2 : g.rho1;
      face = Math.abs(rho - lockRho) < 1e-5 ? 'lock' : 'impulse';
    }
  }
  return { pallet: c.pallet as 0 | 1, face, gap: c.gap, tipGap: best, tooth, point };
}
