/**
 * Recoil anchor escapement, 2D geometry + a quasi-static contact solver. Shared by the 3D
 * plate (extruded) and its SVG poster. Units: escape wheel tip radius = 1, wheel centre at
 * the origin, anchor pivot straight above it. Angles in radians. The wheel turns CLOCKWISE
 * on screen (its angle `w` grows clockwise); the anchor angle `a` is counter-clockwise.
 *
 * The solver has no dynamics: every frame the weight pushes the wheel forward until a tooth
 * tip touches a pallet; if the anchor has moved a pallet into a tooth, the wheel is pushed
 * back (recoil). That is exactly how a recoil anchor behaves, minus the bounce.
 */
export const N_TEETH = 30;
export const PITCH = (2 * Math.PI) / N_TEETH;
export const R_TIP = 1;
export const R_ROOT = 0.8;
/** Anchor spans 7.5 teeth (90°): pallets sit where tangents from the pivot touch the wheel. */
export const SPAN = 7.5 * PITCH;
export const PIVOT_Y = R_TIP / Math.cos(SPAN / 2);
/** Anchor (and pendulum) amplitude. */
export const AMP = (4.2 * Math.PI) / 180;

export type Pt = [number, number];

const rot = (p: Pt, c: Pt, a: number): Pt => {
  const s = Math.sin(a);
  const k = Math.cos(a);
  const x = p[0] - c[0];
  const y = p[1] - c[1];
  return [c[0] + x * k - y * s, c[1] + x * s + y * k];
};

/** One tooth as a polygon at wheel angle 0, tooth index i (tip at math angle 90° - i·pitch). */
export function toothPoly(i: number, w = 0): Pt[] {
  const th = Math.PI / 2 - i * PITCH - w; // clockwise
  const P = (r: number, t: number): Pt => [r * Math.cos(t), r * Math.sin(t)];
  // tip leads (downstream = clockwise = smaller math angle). Front face slightly undercut,
  // back face slopes up from the root behind.
  return [
    P(R_TIP, th),
    P(R_ROOT, th + 0.12 * PITCH),
    P(R_ROOT, th + 0.55 * PITCH),
  ];
}

/** Full wheel outline (for drawing), at wheel angle w. */
export function wheelOutline(w = 0): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < N_TEETH; i++) {
    const t = toothPoly(i, w);
    out.push(t[2], t[0], t[1]);
  }
  return out.reverse();
}

/**
 * Pallets in the anchor's neutral pose. Each is a quadrilateral whose UPSTREAM face is the
 * working face, slanted so that moving the pallet outward lets the wheel advance (impulse)
 * and moving it inward pushes the wheel back (recoil).
 * Local frame at the pallet point: r = outward radial unit, d = downstream tangent unit.
 */
const DEPTH = 0.04; // pallet tip sits this far inside the tip circle at neutral
const SLOPE = 0.9; // tangential run of the face per unit of radial depth
function pallet(theta: number): Pt[] {
  const r: Pt = [Math.cos(theta), Math.sin(theta)];
  const d: Pt = [Math.sin(theta), -Math.cos(theta)]; // clockwise tangent
  const at = (rad: number, dn: number): Pt => [
    rad * r[0] + dn * d[0],
    rad * r[1] + dn * d[1],
  ];
  const tipR = R_TIP - DEPTH;
  const topR = R_TIP + 0.16;
  return [
    at(tipR, 0), // tip (inner, downstream end of the working face)
    at(topR, -(topR - tipR) * SLOPE), // outer, upstream end of the working face
    at(topR, 0.06),
    at(tipR + 0.01, 0.035), // back of the pallet
  ];
}
export const ENTRY_THETA = Math.PI / 2 + SPAN / 2;
export const EXIT_THETA = Math.PI / 2 - SPAN / 2;
export const PALLETS: Pt[][] = [pallet(ENTRY_THETA), pallet(EXIT_THETA)];
export const PIVOT: Pt = [0, PIVOT_Y];

export function palletsAt(a: number): Pt[][] {
  return PALLETS.map((p) => p.map((q) => rot(q, PIVOT, a)));
}

function inside(p: Pt, poly: Pt[]) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** Teeth that can reach a pallet: those in the top half of the wheel. */
function hits(w: number, pals: Pt[][]) {
  for (let i = 0; i < N_TEETH; i++) {
    const th = Math.PI / 2 - i * PITCH - w;
    if (Math.sin(th) < 0.3) continue;
    const t = toothPoly(i, w);
    for (const p of pals) {
      if (inside(t[0], p)) return true;
      for (const q of p) if (inside(q, t)) return true;
    }
  }
  return false;
}

/**
 * Advance the wheel for a new anchor angle. `maxStep` limits how far the wheel may run
 * forward this frame (so drops are visible), in radians.
 */
export function solveWheel(w: number, a: number, maxStep = 0.05): number {
  const pals = palletsAt(a);
  const h = 0.0004;
  let k = 0;
  // recoil: pushed back out of a pallet
  while (hits(w, pals) && k++ < 400) w -= h;
  // drive: run forward to contact
  let run = 0;
  while (run < maxStep && !hits(w + h, pals)) {
    w += h;
    run += h;
  }
  return w;
}

/** Largest anchor rotation the contact solver handles in one go (radians). Bigger moves skip
 *  past a tooth (the wheel then stalls or runs backwards), so `stepMovement` substeps. */
const MAX_DA = (0.12 * Math.PI) / 180;

/**
 * Advance the movement by `dt` seconds: pendulum phase and wheel angle, substepped so the
 * anchor never moves more than MAX_DA between two solver calls. Correct at any frame rate
 * and any period (checked by scripts: one tooth per period, from 0.1 m to 4 m, 20 to 144 fps).
 */
export function stepMovement(
  s: { phase: number; w: number },
  dt: number,
  period: number,
): { phase: number; w: number } {
  const dPhase = (2 * Math.PI * dt) / period;
  const maxDa = AMP * Math.abs(dPhase); // upper bound of |Δa| for this frame
  const n = Math.max(1, Math.min(200, Math.ceil(maxDa / MAX_DA)));
  let { phase, w } = s;
  const sub = dt / n;
  for (let i = 0; i < n; i++) {
    phase += dPhase / n;
    w = solveWheel(w, AMP * Math.sin(phase), Math.max(0.004 / n, sub * 2.2));
  }
  return { phase, w };
}
