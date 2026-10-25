/**
 * Episode 4, verge and foliot: a simple honest model of the swing, for VergePlate.tsx, swept by
 * time.sweep.ts.
 *
 * The foliot has no spring and no gravity restoring it: it swings only because the crown wheel's
 * teeth push the verge's pallets, first one way, then the other. Model: the drive gives a torque
 * of constant size τ that always pushes the foliot back toward the middle (it changes sides as
 * the other pallet takes over), the foliot swings out to the angle A the pallets allow. Then
 *     θ'' = −(τ / I) · sign θ,   so   T = 4 √(2 A I / τ)
 * (a ball rolling in a V, not a pendulum). I = I_BAR + 2 m r²: the two regulating weights at r.
 * The period scales as √(I/τ): move the weights out and it slows; change the drive and it
 * changes too. That last line is the whole problem of this clock.
 *
 * Units: r in bar half-lengths (0.2 .. 1), drive relative to the clock's regular weight, time in
 * seconds. Constants are chosen so the regulated clock (r = R_SET, drive 1) beats every 2 s.
 */

export const A_SWING = (40 * Math.PI) / 180; // the foliot swings ±40° (verge pallets ~90° apart)
export const I_BAR = 0.2;
export const M_W = 1;
export const R_MIN = 0.2;
export const R_MAX = 1;
export const R_STEP = 0.002;
/** Where the weights must sit for the clock to keep time with the regular drive. */
export const R_SET = 0.62;
export const DRIVE_MIN = 0.7;
export const DRIVE_MAX = 1.3;
export const T_SET = 2; // seconds per full swing when regulated
/** Teeth on the crown wheel; one tooth escapes per full swing. */
export const CROWN_TEETH = 15;

export const inertia = (r: number) => I_BAR + 2 * M_W * r * r;
/** τ for drive 1, from T_SET at R_SET. */
const TAU1 = (32 * A_SWING * inertia(R_SET)) / (T_SET * T_SET);
export const period = (r: number, drive: number) => 4 * Math.sqrt((2 * A_SWING * inertia(r)) / (TAU1 * drive));
/** The clock counts one swing as T_SET seconds; seconds gained per day (negative = loses). */
export const errPerDay = (r: number, drive: number) => 86400 * (T_SET / period(r, drive) - 1);
/** "Keeps time" within 15 minutes a day: the best a foliot clock ever did (Wikipedia, Verge escapement). */
export const KEEP_TOL = 900;

/** Foliot angle (rad) at time t for a period T: two parabolic arcs per swing. */
export function foliotAngle(t: number, T: number) {
  const a = (32 * A_SWING) / (T * T); // τ/I that gives this T
  const s = ((t % T) + T) % T;
  if (s < T / 2) return A_SWING - (a / 2) * (s - T / 4) ** 2;
  return -(A_SWING - (a / 2) * (s - (3 * T) / 4) ** 2);
}
/** Crown wheel position in teeth: it advances half a tooth while each pallet is pushed (the
 *  half swing from an extreme back through the middle), and rests while the other pallet holds. */
export function crownTeeth(t: number, T: number) {
  const half = Math.floor((2 * t) / T);
  const s = ((2 * t) / T) % 1; // 0..1 inside this half swing
  // each half swing starts at the middle: first the foliot coasts out to its extreme (wheel
  // held), then the tooth drives it back to the middle (wheel advances half a tooth)
  const push = Math.max(0, Math.min(1, s * 2 - 1));
  return half * 0.5 + push * 0.5;
}

/* ───────────────────────────── screen layout ───────────────────────────── */

/** Top view of the foliot: a bar through the verge with a weight at ±r. */
export function foliotScene(theta: number, r: number, W: number) {
  const H = 170;
  const cx = W < 420 ? W / 2 : W * 0.3;
  const cy = 84;
  const L = Math.min(cx - 16, 120); // bar half-length in px
  const end = (s: 1 | -1) => ({ x: cx + s * L * Math.cos(theta), y: cy - s * L * Math.sin(theta) });
  const wpos = (s: 1 | -1) => ({ x: cx + s * r * L * Math.cos(theta), y: cy - s * r * L * Math.sin(theta) });
  return { H, cx, cy, L, ends: [end(1), end(-1)], weights: [wpos(1), wpos(-1)] };
}
