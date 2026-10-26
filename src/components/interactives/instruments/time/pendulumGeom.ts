/**
 * Episode 5, the pendulum clock: numbers for PendulumPlate.tsx (swept by time.sweep.ts).
 * T = 2π √(L/g) for small swings. The dial is geared for a seconds pendulum (T = 2 s: one beat
 * per second, the 30-tooth escape wheel once a minute), so a pendulum of another length makes
 * the clock gain or lose 86 400 · (2/T − 1) seconds a day. A rod longer by ΔL loses about
 * 86 400 · ΔL / (2L) a day.
 */
export const G = 9.80665;
export const period = (L: number) => 2 * Math.PI * Math.sqrt(L / G);
export const SECONDS_L = G / (Math.PI * Math.PI); // 0.9936 m
export const L_MIN = 0.1;
export const L_MAX = 4;
export const L_STEP = 0.001;
/** The length slider snaps to the seconds pendulum within this (m). */
export const SNAP = 0.006;
export const snapL = (v: number) => (Math.abs(v - SECONDS_L) < SNAP ? SECONDS_L : v);
/** "Beats seconds" within 1 mm. */
export const isSeconds = (L: number) => Math.abs(L - SECONDS_L) < 0.001;
export const DL_MAX = 2; // mm the rod can stretch on the heat step
/** Seconds gained a day (negative = lost) for length L plus a stretch of dLmm millimetres. */
export const errPerDay = (L: number, dLmm = 0) => 86400 * (2 / period(L + dLmm / 1000) - 1);
/** Drawn rod length in 3D/2D world units (wheel tip radius = 1), compressed so 0.1-4 m fit. */
export const rodWorld = (L: number) => 3.0 + 2.6 * Math.sqrt((L - L_MIN) / (L_MAX - L_MIN));
