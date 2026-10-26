/**
 * Episode 6, the pendulum at sea and Harrison's answer: the numbers of SeaClockPlate.tsx (swept
 * by time.sweep.ts). All rates in seconds GAINED per day (negative = lost).
 *
 * The voyage: regulated in Portsmouth (50.8° N) on land, sailed to Jamaica (18° N) in the Longitude
 * Act's trial time, "about six weeks" (Wikipedia, Longitude rewards): 42 days. Latitude falls
 * linearly through the voyage; the sea temperature rises with it (TEMP0 at Portsmouth to TEMP1 at
 * 18° N, an assumption stated in the caption).
 *
 * Pendulum, T = 2π √(L/g_eff):
 *  - heave: the ship's vertical acceleration a(t) = A ω² sin ωt adds to g. Averaged over a wave,
 *    the rate (∝ √g_eff) loses ⟨a²⟩/(8g²) = a_max²/(16 g²) of its value (second order in a/g).
 *    Real seas are irregular, so each day also scatters: a seeded ±50% of that loss
 *    (illustrative; the point is that it cannot be predicted and so cannot be corrected).
 *  - gravity: g(φ) from the WGS-84 formula; rate × √(g(φ)/g(φ₀)). Paris→Cayenne gives 128 s/day,
 *    Richer saw 148 s/day (2 min 28 s, Newton's Principia, Wikipedia Jean Richer).
 *  - heat: a steel rod (α = 11.5 × 10⁻⁶ /°C) lengthens: −86 400 · αΔT/2 per day.
 * Balance and spiral spring, T = 2π √(I/κ): no g, so heave and latitude terms are zero; the
 * residual is H4's own average on the Jamaica trial: 5 s slow in 81 days 5 h after the rate
 * allowance (Wikipedia, John Harrison), −0.0616 s/day.
 * Clock error → longitude: 4 min of time per degree (RMG), so 1 s = 0.25′ = 463 m · cos φ.
 */
export const G0 = 9.80665;
export const PORTSMOUTH = 50.8;
export const JAMAICA = 18;
export const VOYAGE_DAYS = 42;
export const TEMP0 = 12;
export const TEMP1 = 27;
export const ALPHA_STEEL = 11.5e-6;
export const WAVE_PERIOD = 8; // s
export const HEAVE_MAX = 3; // m, amplitude
export const H4_RATE = -5 / (81 + 5 / 24);
export const H4_DAYS = 81 + 5 / 24;
/** The Longitude Act's three prizes: longitude error allowed (arcminutes). */
export const PRIZES = [
  { pounds: 20000, arcmin: 30 },
  { pounds: 15000, arcmin: 40 },
  { pounds: 10000, arcmin: 60 },
];
export const M_PER_S_EQ = 463.8; // 40 075 km / 86 400 s

const R = Math.PI / 180;
export const gravity = (phi: number) => {
  const s = Math.sin(phi * R);
  const s2 = Math.sin(2 * phi * R);
  return 9.780327 * (1 + 0.0053024 * s * s - 0.0000058 * s2 * s2);
};
export const heaveAccel = (A: number) => A * ((2 * Math.PI) / WAVE_PERIOD) ** 2;
/** Mean loss from heave (s/day, negative). */
export const heaveRate = (A: number) => -86400 * heaveAccel(A) ** 2 / (16 * G0 * G0);
export const gravityRate = (phi: number, phi0 = PORTSMOUTH) => 86400 * (Math.sqrt(gravity(phi) / gravity(phi0)) - 1);
export const heatRate = (dT: number) => (-86400 * ALPHA_STEEL * dT) / 2;

/** Seeded PRNG (mulberry32), so the "irregular sea" is the same on every render. */
function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const SCATTER = (() => {
  const r = mulberry32(1714);
  return Array.from({ length: 120 }, () => 2 * r() - 1);
})();

export type Clock = 'pendulum' | 'balance';
export type Voyage = {
  clock: Clock;
  heave: number;
  /** Southern end of the voyage: the ship goes from Portsmouth to this latitude. */
  dest: number;
  /** Whether the temperature changes with the latitude (else it stays at TEMP0). */
  heat: boolean;
};
export const latOnDay = (v: Voyage, day: number) =>
  PORTSMOUTH + (v.dest - PORTSMOUTH) * Math.min(1, Math.max(0, day / VOYAGE_DAYS));
export const tempAt = (v: Voyage, phi: number) =>
  v.heat ? TEMP0 + ((TEMP1 - TEMP0) * (PORTSMOUTH - phi)) / (PORTSMOUTH - JAMAICA) : TEMP0;

/** The clock's rate on a given day (s/day) and its parts. */
export function rateOn(v: Voyage, day: number) {
  if (v.clock === 'balance') return { total: H4_RATE, heave: 0, gravity: 0, heat: 0, scatter: 0 };
  const phi = latOnDay(v, day + 0.5);
  const heave = heaveRate(v.heave);
  const scatter = 0.5 * heave * SCATTER[Math.floor(day) % SCATTER.length];
  const gravity = gravityRate(phi);
  const heat = heatRate(tempAt(v, phi) - TEMP0);
  return { total: heave + scatter + gravity + heat, heave, gravity, heat, scatter };
}
/** Clock error (s, + = fast) after `days`, summing day by day (fractional last day). */
export function errorAfter(v: Voyage, days: number) {
  let e = 0;
  const n = Math.floor(days);
  for (let d = 0; d < n; d++) e += rateOn(v, d).total;
  if (days > n) e += rateOn(v, n).total * (days - n);
  return e;
}
export const arcminOf = (errSec: number) => Math.abs(errSec) / 4; // 1 s of time = 15″ = 0.25′
export const kmOf = (errSec: number, phi: number) => (Math.abs(errSec) * M_PER_S_EQ * Math.cos(phi * R)) / 1000;
/** Best prize the error still wins, or null. */
export function prizeFor(errSec: number) {
  const m = arcminOf(errSec);
  return PRIZES.find((p) => m <= p.arcmin) ?? null;
}

/* ───────────────────────────── the chart ───────────────────────────── */

export const DAYS_MAX = 90;
/** Longitude error chart: x = days 0..DAYS_MAX, y = arcminutes on a LOG scale from Y_MIN to
 *  Y_MAX (so H4's 1.25′ and a pendulum's 30° both read); smaller errors sit on the floor,
 *  larger ones are clipped and flagged. */
export const Y_MIN = 0.5;
export const Y_MAX = 3000; // 50°
export function chartFrame(W: number) {
  const H = 200;
  const m = { l: 46, r: 14, t: 14, b: 30 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const x = (d: number) => m.l + (Math.min(DAYS_MAX, Math.max(0, d)) / DAYS_MAX) * iw;
  const l0 = Math.log10(Y_MIN);
  const l1 = Math.log10(Y_MAX);
  const y = (a: number) => m.t + ih - ((Math.log10(Math.min(Y_MAX, Math.max(Y_MIN, a))) - l0) / (l1 - l0)) * ih;
  return { H, m, iw, ih, x, y };
}
export const Y_TICKS: [number, string][] = [
  [1, '1′'],
  [10, '10′'],
  [60, '1°'],
  [600, '10°'],
];
export function errorCurve(v: Voyage, W: number, upTo: number) {
  const f = chartFrame(W);
  const pts: { x: number; y: number; a: number }[] = [];
  const n = Math.max(1, Math.ceil(upTo * 2));
  for (let i = 0; i <= n; i++) {
    const d = (upTo * i) / n;
    const a = arcminOf(errorAfter(v, d));
    pts.push({ x: f.x(d), y: f.y(a), a });
  }
  return pts;
}
