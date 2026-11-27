/**
 * Growth rates for "Fast, Slow, and Never". Everything is in log10 so huge numbers (100! has
 * 158 digits) stay finite and the curves can be drawn from the same function the readouts use.
 * The machine does a billion steps a second.
 */
export const RATE = 1e9; // steps per second
export const YEAR = 365.25 * 86400; // seconds
export const LOG_LIFETIME = Math.log10(80 * YEAR); // 80 years, 9.40
export const LOG_UNIVERSE = Math.log10(13.8e9 * YEAR); // 13.8 billion years, 17.64

/** log10 of Gamma(x) for x > 0 (Lanczos, g = 7). Continuous, so n! draws as a smooth curve. */
export function log10Gamma(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log10(Math.PI / Math.abs(Math.sin(Math.PI * x))) - log10Gamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return (0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)) / Math.LN10;
}

export type CurveId = 'n' | 'nlogn' | 'n2' | 'n3' | 'exp' | 'fact';

export type Curve = {
  id: CurveId;
  label: string; // shown on the chart, next to the curve
  /** log10 of the number of steps for input size n (n >= 1). */
  logSteps: (n: number) => number;
  /** Exact step count, for the polynomial curves (whole-number search in maxN). */
  steps?: (n: number) => number;
  poly: boolean;
};

export const CURVES: Curve[] = [
  { id: 'n', label: 'n', logSteps: (n) => Math.log10(n), steps: (n) => n, poly: true },
  {
    id: 'nlogn',
    label: 'n log n',
    logSteps: (n) => Math.log10(n) + Math.log10(Math.max(1e-9, Math.log2(n))),
    steps: (n) => n * Math.log2(n),
    poly: true,
  },
  { id: 'n2', label: 'n²', logSteps: (n) => 2 * Math.log10(n), steps: (n) => n * n, poly: true },
  {
    id: 'n3',
    label: 'n³',
    logSteps: (n) => 3 * Math.log10(n),
    steps: (n) => n * n * n,
    poly: true,
  },
  { id: 'exp', label: '2ⁿ', logSteps: (n) => n * Math.log10(2), poly: false },
  { id: 'fact', label: 'n!', logSteps: (n) => log10Gamma(n + 1), poly: false },
];

/** Seconds (log10) for a given number of steps (log10). */
export const logSeconds = (logSteps: number) => logSteps - Math.log10(RATE);

const SUP: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
};
export const sup = (n: number) =>
  String(n)
    .split('')
    .map((c) => SUP[c] ?? c)
    .join('');

/** Two significant figures, written plainly ("36", "1.1", "580"). */
export function sig2(v: number): string {
  if (v >= 100) {
    const m = Math.pow(10, Math.floor(Math.log10(v)) - 1);
    return (Math.round(v / m) * m).toLocaleString('en-US');
  }
  if (v >= 10) return String(Math.round(v));
  const r = Math.round(v * 10) / 10;
  return r >= 10 ? '10' : r.toFixed(1).replace(/\.0$/, '');
}

/** A step count as words: "3,600" or "1.2 × 10¹⁸". */
export function fmtSteps(logSteps: number): string {
  if (logSteps < 6) return Math.round(Math.pow(10, logSteps)).toLocaleString('en-US');
  const e = Math.floor(logSteps + 1e-9);
  const m = Math.pow(10, logSteps - e);
  const ms = m.toFixed(1);
  if (ms === '10.0') return `10${sup(e + 1)}`;
  return `${ms} × 10${sup(e)}`;
}

/** A duration from log10(seconds): value + unit, short enough for a readout. */
export function fmtTime(ls: number): { value: string; unit: string } {
  const s = Math.pow(10, ls);
  if (ls < -6) return { value: sig2(s * 1e9), unit: 'ns' };
  if (ls < -3) return { value: sig2(s * 1e6), unit: 'μs' };
  if (ls < 0) return { value: sig2(s * 1e3), unit: 'ms' };
  if (s < 60) return { value: sig2(s), unit: 'seconds' };
  if (s < 3600) return { value: sig2(s / 60), unit: 'minutes' };
  if (s < 86400) return { value: sig2(s / 3600), unit: 'hours' };
  if (s < YEAR) return { value: sig2(s / 86400), unit: 'days' };
  const ly = ls - Math.log10(YEAR);
  if (ly < 6) return { value: sig2(Math.pow(10, ly)), unit: 'years' };
  if (ly < 9) return { value: sig2(Math.pow(10, ly - 6)), unit: 'million years' };
  if (ls < LOG_UNIVERSE) return { value: sig2(Math.pow(10, ly - 9)), unit: 'billion years' };
  const u = ls - LOG_UNIVERSE; // multiples of the age of the universe
  if (u < 6) return { value: sig2(Math.pow(10, u)), unit: 'ages of the universe' };
  let e = Math.floor(u);
  let m = Math.round(Math.pow(10, u - e));
  if (m >= 10) {
    m = 1;
    e += 1;
  }
  return { value: m === 1 ? `10${sup(e)}` : `${m} × 10${sup(e)}`, unit: 'ages of the universe' };
}

/* ───────────── a faster computer: the biggest n that fits in one hour ───────────── */

export const HOUR_STEPS = 3600 * RATE; // 3.6 × 10¹²

/** Largest whole n whose step count fits in `budget` steps. Integer search, exact. */
export function maxN(c: Curve, budget: number): number {
  const fits = (n: number) =>
    c.steps ? c.steps(n) <= budget : c.logSteps(n) <= Math.log10(budget);
  let lo = 1;
  let hi = 2;
  while (fits(hi)) hi *= 2;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (fits(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}
