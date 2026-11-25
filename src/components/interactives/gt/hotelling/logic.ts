/**
 * Pure logic for the Hotelling page (no React). Customers / voters live on [0, 1]
 * with cumulative distribution F; each goes to the nearest seller, ties split evenly.
 */

export type Cdf = (x: number) => number;
export const uniformCdf: Cdf = (x) => Math.max(0, Math.min(1, x));

/** Share of the population won by each seller at `pos` (co-located sellers split). */
export function shares(pos: number[], F: Cdf = uniformCdf): number[] {
  const uniq = Array.from(new Set(pos)).sort((a, b) => a - b);
  const out = pos.map(() => 0);
  uniq.forEach((u, i) => {
    const lo = i === 0 ? 0 : (uniq[i - 1] + u) / 2;
    const hi = i === uniq.length - 1 ? 1 : (u + uniq[i + 1]) / 2;
    const mass = F(hi) - F(lo);
    const who = pos.map((p, k) => (p === u ? k : -1)).filter((k) => k >= 0);
    who.forEach((k) => (out[k] = mass / who.length));
  });
  return out;
}

/** Mean distance from each customer at points xs to the nearest seller. */
export function meanWalk(pos: number[], xs: number[]): number {
  let s = 0;
  for (const x of xs) s += Math.min(...pos.map((p) => Math.abs(p - x)));
  return s / xs.length;
}

const snap = (x: number, step: number) => Math.round(x / step) * step;

/**
 * One turn of local improvement: seller `who` tries a step left or right (size `step`)
 * and keeps whichever gives the biggest share (stays put on ties).
 */
export function nudge(pos: number[], who: number, step: number, F: Cdf = uniformCdf): number[] {
  const cur = shares(pos, F)[who];
  let best = pos[who];
  let bestShare = cur;
  for (const d of [-step, step]) {
    const x = snap(pos[who] + d, step);
    if (x < 0 || x > 1 + 1e-9) continue;
    const trial = pos.slice();
    trial[who] = Math.min(1, x);
    const s = shares(trial, F)[who];
    if (s > bestShare + 1e-9) {
      bestShare = s;
      best = trial[who];
    }
  }
  const next = pos.slice();
  next[who] = best;
  return next;
}

/** Global best response on a grid of positions 0, step, 2*step, ..., 1. */
export function bestResponse(
  pos: number[],
  who: number,
  step: number,
  F: Cdf = uniformCdf,
): number {
  let best = pos[who];
  let bestShare = shares(pos, F)[who];
  const n = Math.round(1 / step);
  for (let i = 0; i <= n; i++) {
    const trial = pos.slice();
    trial[who] = snap(i * step, step);
    const s = shares(trial, F)[who];
    if (s > bestShare + 1e-9) {
      bestShare = s;
      best = trial[who];
    }
  }
  return best;
}

/** Build a CDF from density samples on an even grid over [0, 1]. */
export function cdfFromDensity(dens: number[]): { F: Cdf; median: number; mean: number } {
  const n = dens.length;
  const total = dens.reduce((a, b) => a + b, 0);
  const cum = [0];
  for (let i = 0; i < n; i++) cum.push(cum[i] + dens[i] / total);
  const F: Cdf = (x) => {
    const t = Math.max(0, Math.min(1, x)) * n;
    const i = Math.min(n - 1, Math.floor(t));
    return cum[i] + (cum[i + 1] - cum[i]) * (t - i);
  };
  let median = 0.5;
  for (let i = 0; i < n; i++) {
    if (cum[i + 1] >= 0.5) {
      median = (i + (0.5 - cum[i]) / (cum[i + 1] - cum[i])) / n;
      break;
    }
  }
  let mean = 0;
  for (let i = 0; i < n; i++) mean += ((i + 0.5) / n) * (dens[i] / total);
  return { F, median, mean };
}

export type Preset = 'even' | 'lopsided' | 'polarized';
const gauss = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2);

/** Voter density presets, sampled at n grid cells. */
export function presetDensity(p: Preset, n = 200): number[] {
  return Array.from({ length: n }, (_, i) => {
    const x = (i + 0.5) / n;
    if (p === 'even') return 1;
    if (p === 'lopsided') return 1.0 * gauss(x, 0.3, 0.09) + 0.35 * gauss(x, 0.72, 0.12);
    return gauss(x, 0.2, 0.08) + gauss(x, 0.8, 0.08) + 0.04;
  });
}
