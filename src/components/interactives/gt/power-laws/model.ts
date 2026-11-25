/**
 * The copying model of Easley & Kleinberg, Networks, Crowds, and Markets, ch. 18.3.
 * Pages arrive one at a time; each new page j makes one link to an earlier page:
 *   with probability p it picks an earlier page uniformly at random;
 *   otherwise it picks a random earlier page k and links to whatever k links to
 *   (equivalently: it links to an earlier page with probability proportional to its in-links).
 * Predicted: the fraction of pages with k in-links falls like k^-(1 + 1/(1-p)).
 */

/** Deterministic PRNG (same as shared.tsx; duplicated so this file runs under plain node). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Run the copying model for n pages. Returns target[j] = the page j links to (-1 for page 0)
 * and indeg[i]. Pass `rand` to share one stream.
 */
export function copyingModel(n: number, p: number, seed: number) {
  const rand = rng(seed);
  const target = new Int32Array(n).fill(-1);
  const indeg = new Int32Array(n);
  for (let j = 1; j < n; j++) {
    let t: number;
    if (j === 1 || rand() < p) {
      t = Math.floor(rand() * j);
    } else {
      // a random earlier page that has a link (pages 1..j-1), copy its target
      const k = 1 + Math.floor(rand() * (j - 1));
      t = target[k];
    }
    target[j] = t;
    indeg[t]++;
  }
  return { target, indeg };
}

/** Predicted exponent of the in-link distribution (density, not cumulative). */
export const predictedAlpha = (p: number) => 1 + 1 / (1 - p);

/**
 * Log-binned density of a list of positive values: bins of width x factor `ratio`.
 * Returns points {x (geometric centre), y (fraction per unit of x)}.
 */
export function logBinned(values: ArrayLike<number>, xmin: number, ratio = 1.5) {
  let max = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
    n++;
  }
  const out: { x: number; y: number; count: number }[] = [];
  // integer data: bins [lo, hi) on integers
  let lo = xmin;
  while (lo <= max) {
    const hi = Math.max(lo + 1, Math.ceil(lo * ratio));
    let c = 0;
    for (let i = 0; i < values.length; i++) if (values[i] >= lo && values[i] < hi) c++;
    if (c > 0) out.push({ x: Math.sqrt(lo * (hi - 1)), y: c / n / (hi - lo), count: c });
    lo = hi;
  }
  return out;
}

/** Least-squares slope of log y on log x. */
export function fitSlope(pts: { x: number; y: number }[]) {
  const xs = pts.map((q) => Math.log10(q.x));
  const ys = pts.map((q) => Math.log10(q.y));
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const my = ys.reduce((a, b) => a + b, 0) / ys.length;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

/** Fit the power-law exponent on the clean middle of the tail (bins with enough counts). */
export function fitAlpha(indeg: ArrayLike<number>) {
  const pts = logBinned(indeg, 1, 1.6).filter((q) => q.count >= 8 && q.x >= 2);
  const { slope, intercept } = fitSlope(pts);
  return { alpha: -slope, intercept, pts };
}

/** Standard normal sample (Box-Muller). */
export function gauss(rand: () => number) {
  const u = Math.max(1e-12, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Share of the total held by the top `frac` of items. sorted must be descending. */
export function topShare(sortedDesc: ArrayLike<number>, frac: number) {
  let total = 0;
  for (let i = 0; i < sortedDesc.length; i++) total += sortedDesc[i];
  const m = Math.max(1, Math.round(sortedDesc.length * frac));
  let top = 0;
  for (let i = 0; i < m; i++) top += sortedDesc[i];
  return top / total;
}

/**
 * Same copying rule, but each new page makes m links (each one: random page with probability
 * p, else copy the target of a random existing link). Returns in-links per page.
 */
export function copyingModelMulti(n: number, m: number, p: number, seed: number) {
  const rand = rng(seed);
  const indeg = new Array<number>(n).fill(0);
  const targets: number[] = [];
  for (let j = 1; j < n; j++) {
    for (let l = 0; l < m; l++) {
      const t =
        targets.length === 0 || rand() < p
          ? Math.floor(rand() * j)
          : targets[Math.floor(rand() * targets.length)];
      indeg[t]++;
      targets.push(t);
    }
  }
  return indeg;
}
