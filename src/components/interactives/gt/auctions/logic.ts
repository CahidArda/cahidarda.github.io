/**
 * Pure logic for the auctions page (no React). Private values, sealed bids.
 */

export type Format = 'second' | 'first';

/** Your payoff given your value v, your bid b and the highest rival bid r (ties go to the rival). */
export function payoff(format: Format, v: number, b: number, r: number): number {
  if (b <= r) return 0;
  return format === 'second' ? v - r : v - b;
}

/** Symmetric equilibrium first-price bid, n bidders, values uniform on [0, 1]. */
export const shade = (n: number, v: number) => ((n - 1) / n) * v;

/** Expected seller revenue in either format (revenue equivalence), uniform values. */
export const expectedRevenue = (n: number) => (n - 1) / (n + 1);

/**
 * Run `count` auctions with n bidders (values from rand). Returns the seller's revenue in
 * each format: second price = second-highest value (everyone bids truthfully), first price =
 * the highest equilibrium bid, shade(n, highest value).
 */
export function simulate(n: number, count: number, rand: () => number) {
  const second: number[] = [];
  const first: number[] = [];
  for (let k = 0; k < count; k++) {
    let hi = -1;
    let hi2 = -1;
    for (let i = 0; i < n; i++) {
      const v = rand();
      if (v > hi) {
        hi2 = hi;
        hi = v;
      } else if (v > hi2) hi2 = v;
    }
    second.push(hi2);
    first.push(shade(n, hi));
  }
  return { second, first };
}

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export function histogram(xs: number[], bins: number): number[] {
  const h = new Array(bins).fill(0);
  for (const x of xs) h[Math.min(bins - 1, Math.floor(x * bins))]++;
  return h;
}
