/**
 * Numbers behind the factoring figures. Everything here is exact or a stated formula:
 *  - multiplying two numbers of a and b digits by hand takes a·b single-digit products;
 *  - trial division of a d-digit number with two equal-size prime factors tries 2, 3, 5, 7, ...
 *    (2 and then the odd numbers) up to the smaller factor, about √N / 2 ≈ 10^(d/2) / 2 tries;
 *  - times assume a billion steps per second.
 */

export const SPEED = 1e9; // steps per second
export const YEAR = 365.25 * 24 * 3600; // seconds
export const UNIVERSE_YEARS = 13.8e9; // age of the universe (Planck 2018: 13.8 billion years)

/** log10 of single-digit products to multiply two primes making a d-digit number. */
export const logMul = (d: number) => Math.log10(Math.ceil(d / 2) * Math.floor(d / 2) || 1);
/** log10 of trial divisions to find a factor of a d-digit number with equal-size factors. */
export const logTrial = (d: number) => d / 2 - Math.log10(2);

/** RSA-250 (829 bits): factored 28 Feb 2020 in about 2700 core-years (Boudot et al.). */
export const RSA250 = { digits: 250, coreYears: 2700 };
/** RSA-2048, the usual key size today: 2048 bits = 617 decimal digits. */
export const RSA2048_DIGITS = Math.ceil(2048 * Math.log10(2));

function isPrime(n: number): boolean {
  if (n < 2) return false;
  if (n % 2 === 0) return n === 2;
  for (let f = 3; f * f <= n; f += 2) if (n % f === 0) return false;
  return true;
}
/** Deterministic PRNG (mulberry32), so the examples are the same on every load. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** A random prime with exactly k digits, at least `lo` times 10^k. */
function randomPrime(k: number, lo: number, r: () => number): number {
  const base = 10 ** k;
  let n = Math.floor(base * (lo + (0.97 - lo) * r())) | 1;
  if (k === 1) n = [3, 5, 7][Math.floor(r() * 3)];
  while (!isPrime(n)) n += 2;
  return n;
}

/**
 * A real example for a d-digit product (2 <= d <= 16): two primes, the product, and how many
 * tries trial division needs to reach the smaller one (2, then every odd number up to it).
 */
export function example(d: number) {
  const r = rng(1000 + d);
  const k = Math.floor(d / 2);
  for (;;) {
    const big = randomPrime(d - k, 0.55, r); // ceil(d/2) digits
    const small = randomPrime(k, 0.2, r); // floor(d/2) digits
    const product = BigInt(big) * BigInt(small);
    if (big !== small && String(product).length === d && small < big)
      return { big, small, product, tries: (small + 1) / 2, mults: (d - k) * k };
  }
}

/** "x seconds" in words, from log10(seconds). Two significant figures, whole units. */
export function fmtTime(logSec: number): { v: string; unit: string } {
  const s = 10 ** logSec;
  const sig = (x: number) => {
    if (x >= 100) return Math.round(x).toLocaleString('en-US');
    if (x >= 10) return String(Math.round(x));
    return (Math.round(x * 10) / 10).toString();
  };
  if (s < 1e-6) return { v: sig(s * 1e9), unit: 'nanoseconds' };
  if (s < 1e-3) return { v: sig(s * 1e6), unit: 'microseconds' };
  if (s < 1) return { v: sig(s * 1e3), unit: 'milliseconds' };
  if (s < 60) return { v: sig(s), unit: 'seconds' };
  if (s < 3600) return { v: sig(s / 60), unit: 'minutes' };
  if (s < 86400) return { v: sig(s / 3600), unit: 'hours' };
  if (s < YEAR) return { v: sig(s / 86400), unit: 'days' };
  const ly = logSec - Math.log10(YEAR);
  if (ly < 6) return { v: sig(10 ** ly), unit: 'years' };
  if (ly < 9) return { v: sig(10 ** (ly - 6)), unit: 'million years' };
  if (ly < 12) return { v: sig(10 ** (ly - 9)), unit: 'billion years' };
  return { v: `10^${Math.floor(ly)}`, unit: 'years' };
}
