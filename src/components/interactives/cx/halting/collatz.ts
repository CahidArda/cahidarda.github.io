/** Collatz helpers (pure, unit-checkable with node --experimental-strip-types). */

/** The full sequence from n down to 1 (inclusive). */
export function collatz(n: number): number[] {
  const out = [n];
  let x = n;
  while (x !== 1) {
    x = x % 2 === 0 ? x / 2 : 3 * x + 1;
    out.push(x);
  }
  return out;
}

/** Steps to reach 1 for every start 1..max (index 0 unused). */
export function stoppingTimes(max: number): number[] {
  const st = new Array<number>(max + 1).fill(0);
  for (let n = 2; n <= max; n++) st[n] = collatz(n).length - 1;
  return st;
}

/** About `count` round tick values from 0 up to at most `max`. */
export function niceTicks(max: number, count = 3): number[] {
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const f = raw / mag;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}
