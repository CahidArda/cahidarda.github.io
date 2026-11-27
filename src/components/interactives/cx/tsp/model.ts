/**
 * Travelling-salesman logic for CX-06: tour length, nearest neighbour, best-improvement
 * 2-opt (one move at a time, so a widget can animate each), and an exact optimum by
 * Held-Karp dynamic programming (fine up to ~13 cities).
 */
export type Pt = { x: number; y: number };

export const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

export function tourLength(pts: Pt[], order: number[]): number {
  let s = 0;
  for (let i = 0; i < order.length; i++)
    s += dist(pts[order[i]], pts[order[(i + 1) % order.length]]);
  return s;
}

/** Start at city 0, always go to the closest city not yet visited. */
export function nearestNeighbour(pts: Pt[], start = 0): number[] {
  const n = pts.length;
  const seen = new Array(n).fill(false);
  const order = [start];
  seen[start] = true;
  for (let k = 1; k < n; k++) {
    const cur = pts[order[order.length - 1]];
    let best = -1;
    let bd = Infinity;
    for (let j = 0; j < n; j++) {
      if (seen[j]) continue;
      const d = dist(cur, pts[j]);
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    seen[best] = true;
    order.push(best);
  }
  return order;
}

/**
 * One 2-opt move: drop roads (a,b) and (c,d), add (a,c) and (b,d), reverse b..c.
 * Positions i < j in the order: a = order[i], b = order[i+1], c = order[j], d = order[j+1].
 */
export type Move = { a: number; b: number; c: number; d: number; gain: number; next: number[] };

/** The single best improving 2-opt move, or null if the tour is 2-optimal. */
export function bestTwoOpt(pts: Pt[], order: number[]): Move | null {
  const n = order.length;
  let best: Move | null = null;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue; // the two roads share a city
      const a = order[i];
      const b = order[i + 1];
      const c = order[j];
      const d = order[(j + 1) % n];
      const gain =
        dist(pts[a], pts[b]) + dist(pts[c], pts[d]) - dist(pts[a], pts[c]) - dist(pts[b], pts[d]);
      if (gain > 1e-9 && (!best || gain > best.gain)) {
        const next = [
          ...order.slice(0, i + 1),
          ...order.slice(i + 1, j + 1).reverse(),
          ...order.slice(j + 1),
        ];
        best = { a, b, c, d, gain, next };
      }
    }
  }
  return best;
}

/** Every move best-improvement 2-opt makes from `order` until no move helps. */
export function twoOptRun(pts: Pt[], order: number[], cap = 60): Move[] {
  const moves: Move[] = [];
  let cur = order;
  for (let k = 0; k < cap; k++) {
    const m = bestTwoOpt(pts, cur);
    if (!m) break;
    moves.push(m);
    cur = m.next;
  }
  return moves;
}

/** Exact shortest tour by Held-Karp: O(n² 2ⁿ). */
export function optimalTour(pts: Pt[]): { order: number[]; length: number } {
  const n = pts.length;
  if (n < 3)
    return {
      order: pts.map((_, i) => i),
      length: tourLength(
        pts,
        pts.map((_, i) => i),
      ),
    };
  const m = n - 1; // city 0 is the fixed start; subsets over cities 1..n-1
  const N = 1 << m;
  const D: number[][] = pts.map((p) => pts.map((q) => dist(p, q)));
  const dp = new Float64Array(N * m).fill(Infinity);
  const par = new Int8Array(N * m).fill(-1);
  for (let k = 0; k < m; k++) dp[(1 << k) * m + k] = D[0][k + 1];
  for (let S = 1; S < N; S++) {
    for (let k = 0; k < m; k++) {
      if (!(S & (1 << k))) continue;
      const v = dp[S * m + k];
      if (v === Infinity) continue;
      for (let j = 0; j < m; j++) {
        if (S & (1 << j)) continue;
        const T = S | (1 << j);
        const w = v + D[k + 1][j + 1];
        if (w < dp[T * m + j]) {
          dp[T * m + j] = w;
          par[T * m + j] = k;
        }
      }
    }
  }
  const full = N - 1;
  let best = Infinity;
  let last = -1;
  for (let k = 0; k < m; k++) {
    const w = dp[full * m + k] + D[k + 1][0];
    if (w < best) {
      best = w;
      last = k;
    }
  }
  const order: number[] = [];
  let S = full;
  let k = last;
  while (k >= 0) {
    order.push(k + 1);
    const p = par[S * m + k];
    S &= ~(1 << k);
    k = p;
  }
  order.push(0);
  order.reverse();
  return { order, length: best };
}

/** Number of distinct round trips through n cities: (n-1)!/2. */
export function tourCount(n: number): number {
  if (n < 3) return 1;
  let f = 1;
  for (let k = 2; k <= n - 1; k++) f *= k;
  return f / 2;
}
