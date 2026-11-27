/**
 * Subset sum for CX-03.2 (the search tree). With n numbers there are 2^n ways to pick a
 * subset: a binary tree n levels deep, one leaf per subset. Level d decides number d
 * (left = skip, right = take). Each target below has exactly one subset that hits it, placed
 * about 70% of the way through a left-to-right search (checked in node).
 */
export const NUMS = [12, 7, 23, 18, 4, 31, 15, 26, 9, 40];
export const TARGETS: Record<number, number> = {
  3: 35,
  4: 53,
  5: 53,
  6: 84,
  7: 99,
  8: 96,
  9: 125,
  10: 165,
};
export const N_MIN = 3;
export const N_MAX = 10;

/** Leaf index (bits MSB first: 1 = take) of the unique subset that hits the target. */
export function solutionLeaf(n: number): number {
  const xs = NUMS.slice(0, n);
  const t = TARGETS[n];
  let found = -1;
  for (let m = 0; m < 1 << n; m++) {
    let s = 0;
    for (let i = 0; i < n; i++) if ((m >> (n - 1 - i)) & 1) s += xs[i];
    if (s === t) {
      if (found >= 0) throw new Error(`two subsets hit ${t} for n=${n}`);
      found = m;
    }
  }
  if (found < 0) throw new Error(`no subset hits ${t}`);
  return found;
}

export type Pt = { x: number; y: number };
export type Geom = { left: number; right: number; top: number; bottom: number };

/** Position of node j (0-based, left to right) at depth d. */
export function nodeAt(g: Geom, n: number, d: number, j: number): Pt {
  const w = (g.right - g.left) / (1 << d);
  return { x: g.left + (j + 0.5) * w, y: g.top + (d / n) * (g.bottom - g.top) };
}

/**
 * Depth-first walk from the root that visits leaves left to right and stops at `stop`.
 * Returns the polyline the searcher's token follows, the cumulative length at every point,
 * and the point index at which each leaf is reached (undefined if the walk stops first).
 */
export function dfsWalk(g: Geom, n: number, stop: number) {
  const pts: Pt[] = [nodeAt(g, n, 0, 0)];
  const leafAt: number[] = [];
  let done = false;
  const rec = (d: number, j: number) => {
    if (done) return;
    if (d === n) {
      leafAt[j] = pts.length - 1;
      if (j === stop) done = true;
      return;
    }
    for (const c of [2 * j, 2 * j + 1]) {
      pts.push(nodeAt(g, n, d + 1, c));
      rec(d + 1, c);
      if (done) return;
      pts.push(nodeAt(g, n, d, j));
    }
  };
  rec(0, 0);
  const cum = [0];
  for (let i = 1; i < pts.length; i++)
    cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  return { pts, cum, leafPt: leafAt };
}

/** The lucky guesser's path: straight down to `leaf`. */
export function guessPath(g: Geom, n: number, leaf: number) {
  const pts: Pt[] = [];
  for (let d = 0; d <= n; d++) pts.push(nodeAt(g, n, d, leaf >> (n - d)));
  const cum = [0];
  for (let i = 1; i < pts.length; i++)
    cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  return { pts, cum };
}

/** Point at length s along a polyline, plus the index of the segment it is on. */
export function along(pts: Pt[], cum: number[], s: number): { p: Pt; i: number } {
  const L = cum[cum.length - 1];
  if (s <= 0) return { p: pts[0], i: 0 };
  if (s >= L) return { p: pts[pts.length - 1], i: pts.length - 1 };
  let lo = 0;
  let hi = cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] <= s) lo = mid;
    else hi = mid;
  }
  const f = (s - cum[lo]) / (cum[hi] - cum[lo]);
  return {
    p: { x: pts[lo].x + (pts[hi].x - pts[lo].x) * f, y: pts[lo].y + (pts[hi].y - pts[lo].y) * f },
    i: lo,
  };
}
