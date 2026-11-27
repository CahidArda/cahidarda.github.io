/**
 * Models for "What a Computer Can Just Do": a road map with Dijkstra's algorithm, the number of
 * routes a brute-force search would have to compare, and two sorting algorithms recorded as a
 * list of states (one per comparison) so a figure can play them back on a clock.
 */
/** Seeded PRNG (same as gt/shared's), inlined so this module runs under plain node. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ─────────────────────────── the road map ─────────────────────────── */

export type MapData = {
  k: number; // crossings per side
  pts: { x: number; y: number }[]; // in [0, 1]²
  edges: [number, number][];
  adj: number[][]; // neighbour lists
  dist: number[]; // from A (Infinity if never reached before B was settled)
  parent: number[];
  settled: boolean[]; // settled by the time B is settled
  dB: number; // distance to B
  path: number[]; // A ... B, along roads
  checks: number; // roads looked at: every road of every settled crossing
};

/**
 * A k × k grid of crossings, nudged by a seeded jitter so it looks like a town, with a road
 * between grid neighbours. Road length = straight-line distance. A is top-left, B bottom-right.
 */
export function makeMap(k: number): MapData {
  const rnd = mulberry32(1000 + k);
  const pts: { x: number; y: number }[] = [];
  const s = 1 / (k - 1);
  for (let r = 0; r < k; r++)
    for (let c = 0; c < k; c++) {
      const edge = r === 0 || c === 0 || r === k - 1 || c === k - 1;
      const j = edge ? 0.12 : 0.28;
      pts.push({
        x: Math.min(
          1,
          Math.max(0, c * s + (rnd() - 0.5) * j * s * (c === 0 || c === k - 1 ? 0 : 1)),
        ),
        y: Math.min(
          1,
          Math.max(0, r * s + (rnd() - 0.5) * j * s * (r === 0 || r === k - 1 ? 0 : 1)),
        ),
      });
    }
  const edges: [number, number][] = [];
  const adj: number[][] = pts.map(() => []);
  const id = (r: number, c: number) => r * k + c;
  for (let r = 0; r < k; r++)
    for (let c = 0; c < k; c++) {
      if (c + 1 < k) edges.push([id(r, c), id(r, c + 1)]);
      if (r + 1 < k) edges.push([id(r, c), id(r + 1, c)]);
    }
  for (const [u, v] of edges) {
    adj[u].push(v);
    adj[v].push(u);
  }
  const len = (u: number, v: number) => Math.hypot(pts[u].x - pts[v].x, pts[u].y - pts[v].y);

  // Dijkstra from A, stopping as soon as B is settled
  const n = k * k;
  const A = 0;
  const B = n - 1;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(-1);
  const settled = new Array(n).fill(false);
  dist[A] = 0;
  let checks = 0;
  for (;;) {
    let u = -1;
    for (let i = 0; i < n; i++) if (!settled[i] && (u < 0 || dist[i] < dist[u])) u = i;
    if (u < 0 || dist[u] === Infinity) break;
    settled[u] = true;
    if (u === B) break;
    for (const v of adj[u]) {
      checks++;
      const d = dist[u] + len(u, v);
      if (d < dist[v]) {
        dist[v] = d;
        parent[v] = u;
      }
    }
  }
  const path: number[] = [];
  for (let v = B; v >= 0; v = parent[v]) path.unshift(v);
  return { k, pts, edges, adj, dist, parent, settled, dB: dist[B], path, checks };
}

/**
 * Self-avoiding routes between opposite corners of a k × k grid of crossings: OEIS A007764
 * (https://oeis.org/A007764), checked against a brute-force count for k <= 6 in the tests.
 */
export const ROUTES: Record<number, bigint> = {
  2: 2n,
  3: 12n,
  4: 184n,
  5: 8512n,
  6: 1262816n,
  7: 575780564n,
  8: 789360053252n,
  9: 3266598486981642n,
  10: 41044208702632496804n,
};

/** Brute force: count simple paths from corner to corner (only for small k). */
export function countRoutes(k: number): number {
  const n = k * k;
  const seen = new Array(n).fill(false);
  let count = 0;
  const go = (u: number) => {
    if (u === n - 1) {
      count++;
      return;
    }
    seen[u] = true;
    const r = Math.floor(u / k);
    const c = u % k;
    if (c + 1 < k && !seen[u + 1]) go(u + 1);
    if (c > 0 && !seen[u - 1]) go(u - 1);
    if (r + 1 < k && !seen[u + k]) go(u + k);
    if (r > 0 && !seen[u - k]) go(u - k);
    seen[u] = false;
  };
  go(0);
  return count;
}

/* ─────────────────────────── sorting ─────────────────────────── */

export type Run = {
  /** frames[c] = order of item ids after c comparisons (frames[0] is the start). */
  frames: number[][];
  /** after c comparisons, how many leading positions are final (selection sort) */
  done: number[];
  /** selection sort only: scan[c] = slot compared at comparison c and the id of the smallest so far */
  scan?: { j: number; min: number }[];
};

export function shuffled(n: number, seed: number): number[] {
  const r = mulberry32(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Selection sort: scan for the smallest, swap it to the front. n(n-1)/2 comparisons. */
export function selectionSort(start: number[]): Run {
  const a = start.slice();
  const frames = [a.slice()];
  const done = [0];
  const n = a.length;
  const scan = [{ j: 0, min: a[0] }];
  for (let i = 0; i < n - 1; i++) {
    let m = i;
    for (let j = i + 1; j < n; j++) {
      if (a[j] < a[m]) m = j;
      scan.push({ j, min: a[m] });
      if (j === n - 1) [a[i], a[m]] = [a[m], a[i]]; // swap lands with the pass's last comparison
      frames.push(a.slice());
      done.push(j === n - 1 ? i + 1 : i);
    }
  }
  done[done.length - 1] = n;
  return { frames, done, scan };
}

/**
 * Bottom-up merge sort, drawn in place: while merging two neighbouring runs, each comparison
 * moves the smaller head to the end of the merged prefix (the items in between shift right),
 * so every frame is a permutation of the items. A run that is left over is already in place.
 */
export function mergeSort(start: number[]): Run {
  const a = start.slice();
  const frames = [a.slice()];
  const n = a.length;
  for (let w = 1; w < n; w *= 2) {
    for (let lo = 0; lo + w < n; lo += 2 * w) {
      let i = lo; // head of the left run (moves right as the prefix grows)
      let mid = lo + w; // head of the right run
      const hi = Math.min(lo + 2 * w, n);
      while (i < mid && mid < hi) {
        if (a[mid] < a[i]) {
          const v = a[mid];
          a.splice(mid, 1);
          a.splice(i, 0, v);
          mid++;
        }
        i++;
        frames.push(a.slice());
      }
    }
  }
  return { frames, done: frames.map((_, c) => (c === frames.length - 1 ? n : 0)) };
}
