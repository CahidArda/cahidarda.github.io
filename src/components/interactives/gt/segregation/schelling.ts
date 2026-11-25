/**
 * Schelling's model of segregation (Schelling 1971; Easley & Kleinberg, NCM 4.5).
 * Two kinds of residents on a grid with some empty lots. A resident looks at its (up to 8)
 * occupied neighbouring cells and is satisfied when the share that are its own kind is at
 * least t. Each sweep, every unsatisfied resident moves to a random empty lot.
 * Cells: 0 = empty, 1 = kind A, 2 = kind B. No wrap-around at the edges.
 */
export type Grid = Uint8Array;

export function makeGrid(n: number, rnd: () => number, emptyFrac = 0.1): Grid {
  const g = new Uint8Array(n * n);
  const empty = Math.round(n * n * emptyFrac);
  const agents = n * n - empty;
  for (let i = 0; i < n * n; i++) g[i] = i < agents ? (i % 2 === 0 ? 1 : 2) : 0;
  for (let i = g.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = g[i];
    g[i] = g[j];
    g[j] = tmp;
  }
  return g;
}

/** [alike, occupied] neighbour counts for the resident at cell i (or as if kind k lived there). */
export function counts(g: Grid, n: number, i: number, k = g[i]): [number, number] {
  const r = (i / n) | 0;
  const c = i % n;
  let alike = 0;
  let occ = 0;
  for (let dr = -1; dr <= 1; dr++) {
    const rr = r + dr;
    if (rr < 0 || rr >= n) continue;
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const cc = c + dc;
      if (cc < 0 || cc >= n) continue;
      const v = g[rr * n + cc];
      if (v === 0) continue;
      occ++;
      if (v === k) alike++;
    }
  }
  return [alike, occ];
}

/** Satisfied if alike / occupied >= t (a resident with no neighbours is satisfied). */
export function happy(g: Grid, n: number, i: number, t: number, k = g[i]): boolean {
  const [a, o] = counts(g, n, i, k);
  return o === 0 || a / o >= t - 1e-9;
}

/** One sweep: every unsatisfied resident, in random order, moves to a random empty lot. */
export function sweep(g: Grid, n: number, t: number, rnd: () => number): number {
  const movers: number[] = [];
  const empties: number[] = [];
  for (let i = 0; i < g.length; i++) {
    if (g[i] === 0) empties.push(i);
    else if (!happy(g, n, i, t)) movers.push(i);
  }
  for (let i = movers.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = movers[i];
    movers[i] = movers[j];
    movers[j] = tmp;
  }
  let moved = 0;
  for (const from of movers) {
    if (g[from] === 0 || happy(g, n, from, t)) continue;
    const e = Math.floor(rnd() * empties.length);
    const to = empties[e];
    g[to] = g[from];
    g[from] = 0;
    empties[e] = from;
    moved++;
  }
  return moved;
}

/** Average share of a resident's occupied neighbours that are its own kind, and share unsatisfied. */
export function stats(g: Grid, n: number, t: number) {
  let sim = 0;
  let cnt = 0;
  let unhappy = 0;
  let agents = 0;
  for (let i = 0; i < g.length; i++) {
    if (g[i] === 0) continue;
    agents++;
    const [a, o] = counts(g, n, i);
    if (o > 0) {
      sim += a / o;
      cnt++;
    }
    if (o > 0 && a / o < t - 1e-9) unhappy++;
  }
  return { similarity: cnt ? sim / cnt : 0, unhappy: agents ? unhappy / agents : 0 };
}
