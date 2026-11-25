/**
 * Watts-Strogatz small world (Watts & Strogatz 1998; Easley & Kleinberg ch. 20).
 * A ring of n nodes, each tied to its k nearest neighbours (k/2 per side). Each lattice edge
 * (i, i+d) gets a fixed random number u and a fixed random new endpoint; at rewiring
 * probability p the edge is rewired iff u < p. Raising p only ever adds rewired edges, so a
 * slider sweeps one consistent history instead of reshuffling the whole graph.
 */

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

export type Edge = { a: number; b: number; rewired: boolean };

export type Plan = { n: number; k: number; base: { a: number; b: number; u: number; pick: number }[] };

/** Fix the random numbers once per world. */
export function makePlan(n: number, k: number, seed: number): Plan {
  const r = rng(seed);
  const base: Plan['base'] = [];
  for (let d = 1; d <= k / 2; d++)
    for (let i = 0; i < n; i++) base.push({ a: i, b: (i + d) % n, u: r(), pick: r() });
  return { n, k, base };
}

/** The graph at rewiring probability p: rewired edges keep endpoint a, get a new endpoint. */
export function graphAt(plan: Plan, p: number): Edge[] {
  const { n } = plan;
  const key = (a: number, b: number) => (a < b ? a * n + b : b * n + a);
  const used = new Set<number>();
  for (const e of plan.base) used.add(key(e.a, e.b));
  const edges: Edge[] = [];
  for (const e of plan.base) {
    if (e.u >= p) {
      edges.push({ a: e.a, b: e.b, rewired: false });
      continue;
    }
    // choose a new endpoint: walk from a pseudo-random start to the first free node
    used.delete(key(e.a, e.b));
    let c = Math.floor(e.pick * n);
    let tries = 0;
    while ((c === e.a || used.has(key(e.a, c))) && tries < n) {
      c = (c + 1) % n;
      tries++;
    }
    if (tries >= n) {
      used.add(key(e.a, e.b));
      edges.push({ a: e.a, b: e.b, rewired: false });
      continue;
    }
    used.add(key(e.a, c));
    edges.push({ a: e.a, b: c, rewired: true });
  }
  return edges;
}

export function adjacency(n: number, edges: { a: number; b: number }[]) {
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const e of edges) {
    adj[e.a].push(e.b);
    adj[e.b].push(e.a);
  }
  return adj;
}

/** BFS distances and parents from s. */
export function bfs(adj: number[][], s: number) {
  const n = adj.length;
  const dist = new Int32Array(n).fill(-1);
  const par = new Int32Array(n).fill(-1);
  const q = [s];
  dist[s] = 0;
  for (let h = 0; h < q.length; h++) {
    const v = q[h];
    for (const w of adj[v])
      if (dist[w] < 0) {
        dist[w] = dist[v] + 1;
        par[w] = v;
        q.push(w);
      }
  }
  return { dist, par };
}

export function shortestPath(adj: number[][], s: number, t: number): number[] {
  const { dist, par } = bfs(adj, s);
  if (dist[t] < 0) return [];
  const path = [t];
  while (path[path.length - 1] !== s) path.push(par[path[path.length - 1]]);
  return path.reverse();
}

/** Average shortest-path length over reachable ordered pairs. */
export function avgPathLength(adj: number[][]) {
  let sum = 0;
  let cnt = 0;
  for (let s = 0; s < adj.length; s++) {
    const { dist } = bfs(adj, s);
    for (let t = 0; t < adj.length; t++)
      if (t !== s && dist[t] > 0) {
        sum += dist[t];
        cnt++;
      }
  }
  return sum / cnt;
}

/** Average local clustering coefficient: links among i's friends / (k_i choose 2). */
export function clustering(adj: number[][]) {
  const n = adj.length;
  const sets = adj.map((l) => new Set(l));
  let total = 0;
  for (let i = 0; i < n; i++) {
    const nb = adj[i];
    const k = nb.length;
    if (k < 2) continue;
    let links = 0;
    for (let x = 0; x < k; x++) for (let y = x + 1; y < k; y++) if (sets[nb[x]].has(nb[y])) links++;
    total += links / ((k * (k - 1)) / 2);
  }
  return total / n;
}

/** L(p)/L(0) and C(p)/C(0) averaged over several worlds, on a log grid of p. */
export function curves(n: number, k: number, seeds: number[], ps: number[]) {
  const L0 = avgPathLength(adjacency(n, graphAt(makePlan(n, k, 1), 0)));
  const C0 = clustering(adjacency(n, graphAt(makePlan(n, k, 1), 0)));
  return ps.map((p) => {
    let L = 0;
    let C = 0;
    for (const s of seeds) {
      const adj = adjacency(n, graphAt(makePlan(n, k, s), p));
      L += avgPathLength(adj);
      C += clustering(adj);
    }
    return { p, L: L / seeds.length / L0, C: C / seeds.length / C0 };
  });
}
