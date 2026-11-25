/**
 * PageRank as in Easley & Kleinberg ch. 14.3.
 * Basic update rule: each page splits its current rank equally over its out-links
 * (a page with no out-links keeps its rank). Scaled rule: apply the basic rule, scale
 * everything by s, and spread the remaining 1 - s equally over all n pages.
 */
export type Edge = [number, number];

export function outLists(n: number, edges: Edge[]): number[][] {
  const out: number[][] = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) out[a].push(b);
  return out;
}

/** One application of the scaled update rule (s = 1 gives the basic rule). */
export function update(r: number[], out: number[][], s: number): number[] {
  const n = r.length;
  const next = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    const o = out[j];
    if (o.length === 0) next[j] += r[j];
    else for (const i of o) next[i] += r[j] / o.length;
  }
  return next.map((v) => s * v + (1 - s) / n);
}

export function pagerank(n: number, edges: Edge[], s: number, iters = 200): number[] {
  const out = outLists(n, edges);
  let r = new Array(n).fill(1 / n);
  for (let k = 0; k < iters; k++) r = update(r, out, s);
  return r;
}
