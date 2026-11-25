/**
 * Network cascades (Easley & Kleinberg, NCM 19.2-19.3). Each node plays a coordination
 * game with every neighbour. Payoff a if both use the new behaviour, b if both use the
 * old one, 0 if they differ. With a fraction p of neighbours on the new behaviour, the
 * new one is at least as good when p*a >= (1-p)*b, i.e. p >= q = b / (a + b).
 * Adoption is monotone here: once a node switches it stays switched.
 */
export interface Node {
  id: number;
  x: number;
  y: number;
}

// Hand-laid in a 600 x 360 box: a loose town on the left (where the idea starts), a
// chain of acquaintances, an open middle group and a tight-knit group on the right.
export const NODES: Node[] = [
  // left: loose town (0-8)
  { id: 0, x: 70, y: 150 },
  { id: 1, x: 120, y: 95 },
  { id: 2, x: 130, y: 205 },
  { id: 3, x: 55, y: 255 },
  { id: 4, x: 185, y: 145 },
  { id: 5, x: 190, y: 270 },
  { id: 6, x: 60, y: 60 },
  { id: 7, x: 115, y: 315 },
  { id: 8, x: 245, y: 210 },
  // middle top: open group (9-16)
  { id: 9, x: 255, y: 80 },
  { id: 10, x: 315, y: 40 },
  { id: 11, x: 330, y: 125 },
  { id: 12, x: 385, y: 75 },
  { id: 13, x: 300, y: 185 },
  { id: 14, x: 360, y: 260 },
  { id: 15, x: 290, y: 315 },
  { id: 16, x: 420, y: 160 },
  // right: tight-knit group (17-23)
  { id: 17, x: 470, y: 250 },
  { id: 18, x: 520, y: 200 },
  { id: 19, x: 560, y: 260 },
  { id: 20, x: 530, y: 320 },
  { id: 21, x: 460, y: 315 },
  { id: 22, x: 470, y: 60 },
  { id: 23, x: 545, y: 110 },
];

export const EDGES: [number, number][] = [
  // left town
  [0, 1], [0, 2], [0, 3], [0, 6], [1, 6], [1, 4], [2, 4], [2, 3], [2, 5], [3, 7], [5, 7], [4, 8], [5, 8],
  // bridges
  [4, 9], [1, 9], [8, 13], [5, 15],
  // middle group
  [9, 10], [9, 11], [10, 12], [11, 12], [11, 13], [12, 16], [13, 14], [14, 15], [11, 16], [13, 16], [14, 16],
  [12, 22], [22, 23], [16, 23],
  // tight-knit group
  [17, 18], [17, 19], [17, 20], [17, 21], [18, 19], [18, 20], [19, 20], [19, 21], [20, 21], [18, 21],
  [14, 17], [16, 18],
];

export function neighbours(n: number, edges = EDGES): number[][] {
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adj[a].push(b);
    adj[b].push(a);
  }
  return adj;
}

export const ADJ = neighbours(NODES.length);

/** One synchronous round: returns the nodes that switch this round. */
export function round(adopted: boolean[], q: number, adj = ADJ): number[] {
  const out: number[] = [];
  adj.forEach((nb, i) => {
    if (adopted[i] || nb.length === 0) return;
    const k = nb.filter((j) => adopted[j]).length;
    if (k / nb.length >= q - 1e-9) out.push(i);
  });
  return out;
}

/** Run to a fixed point. Returns the round each node adopted in (0 = seed, -1 = never). */
export function runAll(seeds: number[], q: number, adj = ADJ): number[] {
  const when = adj.map(() => -1);
  const adopted = adj.map(() => false);
  for (const s of seeds) {
    adopted[s] = true;
    when[s] = 0;
  }
  for (let r = 1; r <= adj.length; r++) {
    const next = round(adopted, q, adj);
    if (next.length === 0) break;
    for (const i of next) {
      adopted[i] = true;
      when[i] = r;
    }
  }
  return when;
}

/**
 * The holdouts of a stalled cascade form a cluster of density > 1 - q (NCM 19.3):
 * every holdout has more than 1 - q of its friends inside the holdout set.
 * Returns the smallest such inside-fraction (the cluster's density).
 */
export function clusterDensity(holdouts: number[], adj = ADJ): number {
  const set = new Set(holdouts);
  let min = 1;
  for (const i of holdouts) {
    const nb = adj[i];
    if (nb.length === 0) continue;
    min = Math.min(min, nb.filter((j) => set.has(j)).length / nb.length);
  }
  return min;
}
