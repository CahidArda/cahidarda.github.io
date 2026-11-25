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

// Hand-laid in a 600 x 360 box: a loose town on the left (where the idea starts), thin
// chains of acquaintances in the middle, a small triangle of friends at the top right and a
// tight-knit group of seven at the bottom right.
export const W = 600;
export const H = 360;

const P: [number, number][] = [
  // 0-8: the loose town
  [70, 170],
  [125, 100],
  [130, 215],
  [50, 265],
  [190, 150],
  [185, 275],
  [55, 75],
  [110, 320],
  [245, 220],
  // 9-16: the middle chains
  [250, 90],
  [310, 45],
  [320, 130],
  [380, 40],
  [300, 205],
  [355, 270],
  [270, 310],
  [395, 180],
  // 17-21: the tight-knit group (with 25, 26 below)
  [430, 285],
  [465, 205],
  [505, 260],
  [570, 265],
  [470, 330],
  // 22-24: the triangle
  [455, 60],
  [470, 130],
  [540, 80],
  // 25-26: rest of the tight-knit group
  [555, 185],
  [540, 330],
];
export const NODES: Node[] = P.map(([x, y], id) => ({ id, x, y }));

export const TIGHT = [17, 18, 19, 20, 21, 25, 26];
export const SEEDS = [0, 2, 3];

export const EDGES: [number, number][] = [
  // the loose town
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 6],
  [1, 6],
  [1, 4],
  [2, 4],
  [2, 3],
  [2, 5],
  [3, 7],
  [5, 7],
  [4, 8],
  [5, 8],
  // bridges out of town
  [1, 9],
  [4, 9],
  [8, 13],
  [5, 15],
  [8, 15],
  // the middle chains
  [9, 10],
  [9, 11],
  [13, 14],
  [15, 14],
  [10, 12],
  [11, 16],
  [14, 16],
  [14, 17],
  [16, 18],
  [12, 22],
  [16, 23],
  // the triangle
  [22, 23],
  [22, 24],
  [23, 24],
  [24, 25],
  // the tight-knit group
  [17, 18],
  [17, 19],
  [17, 21],
  [18, 19],
  [18, 20],
  [19, 20],
  [19, 21],
  [20, 21],
  [18, 25],
  [19, 25],
  [20, 26],
  [21, 26],
  [25, 26],
  [17, 26],
  [25, 20],
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
