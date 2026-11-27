/**
 * Exam timetabling as graph colouring. Courses are nodes, an edge joins two courses that share
 * a student, slots are colours. Pure logic, no React (unit-tested with node --experimental-strip-types).
 * The course list and overlaps are made up for illustration.
 */

export type Graph = { n: number; adj: number[][] };

export const COURSES = [
  'Art',
  'Bio',
  'Calc',
  'Chem',
  'Econ',
  'Hist',
  'Law',
  'Ling',
  'Phil',
  'Phys',
  'Prog',
  'Stat',
] as const;
export const FULL: Record<string, string> = {
  Art: 'Art History',
  Bio: 'Biology',
  Calc: 'Calculus',
  Chem: 'Chemistry',
  Econ: 'Economics',
  Hist: 'History',
  Law: 'Law',
  Ling: 'Linguistics',
  Phil: 'Philosophy',
  Phys: 'Physics',
  Prog: 'Programming',
  Stat: 'Statistics',
};

/** Groups of courses some student takes together; every pair in a group clashes. */
export const GROUPS: string[][] = [
  ['Calc', 'Phys', 'Prog', 'Stat'],
  ['Bio', 'Chem', 'Stat'],
  ['Chem', 'Phys', 'Calc'],
  ['Econ', 'Stat', 'Law'],
  ['Hist', 'Phil', 'Art'],
  ['Ling', 'Phil', 'Prog'],
  ['Law', 'Hist'],
  ['Econ', 'Calc'],
  ['Econ', 'Hist'],
];

export function fromGroups(names: readonly string[], groups: string[][]): Graph {
  const n = names.length;
  const sets = Array.from({ length: n }, () => new Set<number>());
  for (const g of groups)
    for (const a of g) for (const b of g) if (a !== b) sets[names.indexOf(a)].add(names.indexOf(b));
  return { n, adj: sets.map((s) => [...s].sort((a, b) => a - b)) };
}

export const EXAMS = fromGroups(COURSES, GROUPS);

export function edges(g: Graph): [number, number][] {
  const out: [number, number][] = [];
  g.adj.forEach((nb, a) => nb.forEach((b) => a < b && out.push([a, b])));
  return out;
}

const firstFree = (g: Graph, col: number[], v: number) => {
  const used = new Set(g.adj[v].map((u) => col[u]));
  let k = 0;
  while (used.has(k)) k++;
  return k;
};

/** Greedy: visit in the given order, give each node the lowest slot free of clashes. */
export function greedy(g: Graph, order: number[]) {
  const col = Array(g.n).fill(-1);
  for (const v of order) col[v] = firstFree(g, col, v);
  return { order, col };
}

/**
 * DSatur (Brélaz 1979): next node is the uncoloured one whose neighbours already use the most
 * different slots; ties go to more uncoloured neighbours, then to list order.
 */
export function dsatur(g: Graph) {
  const col = Array(g.n).fill(-1);
  const order: number[] = [];
  for (let s = 0; s < g.n; s++) {
    let best = -1;
    let bs = -1;
    let bd = -1;
    for (let v = 0; v < g.n; v++) {
      if (col[v] >= 0) continue;
      const sat = new Set(g.adj[v].map((u) => col[u]).filter((x) => x >= 0)).size;
      const deg = g.adj[v].filter((u) => col[u] < 0).length;
      if (sat > bs || (sat === bs && deg > bd)) {
        best = v;
        bs = sat;
        bd = deg;
      }
    }
    col[best] = firstFree(g, col, best);
    order.push(best);
  }
  return { order, col };
}

export const slotsUsed = (col: number[]) => new Set(col.filter((c) => c >= 0)).size;

/** Number of clashing pairs (edges whose ends share a slot). */
export const clashes = (g: Graph, col: number[]) =>
  edges(g).filter(([a, b]) => col[a] >= 0 && col[a] === col[b]).length;

/** Chromatic number by exhaustive backtracking (fine for a dozen nodes). */
export function chromatic(g: Graph): number {
  for (let k = 1; k <= g.n; k++) {
    const col = Array(g.n).fill(-1);
    const bt = (i: number): boolean => {
      if (i === g.n) return true;
      for (let c = 0; c < k; c++) {
        if (g.adj[i].some((u) => col[u] === c)) continue;
        col[i] = c;
        if (bt(i + 1)) return true;
        col[i] = -1;
      }
      return false;
    };
    if (bt(0)) return k;
  }
  return g.n;
}

/**
 * Crown graph on n pairs: top i clashes with bottom j for every j ≠ i. Nodes 0..n-1 are the
 * top row, n..2n-1 the bottom row. Two slots always suffice (top / bottom), yet greedy in the
 * order top 1, bottom 1, top 2, bottom 2, ... needs n.
 */
export function crown(n: number): Graph {
  const adj = Array.from({ length: 2 * n }, () => [] as number[]);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (i !== j) {
        adj[i].push(n + j);
        adj[n + j].push(i);
      }
  adj.forEach((a) => a.sort((x, y) => x - y));
  return { n: 2 * n, adj };
}
export const zigzag = (n: number) =>
  Array.from({ length: 2 * n }, (_, k) => (k % 2 ? n : 0) + (k >> 1));
export const rowsFirst = (n: number) => Array.from({ length: 2 * n }, (_, k) => k);
