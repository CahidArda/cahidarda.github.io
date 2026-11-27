/**
 * 6x6 Sudoku logic for CX-03.1 (check vs solve). Boxes are 2 rows x 3 columns.
 * Both machines are written as a list of "steps" so the widget can replay them on one clock:
 *   - the checker reads every cell of every row, column and box: 18 groups x 6 = 108 steps;
 *   - the solver is plain backtracking: walk the blanks in reading order, try 1..6 in each,
 *     one step per digit tried, back up when a cell runs out of digits.
 */
export const N = 6;
export type Grid = number[]; // 36 cells, 0 = blank

// 10 givens, one solution; plain backtracking needs 3,013 digit tries (checked in node)
export const PUZZLE: Grid = [
  0, 0, 5, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 5, 0, 0, 0, 6, 0, 0, 6,
  0, 0, 0,
];

export const rowOf = (i: number) => Math.floor(i / N);
export const colOf = (i: number) => i % N;
export const boxOf = (i: number) => Math.floor(rowOf(i) / 2) * 2 + Math.floor(colOf(i) / 3);

/** The 18 groups (6 rows, 6 columns, 6 boxes), each a list of 6 cell indices. */
export const GROUPS: number[][] = (() => {
  const g: number[][] = [];
  for (let r = 0; r < N; r++) g.push(Array.from({ length: N }, (_, c) => r * N + c));
  for (let c = 0; c < N; c++) g.push(Array.from({ length: N }, (_, r) => r * N + c));
  for (let b = 0; b < N; b++) {
    const r0 = Math.floor(b / 2) * 2;
    const c0 = (b % 2) * 3;
    const cells: number[] = [];
    for (let r = r0; r < r0 + 2; r++) for (let c = c0; c < c0 + 3; c++) cells.push(r * N + c);
    g.push(cells);
  }
  return g;
})();

export const CHECK_STEPS = GROUPS.length * N; // 108

/** Which groups are broken (a repeated or missing digit). */
export function badGroups(g: Grid): boolean[] {
  return GROUPS.map((cells) => {
    const seen = new Set(cells.map((i) => g[i]));
    return seen.size !== N || seen.has(0);
  });
}

function fits(g: Grid, i: number, d: number) {
  const r = rowOf(i);
  const c = colOf(i);
  const b = boxOf(i);
  for (let j = 0; j < N * N; j++) {
    if (j === i || g[j] !== d) continue;
    if (rowOf(j) === r || colOf(j) === c || boxOf(j) === b) return false;
  }
  return true;
}

/** One solver step: cell `cell` shows `digit` (0 = erased while backing up). */
export type Step = { cell: number; digit: number };

/** Backtracking trace: every digit tried is one step; erasing on the way back is free. */
export function solveTrace(p: Grid): { steps: Step[]; solution: Grid } {
  const g = p.slice();
  const blanks = g.map((v, i) => (v === 0 ? i : -1)).filter((i) => i >= 0);
  const steps: Step[] = [];
  let k = 0;
  while (k >= 0 && k < blanks.length) {
    const i = blanks[k];
    let d = g[i] + 1;
    let placed = false;
    while (d <= N) {
      steps.push({ cell: i, digit: d });
      if (fits(g, i, d)) {
        g[i] = d;
        placed = true;
        break;
      }
      d++;
    }
    if (placed) k++;
    else {
      g[i] = 0;
      steps.push({ cell: i, digit: 0 });
      k--;
    }
  }
  return { steps, solution: g };
}

export function countSolutions(p: Grid, cap = 2): number {
  const g = p.slice();
  let n = 0;
  const rec = (k: number): void => {
    if (n >= cap) return;
    const i = g.indexOf(0, k);
    if (i < 0) {
      n++;
      return;
    }
    for (let d = 1; d <= N; d++)
      if (fits(g, i, d)) {
        g[i] = d;
        rec(i + 1);
        g[i] = 0;
      }
  };
  rec(0);
  return n;
}
