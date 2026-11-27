/**
 * Boards and gadgets from Richard Kaye, "Minesweeper is NP-complete",
 * The Mathematical Intelligencer 22(2), 2000, pp. 9-15.
 *
 * A board is a list of rows. Tokens: '0'..'8' a revealed number, '.' a hidden square,
 * '*' a known mine, 'x' / 'y' the two kinds of hidden wire square (Kaye's x and x'),
 * ' ' outside the board. Pure logic, no React: unit-tested with node --experimental-strip-types.
 */

export type Cell = { r: number; c: number; t: string };

export function parse(rows: string[]): Cell[] {
  const out: Cell[] = [];
  rows.forEach((row, r) => [...row].forEach((t, c) => t !== ' ' && out.push({ r, c, t })));
  return out;
}

export const isNum = (t: string) => t >= '0' && t <= '8';
export const isHidden = (t: string) => t === '.' || t === 'x' || t === 'y';

/** For every revealed number: its value and the indices (into `hidden`) of its hidden neighbours. */
export function constraints(rows: string[]) {
  const cells = parse(rows);
  const hidden = cells.filter((k) => isHidden(k.t));
  const at = new Map(hidden.map((k, i) => [`${k.r},${k.c}`, i]));
  const mines = new Set(cells.filter((k) => k.t === '*').map((k) => `${k.r},${k.c}`));
  const nums = cells
    .filter((k) => isNum(k.t))
    .map((k) => {
      const nb: number[] = [];
      let fixed = 0;
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          const key = `${k.r + dr},${k.c + dc}`;
          const i = at.get(key);
          if (i !== undefined) nb.push(i);
          if (mines.has(key)) fixed++;
        }
      return { r: k.r, c: k.c, v: Number(k.t), nb, fixed };
    });
  return { hidden, nums };
}

/** Every mine placement (bitmask over `hidden`) that fits every number. Brute force. */
export function solve(rows: string[]): number[] {
  const { hidden, nums } = constraints(rows);
  const n = hidden.length;
  if (n > 24) throw new Error('too many hidden squares for brute force');
  const sols: number[] = [];
  for (let m = 0; m < 1 << n; m++) {
    let ok = true;
    for (const q of nums) {
      let s = q.fixed;
      for (const k of q.nb) s += (m >> k) & 1;
      if (s !== q.v) {
        ok = false;
        break;
      }
    }
    if (ok) sols.push(m);
  }
  return sols;
}

/* ───────────── EX 1: "is this board possible?" (Kaye's Figure 4) ───────────── */

/** Kaye's Figure 4: a 6 x 6 board with sixteen squares revealed. */
export const FIG4 = ['......', '.2222.', '.2002.', '.2002.', '.2222.', '......'];
/** The same board with the bottom-right 2 changed to a 1: no placement fits. */
export const BROKEN = ['......', '.2222.', '.2002.', '.2002.', '.2221.', '......'];
/** 20 hidden squares, so 2^20 placements to search. */
export const SPACE = 1 << 20;
/** The only placement that fits FIG4 (bitmask over its hidden squares, row-major). */
export const FIG4_SOLUTION = 0b110000111100001100;
/** Mines of that placement, as "r,c" keys, in the order the demo places them (reading order). */
export const FIG4_MINES = ['0,2', '0,3', '2,0', '2,5', '3,0', '3,5', '5,2', '5,3'];

/* ───────────── EX 2: the wire and the NOT gate (Kaye's Figures 6 and 9) ───────────── */

/**
 * A wire into a NOT gate and out again. 'x' squares hold mines when the input is TRUE,
 * 'y' (Kaye's x') when it is FALSE. Middle three columns are Figure 9's gate; the zero rows
 * are Figure 6's border. `side` wire squares on each side of the gate.
 */
export function notGate(side: 5 | 8): string[] {
  const L = side === 8 ? 'yx1yx1yx' : 'yx1yx';
  const R = side === 8 ? 'xy1xy1xy' : 'xy1xy';
  const ones = '1'.repeat(side);
  const zeros = '0'.repeat(side);
  return [
    zeros + '111' + zeros,
    ones + '2*2' + ones,
    L + '3y3' + R,
    ones + '2*2' + ones,
    zeros + '111' + zeros,
  ];
}

/** Is the square a mine when the wire's input is `v`? (Gate centre is a 'y': mine iff input FALSE.) */
export const mineAt = (t: string, v: boolean) => (t === 'x' ? v : t === 'y' ? !v : t === '*');
