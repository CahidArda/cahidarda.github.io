/**
 * The five 2x2 games of GT-01. Payoffs u[r][c] = [row player's, column player's].
 * Row player = blue (GT.a), column player = red (GT.b). Pure logic, no React.
 */
export type GameId = 'pd' | 'stag' | 'chicken' | 'coord' | 'pennies';

export interface Game {
  id: GameId;
  name: string; // tab label
  rows: [string, string]; // row player's strategies
  cols: [string, string];
  u: [number, number][][];
  note: string; // one-line caption
}

export const GAMES: Game[] = [
  {
    id: 'pd',
    name: "Prisoner's",
    rows: ['Cooperate', 'Defect'],
    cols: ['Cooperate', 'Defect'],
    u: [
      [
        [3, 3],
        [0, 5],
      ],
      [
        [5, 0],
        [1, 1],
      ],
    ],
    note: 'Defect beats Cooperate whatever the other does. One equilibrium, and it is the bad one.',
  },
  {
    id: 'stag',
    name: 'Stag hunt',
    rows: ['Stag', 'Hare'],
    cols: ['Stag', 'Hare'],
    u: [
      [
        [4, 4],
        [0, 3],
      ],
      [
        [3, 0],
        [3, 3],
      ],
    ],
    note: 'Hunt the stag together or settle for hares. Both are equilibria; trust decides which.',
  },
  {
    id: 'chicken',
    name: 'Hawk-Dove',
    rows: ['Dove', 'Hawk'],
    cols: ['Dove', 'Hawk'],
    u: [
      [
        [3, 3],
        [1, 5],
      ],
      [
        [5, 1],
        [0, 0],
      ],
    ],
    note: 'Each wants to be the hawk while the other backs down. Two equilibria, one per bully.',
  },
  {
    id: 'coord',
    name: 'Coordination',
    rows: ['Slides', 'Docs'],
    cols: ['Slides', 'Docs'],
    u: [
      [
        [2, 1],
        [0, 0],
      ],
      [
        [0, 0],
        [1, 2],
      ],
    ],
    note: 'Both want to use the same tool, but each prefers a different one. Two equilibria.',
  },
  {
    id: 'pennies',
    name: 'Pennies',
    rows: ['Heads', 'Tails'],
    cols: ['Heads', 'Tails'],
    u: [
      [
        [1, -1],
        [-1, 1],
      ],
      [
        [-1, 1],
        [1, -1],
      ],
    ],
    note: 'Blue wants the coins to match, red wants them to differ. Someone always wants to move.',
  },
];

export const GAME_BY_ID = Object.fromEntries(GAMES.map((g) => [g.id, g])) as Record<GameId, Game>;

export type Cell = [number, number]; // [row, col]

export interface Arrow {
  from: Cell;
  to: Cell;
  who: 'row' | 'col';
  gain: number;
}

/** Every strictly improving unilateral deviation. */
export function arrows(g: Game): Arrow[] {
  const out: Arrow[] = [];
  for (let c = 0; c < 2; c++) {
    const d = g.u[1][c][0] - g.u[0][c][0];
    if (d > 0) out.push({ from: [0, c], to: [1, c], who: 'row', gain: d });
    if (d < 0) out.push({ from: [1, c], to: [0, c], who: 'row', gain: -d });
  }
  for (let r = 0; r < 2; r++) {
    const d = g.u[r][1][1] - g.u[r][0][1];
    if (d > 0) out.push({ from: [r, 0], to: [r, 1], who: 'col', gain: d });
    if (d < 0) out.push({ from: [r, 1], to: [r, 0], who: 'col', gain: -d });
  }
  return out;
}

const same = (a: Cell, b: Cell) => a[0] === b[0] && a[1] === b[1];

/** Pure Nash equilibria = cells with no outgoing improvement arrow. */
export function equilibria(g: Game): Cell[] {
  const as = arrows(g);
  const cells: Cell[] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  return cells.filter((c) => !as.some((a) => same(a.from, c)));
}

/** Next hop for a token at `at`. Prefers the player who did not move last. */
export function nextHop(g: Game, at: Cell, lastWho: 'row' | 'col' | null): Arrow | null {
  const out = arrows(g).filter((a) => same(a.from, at));
  if (!out.length) return null;
  return out.find((a) => a.who !== lastWho) ?? out[0];
}

/** Pareto-optimal outcomes: no other cell is at least as good for both and better for one. */
export function paretoOptimal(g: Game): Cell[] {
  const cells: Cell[] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  const u = (c: Cell) => g.u[c[0]][c[1]];
  return cells.filter(
    (c) =>
      !cells.some((d) => {
        const [a1, b1] = u(d);
        const [a0, b0] = u(c);
        return a1 >= a0 && b1 >= b0 && (a1 > a0 || b1 > b0);
      }),
  );
}

export const isCell = (cs: Cell[], c: Cell) => cs.some((d) => same(c, d));

export const NASH_EVENT = 'gt-nash-game';
