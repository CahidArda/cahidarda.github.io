/**
 * The three tile sets of CX-14.1 and what Wang's procedure finds for each. The two small sets
 * were picked by a random search: one repeats with a 4 x 4
 * block and no smaller one, one fills squares up to 5 x 5 and no 6 x 6 square at all.
 */
import { JEANDEL_RAO, fill, periodicBlock, type Grid, type Tile } from './wang';
import { JR_PATCH } from './jrPatch';

export type SetId = 'repeats' | 'stuck' | 'jr';

export const REPEATS: Tile[] = [
  [2, 2, 0, 1],
  [1, 0, 0, 2],
  [1, 2, 1, 0],
  [0, 0, 0, 1],
  [0, 1, 1, 2],
  [1, 0, 2, 0],
];

export const STUCK: Tile[] = [
  [0, 2, 3, 2],
  [2, 0, 0, 3],
  [1, 0, 3, 3],
  [1, 3, 0, 0],
  [3, 2, 1, 2],
];

export type Outcome = {
  tiles: Tile[];
  /** a patch whose top-left n x n squares are what the searcher shows */
  patch: Grid;
  /** square size at which the answer arrives (Infinity = never) */
  decideAt: number;
  verdict: 'yes' | 'no' | 'searching';
  /** side of the repeating block (yes), or placements tried before giving up (no) */
  block?: number;
  tries?: number;
  /** last ring drawn */
  maxN: number;
};

function repeats(): Outcome {
  const block = periodicBlock(REPEATS, 4)!;
  const k = block.length;
  const N = 12;
  const patch = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (_, c) => block[r % k][c % block[0].length]),
  );
  return { tiles: REPEATS, patch, decideAt: k, verdict: 'yes', block: k, maxN: N };
}

function stuck(): Outcome {
  const g = fill(STUCK, 5, 5).grid!;
  const six = fill(STUCK, 6, 6);
  return { tiles: STUCK, patch: g, decideAt: 6, verdict: 'no', tries: six.tries, maxN: 6 };
}

function jr(): Outcome {
  const patch = JR_PATCH.map((row) => [...row].map((ch) => ch.charCodeAt(0) - 97));
  return {
    tiles: JEANDEL_RAO,
    patch,
    decideAt: Infinity,
    verdict: 'searching',
    maxN: patch.length,
  };
}

export const OUTCOMES: Record<SetId, Outcome> = { repeats: repeats(), stuck: stuck(), jr: jr() };
