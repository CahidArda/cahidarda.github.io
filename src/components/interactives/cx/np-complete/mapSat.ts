/**
 * The CX-05.1 reduction: 3-colouring a 4-country map, written as SAT clauses.
 * Each country X gets three yes/no variables X1 X2 X3 ("X has colour c").
 *   some(X):      X1 ∨ X2 ∨ X3          every country gets a colour     (4 clauses)
 *   diff(X,Y,c): ¬Xc ∨ ¬Yc             neighbours are not both colour c (5 borders × 3 = 15)
 */
export type Region = 'A' | 'B' | 'C' | 'D';
export const REGIONS: Region[] = ['A', 'B', 'C', 'D'];
/** Shared borders. A and D do not touch. */
export const BORDERS: [Region, Region][] = [
  ['A', 'B'],
  ['A', 'C'],
  ['B', 'C'],
  ['B', 'D'],
  ['C', 'D'],
];
export const COLOR_NAMES = ['blue', 'amber', 'violet'];

/** A colouring: one colour index (0..2) per country, or null for none. */
export type Colouring = Record<Region, number | null>;

export type Clause =
  | { kind: 'some'; r: Region }
  | { kind: 'diff'; a: Region; b: Region; c: number };

export const CLAUSES: Clause[] = [
  ...REGIONS.map((r) => ({ kind: 'some' as const, r })),
  ...BORDERS.flatMap(([a, b]) => [0, 1, 2].map((c) => ({ kind: 'diff' as const, a, b, c }))),
];

/** Variable X_c is true exactly when country X has colour c. */
export function clauseTrue(cl: Clause, col: Colouring): boolean {
  if (cl.kind === 'some') return col[cl.r] !== null;
  return !(col[cl.a] === cl.c && col[cl.b] === cl.c);
}

/** True when every country has a colour and no border joins two of the same colour. */
export function properColouring(col: Colouring): boolean {
  return REGIONS.every((r) => col[r] !== null) && BORDERS.every(([a, b]) => col[a] !== col[b]);
}
