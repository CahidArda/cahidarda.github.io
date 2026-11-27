/**
 * CX-05.2: famous NP-complete problems and one published reduction into each, drawn the
 * way speed flows (child → parent: a fast solver for the child also solves the parent).
 *   3-SAT  ← SAT          Karp 1972 (SATISFIABILITY ∝ SAT WITH ≤3 LITERALS PER CLAUSE)
 *   Clique ← SAT          Karp 1972
 *   Vertex cover ← Clique Karp 1972 (NODE COVER)
 *   Ham cycle ← Vertex cover  Karp 1972 (via DIRECTED HAMILTON CIRCUIT)
 *   TSP ← Ham cycle       standard textbook reduction (Garey & Johnson 1979)
 *   3-colouring ← 3-SAT   Karp 1972 (CHROMATIC NUMBER); Sipser 3rd ed. §7.5 problems
 *   Subset sum ← 3-SAT    Sipser 3rd ed. §7.5 (SUBSET-SUM)
 *   Latin square ← 3-SAT  completing a partial Latin square is NP-complete (Colbourn 1984)
 *   Sudoku ← Latin square Yato & Seta 2003
 */
export type NodeId =
  | 'SAT'
  | '3SAT'
  | 'CLIQUE'
  | 'VC'
  | 'HAM'
  | 'TSP'
  | 'COL'
  | 'SUBSET'
  | 'LATIN'
  | 'SUDOKU';

export const NODES: { id: NodeId; name: string; say: string; parent?: NodeId }[] = [
  { id: 'SAT', name: 'SAT', say: 'SAT' },
  { id: '3SAT', name: '3-SAT', say: '3-SAT', parent: 'SAT' },
  { id: 'CLIQUE', name: 'Clique', say: 'clique', parent: 'SAT' },
  { id: 'COL', name: '3-colouring', say: '3-colouring', parent: '3SAT' },
  { id: 'SUBSET', name: 'Subset sum', say: 'subset sum', parent: '3SAT' },
  { id: 'LATIN', name: 'Latin square', say: 'Latin squares', parent: '3SAT' },
  { id: 'VC', name: 'Vertex cover', say: 'vertex cover', parent: 'CLIQUE' },
  { id: 'SUDOKU', name: 'Sudoku', say: 'Sudoku', parent: 'LATIN' },
  { id: 'HAM', name: 'Ham cycle', say: 'Hamiltonian cycle', parent: 'VC' },
  { id: 'TSP', name: 'TSP', say: 'TSP', parent: 'HAM' },
];

/** The chain of problems a fast solver for `id` also solves, ending at SAT. */
export function pathToSat(id: NodeId): NodeId[] {
  const out: NodeId[] = [id];
  let cur = NODES.find((n) => n.id === id);
  while (cur?.parent) {
    out.push(cur.parent);
    cur = NODES.find((n) => n.id === cur!.parent);
  }
  return out;
}
