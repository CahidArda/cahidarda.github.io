/**
 * Random 3-SAT and a plain DPLL solver (unit propagation + branching), the setup of
 * Mitchell, Selman and Levesque (AAAI 1992). Pure logic, no React: used at build time by the
 * precompute script and in the browser for the search-tree figure.
 *
 * A literal is +v or -v for variable v in 1..n. A clause is three literals on distinct variables.
 */

export type Clause = [number, number, number];

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** m clauses over n variables: each picks 3 distinct variables, each negated with probability 1/2. */
export function randomFormula(n: number, m: number, seed: number): Clause[] {
  const r = rng(seed);
  const out: Clause[] = [];
  for (let i = 0; i < m; i++) {
    const vs: number[] = [];
    while (vs.length < 3) {
      const v = 1 + Math.floor(r() * n);
      if (!vs.includes(v)) vs.push(v);
    }
    out.push(vs.map((v) => (r() < 0.5 ? -v : v)) as Clause);
  }
  return out;
}

/** One node of the search tree, in the order the solver visits them. */
export type Node = {
  depth: number; // number of branching decisions above this node
  parent: number; // index of the parent node, -1 for the root
  lit: number; // the decision that led here (0 for the root)
  forced: number; // variables set by unit propagation at this node
  end: 'open' | 'conflict' | 'sat'; // open = branched further
};

export type Result = { sat: boolean; calls: number; nodes?: Node[] };

/**
 * DPLL. Each call does unit propagation; on a conflict it backtracks; if every clause is
 * satisfied it stops; otherwise it branches on a literal of the shortest open clause, trying
 * that literal true first, then false. `calls` counts the calls (nodes of the search tree).
 */
export function dpll(n: number, clauses: Clause[], record = false, limit = 1e7): Result {
  const val = new Int8Array(n + 1); // 0 unset, 1 true, -1 false
  const nodes: Node[] = [];
  let calls = 0;

  const litVal = (l: number) => (l > 0 ? val[l] : -val[-l]);

  // returns false on conflict; pushes assigned vars onto trail
  function propagate(trail: number[]): boolean {
    let changed = true;
    while (changed) {
      changed = false;
      for (const c of clauses) {
        let unset = 0;
        let last = 0;
        let sat = false;
        for (const l of c) {
          const x = litVal(l);
          if (x === 1) {
            sat = true;
            break;
          }
          if (x === 0) {
            unset++;
            last = l;
          }
        }
        if (sat) continue;
        if (unset === 0) return false;
        if (unset === 1) {
          const v = Math.abs(last);
          val[v] = last > 0 ? 1 : -1;
          trail.push(v);
          changed = true;
        }
      }
    }
    return true;
  }

  function pick(): number {
    let best = 0;
    let bestLen = 4;
    for (const c of clauses) {
      let unset = 0;
      let first = 0;
      let sat = false;
      for (const l of c) {
        const x = litVal(l);
        if (x === 1) {
          sat = true;
          break;
        }
        if (x === 0) {
          unset++;
          if (!first) first = l;
        }
      }
      if (!sat && unset > 0 && unset < bestLen) {
        bestLen = unset;
        best = first;
        if (unset === 2) break;
      }
    }
    return best;
  }

  function solve(depth: number, parent: number, lit: number): boolean {
    calls++;
    if (calls > limit) throw new Error('limit');
    const trail: number[] = [];
    if (lit) {
      const v = Math.abs(lit);
      val[v] = lit > 0 ? 1 : -1;
      trail.push(v);
    }
    const ok = propagate(trail);
    const me = nodes.length;
    if (record)
      nodes.push({ depth, parent, lit, forced: trail.length - (lit ? 1 : 0), end: 'open' });
    let found = false;
    if (!ok) {
      if (record) nodes[me].end = 'conflict';
    } else {
      const l = pick();
      if (l === 0) {
        if (record) nodes[me].end = 'sat';
        found = true;
      } else {
        found = solve(depth + 1, me, l) || solve(depth + 1, me, -l);
      }
    }
    if (!found) for (const v of trail) val[v] = 0;
    return found;
  }

  const sat = solve(0, -1, 0);
  return { sat, calls, nodes: record ? nodes : undefined };
}

/** Brute-force check for small n (tests only). */
export function bruteSat(n: number, clauses: Clause[]): boolean {
  for (let mask = 0; mask < 1 << n; mask++) {
    let all = true;
    for (const c of clauses) {
      let s = false;
      for (const l of c) {
        const bit = (mask >> (Math.abs(l) - 1)) & 1;
        if ((l > 0 && bit) || (l < 0 && !bit)) {
          s = true;
          break;
        }
      }
      if (!s) {
        all = false;
        break;
      }
    }
    if (all) return true;
  }
  return false;
}
