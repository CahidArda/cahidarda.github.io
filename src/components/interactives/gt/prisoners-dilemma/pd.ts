/**
 * Prisoner's Dilemma logic for GT-02, shared by every widget on the page. Pure TS (no
 * React) so it can be unit-checked with `node --experimental-strip-types`.
 *
 * Reworked from the earlier `interactives/pd/engine.ts` + `strategies.ts`:
 *  - strategies see the full history (Tit for Two Tats needs two rounds of memory);
 *  - memory-1 strategies also expose their reflex as a probability table, which the
 *    evolution model folds into an exact 4-state Markov chain (no sampling);
 *  - evolution is the textbook discrete replicator x_i' = x_i f_i / f_bar, plus a tiny
 *    mutation flow so extinct strategies can come back. Fully deterministic.
 */
export type Move = 'C' | 'D';

export interface Payoffs {
  T: number; // temptation: defect on a cooperator
  R: number; // reward: both cooperate
  P: number; // punishment: both defect
  S: number; // sucker: cooperate against a defector
}

/** Axelrod's tournament values. */
export const PAY: Payoffs = { T: 5, R: 3, P: 1, S: 0 };

export function payoff(mine: Move, theirs: Move, p: Payoffs = PAY): number {
  if (mine === 'C') return theirs === 'C' ? p.R : p.S;
  return theirs === 'C' ? p.T : p.P;
}

/** Deterministic PRNG (mulberry32). */
export function rng32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Strategy {
  id: string;
  name: string;
  short: string; // 2-4 letter tag for tight spaces
  /** Never defects first. */
  nice: boolean;
  rule: string; // one-line description
  decide: (me: Move[], them: Move[], rnd: () => number) => Move;
  /** Memory-1 form: [first-move pC, pC after CC, CD, DC, DD] (my move first). */
  m1?: [number, number, number, number, number];
}

const last = (xs: Move[]) => xs[xs.length - 1];

export const STRATS: Record<string, Strategy> = {
  tft: {
    id: 'tft',
    name: 'Tit for Tat',
    short: 'TFT',
    nice: true,
    rule: 'Cooperate first, then copy their last move.',
    decide: (_m, t) => (t.length ? last(t) : 'C'),
    m1: [1, 1, 0, 1, 0],
  },
  tf2t: {
    id: 'tf2t',
    name: 'Tit for Two Tats',
    short: 'TF2T',
    nice: true,
    rule: 'Defect only after two defections in a row.',
    decide: (_m, t) =>
      t.length >= 2 && t[t.length - 1] === 'D' && t[t.length - 2] === 'D' ? 'D' : 'C',
  },
  grudger: {
    id: 'grudger',
    name: 'Grudger',
    short: 'GRDG',
    nice: true,
    rule: 'Cooperate until betrayed once, then defect forever.',
    decide: (_m, t) => (t.includes('D') ? 'D' : 'C'),
  },
  pavlov: {
    id: 'pavlov',
    name: 'Pavlov',
    short: 'PAV',
    nice: true,
    rule: 'Win-stay, lose-shift: cooperate iff both moved alike last round.',
    decide: (m, t) => (m.length ? (last(m) === last(t) ? 'C' : 'D') : 'C'),
    m1: [1, 1, 0, 0, 1],
  },
  allc: {
    id: 'allc',
    name: 'Always Cooperate',
    short: 'ALLC',
    nice: true,
    rule: 'Always cooperate.',
    decide: () => 'C',
    m1: [1, 1, 1, 1, 1],
  },
  random: {
    id: 'random',
    name: 'Random',
    short: 'RAND',
    nice: false,
    rule: 'Flip a coin every round.',
    decide: (_m, _t, r) => (r() < 0.5 ? 'C' : 'D'),
  },
  alld: {
    id: 'alld',
    name: 'Always Defect',
    short: 'ALLD',
    nice: false,
    rule: 'Always defect.',
    decide: () => 'D',
    m1: [0, 0, 0, 0, 0],
  },
  joss: {
    id: 'joss',
    name: 'Joss',
    short: 'JOSS',
    nice: false,
    rule: 'Tit for Tat, but sneaks in a defection 10% of the time.',
    decide: (_m, t, r) => (t.length && last(t) === 'D' ? 'D' : r() < 0.1 ? 'D' : 'C'),
  },
};

export interface Match {
  a: Move[];
  b: Move[];
  scoreA: number; // average per round
  scoreB: number;
}

/** Play `rounds` of A vs B. `noise` flips each intended move with that probability. */
export function playMatch(
  a: Strategy,
  b: Strategy,
  rounds = 200,
  seed = 1,
  noise = 0,
  p: Payoffs = PAY,
): Match {
  const r = rng32(seed);
  const am: Move[] = [];
  const bm: Move[] = [];
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < rounds; i++) {
    let x = a.decide(am, bm, r);
    let y = b.decide(bm, am, r);
    if (noise && r() < noise) x = x === 'C' ? 'D' : 'C';
    if (noise && r() < noise) y = y === 'C' ? 'D' : 'C';
    am.push(x);
    bm.push(y);
    sa += payoff(x, y, p);
    sb += payoff(y, x, p);
  }
  return { a: am, b: bm, scoreA: sa / rounds, scoreB: sb / rounds };
}

/**
 * Round robin, everyone vs everyone including a twin of itself (as in Axelrod's
 * tournaments). score[i][j] = i's average per-round payoff against j, averaged over
 * `reps` seeded replays (only matters for the random strategies).
 */
export function roundRobin(ids: string[], rounds = 200, reps = 5) {
  const n = ids.length;
  const score = ids.map(() => new Array<number>(n).fill(0));
  const sample: Match[][] = ids.map(() => new Array<Match>(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < reps; k++) {
        const m = playMatch(STRATS[ids[i]], STRATS[ids[j]], rounds, 1000 + 97 * i + 13 * j + k);
        if (k === 0) sample[i][j] = m;
        s += m.scoreA;
      }
      score[i][j] = s / reps;
    }
  const total = score.map((row) => row.reduce((a, b) => a + b, 0) / n);
  return { score, total, sample };
}

/* ── evolution ── */

/**
 * Expected per-round payoff of memory-1 strategy A against B over `rounds` rounds, with
 * each intended move flipped with probability `noise`. Exact: a 4-state Markov chain on
 * (A's last move, B's last move).
 */
export function expectedPayoff(
  a: Strategy,
  b: Strategy,
  noise: number,
  rounds = 200,
  p: Payoffs = PAY,
): number {
  if (!a.m1 || !b.m1) throw new Error('memory-1 strategies only');
  const sq = (c: number) => c * (1 - noise) + (1 - c) * noise;
  // state index: 0=CC, 1=CD, 2=DC, 3=DD from A's point of view (A's move first)
  const pay = [p.R, p.S, p.T, p.P];
  const mirror = [0, 2, 1, 3]; // the same state seen from B
  let pa = sq(a.m1[0]);
  let pb = sq(b.m1[0]);
  let dist = [pa * pb, pa * (1 - pb), (1 - pa) * pb, (1 - pa) * (1 - pb)];
  let total = 0;
  for (let r = 0; r < rounds; r++) {
    for (let s = 0; s < 4; s++) total += dist[s] * pay[s];
    const next = [0, 0, 0, 0];
    for (let s = 0; s < 4; s++) {
      if (!dist[s]) continue;
      pa = sq(a.m1[1 + s]);
      pb = sq(b.m1[1 + mirror[s]]);
      next[0] += dist[s] * pa * pb;
      next[1] += dist[s] * pa * (1 - pb);
      next[2] += dist[s] * (1 - pa) * pb;
      next[3] += dist[s] * (1 - pa) * (1 - pb);
    }
    dist = next;
  }
  return total / rounds;
}

export const EVO_IDS = ['alld', 'allc', 'tft', 'pavlov'] as const;

/**
 * Replicator dynamics: x_i <- x_i f_i / f_bar, then a mutation flow mu toward uniform.
 * Returns the shares per generation and the population's mean payoff f_bar.
 */
export function evolve(
  noise: number,
  gens = 800,
  start = [0.6, 0.15, 0.15, 0.1],
  mu = 0.0005,
) {
  const set = EVO_IDS.map((id) => STRATS[id]);
  const M = set.map((a) => set.map((b) => expectedPayoff(a, b, noise)));
  const n = set.length;
  let x = [...start];
  const shares: number[][] = [];
  const mean: number[] = [];
  for (let g = 0; g <= gens; g++) {
    const f = x.map((_, i) => x.reduce((s, xj, j) => s + xj * M[i][j], 0));
    const fb = x.reduce((s, xi, i) => s + xi * f[i], 0);
    shares.push(x);
    mean.push(fb);
    x = x.map((xi, i) => (1 - mu) * ((xi * f[i]) / fb) + mu / n);
  }
  return { shares, mean, M };
}
