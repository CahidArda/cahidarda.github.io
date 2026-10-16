/**
 * The simulation core. Two layers:
 *
 *  1. `playRound` / `simulateMatch` - concrete round-by-round play, used by the
 *     interactive playground and the tournament.
 *  2. `payoffMatrix` + `evoStep` - the evolution engine. Expected pairwise payoffs are
 *     computed analytically as the stationary behaviour of a 16-state joint Markov chain
 *     (no sampling, fully deterministic), then a finite-population replicator update
 *     drives the dynamics. `evoStep(state) -> state` is deliberately a single pure
 *     function so the JS loop can later be swapped for the Rust/rayon WASM core behind
 *     the exact same interface.
 */
import type { Move } from './shared';
import { payoff, STRATEGIES, type Payoffs, type Strategy } from './strategies';

/* ── Deterministic PRNG (mulberry32) so every run is reproducible & WASM-portable ── */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const flip = (m: Move): Move => (m === 'C' ? 'D' : 'C');
const tremble = (intended: Move, noise: number, rng: () => number): Move =>
  noise > 0 && rng() < noise ? flip(intended) : intended;

/* ── Concrete play (playground, tournament) ── */

export function nextMove(
  strat: Strategy,
  myMoves: Move[],
  oppMoves: Move[],
  rng: () => number,
): Move {
  const round = myMoves.length;
  if (round === 0) return strat.opening;
  return strat.decide({
    round,
    myLast: myMoves[round - 1],
    oppLast: oppMoves[round - 1],
    oppEverDefected: oppMoves.includes('D'),
    rng,
  });
}

export interface MatchResult {
  a: Move[];
  b: Move[];
  scoreA: number;
  scoreB: number;
}

/** Simulate `rounds` of A vs B with optional trembling-hand `noise`. Deterministic in `seed`. */
export function simulateMatch(
  a: Strategy,
  b: Strategy,
  rounds: number,
  p: Payoffs,
  noise = 0,
  seed = 1,
): MatchResult {
  const rng = mulberry32(seed);
  const am: Move[] = [];
  const bm: Move[] = [];
  let scoreA = 0;
  let scoreB = 0;
  for (let r = 0; r < rounds; r++) {
    const ai = tremble(nextMove(a, am, bm, rng), noise, rng);
    const bi = tremble(nextMove(b, bm, am, rng), noise, rng);
    am.push(ai);
    bm.push(bi);
    scoreA += payoff(ai, bi, p);
    scoreB += payoff(bi, ai, p);
  }
  return { a: am, b: bm, scoreA, scoreB };
}

/* ── Analytic expected pairwise payoffs: a 16-state joint Markov chain ──
   State = (aLast, bLast, aSeenD, bSeenD), each one bit. aSeenD = "B has ever defected
   against A" (the only history bit Grudger needs). We propagate a probability
   distribution forward and average the per-round payoff to A. */

const S = 4; // bits -> 16 states
const NSTATES = 1 << S;
const enc = (aLast: number, bLast: number, aSeen: number, bSeen: number) =>
  (aLast << 3) | (bLast << 2) | (aSeen << 1) | bSeen;

function expectedPayoff(
  a: Strategy,
  b: Strategy,
  p: Payoffs,
  noise: number,
  rounds: number,
): number {
  // pC after squeezing intended cooperate-probability through the trembling hand.
  const squeeze = (pc: number) => pc * (1 - noise) + (1 - pc) * noise;

  let dist = new Float64Array(NSTATES);
  let total = 0;

  // Round 0: openings.
  const a0 = squeeze(a.opening === 'C' ? 1 : 0);
  const b0 = squeeze(b.opening === 'C' ? 1 : 0);
  for (const [aMove, aP] of [
    [0, a0],
    [1, 1 - a0],
  ] as const) {
    for (const [bMove, bP] of [
      [0, b0],
      [1, 1 - b0],
    ] as const) {
      const prob = aP * bP;
      if (prob === 0) continue;
      const am: Move = aMove === 0 ? 'C' : 'D';
      const bm: Move = bMove === 0 ? 'C' : 'D';
      total += prob * payoff(am, bm, p);
      dist[enc(aMove, bMove, bMove === 1 ? 1 : 0, aMove === 1 ? 1 : 0)] += prob;
    }
  }

  // Rounds 1..rounds-1.
  for (let r = 1; r < rounds; r++) {
    const next = new Float64Array(NSTATES);
    for (let s = 0; s < NSTATES; s++) {
      const prob = dist[s];
      if (prob === 0) continue;
      const aLast = (s >> 3) & 1;
      const bLast = (s >> 2) & 1;
      const aSeen = (s >> 1) & 1;
      const bSeen = s & 1;
      const aLastM: Move = aLast === 0 ? 'C' : 'D';
      const bLastM: Move = bLast === 0 ? 'C' : 'D';
      const apC = squeeze(a.pCooperate(aLastM, bLastM, aSeen === 1));
      const bpC = squeeze(b.pCooperate(bLastM, aLastM, bSeen === 1));
      for (const [aMove, aP] of [
        [0, apC],
        [1, 1 - apC],
      ] as const) {
        if (aP === 0) continue;
        for (const [bMove, bP] of [
          [0, bpC],
          [1, 1 - bpC],
        ] as const) {
          const pr = prob * aP * bP;
          if (pr === 0) continue;
          const am: Move = aMove === 0 ? 'C' : 'D';
          const bm: Move = bMove === 0 ? 'C' : 'D';
          total += pr * payoff(am, bm, p);
          const nASeen = aSeen | (bMove === 1 ? 1 : 0);
          const nBSeen = bSeen | (aMove === 1 ? 1 : 0);
          next[enc(aMove, bMove, nASeen, nBSeen)] += pr;
        }
      }
    }
    dist = next;
  }
  return total / rounds;
}

/** N×N matrix of expected per-round payoffs (row plays column). Order follows `set`. */
export function payoffMatrix(
  p: Payoffs,
  noise: number,
  rounds = 200,
  set: Strategy[] = STRATEGIES,
): number[][] {
  return set.map((row) => set.map((col) => expectedPayoff(row, col, p, noise, rounds)));
}

/* ── Evolution: finite-population replicator dynamics ── */

export interface EvoParams {
  N: number; // population size - controls drift (~1/sqrt(N))
  mu: number; // mutation rate
  noise: number; // trembling-hand
  payoffs: Payoffs;
  select: number; // selection strength
}

export interface EvoState {
  gen: number;
  freqs: number[]; // length STRATEGIES.length, sums to 1
  seed: number;
  params: EvoParams;
  matrix: number[][];
}

export function initEvo(
  params: EvoParams,
  seed = 12345,
  start?: number[],
  set: Strategy[] = STRATEGIES,
): EvoState {
  const n = set.length;
  const raw = start ? [...start] : new Array(n).fill(1 / n);
  const sum = raw.reduce((a, b) => a + b, 0) || 1;
  const freqs = raw.map((x) => x / sum);
  return {
    gen: 0,
    freqs,
    seed,
    params,
    matrix: payoffMatrix(params.payoffs, params.noise, 200, set),
  };
}

/** One pure generation step. Swap-compatible with a future WASM `step(state) -> state`. */
export function evoStep(state: EvoState): EvoState {
  const { freqs, matrix, params } = state;
  const n = freqs.length;
  const rng = mulberry32(state.seed);

  // Fitness = expected payoff against the current population (self-play included).
  const fit = new Array(n).fill(0);
  let avg = 0;
  for (let i = 0; i < n; i++) {
    let f = 0;
    for (let j = 0; j < n; j++) f += freqs[j] * matrix[i][j];
    fit[i] = f;
    avg += freqs[i] * f;
  }

  // Replicator via exponential (softmax) weighting - always positive, smooth.
  const w = new Array(n);
  let wsum = 0;
  for (let i = 0; i < n; i++) {
    w[i] = freqs[i] * Math.exp(params.select * (fit[i] - avg));
    wsum += w[i];
  }
  let p = w.map((x) => x / wsum);

  // Mutation toward the uniform mix.
  const u = 1 / n;
  p = p.map((x) => (1 - params.mu) * x + params.mu * u);

  // Finite-population drift: Gaussian approximation to the multinomial sampling,
  // magnitude ~ 1/sqrt(N). Large N -> near-deterministic; small N -> wandering.
  const next = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const mean = p[i];
    const sd = Math.sqrt((mean * (1 - mean)) / Math.max(1, params.N));
    // Box-Muller.
    const z = Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
    next[i] = Math.max(0, mean + sd * z);
    sum += next[i];
  }
  for (let i = 0; i < n; i++) next[i] /= sum || 1;

  return {
    ...state,
    gen: state.gen + 1,
    freqs: next,
    seed: (state.seed * 1664525 + 1013904223) >>> 0,
  };
}
