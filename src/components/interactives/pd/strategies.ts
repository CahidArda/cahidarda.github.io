/**
 * The cast of strategies, defined ONCE and shared by the playground, the zoo, the
 * tournament and the evolution sim. Each strategy is a small reflex described by
 * `decide`, a pure function of what it can see: its own last move, the opponent's last
 * move, and whether the opponent has ever defected (the one bit Grudger needs).
 *
 * `pCooperate` exposes the same reflex as a *probability*, which lets the evolution
 * engine fold every strategy - deterministic or random - into one Markov chain.
 */
import type { Move } from './shared';

export interface DecideCtx {
  round: number; // 0-based
  myLast: Move | null;
  oppLast: Move | null;
  oppEverDefected: boolean;
  rng: () => number;
}

export interface Strategy {
  id: string;
  name: string;
  /** One-sentence reflex, used as the card caption in the zoo. No paragraphs. */
  short: string;
  opening: Move;
  decide: (ctx: DecideCtx) => Move;
  /** Probability of cooperating in the given situation - for the analytic engine. */
  pCooperate: (myLast: Move, oppLast: Move, oppEverDefected: boolean) => number;
}

const det =
  (f: (c: DecideCtx) => Move): ((m: Move, o: Move, e: boolean) => number) =>
  (myLast, oppLast, oppEverDefected) =>
    f({ round: 1, myLast, oppLast, oppEverDefected, rng: () => 0 }) === 'C' ? 1 : 0;

export const STRATEGIES: Strategy[] = [
  {
    id: 'tft',
    name: 'Tit-for-Tat',
    short: 'Open with cooperation, then echo whatever you did last.',
    opening: 'C',
    decide: ({ oppLast }) => oppLast ?? 'C',
    pCooperate: (_m, oppLast) => (oppLast === 'C' ? 1 : 0),
  },
  {
    id: 'alld',
    name: 'Always Defect',
    short: 'Never cooperate. Ever.',
    opening: 'D',
    decide: () => 'D',
    pCooperate: () => 0,
  },
  {
    id: 'allc',
    name: 'Always Cooperate',
    short: 'Always cooperate, no matter what.',
    opening: 'C',
    decide: () => 'C',
    pCooperate: () => 1,
  },
  {
    id: 'grudger',
    name: 'Grudger',
    short: 'Cooperate until betrayed once, then defect forever.',
    opening: 'C',
    decide: ({ oppEverDefected }) => (oppEverDefected ? 'D' : 'C'),
    pCooperate: (_m, _o, oppEverDefected) => (oppEverDefected ? 0 : 1),
  },
  {
    id: 'pavlov',
    name: 'Pavlov',
    short: 'Win-stay, lose-shift: repeat when the last round matched, switch when it clashed.',
    opening: 'C',
    decide: ({ myLast, oppLast }) => (myLast === oppLast ? 'C' : 'D'),
    pCooperate: (myLast, oppLast) => (myLast === oppLast ? 1 : 0),
  },
  {
    id: 'random',
    name: 'Random',
    short: 'Flip a coin every round.',
    opening: 'C',
    decide: ({ rng }) => (rng() < 0.5 ? 'C' : 'D'),
    pCooperate: () => 0.5,
  },
];

export const STRAT_BY_ID: Record<string, Strategy> = Object.fromEntries(
  STRATEGIES.map((s) => [s.id, s]),
);

// Keep the deterministic `det` helper referenced (and the two forms in sync) without
// duplicating logic: pCooperate is authored by hand above for clarity/perf.
void det;

/* ── Scoring ── */

export interface Payoffs {
  T: number; // temptation - defect on a cooperator
  R: number; // reward - mutual cooperation
  P: number; // punishment - mutual defection
  S: number; // sucker - cooperate against a defector
}

export const CLASSIC: Payoffs = { T: 5, R: 3, P: 1, S: 0 };

/** Payoff to the *row* player given both moves. */
export function payoff(mine: Move, opp: Move, p: Payoffs): number {
  if (mine === 'C') return opp === 'C' ? p.R : p.S;
  return opp === 'C' ? p.T : p.P;
}

/** The two Prisoner's-Dilemma conditions, named, for the matrix validator. */
export function validatePD(p: Payoffs): { ok: boolean; message: string } {
  const { T, R, P, S } = p;
  if (!(T > R))
    return {
      ok: false,
      message: 'Need T > R: defecting on a cooperator must beat mutual cooperation.',
    };
  if (!(R > P))
    return { ok: false, message: 'Need R > P: mutual cooperation must beat mutual defection.' };
  if (!(P > S))
    return { ok: false, message: 'Need P > S: mutual defection must beat being the lone sucker.' };
  if (!(2 * R > T + S))
    return {
      ok: false,
      message:
        'Need 2R > T + S: otherwise alternating exploitation pays more than cooperating — no longer a social dilemma.',
    };
  return { ok: true, message: 'Valid Prisoner’s Dilemma.' };
}
