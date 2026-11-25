/**
 * The urn experiment (Easley & Kleinberg, NCM 16.2-16.3). The urn is majority blue
 * (2 blue, 1 red) or majority red, 50/50. Each person draws one ball privately and
 * announces a guess. Everyone can see earlier guesses, not earlier balls.
 *
 * Bayesian play with the tie rule "when indifferent, follow your own ball": while the
 * public count of guesses is within one of even, each guess reveals that person's ball.
 * Once one colour leads by two, a single contrary ball cannot outweigh it, so every later
 * person repeats the leading colour whatever they drew (a cascade). Those guesses carry
 * no information, so the lead stays at two forever.
 */
export type Colour = 'blue' | 'red';

export interface Turn {
  ball: Colour;
  guess: Colour;
  /** true once a cascade is running: this person ignores their ball */
  following: boolean;
  /** public lead (blue minus red guesses) this person saw before deciding */
  leadBefore: number;
}

export function play(balls: Colour[]): Turn[] {
  let lead = 0;
  return balls.map((ball) => {
    const leadBefore = lead;
    if (Math.abs(lead) >= 2) {
      return { ball, guess: lead > 0 ? 'blue' : 'red', following: true, leadBefore };
    }
    lead += ball === 'blue' ? 1 : -1;
    return { ball, guess: ball, following: false, leadBefore };
  });
}

export function drawBalls(urn: Colour, n: number, rnd: () => number): Colour[] {
  const other: Colour = urn === 'blue' ? 'red' : 'blue';
  return Array.from({ length: n }, () => (rnd() < 2 / 3 ? urn : other));
}

/** The crowd's answer: the last person's public guess. */
export function crowdAnswer(turns: Turn[]): Colour {
  return turns[turns.length - 1].guess;
}

/** Majority of everybody's private balls (ties broken by a coin). */
export function pooledAnswer(balls: Colour[], rnd: () => number): Colour {
  const b = balls.filter((x) => x === 'blue').length;
  const r = balls.length - b;
  if (b === r) return rnd() < 0.5 ? 'blue' : 'red';
  return b > r ? 'blue' : 'red';
}
