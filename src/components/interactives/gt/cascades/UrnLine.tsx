import { useEffect, useRef, useState } from 'react';
import { BigStat, Btn, GT, Specimen, mulberry32, useInView, useReducedMotion } from '../shared';
import { crowdAnswer, drawBalls, play, pooledAnswer, type Colour, type Turn } from './urn';
import { prefersReduced } from './motion';

// NCM 16.2: the urn experiment. Twelve people in a line, one private ball each, one public
// guess each. The first line is scripted: the urn is blue but the first two people draw
// red, and the whole line herds on red while 8 of the 12 balls are blue.

const N = 12;
const STEP_MS = 850;
const SCRIPT: { urn: Colour; balls: Colour[] } = {
  urn: 'blue',
  balls: [
    'red',
    'red',
    'blue',
    'blue',
    'red',
    'blue',
    'blue',
    'blue',
    'blue',
    'red',
    'blue',
    'blue',
  ],
};

// 10,000 simulated lines (deterministic), computed once at module load so SSR has the numbers.
const RUNS = 10000;
const SIM = (() => {
  const rnd = mulberry32(2010);
  let crowd = 0;
  let pooled = 0;
  for (let i = 0; i < RUNS; i++) {
    const urn: Colour = rnd() < 0.5 ? 'blue' : 'red';
    const balls = drawBalls(urn, N, rnd);
    if (crowdAnswer(play(balls)) !== urn) crowd++;
    if (pooledAnswer(balls, rnd) !== urn) pooled++;
  }
  return { crowd: crowd / RUNS, pooled: pooled / RUNS };
})();

const col = (c: Colour) => (c === 'blue' ? GT.a : GT.b);

function lineFor(run: number) {
  if (run === 0) return SCRIPT;
  const rnd = mulberry32(run * 9973 + 17);
  const urn: Colour = rnd() < 0.5 ? 'blue' : 'red';
  return { urn, balls: drawBalls(urn, N, rnd) };
}

function narrate(t: Turn | undefined, i: number): string {
  if (!t) return 'Press play. Each person sees every earlier guess, never an earlier ball.';
  const who = `Person ${i + 1}`;
  const lead = t.leadBefore;
  if (!t.following) {
    if (lead === 0) return `${who} sees a tied line, so their own ball decides: says ${t.guess}.`;
    const side = lead > 0 ? 'blue' : 'red';
    if (t.ball === side)
      return `${who} sees ${side} ahead by 1 and drew ${t.ball}: says ${t.guess}.`;
    return `${who} sees ${side} ahead by 1 but drew ${t.ball}: a tie in evidence, so they trust their ball.`;
  }
  if (t.ball !== t.guess)
    return `${who} drew ${t.ball}, but ${t.guess} leads by 2. One ball cannot outweigh two: says ${t.guess}.`;
  return `${who} drew ${t.ball} and ${t.guess} leads by 2. They would say ${t.guess} whatever they drew.`;
}

export default function UrnLine() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [run, setRun] = useState(0);
  const [shown, setShown] = useState(N); // SSR / reduced motion: the finished line
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);

  const { urn, balls } = lineFor(run);
  const turns = play(balls);

  // Auto-play the scripted line once, when it scrolls into view.
  useEffect(() => {
    if (reduced || prefersReduced() || started.current || !inView) return;
    started.current = true;
    setShown(0);
    setPlaying(true);
  }, [reduced, inView]);

  useEffect(() => {
    if (!playing || !inView) return;
    if (shown >= N) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setShown((s) => s + 1), shown === 0 ? 500 : STEP_MS);
    return () => clearTimeout(id);
  }, [playing, shown, inView]);

  const step = () => {
    setPlaying(false);
    setShown((s) => Math.min(N, s + 1));
  };
  const toggle = () => {
    if (shown >= N) setShown(0);
    setPlaying((p) => !(p && shown < N));
  };
  const fresh = () => {
    setRun((r) => r + 1);
    setShown(0);
    setPlaying(!reduced);
    if (reduced) setShown(N);
  };

  const done = shown >= N;
  const blueBalls = balls.filter((b) => b === 'blue').length;
  const answer = crowdAnswer(turns);
  const wrong = answer !== urn;
  const ignored = turns.filter((t) => t.following && t.ball !== t.guess).length;

  return (
    <Specimen
      code="GT-08.1"
      title="the urn"
      kicker="NCM 16.2 · 2 of one colour, 1 of the other"
      rootRef={rootRef}
      hint="play the line, then draw a new urn"
      footer={run === 0 ? 'line 1 (the urn is blue)' : `line ${run + 1}`}
    >
      {/* the line of people: the chart, first */}
      <div className="grid grid-cols-6 gap-1.5 sm:grid-cols-12">
        {turns.map((t, i) => {
          const on = i < shown;
          const current = on && i === shown - 1 && !done;
          const ign = t.following && t.ball !== t.guess;
          return (
            <div
              key={`${run}-${i}`}
              className="relative flex flex-col items-center border bg-paper-raised px-0.5 pb-1.5 pt-1 transition-colors"
              style={{
                borderColor: current ? GT.ink : on && t.following ? GT.warn : GT.line,
                borderStyle: on ? 'solid' : 'dashed',
                // constant box size: the "following" band is an inset shadow, not a thicker border
                boxShadow: on && t.following ? `inset 0 3px 0 ${GT.warn}` : 'none',
              }}
            >
              <span className="font-mono text-[0.65rem] text-muted">
                {String(i + 1).padStart(2, '0')}
              </span>
              {/* public guess: big */}
              <span
                className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-full font-display text-base leading-none transition-all duration-300"
                style={{
                  background: on ? col(t.guess) : 'transparent',
                  border: on ? 'none' : `1px dashed ${GT.lineStrong}`,
                  color: on ? GT.paper : GT.muted,
                  transform: on ? 'scale(1)' : 'scale(0.8)',
                }}
                aria-label={on ? `says ${t.guess}` : 'not yet'}
              >
                {on ? (t.guess === 'blue' ? 'B' : 'R') : '?'}
              </span>
              {/* private ball: small */}
              <span className="mt-1.5 flex items-center gap-1">
                <span
                  className="relative inline-block h-2.5 w-2.5 rounded-full"
                  style={{
                    background: on ? col(t.ball) : 'transparent',
                    border: on ? 'none' : `1px dashed ${GT.line}`,
                    opacity: on && ign ? 0.45 : 1,
                  }}
                >
                  {on && ign && (
                    <span
                      className="absolute left-[-3px] top-1/2 block h-px w-4 -rotate-45"
                      style={{ background: GT.ink }}
                    />
                  )}
                </span>
                <span className="font-mono text-[0.62rem] text-muted">
                  {on && ign ? 'ignored' : 'ball'}
                </span>
              </span>
            </div>
          );
        })}
      </div>
      {/* direct key under the line, then the running story (fixed height: no reflow) */}
      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 font-mono text-[0.68rem] text-muted">
        <span>
          big = said aloud · small = drawn in private ·{' '}
          <span style={{ color: GT.warn }}>gold = following</span>
        </span>
        <span className="tabular-nums">
          balls <span style={{ color: GT.a }}>{blueBalls} blue</span> :{' '}
          <span style={{ color: GT.b }}>{N - blueBalls} red</span>
        </span>
      </div>
      <p className="mt-3 h-[4.2rem] overflow-hidden text-[0.85rem] leading-snug text-ink-soft sm:h-[2.6rem]">
        {done
          ? wrong
            ? `The urn was ${urn}. The crowd said ${answer}: ${ignored} people threw away a ${urn} ball to repeat it.`
            : `The urn was ${urn} and the crowd got it right, for the same reason it can get it wrong: the first few balls decided.`
          : narrate(turns[shown - 1], shown - 1)}
      </p>

      {/* one supporting readout: over many lines */}
      <div className="mt-3 flex flex-wrap items-end justify-between gap-x-6 gap-y-3 border-t border-line pt-4">
        <div className="flex gap-8">
          <BigStat
            label="crowd wrong"
            value={Math.round(SIM.crowd * 100)}
            unit="%"
            color={GT.warn}
            sub="of 10,000 lines"
          />
          <BigStat
            label="pooled balls wrong"
            value={Math.round(SIM.pooled * 100)}
            unit="%"
            color={GT.opt}
            sub="majority of all 12"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <Btn primary onClick={toggle}>
            <span className="inline-block w-[6ch] text-center">
              {playing && !done ? 'pause' : done ? 'replay' : 'play'}
            </span>
          </Btn>
          <Btn onClick={step} disabled={done}>
            step
          </Btn>
          <Btn onClick={fresh}>new urn</Btn>
        </div>
      </div>
      <span className="sr-only">
        Urn colour {urn}. Guesses: {turns.map((t) => t.guess).join(', ')}.
      </span>
    </Specimen>
  );
}
