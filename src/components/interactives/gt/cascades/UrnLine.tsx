import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Specimen,
  mulberry32,
  useInView,
  useReducedMotion,
} from '../shared';
import { crowdAnswer, drawBalls, play, pooledAnswer, type Colour, type Turn } from './urn';

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

// 1000 simulated lines (deterministic), computed once at module load so SSR has the numbers.
const RUNS = 1000;
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
    if (reduced || started.current || !inView) return;
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
  const visible = turns.slice(0, shown);
  const blueGuesses = visible.filter((t) => t.guess === 'blue').length;
  const redGuesses = visible.length - blueGuesses;
  const cascadeAt = turns.findIndex((t) => t.following);
  const blueBalls = balls.filter((b) => b === 'blue').length;
  const answer = crowdAnswer(turns);
  const wrong = answer !== urn;
  const herders = turns.filter((t) => t.following).length;
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
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Btn primary onClick={toggle}>
          {playing && !done ? 'pause' : done ? 'replay' : 'play'}
        </Btn>
        <Btn onClick={step} disabled={done}>
          step
        </Btn>
        <Btn onClick={fresh}>new urn</Btn>
        <span className="ml-auto flex flex-wrap gap-3">
          <Chip color={GT.a}>blue</Chip>
          <Chip color={GT.b}>red</Chip>
          <Chip color={GT.eq}>following the crowd</Chip>
        </span>
      </div>

      {/* the line of people */}
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
                borderColor: current ? GT.ink : on && t.following ? GT.eq : GT.line,
                borderStyle: on ? 'solid' : 'dashed',
                borderTopWidth: on && t.following ? 3 : 1,
              }}
            >
              <span className="font-mono text-[0.6rem] text-muted">
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
                <span className="font-mono text-[0.55rem] text-muted">
                  {on ? (ign ? 'ignored' : 'ball') : 'ball'}
                </span>
              </span>
            </div>
          );
        })}
      </div>

      {/* tally + narration */}
      <div className="mt-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 font-mono text-[0.7rem]">
        <span className="text-ink-soft">
          public guesses: <span style={{ color: GT.a }}>blue {blueGuesses}</span> ·{' '}
          <span style={{ color: GT.b }}>red {redGuesses}</span>
        </span>
        <span className="text-muted">
          {cascadeAt >= 0 && shown > cascadeAt
            ? `cascade from person ${cascadeAt + 1}: ${herders} people follow`
            : 'no cascade yet'}
        </span>
      </div>
      <Caption>
        {done
          ? wrong
            ? `The urn was ${urn}. ${ignored} people threw away a ${urn} ball to repeat the crowd.`
            : `The urn was ${urn}. The crowd got it right this time, for the same reason it can get it wrong: the first few balls decided.`
          : narrate(turns[shown - 1], shown - 1)}
      </Caption>

      {/* readouts */}
      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-line pt-4 sm:grid-cols-4">
        <BigStat
          label="balls in this line"
          value={
            <span>
              <span style={{ color: GT.a }}>{blueBalls}</span>
              <span className="text-muted">:</span>
              <span style={{ color: GT.b }}>{N - blueBalls}</span>
            </span>
          }
          sub="blue : red, all private"
        />
        <BigStat
          label="the crowd says"
          value={done ? answer : '…'}
          color={done ? col(answer) : GT.muted}
          sub={done ? `urn was ${urn}: ${wrong ? 'wrong' : 'right'}` : 'wait for the line'}
        />
        <BigStat
          label="crowd wrong"
          value={Math.round(SIM.crowd * 100)}
          unit="%"
          color={GT.eq}
          sub={`of ${RUNS} simulated lines`}
        />
        <BigStat
          label="pooling the balls"
          value={Math.round(SIM.pooled * 100)}
          unit="%"
          color={GT.opt}
          sub="wrong, by majority of 12"
        />
      </div>
      <span className="sr-only">
        Urn colour {urn}. Guesses: {turns.map((t) => t.guess).join(', ')}.
      </span>
    </Specimen>
  );
}
