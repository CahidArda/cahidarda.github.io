import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';

/**
 * CX-10.3 · proof search can't settle it. Turing 1936 §11: a machine that lists every proof
 * in order and checks each last line for "P halts" or "P never halts". Proofs ride a belt
 * past a checker. For Collatz-from-27 (halts after 111 steps) and an empty infinite loop it
 * stops on a match; for "hunt for a proof of 0 = 1" it never does: if the system is
 * consistent that program never halts, and Gödel's second theorem says the system cannot
 * prove it. Proof numbers are illustrative. Belt position s(t) is one continuous function of
 * one clock (accelerate, cruise, and for a found proof decelerate onto the match).
 */

type Prog = 'collatz' | 'loop' | 'hunt';
type Spec = {
  label: string;
  code: [string, string];
  match: number | null; // proof number that settles it
  verdict: 0 | 1; // 1 = halts, 0 = never halts
  caps: [string, string];
};

const SPECS: Record<Prog, Spec> = {
  collatz: {
    label: 'Collatz from 27',
    code: [
      'P:  n = 27; while n ≠ 1: halve n if even, else n = 3n + 1',
      'P: Collatz steps from n = 27',
    ],
    match: 7,
    verdict: 1,
    caps: [
      'The machine lists every proof in order and checks how each one ends. Here P runs the Collatz steps from 27.',
      'P halts after 111 steps, and writing out the run is a proof. Sooner or later the machine reaches it.',
    ],
  },
  loop: {
    label: 'empty loop',
    code: ['P:  while true: do nothing', 'P: while true: do nothing'],
    match: 4,
    verdict: 0,
    caps: [
      'This P spins forever. Is there a proof that says so?',
      'Yes: a short proof shows P never halts, and the machine finds it.',
    ],
  },
  hunt: {
    label: 'hunt for 0 = 1',
    code: ['P:  check proofs 1, 2, 3, …; halt if one proves 0 = 1', 'P: halt on a proof of 0 = 1'],
    match: null,
    verdict: 0,
    caps: [
      'This P searches the system’s own proofs for one that ends in 0 = 1. If the system is consistent, P never halts.',
      'A proof of “P never halts” would prove the system consistent, which Gödel’s second theorem rules out. The search never ends.',
    ],
  },
};
const ORDER: Prog[] = ['collatz', 'loop', 'hunt'];

// What other proofs end with (all true, none about P).
const OTHER = [
  '2 + 2 = 4',
  'x + 0 = x',
  '7 is prime',
  'n < n + 1',
  'a+b = b+a',
  '0 ≠ 1',
  '3 · 3 = 9',
  '√2 ∉ ℚ',
  'x · 1 = x',
  '13 is prime',
];

const V = 2.1; // cards per second while cruising
const TA = 0.5; // seconds to speed up or slow down
const DELAY = 0.4; // belt rests this long before it starts
const HOLD = 1.6;
const HUNT_SECOND = 5.5; // when the hunt caption turns to the conclusion

/** Belt position (in cards) at time t: rest, speed up, cruise, and for a match slow onto it. */
function beltPos(t: number, match: number | null): number {
  const u = t - DELAY;
  if (u <= 0) return 0;
  const acc = (x: number) => (V * x * x) / (2 * TA);
  if (match === null) return u < TA ? acc(u) : (V * TA) / 2 + V * (u - TA);
  const total = match / V + TA;
  if (u >= total) return match;
  if (u < TA) return acc(u);
  if (u > total - TA) return match - acc(total - u);
  return (V * TA) / 2 + V * (u - TA);
}
const foundAt = (m: number) => DELAY + m / V + TA;

export default function ProofSearch() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const W = compact ? 360 : 520;
  const LX = compact ? 6 : 20;
  const PITCH = compact ? 74 : 92;
  const CWD = compact ? 66 : 82;
  const CHT = 52;
  const BOX_T = 26;
  const BOX_H = 42;
  const BELT_Y = 112;
  const H = BELT_Y + CHT + 30;
  const CX = W / 2;
  const [svgRef, k] = useBoost(W, 10, 9);

  const [sel, setSel] = useState<Prog>('collatz');
  const [prog, bindRaw] = useHoverPreview(sel);
  const [t, setT] = useState(0);
  const tRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const auto = useRef(true);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const S = SPECS[prog];
  const end = S.match === null ? Infinity : foundAt(S.match) + 0.6;

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      auto.current = false;
      tRef.current = 30;
      setT(30);
      return;
    }
    timer.current = setTimeout(() => setPlaying(true), 400);
  }, [inView, reduced]);
  useEffect(() => {
    if (!reduced) return;
    auto.current = false;
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    tRef.current = 30;
    setT(30);
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const running = playing && inView && t < end;
  useFrame(
    (dt) => {
      const n = tRef.current + dt;
      tRef.current = n;
      setT(n);
      const i = ORDER.indexOf(sel);
      if (
        auto.current &&
        prog === sel &&
        SPECS[sel].match !== null &&
        n >= foundAt(SPECS[sel].match!) + HOLD
      ) {
        setSel(ORDER[i + 1]);
        tRef.current = 0;
        setT(0);
      }
    },
    running || (playing && inView && auto.current),
  );

  const takeOver = () => {
    auto.current = false;
  };
  const bind = (v: Prog) => {
    const b = bindRaw(v);
    return {
      ...b,
      onPointerEnter: (e: React.PointerEvent<Element>) => {
        if (e.pointerType !== 'touch') takeOver();
        b.onPointerEnter(e);
      },
    };
  };
  const choose = (v: Prog) => {
    takeOver();
    setSel(v);
    if (v !== sel) {
      // a new program: restart its search from proof 1
      if (timer.current) clearTimeout(timer.current);
      tRef.current = reduced ? 30 : 0;
      setT(tRef.current);
      setPlaying(!reduced);
    }
  };
  const replay = () => {
    takeOver();
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    tRef.current = reduced ? 30 : 0;
    setT(tRef.current);
    if (!reduced) timer.current = setTimeout(() => setPlaying(true), 1000);
  };

  const s = beltPos(t, S.match);
  const found = S.match !== null && t >= foundAt(S.match);
  const checked = Math.max(0, Math.floor(s + 1e-6));
  const capIdx =
    ORDER.indexOf(prog) * 2 + (found || (S.match === null && t >= HUNT_SECOND) ? 1 : 0);

  const fs = (compact ? 9.5 : 10) * k;
  const fsC = (compact ? 8.5 : 9.5) * k;
  const conclusion = (i: number): [string, string?] => {
    if (S.match === i) return S.verdict ? ['P halts'] : ['P never', 'halts'];
    return [OTHER[(i * 7 + 3) % OTHER.length]];
  };

  const boxes = [
    { x0: LX, x1: CX - 6, title: compact ? '“P halts”' : 'a proof of “P halts”', v: 1 },
    {
      x0: CX + 6,
      x1: W - LX,
      title: compact ? '“P never halts”' : 'a proof of “P never halts”',
      v: 0,
    },
  ];
  const linkP = found ? clamp((t - foundAt(S.match!)) / 0.4, 0, 1) : 0;

  // visible cards: those within the belt window
  const first = Math.max(1, Math.floor(s - CX / PITCH) - 1);
  const last = Math.ceil(s + CX / PITCH) + 1;
  const cards = [];
  for (let i = first; i <= last; i++) cards.push(i);

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-10.3"
      title="proof search can't settle it"
      kicker="Turing 1936, §11"
      hint="hover or tap a program · proof numbers illustrative"
      footer="assumes the system proves only true halting claims"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`A machine checks proofs one by one for “P halts” or “P never halts”. ${
            found ? `It finds proof ${S.match}.` : 'It is still searching.'
          }`}
        >
          <T x={LX} y={10} anchor="start" size={fs} color={GT.ink}>
            {S.code[compact ? 1 : 0]}
          </T>
          {/* the two things the machine is looking for */}
          {boxes.map((b) => {
            const hit = found && S.verdict === b.v;
            const c = b.v ? GT.opt : GT.b;
            const bx = (b.x0 + b.x1) / 2;
            return (
              <g key={b.v}>
                <path
                  d={`M${CX} ${BELT_Y - 8} V${BELT_Y - 22} H${bx} V${BOX_T + BOX_H}`}
                  style={{ fill: 'none', stroke: GT.lineStrong, strokeDasharray: '3 3' }}
                />
                {hit && (
                  <path
                    d={`M${CX} ${BELT_Y - 8} V${BELT_Y - 22} H${bx} V${BOX_T + BOX_H}`}
                    pathLength={1}
                    style={{
                      fill: 'none',
                      stroke: c,
                      strokeWidth: 2,
                      strokeDasharray: 1,
                      strokeDashoffset: 1 - linkP,
                    }}
                  />
                )}
                <rect
                  x={b.x0}
                  y={BOX_T}
                  width={b.x1 - b.x0}
                  height={BOX_H}
                  style={{
                    fill: hit ? (b.v ? GT.optSoft : GT.bSoft) : GT.paperRaised,
                    stroke: c,
                    strokeWidth: hit ? 2 : 1,
                    fillOpacity: hit ? linkP : 1,
                  }}
                />
                <T x={b.x0 + 10} y={BOX_T + 14} anchor="start" size={fs} color={c} weight={700}>
                  {b.title}
                </T>
                <T
                  x={b.x0 + 10}
                  y={BOX_T + 30}
                  anchor="start"
                  size={fs}
                  color={hit ? c : GT.muted}
                  weight={hit ? 700 : 400}
                >
                  {hit ? `found: proof #${S.match}` : found ? 'not needed' : 'not found yet'}
                </T>
              </g>
            );
          })}

          {/* the belt of proofs */}
          <line x1={0} x2={W} y1={BELT_Y - 6} y2={BELT_Y - 6} style={{ stroke: GT.line }} />
          <line
            x1={0}
            x2={W}
            y1={BELT_Y + CHT + 6}
            y2={BELT_Y + CHT + 6}
            style={{ stroke: GT.line }}
          />
          <clipPath id="ps-belt">
            <rect x={0} y={BELT_Y - 4} width={W} height={CHT + 8} />
          </clipPath>
          <g clipPath="url(#ps-belt)">
            {cards.map((i) => {
              const x = CX + (i - s) * PITCH;
              const d = Math.abs(x - CX) / (W / 2);
              const op = clamp(1.15 - d, 0.12, 1);
              const isMatch = S.match === i;
              const done = i <= s + 1e-6;
              const lit = isMatch && found;
              const [l1, l2] = conclusion(i);
              const c = lit ? (S.verdict ? GT.opt : GT.b) : done ? GT.muted : GT.ink;
              return (
                <g key={i} opacity={op}>
                  <rect
                    x={x - CWD / 2}
                    y={BELT_Y}
                    width={CWD}
                    height={CHT}
                    style={{
                      fill: lit ? (S.verdict ? GT.optSoft : GT.bSoft) : GT.paper,
                      stroke: lit ? c : GT.lineStrong,
                      strokeWidth: lit ? 2 : 1,
                    }}
                  />
                  <T x={x - CWD / 2 + 7} y={BELT_Y + 11} anchor="start" size={fsC} color={GT.muted}>
                    {`proof #${i}`}
                  </T>
                  <line
                    x1={x - CWD / 2 + 7}
                    x2={x + CWD / 2 - 7}
                    y1={BELT_Y + 20}
                    y2={BELT_Y + 20}
                    style={{ stroke: GT.line }}
                  />
                  <T
                    x={x}
                    y={BELT_Y + (l2 ? 30 : 36)}
                    size={fsC}
                    color={c}
                    weight={lit ? 700 : 400}
                  >
                    {l1}
                  </T>
                  {l2 && (
                    <T x={x} y={BELT_Y + 42} size={fsC} color={c} weight={lit ? 700 : 400}>
                      {l2}
                    </T>
                  )}
                </g>
              );
            })}
          </g>
          {/* the checker window */}
          <rect
            x={CX - CWD / 2 - 5}
            y={BELT_Y - 8}
            width={CWD + 10}
            height={CHT + 16}
            style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
          />
          <T x={CX} y={BELT_Y + CHT + 20} size={fs} color={GT.eq} weight={700}>
            checker
          </T>
          <T x={LX} y={BELT_Y + CHT + 20} anchor="start" size={fs}>
            {'← checked'}
          </T>
          <T x={W - LX} y={BELT_Y + CHT + 20} anchor="end" size={fs}>
            {'next →'}
          </T>
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
        <div className="flex gap-6">
          <BigStat label="proofs checked" value={checked.toLocaleString('en-US')} size="sm" />
          <div className="w-[9.5rem]">
            <BigStat
              label="settled?"
              value={found ? (S.verdict ? 'halts' : 'never halts') : 'not yet'}
              size="sm"
              color={found ? (S.verdict ? GT.opt : GT.b) : GT.muted}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Seg
            options={ORDER.map((p) => ({ value: p, label: SPECS[p].label }))}
            value={prog}
            onChange={choose}
            bind={bind}
          />
          <Btn primary onClick={replay}>
            replay
          </Btn>
        </div>
      </div>
      <Reserve
        items={ORDER.flatMap((p) => SPECS[p].caps).map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
