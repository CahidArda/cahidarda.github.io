import { useEffect, useMemo, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { bestResponse, shares } from './logic';
import { useCompact } from './useCompact';
import { Reserve } from '../nash/Reserve';

const H = 132;
const SAND = 80;
const GRIDSTEP = 0.05;
const START = [0.2, 0.5, 0.8];
const TURNS = 48;
const TURN_MS = 1100;
const NAMES = ['A', 'B', 'C'];
const COLORS = [GT.a, GT.b, GT.alt];

/** Precompute the best-response sequence: carts move in turn A, B, C, A, ... */
function sequence() {
  const states = [START.slice()];
  for (let k = 0; k < TURNS; k++) {
    const who = k % 3;
    const cur = states[states.length - 1].slice();
    cur[who] = Math.round(bestResponse(cur, who, GRIDSTEP) * 100) / 100;
    states.push(cur);
  }
  return states;
}

const pct = (s: number) => `${Math.round(s * 100)}`;

export default function ThreeCarts() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact();
  const W = compact ? 400 : 640;
  const X0 = compact ? 14 : 30;
  const X1 = W - X0;
  const fs = compact ? 12 : 10.5;
  const px = (p: number) => Math.round((X0 + p * (X1 - X0)) * 100) / 100;
  const states = useMemo(sequence, []);
  // SSR / poster: a moment mid-cycle where one cart is squeezed in the middle
  const [k, setK] = useState(2);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(1); // hop progress for the cart that moved into state k
  const [started, setStarted] = useState(false);
  const [moves, setMoves] = useState(2);

  useEffect(() => {
    if (reduced) {
      // static poster: a mid-cycle moment with one cart squeezed in the middle
      setPlaying(false);
      setStarted(true);
      return;
    }
    if (started || !inView) return;
    const id = setTimeout(() => {
      setStarted(true);
      setK(0);
      setMoves(0);
      setPlaying(true);
    }, 500);
    return () => clearTimeout(id);
  }, [inView, reduced, started]);

  useEffect(() => {
    if (!playing || !inView) return;
    const id = setTimeout(() => {
      // loop inside the cycle once it is reached (states repeat every 12 moves from move 2 on, so the move after the last one is move 13)
      setK((x) => (x + 1 >= states.length ? 13 : x + 1));
      setMoves((m) => m + 1);
      setT(reduced ? 1 : 0);
    }, TURN_MS);
    return () => clearTimeout(id);
  }, [playing, k, inView, states.length, reduced]);

  useFrame((dt) => setT((x) => Math.min(1, x + dt / 0.55)), !reduced && t < 1);

  const step = () => {
    setPlaying(false);
    setK((x) => (x + 1 >= states.length ? 13 : x + 1));
    setMoves((m) => m + 1);
    setT(reduced ? 1 : 0);
  };

  const cur = states[k];
  const prev = k > 0 ? states[k - 1] : cur;
  const mover = k > 0 ? (k - 1) % 3 : -1;
  const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  const shown = cur.map((p, i) => (i === mover ? prev[i] + (p - prev[i]) * e : p));
  const lift = (i: number) =>
    i === mover
      ? Math.sin(Math.PI * e) * Math.min(46, Math.abs(px(cur[i]) - px(prev[i])) * 0.9)
      : 0;
  const sh = shares(cur);
  const low = sh.indexOf(Math.min(...sh));

  // territory bands under the sand
  const order = [0, 1, 2].sort((a, b) => cur[a] - cur[b]);
  const bounds = order.map((i, j) => {
    const lo = j === 0 ? 0 : (cur[order[j - 1]] + cur[i]) / 2;
    const hi = j === 2 ? 1 : (cur[i] + cur[order[j + 1]]) / 2;
    return { i, lo, hi };
  });

  let note = 'Three carts start evenly spaced.';
  if (mover >= 0) {
    const jumped =
      Math.sign(prev[mover] - prev[(mover + 1) % 3]) !==
        Math.sign(cur[mover] - cur[(mover + 1) % 3]) ||
      Math.sign(prev[mover] - prev[(mover + 2) % 3]) !==
        Math.sign(cur[mover] - cur[(mover + 2) % 3]);
    note = jumped
      ? `${NAMES[mover]} was squeezed between two rivals, so it leaps over one. Now it has ${pct(sh[mover])}%.`
      : `${NAMES[mover]} slides up against a rival and now has ${pct(sh[mover])}%.`;
  }

  return (
    <Specimen
      code="GT-03.3"
      title="three carts"
      kicker="no resting point"
      rootRef={rootRef}
      hint="press step to move one cart at a time"
    >
      <div ref={boxRef}>
        <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={note}>
          {bounds.map(({ i, lo, hi }) => (
            <rect
              key={i}
              x={px(lo)}
              y={SAND}
              width={px(hi) - px(lo)}
              height={12}
              style={{ fill: COLORS[i], opacity: 0.35 }}
            />
          ))}
          <line x1={X0} y1={SAND} x2={X1} y2={SAND} style={{ stroke: GT.ink, strokeWidth: 1.2 }} />
          {[0, 1, 2].map((i) => {
            const x = px(shown[i]);
            const y = SAND - lift(i);
            return (
              <g key={i}>
                <rect
                  x={x - 9}
                  y={y - 22}
                  width={18}
                  height={18}
                  style={{ fill: GT.paperRaised, stroke: COLORS[i], strokeWidth: 2 }}
                />
                <text
                  x={x}
                  y={y - 12.5}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ fill: COLORS[i], fontFamily: GT.display, fontSize: 12, fontWeight: 600 }}
                >
                  {NAMES[i]}
                </text>
                <circle cx={x - 5} cy={y - 2} r={2.5} style={{ fill: GT.ink }} />
                <circle cx={x + 5} cy={y - 2} r={2.5} style={{ fill: GT.ink }} />
              </g>
            );
          })}
          {bounds.map(({ i, lo, hi }) =>
            px(hi) - px(lo) > 26 ? (
              <T key={`s${i}`} x={(px(lo) + px(hi)) / 2} y={SAND + 26} color={COLORS[i]} size={fs}>
                {pct(sh[i])}%
              </T>
            ) : null,
          )}
          <T x={X0} y={SAND + 44} anchor="start" size={fs - 1}>
            0
          </T>
          <T x={X1} y={SAND + 44} anchor="end" size={fs - 1}>
            1 km
          </T>
        </svg>
      </div>
      <div className="mt-4 grid grid-cols-[7rem_minmax(0,1fr)] items-end gap-x-6 gap-y-4 sm:grid-cols-[7rem_10rem_minmax(0,1fr)]">
        <BigStat
          label="squeezed"
          value={pct(sh[low])}
          unit="%"
          color={COLORS[low]}
          sub={`cart ${NAMES[low]}`}
        />
        <BigStat label="moves, no rest" value={moves} color={GT.eq} sub="it never settles" />
        <div className="col-span-2 flex gap-2 sm:col-span-1 sm:justify-end">
          <Btn primary onClick={() => setPlaying((p) => !p)}>
            {playing ? 'pause' : 'play'}
          </Btn>
          <Btn onClick={step}>step</Btn>
        </div>
      </div>
      <Reserve
        items={[
          <Caption key="l">
            A was squeezed between two rivals, so it leaps over one. Now it has 48%.
          </Caption>,
          <Caption key="n">{note}</Caption>,
        ]}
        active={1}
      />
    </Specimen>
  );
}
