import { useEffect, useRef, useState } from 'react';
import {
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { mergeSort, selectionSort, shuffled, type Run } from './model';

/**
 * CX-02.2 "sorting". The same 40 shuffled bars, sorted twice on one clock at the same number
 * of comparisons per second: selection sort (n(n-1)/2 = 780 comparisons) and merge sort (176
 * for this shuffle). Each item is one bar that glides toward its slot after every comparison;
 * two ticks under the
 * baseline mark the items being compared, in both panels. Replay resets and replays after 1 s.
 */
const N = 40;
const START = shuffled(N, 1);
const RUNS: { id: 'sel' | 'merge'; name: string; color: string; run: Run; formula: string }[] = [
  {
    id: 'sel',
    name: 'selection sort',
    color: GT.b,
    run: selectionSort(START),
    formula: 'n(n−1)/2',
  },
  {
    id: 'merge',
    name: 'merge sort',
    color: GT.opt,
    run: mergeSort(START),
    formula: 'at most n log₂ n',
  },
];
const RATE = 170; // comparisons per second, both algorithms
const SMOOTH = 18; // 1/s, how fast a bar catches up with its slot
const VMAX = 150; // slots per second, the fastest a bar may slide
const MARK_SMOOTH = 30; // 1/s, the comparison markers under the bars
const END_S = Math.max(...RUNS.map((r) => r.run.frames.length - 1)) / RATE;
/** unsorted bars: a solid mix of muted and paper, so overlapping bars never show through */
const REST = `color-mix(in srgb, ${GT.muted} 55%, ${GT.paper})`;
const r2 = (v: number) => Math.round(v * 100) / 100;

/** slot of each item id in a frame */
const slots = (frame: number[]) => {
  const s = new Array(N);
  frame.forEach((id, i) => (s[id] = i));
  return s as number[];
};
const FIRST = slots(START);
const firstMarks = () => RUNS.map((R) => R.run.cmp[0].slice() as [number, number]);

function Bars({
  pos,
  marks,
  cs,
  narrow,
}: {
  pos: number[][];
  marks: [number, number][];
  cs: number[];
  narrow: boolean;
}) {
  const W = narrow ? 360 : 640;
  const PH = narrow ? 64 : 80; // bar area height per panel
  const head = 26;
  const panel = head + PH + 22;
  const H = panel * RUNS.length - 8;
  const bw = W / N;
  const mw = Math.min(8, bw - 2); // marker width
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full"
      role="img"
      aria-label="Selection sort and merge sort sorting the same 40 bars"
    >
      {RUNS.map((R, ri) => {
        const y0 = ri * panel;
        const base = y0 + head + PH;
        const total = R.run.frames.length - 1;
        const c = Math.min(cs[ri], total);
        const done = R.run.done[c];
        const finished = c >= total;
        const live = !finished && c > 0;
        const minId = live && R.run.min ? R.run.min[c] : -1;
        const target = slots(R.run.frames[c]);
        return (
          <g key={R.id}>
            <T x={0} y={y0 + 9} anchor="start" size={11} color={R.color} weight={700}>
              {R.name}
            </T>
            <T x={W} y={y0 + 9} anchor="end" size={11} color={GT.inkSoft}>
              {`${c} comparisons${finished ? ' · sorted' : ''}`}
            </T>
            {/* every item is drawn exactly once: one opaque bar, gliding to its slot; bars in
                motion are drawn last so a swap slides in front instead of showing through */}
            {pos[ri]
              .map((p, id) => ({ p, id, moving: Math.abs(p - Math.round(p)) > 0.01 }))
              .sort((u, v) => Number(u.moving) - Number(v.moving) || u.id - v.id)
              .map(({ p, id }) => {
                const h = ((id + 1) / N) * (PH - 4);
                // a bar takes its colour once it has arrived in its final slot
                const fixed = Math.abs(p - target[id]) < 0.5 && target[id] < done;
                return (
                  <rect
                    key={id}
                    data-item={id}
                    x={r2(p * bw + 1)}
                    y={r2(base - h)}
                    width={r2(bw - 2)}
                    height={r2(h)}
                    style={{ fill: id === minId ? GT.ink : fixed ? R.color : REST }}
                  />
                );
              })}
            <line x1={0} x2={W} y1={base + 0.5} y2={base + 0.5} style={{ stroke: GT.lineStrong }} />
            {/* the two items being compared: small ticks under the baseline, same in both panels */}
            {live &&
              marks[ri].map((m, k) => {
                const cx = r2((m + 0.5) * bw);
                return (
                  <path
                    key={k}
                    data-mark={k}
                    d={`M${cx} ${base + 3} l${r2(mw / 2)} 5 h${r2(-mw)} Z`}
                    style={{ fill: R.color }}
                  />
                );
              })}
            {/* progress: share of this algorithm's comparisons done */}
            <rect
              x={0}
              y={base + 12}
              width={r2(W * (c / total))}
              height={3}
              style={{ fill: R.color }}
            />
          </g>
        );
      })}
    </svg>
  );
}

export default function SortRace() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [e, setE] = useState(0);
  const eRef = useRef(0);
  const pos = useRef<number[][]>(RUNS.map(() => FIRST.slice()));
  // the two compared slots per panel, glided like the bars
  const marks = useRef<[number, number][]>(firstMarks());
  const [, setFrame] = useState(0);
  const [running, setRunning] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const finish = () => {
    eRef.current = END_S;
    setE(END_S);
    pos.current = RUNS.map((R) => slots(R.run.frames[R.run.frames.length - 1]));
    setRunning(false);
  };
  const restart = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    eRef.current = 0;
    setE(0);
    pos.current = RUNS.map(() => FIRST.slice());
    marks.current = firstMarks();
    setRunning(false);
    if (reduced) return finish();
    timer.current = setTimeout(() => setRunning(true), delay);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) finish();
    else timer.current = setTimeout(() => setRunning(true), 500);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // one clock: comparisons advance with time; bars and markers ease toward their slots
  useFrame((dt) => {
    const t = Math.min(END_S + 1, eRef.current + Math.max(0, dt));
    eRef.current = t;
    let moving = t < END_S;
    const a = Math.min(1, Math.max(0, dt) * SMOOTH);
    const vmax = VMAX * Math.max(0, dt);
    const k = Math.min(1, Math.max(0, dt) * MARK_SMOOTH);
    RUNS.forEach((R, ri) => {
      const last = R.run.frames.length - 1;
      const cf = Math.min(t * RATE, last);
      const c = Math.floor(cf);
      const target = slots(R.run.frames[c]);
      const p = pos.current[ri];
      for (let i = 0; i < N; i++) {
        const d = target[i] - p[i];
        if (Math.abs(d) < 0.002) p[i] = target[i];
        else {
          // ease in, but never faster than VMAX: a long swap slides instead of leaping
          p[i] += Math.sign(d) * Math.min(Math.abs(d * a), vmax);
          moving = true;
        }
      }
      // a marker that steps one slot per comparison aims at the fractional slot, so it sweeps
      const here = R.run.cmp[c];
      const next = R.run.cmp[Math.min(c + 1, last)];
      const m = marks.current[ri];
      for (let j = 0; j < 2; j++) {
        const aim = next[j] === here[j] + 1 ? here[j] + (cf - c) : here[j];
        m[j] += (aim - m[j]) * k;
      }
      // selection sort: the second marker sits under the smallest-so-far bar itself
      if (R.run.min) m[1] = p[R.run.min[c]];
    });
    setE(t);
    setFrame((f) => f + 1);
    if (!moving) setRunning(false);
  }, running && inView);

  const cs = RUNS.map((R) => Math.min(Math.floor(e * RATE), R.run.frames.length - 1));

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-02.2"
      title="sorting"
      kicker={`${N} items · ${RATE} comparisons a second each`}
      hint="ticks: the two items being compared · ink bar: smallest so far · replay to race again"
    >
      {/* one layout per breakpoint (CSS picks it, so hydrating never resizes the figure) */}
      <div className="hidden sm:block">
        <Bars pos={pos.current} marks={marks.current} cs={cs} narrow={false} />
      </div>
      <div className="sm:hidden">
        <Bars pos={pos.current} marks={marks.current} cs={cs} narrow />
      </div>
      <div className="mt-3 flex items-end justify-between gap-4 border-t border-line pt-3">
        <div className="grid grid-cols-[auto_auto] gap-x-4 gap-y-1 font-mono text-[0.72rem]">
          {RUNS.map((R) => (
            <div key={R.id} className="contents">
              <span style={{ color: R.color }}>{R.name}</span>
              <span className="text-muted">
                {R.run.frames.length - 1} comparisons · {R.formula}
              </span>
            </div>
          ))}
        </div>
        <Btn primary onClick={() => restart(1000)}>
          replay
        </Btn>
      </div>
      <Caption>
        With a billion items, selection sort needs 5 × 10¹⁷ comparisons, 16 years at a billion a
        second. Merge sort needs at most 3 × 10¹⁰: 30 seconds.
      </Caption>
    </Specimen>
  );
}
