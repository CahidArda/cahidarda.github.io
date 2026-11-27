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
 * for this shuffle). Bars glide toward their slot after every comparison, so motion is
 * continuous. Replay resets and plays again after a second.
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
const SCAN_SMOOTH = 30; // 1/s, the scan band
const END_S = Math.max(...RUNS.map((r) => r.run.frames.length - 1)) / RATE;
const r2 = (v: number) => Math.round(v * 100) / 100;

/** slot of each item id in a frame */
const slots = (frame: number[]) => {
  const s = new Array(N);
  frame.forEach((id, i) => (s[id] = i));
  return s as number[];
};
const FIRST = slots(START);

function Bars({
  pos,
  cs,
  scan,
  narrow,
}: {
  pos: number[][];
  cs: number[];
  scan: { x: number };
  narrow: boolean;
}) {
  const W = narrow ? 360 : 640;
  const L = 0;
  const PH = narrow ? 64 : 80; // bar area height per panel
  const head = 26;
  const panel = head + PH + 18;
  const H = panel * RUNS.length - 10;
  const bw = (W - L) / N;
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
        return (
          <g key={R.id}>
            <T x={0} y={y0 + 9} anchor="start" size={11} color={R.color} weight={700}>
              {R.name}
            </T>
            <T x={W} y={y0 + 9} anchor="end" size={11} color={GT.inkSoft}>
              {`${c} comparisons${finished ? ' · sorted' : ''}`}
            </T>
            <line x1={0} x2={W} y1={base + 0.5} y2={base + 0.5} style={{ stroke: GT.lineStrong }} />
            {/* selection sort's scan: a band on the slot being compared (behind the bars) */}
            {R.run.scan && !finished && c > 0 && (
              <rect
                x={r2(L + scan.x * bw)}
                y={y0 + head - 4}
                width={r2(bw)}
                height={PH + 4}
                style={{ fill: R.color, opacity: 0.16 }}
              />
            )}
            {pos[ri].map((p, id) => {
              const h = ((id + 1) / N) * (PH - 4);
              const fixed = finished || Math.round(p) < done;
              // selection sort's smallest-so-far is drawn in ink, so the mark is the bar itself
              const isMin = !!R.run.scan && !finished && c > 0 && R.run.scan[c].min === id;
              return (
                <rect
                  key={id}
                  x={r2(L + p * bw + 1)}
                  y={r2(base - h)}
                  width={r2(bw - 2)}
                  height={r2(h)}
                  style={{
                    fill: isMin ? GT.ink : fixed ? R.color : GT.muted,
                    opacity: isMin ? 0.85 : fixed ? 0.9 : 0.55,
                  }}
                />
              );
            })}
            {/* progress: share of this algorithm's comparisons done */}
            <rect
              x={0}
              y={base + 6}
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
  // selection sort's scan band, in slots, glided like the bars
  const scan = useRef({ x: 0 });
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
    scan.current = { x: 0 };
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

  // one clock: comparisons advance with time; bars ease toward their current slots
  useFrame((dt) => {
    const t = Math.min(END_S + 1, eRef.current + Math.max(0, dt));
    eRef.current = t;
    let moving = t < END_S;
    RUNS.forEach((R, ri) => {
      const c = Math.min(Math.floor(t * RATE), R.run.frames.length - 1);
      const target = slots(R.run.frames[c]);
      const p = pos.current[ri];
      const a = Math.min(1, Math.max(0, dt) * SMOOTH);
      for (let i = 0; i < N; i++) {
        const d = target[i] - p[i];
        if (Math.abs(d) < 0.002) p[i] = target[i];
        else {
          p[i] += d * a;
          moving = true;
        }
      }
    });
    // the scan advances one slot per comparison: aim at the fractional slot, then glide
    const sr = RUNS[0].run;
    if (sr.scan) {
      const last = sr.frames.length - 1;
      const cf = Math.min(t * RATE, last);
      const c = Math.floor(cf);
      const here = sr.scan[c];
      const next = sr.scan[Math.min(c + 1, last)];
      const aim = next.j === here.j + 1 ? here.j + (cf - c) : here.j;
      const k = Math.min(1, Math.max(0, dt) * SCAN_SMOOTH);
      scan.current.x += (aim - scan.current.x) * k;
    }
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
      hint="band: the slot being compared · solid bar: smallest so far · replay to race again"
    >
      <div className="hidden sm:block">
        <Bars pos={pos.current} cs={cs} scan={scan.current} narrow={false} />
      </div>
      <div className="sm:hidden">
        <Bars pos={pos.current} cs={cs} scan={scan.current} narrow />
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
