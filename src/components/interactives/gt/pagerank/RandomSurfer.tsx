import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Seg,
  Slider,
  Specimen,
  SvgArrows,
  T,
  mulberry32,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { useBoost } from '../braess/useBoost';
import { circleExit, control, qAt, type Pt } from './graph';
import { outLists, pagerank, type Edge } from './model';

/* A small web. A is linked by every page on the left; B is linked by only A and C,
   yet B ends up on top because A, the most important page, gives it half its rank. */
const NAMES = 'ABCDEFGH'.split('');
const POS: Pt[] = [
  { x: 175, y: 150 }, // A
  { x: 278, y: 208 }, // B
  { x: 278, y: 92 }, // C
  { x: 156, y: 38 }, // D
  { x: 76, y: 92 }, // E
  { x: 58, y: 150 }, // F
  { x: 76, y: 208 }, // G
  { x: 156, y: 262 }, // H
];
const id = (c: string) => NAMES.indexOf(c);
export const EDGES: Edge[] = 'AB BA BC CB CD DE EF FG GH DA EA FA GA HA'
  .split(' ')
  .map((p) => [id(p[0]), id(p[1])]);
const N = NAMES.length;
const OUT = outLists(N, EDGES);
const VW = 340;
const VH = 300;
const radius = (share: number) => 9 + 34 * Math.sqrt(share);

type Speed = 'walk' | 'run' | 'sprint';
const RATE: Record<Speed, number> = { walk: 1.8, run: 24, sprint: 4000 };
const MAX_HOPS = 200000;

type Hop = { from: number; to: number; teleport: boolean; age: number };

export default function RandomSurfer() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [svgRef, k] = useBoost(VW);
  const [s, setS] = useState(0.85);
  const exact = useMemo(() => pagerank(N, EDGES, s), [s]);
  const [speed, setSpeed] = useState<Speed>('walk');
  const [playing, setPlaying] = useState(false);
  const counts = useRef<number[]>(new Array(N).fill(0));
  const total = useRef(0);
  const cur = useRef(0);
  const hop = useRef<Hop | null>(null);
  const acc = useRef(0);
  const rng = useRef(mulberry32(42));
  const touched = useRef(false);
  const [, setTick] = useState(0);
  const sRef = useRef(s);
  sRef.current = s;

  const doHop = () => {
    const from = cur.current;
    const r = rng.current;
    let to: number;
    let teleport = false;
    if (r() < sRef.current) {
      const o = OUT[from];
      to = o[Math.floor(r() * o.length)];
    } else {
      teleport = true;
      to = Math.floor(r() * N);
    }
    cur.current = to;
    counts.current[to]++;
    total.current++;
    hop.current = { from, to, teleport, age: 0 };
  };

  const resetCounts = () => {
    counts.current = new Array(N).fill(0);
    total.current = 0;
    hop.current = null;
    acc.current = 0;
    rng.current = mulberry32(42);
    cur.current = 0;
  };

  // reduced motion: a static poster of a long walk
  useEffect(() => {
    if (!reduced) return;
    resetCounts();
    for (let i = 0; i < 50000; i++) doHop();
    hop.current = null;
    setTick((t) => t + 1);
  }, [reduced, s]);

  // auto-play once on first view
  useEffect(() => {
    if (!inView || reduced || touched.current || total.current > 0) return;
    const t = setTimeout(() => setPlaying(true), 700);
    return () => clearTimeout(t);
  }, [inView, reduced]);

  useFrame((dt) => {
    // the demo speeds itself up: a few visible hops, then a blur, then a sprint
    if (!touched.current) {
      if (speed === 'walk' && total.current >= 7) setSpeed('run');
      if (speed === 'run' && total.current >= 90) setSpeed('sprint');
    }
    acc.current += dt * RATE[speed];
    let n = Math.floor(acc.current);
    acc.current -= n;
    if (speed === 'walk' && hop.current && hop.current.age < 0.5) n = 0;
    while (n-- > 0) doHop();
    if (hop.current) hop.current.age += dt;
    if (total.current >= MAX_HOPS) setPlaying(false);
    setTick((t) => (t + 1) % 1e6);
  }, playing && inView);

  const touch = () => {
    touched.current = true;
  };

  const tot = total.current;
  const share = counts.current.map((c) => (tot ? c / tot : 1 / N));
  const gap = tot ? 0.5 * share.reduce((g, v, i) => g + Math.abs(v - exact[i]), 0) : null;
  const R = share.map(radius);

  // the token: glides along the edge while walking, otherwise sits on the node
  const h = hop.current;
  let token: Pt = POS[cur.current];
  let tele: { a: Pt; b: Pt; o: number } | null = null;
  let lastEdge: [number, number] | null = null;
  if (h && speed !== 'sprint') {
    const tt = speed === 'walk' ? Math.min(1, h.age / 0.4) : 1;
    if (h.teleport) {
      tele = {
        a: POS[h.from],
        b: POS[h.to],
        o: Math.max(0, 1 - h.age / (speed === 'walk' ? 0.9 : 0.25)),
      };
      token = tt < 1 ? POS[h.from] : POS[h.to];
    } else {
      lastEdge = [h.from, h.to];
      const bend = OUT[h.to].includes(h.from) ? 14 : 0;
      const c = control(POS[h.from], POS[h.to], bend);
      token = qAt(POS[h.from], c, POS[h.to], tt);
    }
  }

  let caption: string;
  if (!tot) caption = 'A surfer starts on A and clicks a random link on each page.';
  else if (h?.teleport && speed === 'walk')
    caption = `Bored, the surfer jumps to a random page (${NAMES[h.to]}). This happens ${Math.round((1 - s) * 100)}% of the time.`;
  else if (speed === 'walk' && !reduced)
    caption = `The surfer follows a random link to ${NAMES[cur.current]}. Each visit is counted.`;
  else
    caption =
      'The share of time spent on each page settles on its PageRank. B wins with two in-links, because one of them is from A.';

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-06.1"
      title="the random surfer"
      kicker={`s = ${s.toFixed(2)}`}
      hint="press play, then change the speed or s"
      footer="Easley & Kleinberg, sec. 14.3"
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <Btn
          primary
          onClick={() => {
            touch();
            if (tot >= MAX_HOPS) resetCounts();
            setPlaying((p) => !p);
          }}
        >
          {playing ? 'pause' : 'play'}
        </Btn>
        <Seg
          options={[
            { value: 'walk', label: 'walk' },
            { value: 'run', label: 'run' },
            { value: 'sprint', label: 'sprint' },
          ]}
          value={speed}
          onChange={(v) => {
            touch();
            setSpeed(v);
            setPlaying(true);
          }}
        />
        <Btn
          onClick={() => {
            touch();
            resetCounts();
            setTick((t) => t + 1);
          }}
        >
          reset
        </Btn>
      </div>

      <div className="grid items-center gap-5 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VW} ${VH}`}
          className="block w-full"
          role="img"
          aria-label="Eight web pages with links between them and a surfer moving along the links"
        >
          <SvgArrows />
          {EDGES.map(([a, b], i) => {
            const bend = OUT[b].includes(a) ? 14 : 0;
            const c = control(POS[a], POS[b], bend);
            const p0 = circleExit(POS[a], R[a] + 2, c);
            const p1 = circleExit(POS[b], R[b] + 3, c);
            const on = lastEdge && lastEdge[0] === a && lastEdge[1] === b;
            return (
              <path
                key={i}
                d={`M${p0.x} ${p0.y} Q${c.x} ${c.y} ${p1.x} ${p1.y}`}
                markerEnd={`url(#gt-arrow-${on ? 'eq' : 'muted'})`}
                style={{
                  fill: 'none',
                  stroke: on ? GT.eq : GT.muted,
                  strokeWidth: on ? 2 : 1.1,
                  opacity: on ? 1 : 0.7,
                }}
              />
            );
          })}
          {tele && (
            <path
              d={`M${tele.a.x} ${tele.a.y} Q${(tele.a.x + tele.b.x) / 2} ${Math.min(tele.a.y, tele.b.y) - 60} ${tele.b.x} ${tele.b.y}`}
              style={{
                fill: 'none',
                stroke: GT.alt,
                strokeWidth: 2,
                strokeDasharray: '3 4',
                opacity: tele.o,
              }}
            />
          )}
          {/* the surfer, under the page labels */}
          {tot > 0 && speed !== 'sprint' && (
            <circle cx={token.x} cy={token.y} r={6} style={{ fill: GT.eq }} />
          )}
          {POS.map((p, i) => (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={R[i]}
                style={{
                  fill: GT.aSoft,
                  stroke: cur.current === i && tot > 0 ? GT.eq : GT.a,
                  strokeWidth: cur.current === i && tot > 0 ? 3 : 1.5,
                  fillOpacity: 0.92,
                }}
              />
              <T x={p.x} y={p.y + 1} size={12 * Math.min(k, 1.3)} font="display" color={GT.ink}>
                {NAMES[i]}
              </T>
            </g>
          ))}
          {tele && tele.o > 0.2 && (
            <T
              x={(tele.a.x + tele.b.x) / 2}
              y={Math.min(tele.a.y, tele.b.y) - 34}
              size={9 * k}
              color={GT.alt}
            >
              teleport
            </T>
          )}
        </svg>

        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
            <Chip color={GT.a}>time spent</Chip>
            <Chip color={GT.eq}>exact PageRank</Chip>
            <Chip color={GT.alt}>teleport</Chip>
          </div>
          <div className="grid gap-1.5">
            {NAMES.map((nm, i) => (
              <div key={nm} className="grid grid-cols-[1rem_1fr_3.2rem] items-center gap-2">
                <span className="font-display text-sm text-ink">{nm}</span>
                <div className="relative h-3.5 border border-line">
                  <div
                    className="absolute inset-y-0 left-0"
                    style={{
                      width: `${(tot ? share[i] : 0) * 250}%`,
                      background: GT.a,
                      maxWidth: '100%',
                    }}
                  />
                  <div
                    className="absolute -inset-y-1 w-0.5"
                    style={{ left: `calc(${exact[i] * 250}% - 1px)`, background: GT.eq }}
                  />
                </div>
                <span className="text-right font-mono text-[0.68rem] tabular-nums text-ink-soft">
                  {(exact[i] * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <BigStat label="clicks" value={tot.toLocaleString('en-US')} size="md" />
            <BigStat
              label="off from exact"
              value={gap === null ? '·' : `${(gap * 100).toFixed(1)}`}
              unit={gap === null ? undefined : '%'}
              size="md"
              color={gap !== null && gap < 0.01 ? GT.opt : GT.ink}
            />
          </div>
          <div className="mt-4">
            <Slider
              label="s, chance of following a link"
              value={s}
              min={0.5}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => {
                touch();
                setS(v);
                resetCounts();
              }}
            />
          </div>
        </div>
      </div>
      <Caption>{caption}</Caption>
    </Specimen>
  );
}
