import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Specimen,
  SvgArrows,
  T,
  clamp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import { NODES, pathToSat, type NodeId } from './web';

/* Arrows point the way speed flows: X → Y means "a fast X solver also solves Y" (Y reduces
 * to X). Suppose one problem is fast: the speed climbs the arrows to SAT, then the Cook-Levin
 * theorem (every NP problem rewrites as SAT) spreads it over all of NP. */
const PULSE = 300; // units per second along an arrow
const RING = 380; // units per second for the Cook-Levin wave
const RING_GAP = 0;

type Layout = {
  W: number;
  H: number;
  bw: number;
  bh: number;
  fs: number;
  pos: Record<NodeId, [number, number]>;
  restY: number;
};
const WIDE: Layout = {
  W: 640,
  H: 372,
  bw: 104,
  bh: 32,
  fs: 12,
  pos: {
    SAT: [320, 34],
    '3SAT': [185, 116],
    CLIQUE: [455, 116],
    COL: [70, 200],
    SUBSET: [185, 200],
    LATIN: [300, 200],
    VC: [455, 200],
    SUDOKU: [300, 282],
    HAM: [455, 282],
    TSP: [580, 282],
  },
  restY: 324,
};
const NARROW: Layout = {
  W: 360,
  H: 424,
  bw: 78,
  bh: 30,
  fs: 10.5,
  pos: {
    SAT: [217, 28],
    '3SAT': [124, 104],
    CLIQUE: [310, 104],
    COL: [42, 182],
    SUBSET: [124, 182],
    LATIN: [206, 182],
    VC: [310, 182],
    SUDOKU: [206, 260],
    HAM: [310, 260],
    TSP: [310, 338],
  },
  restY: 376,
};

/** Point where the segment from box centre `a` toward `b` leaves the box. */
function boxEdge(a: [number, number], b: [number, number], bw: number, bh: number) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const sx = dx === 0 ? Infinity : bw / 2 / Math.abs(dx);
  const sy = dy === 0 ? Infinity : bh / 2 / Math.abs(dy);
  const s = Math.min(sx, sy);
  return [a[0] + dx * s, a[1] + dy * s] as [number, number];
}

/** Everything the figure shows at time t (seconds) after "suppose `origin` is fast". */
function stateAt(L: Layout, origin: NodeId | null, t: number) {
  const lit = new Set<NodeId>();
  const edgeProg = new Map<NodeId, number>(); // child id → 0..1 progress of its upward arrow
  let pulse: [number, number] | null = null;
  let ring = -1;
  let pathDone = false;
  let all = false;
  if (!origin) return { lit, edgeProg, pulse, ring, pathDone, all };
  const path = pathToSat(origin);
  lit.add(origin);
  let clock = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = boxEdge(L.pos[path[i]], L.pos[path[i + 1]], L.bw, L.bh);
    const b = boxEdge(L.pos[path[i + 1]], L.pos[path[i]], L.bw, L.bh);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const dur = len / PULSE;
    const u = clamp((t - clock) / dur, 0, 1);
    edgeProg.set(path[i], u);
    if (u > 0 && u < 1) pulse = [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
    if (u >= 1) lit.add(path[i + 1]);
    clock += dur;
  }
  pathDone = t >= clock;
  const tr = t - clock - (origin === 'SAT' ? 0 : RING_GAP);
  if (tr > 0) {
    ring = tr * RING;
    const [sx, sy] = L.pos.SAT;
    for (const n of NODES) {
      const [x, y] = L.pos[n.id];
      if (Math.hypot(x - sx, y - sy) <= ring) lit.add(n.id);
    }
    all = ring >= maxReach(L);
  }
  return { lit, edgeProg, pulse, ring, pathDone, all };
}

function maxReach(L: Layout) {
  const [sx, sy] = L.pos.SAT;
  return Math.max(Math.hypot(sx, L.H - sy), Math.hypot(L.W - sx, L.H - sy));
}
function endTime(L: Layout, origin: NodeId) {
  const path = pathToSat(origin);
  let c = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const a = boxEdge(L.pos[path[i]], L.pos[path[i + 1]], L.bw, L.bh);
    const b = boxEdge(L.pos[path[i + 1]], L.pos[path[i]], L.bw, L.bh);
    c += Math.hypot(b[0] - a[0], b[1] - a[1]) / PULSE;
  }
  return c + RING_GAP + maxReach(L) / RING + 0.05;
}

/** The rest of NP, by name: a mix of hard ones and easy ones (P sits inside NP too). */
const REST = [
  'knapsack',
  'sorting',
  'Tetris',
  'bin packing',
  'protein folding',
  'shortest path',
  'Battleship',
  'set cover',
  'job-shop scheduling',
  'factoring',
  'Kakuro',
  'max cut',
  'register allocation',
  'Candy Crush',
  'Steiner tree',
  'is it prime?',
  'Lemmings',
  'Nurikabe',
  'partition',
];
const MORE = '…and thousands more';

/** Lay the names out in rows under the heading; each word lights when the wave reaches its middle. */
function restWords(L: Layout) {
  const fs = L.fs - 0.5;
  const cw = fs * 0.56; // average sans glyph width
  const gap = fs * 1.6;
  const out: { w: string; x: number; y: number; cx: number }[] = [];
  let x = 14;
  let y = L.restY + fs * 1.9;
  for (const w of [...REST, MORE]) {
    const width = w.length * cw;
    if (x + width > L.W - 14) {
      x = 14;
      y += fs * 1.75;
    }
    out.push({ w, x, y, cx: x + width / 2 });
    x += width + gap;
  }
  return { words: out, fs, H: y + fs * 1.2 };
}

// the figure ends just under the last row of names
for (const L of [WIDE, NARROW]) L.H = Math.ceil(restWords(L).H);

export default function ReductionWeb() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const L = compact ? NARROW : WIDE;
  const [sel, setSel] = useState<NodeId | null>(null);
  const [active, bind, previewing] = useHoverPreview<NodeId | null>(sel);
  const t = useRef(0);
  const [, setTick] = useState(0);
  const started = useRef(false);

  const commit = (id: NodeId) => {
    started.current = true;
    if (sel === id) {
      setSel(null); // clicking the same problem again undoes the supposition
      return;
    }
    t.current = reduced ? 1e3 : 0;
    setSel(id);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    if (reduced) {
      started.current = true;
      t.current = 1e3;
      setSel('TSP');
      return;
    }
    const id = window.setTimeout(() => {
      if (started.current) return;
      started.current = true;
      t.current = 0;
      setSel('TSP');
    }, 900);
    return () => window.clearTimeout(id);
  }, [inView, reduced]);

  const running = inView && !reduced && sel !== null && t.current < endTime(L, sel);
  useFrame((dt) => {
    t.current += dt;
    setTick((x) => (x + 1) % 1e6);
  }, running);

  // a hover preview shows the climb to SAT, frozen just before the Cook-Levin wave
  const st = previewing
    ? stateAt(L, active, active ? endTime(L, active) - maxReach(L) / RING - RING_GAP - 0.05 : 0)
    : stateAt(L, sel, t.current);
  const rest = restWords(L);
  const [sx, sy] = L.pos.SAT;

  const captions: string[] = [
    'Click a problem to suppose someone found a fast way to solve it.',
    ...NODES.map((n) => {
      const path = pathToSat(n.id)
        .slice(1)
        .map((id) => NODES.find((m) => m.id === id)!.say);
      return n.id === 'SAT'
        ? 'Suppose SAT had a fast solver. By the Cook-Levin theorem every problem in NP can be rewritten as SAT, so every one of them would be fast too.'
        : `Suppose ${n.say} had a fast solver. It also solves ${path.join(', then ')}. And by the Cook-Levin theorem a fast SAT solver solves every problem in NP.`;
    }),
  ];
  const capIdx = active ? 1 + NODES.findIndex((n) => n.id === active) : 0;
  const nLit = st.lit.size;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-05.2"
      title="the web of reductions"
      kicker="X → Y: a fast X also solves Y"
      hint="click a problem; click it again to undo"
      footer="Karp 1972 · Sipser §7.5"
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${L.W} ${L.H}`}
          className="block w-full select-none"
          role="img"
          aria-label="Famous NP-complete problems joined by reductions, all leading to SAT"
        >
          <SvgArrows />
          {/* Cook-Levin wave */}
          {st.ring > 0 && !st.all && (
            <circle
              cx={sx}
              cy={sy}
              r={st.ring}
              style={{ fill: GT.optSoft, fillOpacity: 0.35, stroke: GT.opt, strokeWidth: 2 }}
            />
          )}

          {/* arrows: child → parent */}
          {NODES.filter((n) => n.parent).map((n) => {
            const a = boxEdge(L.pos[n.id], L.pos[n.parent!], L.bw, L.bh);
            const b = boxEdge(L.pos[n.parent!], L.pos[n.id], L.bw, L.bh);
            const u = st.edgeProg.get(n.id) ?? 0;
            return (
              <g key={'e' + n.id}>
                <line
                  x1={a[0]}
                  y1={a[1]}
                  x2={b[0]}
                  y2={b[1]}
                  markerEnd={`url(#gt-arrow-${u >= 1 ? 'opt' : 'muted'})`}
                  style={{ stroke: u >= 1 ? GT.opt : GT.muted, strokeWidth: u >= 1 ? 2 : 1.4 }}
                />
                {u > 0 && u < 1 && (
                  <line
                    x1={a[0]}
                    y1={a[1]}
                    x2={a[0] + (b[0] - a[0]) * u}
                    y2={a[1] + (b[1] - a[1]) * u}
                    style={{ stroke: GT.opt, strokeWidth: 2 }}
                  />
                )}
              </g>
            );
          })}
          {st.pulse && (
            <circle cx={st.pulse[0]} cy={st.pulse[1]} r={4.5} style={{ fill: GT.opt }} />
          )}

          {/* the rest of NP */}
          <T x={14} y={L.restY} anchor="start" size={L.fs - 1}>
            and every other problem in NP, for example
          </T>
          {/* one text per row, so the browser spaces the names evenly */}
          {[...new Set(rest.words.map((r) => r.y))].map((y) => (
            <text
              key={y}
              x={14}
              y={y}
              dominantBaseline="middle"
              style={{ fontFamily: 'var(--font-sans)', fontSize: rest.fs }}
            >
              {rest.words
                .filter((r) => r.y === y)
                .map(({ w, cx }, i) => {
                  const more = w === MORE;
                  const on = !more && st.ring > 0 && Math.hypot(cx - sx, y - sy) <= st.ring;
                  return (
                    <tspan
                      key={w}
                      dx={i ? rest.fs * 1.4 : 0}
                      style={{
                        fill: on ? GT.opt : more ? GT.muted : GT.inkSoft,
                        fontStyle: more ? 'italic' : 'normal',
                        transition: 'fill 160ms',
                      }}
                    >
                      {w}
                    </tspan>
                  );
                })}
            </text>
          ))}

          {/* problems */}
          {NODES.map((n) => {
            const [x, y] = L.pos[n.id];
            const on = st.lit.has(n.id);
            const origin = active === n.id;
            return (
              <g
                key={n.id}
                onClick={() => commit(n.id)}
                {...bind(n.id)}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x - L.bw / 2}
                  y={y - L.bh / 2}
                  width={L.bw}
                  height={L.bh}
                  style={{
                    fill: on ? GT.optSoft : GT.paperRaised,
                    stroke: on ? GT.opt : GT.inkSoft,
                    strokeWidth: origin ? 3 : 1.4,
                    transition: 'fill 160ms, stroke 160ms',
                  }}
                />
                <T
                  x={x}
                  y={y + 1}
                  size={n.id === 'SAT' ? L.fs + 3 : L.fs}
                  color={GT.ink}
                  font={n.id === 'SAT' ? 'display' : 'mono'}
                >
                  {n.name}
                </T>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="named problems fast"
          value={`${nLit} / ${NODES.length}`}
          color={nLit ? GT.opt : GT.ink}
        />
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="all of NP fast"
          value={st.all ? 'yes' : st.ring > 0 ? 'spreading' : 'not yet'}
          color={st.all ? GT.opt : GT.muted}
        />
      </div>
      <Reserve
        active={capIdx}
        items={captions.map((c) => (
          <Caption key={c}>{c}</Caption>
        ))}
      />
    </Specimen>
  );
}
