import { useEffect, useLayoutEffect, useRef, useState } from 'react';
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
  /** how far the wave must spread to cover the whole figure background (measured) */
  reach?: number;
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
  if (L.reach) return L.reach;
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

/** The rest of NP, by name. Problems already in P are left out: they are fast already. */
const REST = [
  'knapsack',
  'Tetris',
  'bin packing',
  'protein folding',
  'Battleship',
  'set cover',
  'job-shop scheduling',
  'factoring',
  'Kakuro',
  'max cut',
  'register allocation',
  'Candy Crush',
  'Steiner tree',
  'Lemmings',
  'Minesweeper',
  'Nurikabe',
  'partition',
  'longest path',
  'dominating set',
  'crossword building',
  'vehicle routing',
  'Light Up',
  'Slitherlink',
];

/**
 * Lay the names out in justified rows under the heading, flush with the figure's edges. Widths
 * are measured once the font is in (estimated before that). Each name lights up when the wave
 * reaches its middle.
 */
function restWords(L: Layout, widths?: number[]) {
  const fs = L.fs - 0.5;
  const gap = fs * 1.3;
  const lh = fs * 1.75;
  const wd = REST.map((w, i) => widths?.[i] ?? w.length * fs * 0.56);
  const rows: number[][] = [[]];
  let used = 0;
  REST.forEach((_, i) => {
    const row = rows[rows.length - 1];
    if (row.length && used + gap + wd[i] > L.W) {
      rows.push([i]);
      used = wd[i];
    } else {
      used += (row.length ? gap : 0) + wd[i];
      row.push(i);
    }
  });
  const out: { w: string; x: number; y: number; cx: number }[] = [];
  rows.forEach((row, r) => {
    const y = L.restY + fs * 1.9 + r * lh;
    const total = row.reduce((t, i) => t + wd[i], 0);
    const fill = (total + gap * (row.length - 1)) / L.W;
    // the last row stays left-aligned unless it is nearly full
    const g =
      row.length > 1 && (r < rows.length - 1 || fill > 0.8)
        ? (L.W - total) / (row.length - 1)
        : gap;
    let x = 0;
    for (const i of row) {
      out.push({ w: REST[i], x, y, cx: x + wd[i] / 2 });
      x += wd[i] + g;
    }
  });
  return { words: out, fs, H: L.restY + fs * 1.9 + (rows.length - 1) * lh + fs * 1.2 };
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
  const [widths, setWidths] = useState<number[] | undefined>();
  const rest = restWords(L, widths);
  const VH = Math.ceil(rest.H);
  const [sx, sy] = L.pos.SAT;

  // measure the names once the font is in, so rows can be justified exactly
  const svgRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    let live = true;
    const measure = () => {
      const els = svgRef.current?.querySelectorAll<SVGTextElement>('[data-rest]');
      if (!live || !els || els.length !== REST.length) return;
      setWidths([...els].map((e) => e.getComputedTextLength()));
    };
    measure();
    document.fonts?.ready.then(measure);
    return () => {
      live = false;
    };
  }, [compact]);

  // the wave is painted on a layer behind the whole figure body, edge to edge
  const bgRef = useRef<HTMLDivElement>(null);
  const [geo, setGeo] = useState<{ ox: number; oy: number; k: number } | null>(null);
  useLayoutEffect(() => {
    const bg = bgRef.current;
    const svg = svgRef.current;
    if (!bg || !svg) return;
    const update = () => {
      const b = bg.getBoundingClientRect();
      const r = svg.getBoundingClientRect();
      const k = r.width / L.W;
      const ox = r.left - b.left;
      const oy = r.top - b.top;
      const cx = ox + sx * k;
      const cy = oy + sy * k;
      L.reach =
        Math.max(
          Math.hypot(cx, cy),
          Math.hypot(b.width - cx, cy),
          Math.hypot(cx, b.height - cy),
          Math.hypot(b.width - cx, b.height - cy),
        ) / k;
      setGeo({ ox, oy, k });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(bg);
    return () => ro.disconnect();
  }, [L, sx, sy]);
  const wave = (() => {
    if (!geo || st.ring <= 0) return 'none';
    const fill = `color-mix(in srgb, ${GT.optSoft} 45%, transparent)`;
    if (st.all) return fill;
    const cx = geo.ox + sx * geo.k;
    const cy = geo.oy + sy * geo.k;
    const R = st.ring * geo.k;
    return `radial-gradient(circle at ${cx}px ${cy}px, ${fill} ${R - 1.5}px, ${GT.opt} ${R - 1.5}px, ${GT.opt} ${R + 0.5}px, transparent ${R + 0.5}px)`;
  })();

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
      <div className="relative isolate">
        <div
          ref={bgRef}
          aria-hidden
          className="pointer-events-none absolute -inset-4 -z-10 sm:-inset-5"
          style={{ background: wave }}
        />
        <div ref={wrapRef}>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${L.W} ${VH}`}
            className="block w-full select-none"
            role="img"
            aria-label="Famous NP-complete problems joined by reductions, all leading to SAT"
          >
            <SvgArrows />
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
            <T x={0} y={L.restY} anchor="start" size={L.fs - 1}>
              and thousands of other problems in NP, such as
            </T>
            {rest.words.map(({ w, x, y, cx }) => {
              const on = st.ring > 0 && Math.hypot(cx - sx, y - sy) <= st.ring;
              return (
                <text
                  key={w}
                  data-rest
                  x={x}
                  y={y}
                  dominantBaseline="middle"
                  style={{
                    fill: on ? GT.opt : GT.inkSoft,
                    fontFamily: 'var(--font-sans)',
                    fontSize: rest.fs,
                    transition: 'fill 160ms',
                  }}
                >
                  {w}
                </text>
              );
            })}

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
      </div>
    </Specimen>
  );
}
