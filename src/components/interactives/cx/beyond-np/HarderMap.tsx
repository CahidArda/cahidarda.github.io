import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import {
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';

/**
 * The landscape above NP as nested boxes, with example problems as dots. Inclusions drawn
 * are the proven ones (P ⊆ NP ∩ coNP, NP ∪ coNP ⊆ PSPACE ⊆ EXPTIME, P ⊆ BQP ⊆ PSPACE).
 * Where a dot sits outside an inner box, that is the believed picture; the captions say
 * which separations are proven (only P ≠ EXPTIME among these, plus decidable vs not).
 */

type RegionId = 'P' | 'BQP' | 'NP' | 'coNP' | 'PSPACE' | 'EXPTIME' | 'DEC' | 'UNDEC';
type Box = [number, number, number, number]; // x0 y0 x1 y1

type Layout = {
  W: number;
  H: number;
  boxes: Record<Exclude<RegionId, 'UNDEC'>, Box>;
  labels: Record<RegionId, { x: number; y: number; anchor: 'start' | 'middle' | 'end' }>;
  dots: Record<DotId, [number, number]>;
};

type DotId =
  | 'sort'
  | 'path'
  | 'prime'
  | 'factor'
  | 'sudoku'
  | 'tsp'
  | 'taut'
  | 'geo'
  | 'chess'
  | 'halt';

const WIDE: Layout = {
  W: 600,
  H: 392,
  boxes: {
    DEC: [8, 24, 456, 384],
    EXPTIME: [22, 54, 442, 370],
    PSPACE: [36, 92, 428, 356],
    NP: [50, 146, 302, 342],
    coNP: [196, 146, 414, 342],
    BQP: [202, 204, 422, 336],
    P: [208, 238, 296, 330],
  },
  labels: {
    DEC: { x: 16, y: 38, anchor: 'start' },
    EXPTIME: { x: 30, y: 68, anchor: 'start' },
    PSPACE: { x: 44, y: 106, anchor: 'start' },
    NP: { x: 58, y: 160, anchor: 'start' },
    coNP: { x: 406, y: 160, anchor: 'end' },
    BQP: { x: 414, y: 324, anchor: 'end' },
    P: { x: 290, y: 252, anchor: 'end' },
    UNDEC: { x: 528, y: 38, anchor: 'middle' },
  },
  dots: {
    sort: [216, 264],
    path: [216, 286],
    prime: [216, 308],
    factor: [216, 220],
    sudoku: [66, 210],
    tsp: [66, 250],
    taut: [312, 180],
    geo: [200, 124],
    chess: [200, 80],
    halt: [490, 204],
  },
};

const COMPACT: Layout = {
  W: 380,
  H: 520,
  boxes: {
    DEC: [6, 24, 374, 452],
    EXPTIME: [16, 54, 364, 440],
    PSPACE: [26, 92, 354, 428],
    NP: [36, 150, 238, 416],
    coNP: [134, 150, 344, 416],
    BQP: [140, 230, 340, 408],
    P: [144, 304, 234, 400],
  },
  labels: {
    DEC: { x: 12, y: 38, anchor: 'start' },
    EXPTIME: { x: 22, y: 68, anchor: 'start' },
    PSPACE: { x: 32, y: 106, anchor: 'start' },
    NP: { x: 42, y: 164, anchor: 'start' },
    coNP: { x: 338, y: 164, anchor: 'end' },
    BQP: { x: 334, y: 398, anchor: 'end' },
    P: { x: 228, y: 318, anchor: 'end' },
    UNDEC: { x: 12, y: 474, anchor: 'start' },
  },
  dots: {
    sort: [152, 332],
    path: [152, 356],
    prime: [152, 380],
    factor: [148, 270],
    sudoku: [48, 214],
    tsp: [48, 254],
    taut: [246, 196],
    geo: [40, 126],
    chess: [30, 82],
    halt: [24, 500],
  },
};

const COLOR: Record<RegionId, string> = {
  P: GT.opt,
  NP: GT.a,
  coNP: GT.warn,
  BQP: GT.alt,
  PSPACE: GT.ink,
  EXPTIME: GT.ink,
  DEC: GT.muted,
  UNDEC: GT.b,
};
const NAME: Record<RegionId, string> = {
  P: 'P',
  NP: 'NP',
  coNP: 'coNP',
  BQP: 'BQP (quantum)',
  PSPACE: 'PSPACE',
  EXPTIME: 'EXPTIME',
  DEC: 'decidable',
  UNDEC: 'undecidable',
};

const DOTS: { id: DotId; label: string; home: RegionId; line: string }[] = [
  {
    id: 'sort',
    label: 'sorting',
    home: 'P',
    line: 'Sorting a list: about n log n comparisons. In P.',
  },
  {
    id: 'path',
    label: 'shortest path',
    home: 'P',
    line: 'Shortest route on a map: Dijkstra’s algorithm does it fast. In P.',
  },
  {
    id: 'prime',
    label: 'primality',
    home: 'P',
    line: 'Is n prime? In P since Agrawal, Kayal and Saxena (2002).',
  },
  {
    id: 'factor',
    label: 'factoring',
    home: 'BQP',
    line: 'Factoring: in NP and coNP, fast on a quantum computer (Shor, 1994). Not known to be in P; not believed NP-complete.',
  },
  {
    id: 'sudoku',
    label: 'Sudoku',
    home: 'NP',
    line: 'Sudoku on n² × n² grids: NP-complete. A filled grid is checked fast; finding one is the hard part.',
  },
  {
    id: 'tsp',
    label: 'salesman',
    home: 'NP',
    line: 'Travelling salesman (is there a tour shorter than k?): NP-complete.',
  },
  {
    id: 'taut',
    label: 'tautology',
    home: 'coNP',
    line: 'Is a logic formula true for every input? One counterexample proves “no”; nobody knows a short proof of “yes”. coNP-complete.',
  },
  {
    id: 'geo',
    label: 'generalized geography',
    home: 'PSPACE',
    line: 'Generalized geography, a two-player word-chain game played on any graph: PSPACE-complete.',
  },
  {
    id: 'chess',
    label: 'chess on n × n boards',
    home: 'EXPTIME',
    line: 'Chess on an n × n board: provably needs time exponential in n (Fraenkel and Lichtenstein, 1981). So it is not in P.',
  },
  {
    id: 'halt',
    label: 'halting problem',
    home: 'UNDEC',
    line: 'The halting problem: no algorithm at all (Turing, 1936).',
  },
];

const REGION_LINE: Record<RegionId, string> = {
  P: 'P: solvable in polynomial time. Fast in practice.',
  NP: 'NP: a “yes” answer comes with a proof you can check fast.',
  coNP: 'coNP: a “no” answer comes with a proof you can check fast. Whether NP = coNP is open.',
  BQP: 'BQP: fast on a quantum computer. Contains P and factoring; believed not to contain the NP-complete problems.',
  PSPACE:
    'PSPACE: solvable with a polynomial amount of memory, however long it takes. Two-player games of polynomial length live here.',
  EXPTIME:
    'EXPTIME: solvable in exponential time. Provably bigger than P (the time hierarchy theorem).',
  DEC: 'Decidable: some algorithm always answers. Above EXPTIME there are bigger classes still.',
  UNDEC: 'Undecidable: no algorithm answers every case, however much time it gets.',
};

const IDLE =
  'Hover a dot or a box. Of the boundaries drawn inside the decidable box, only one separation is proven: P ≠ EXPTIME. Any other pair could still turn out equal.';

// innermost first, for hit testing
const ORDER: Exclude<RegionId, 'UNDEC'>[] = ['P', 'BQP', 'NP', 'coNP', 'PSPACE', 'EXPTIME', 'DEC'];
// the wave that ripples outward when the figure comes into view
const WAVE: Exclude<RegionId, 'UNDEC'>[] = ['P', 'NP', 'coNP', 'BQP', 'PSPACE', 'EXPTIME', 'DEC'];
const WAVE_STEP = 0.28;
const WAVE_DUR = 0.7;
const END = WAVE_STEP * (WAVE.length - 1) + WAVE_DUR;

type Focus = { kind: 'dot'; id: DotId } | { kind: 'region'; id: RegionId } | null;

export default function HarderMap() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const L = compact ? COMPACT : WIDE;
  const [svgRef, k] = useBoost(L.W, 9, 8.5);
  const [focus, setFocus] = useState<Focus>(null);
  const [t, setT] = useState(END);
  const tRef = useRef(END);
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (!inView || started.current || reduced) return;
    started.current = true;
    tRef.current = 0;
    setT(0);
    const id = setTimeout(() => setPlaying(true), 400);
    return () => clearTimeout(id);
  }, [inView, reduced]);
  useFrame((dt) => {
    const v = Math.min(END, tRef.current + Math.max(0, dt));
    tRef.current = v;
    setT(v);
    if (v >= END) setPlaying(false);
  }, playing && inView);

  const inBox = (b: Box, x: number, y: number) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  const hit = (e: RPointerEvent): Focus => {
    if (!svgRef.current) return null;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    for (const d of DOTS) {
      const [x, y] = L.dots[d.id];
      if (Math.hypot(p.x - x, p.y - y) < 11) return { kind: 'dot', id: d.id };
    }
    for (const r of ORDER) if (inBox(L.boxes[r], p.x, p.y)) return { kind: 'region', id: r };
    return { kind: 'region', id: 'UNDEC' };
  };

  const focusRegion: RegionId | null =
    focus === null
      ? null
      : focus.kind === 'region'
        ? focus.id
        : DOTS.find((d) => d.id === focus.id)!.home;
  const capItems = [
    IDLE,
    ...DOTS.map((d) => d.line),
    ...(Object.keys(REGION_LINE) as RegionId[]).map((r) => REGION_LINE[r]),
  ];
  const capIdx =
    focus === null
      ? 0
      : focus.kind === 'dot'
        ? 1 + DOTS.findIndex((d) => d.id === focus.id)
        : 1 + DOTS.length + (Object.keys(REGION_LINE) as RegionId[]).indexOf(focus.id);

  const wave = (r: RegionId) => {
    const i = WAVE.indexOf(r as Exclude<RegionId, 'UNDEC'>);
    if (i < 0) return 0;
    const u = clamp((t - i * WAVE_STEP) / WAVE_DUR, 0, 1);
    return Math.sin(Math.PI * u);
  };
  const tint = (r: RegionId) => Math.max(focusRegion === r ? 1 : 0, wave(r));

  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };
  const fs = 9.5 * k;
  const dec = L.boxes.DEC;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-11.1"
      title="the map above NP"
      kicker="proven inclusions, believed gaps"
      hint="hover or tap a dot or a box"
      footer="Sipser ch. 8-9; Aaronson, P =? NP"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${L.W} ${L.H}`}
          className="block w-full select-none"
          role="img"
          aria-label="Nested boxes: P inside NP and coNP, inside PSPACE, inside EXPTIME, inside the decidable problems, with BQP containing P; the halting problem outside, undecidable."
          onPointerMove={(e) => setFocus(hit(e))}
          onPointerDown={(e) => setFocus(hit(e))}
          onPointerLeave={(e) => {
            if (e.pointerType !== 'touch') setFocus(null);
          }}
        >
          {/* undecidable surround */}
          <rect
            x={0}
            y={0}
            width={L.W}
            height={L.H}
            style={{ fill: GT.b, opacity: 0.05 + 0.08 * tint('UNDEC') }}
          />
          {(['DEC', 'EXPTIME', 'PSPACE', 'NP', 'coNP', 'BQP', 'P'] as const).map((r) => {
            const [x0, y0, x1, y1] = L.boxes[r];
            const c = COLOR[r];
            const a = tint(r);
            return (
              <g key={r}>
                <rect
                  x={x0}
                  y={y0}
                  width={x1 - x0}
                  height={y1 - y0}
                  style={{
                    fill: r === 'DEC' ? GT.paperRaised : 'transparent',
                  }}
                />
                <rect
                  x={x0}
                  y={y0}
                  width={x1 - x0}
                  height={y1 - y0}
                  style={{ fill: c, opacity: 0.1 * a }}
                />
                <rect
                  x={x0}
                  y={y0}
                  width={x1 - x0}
                  height={y1 - y0}
                  style={{
                    fill: 'none',
                    stroke: c,
                    strokeWidth: 1.2 + 1.3 * a,
                    strokeDasharray: r === 'BQP' ? '5 4' : undefined,
                  }}
                />
              </g>
            );
          })}
          {/* box names, placed in each box's own corner */}
          {(Object.keys(L.labels) as RegionId[]).map((r) => (
            <g key={r} style={halo}>
              <T
                x={L.labels[r].x}
                y={L.labels[r].y}
                anchor={L.labels[r].anchor}
                size={(r === 'DEC' || r === 'UNDEC' ? 9 : 11) * k}
                color={COLOR[r]}
                weight={700}
              >
                {NAME[r]}
              </T>
            </g>
          ))}
          <g style={halo}>
            <T x={dec[2] - 8} y={dec[1] + 14} anchor="end" size={8.5 * k}>
              {'… and harder still'}
            </T>
          </g>

          {/* problems */}
          {DOTS.map((d) => {
            const [x, y] = L.dots[d.id];
            const on = focus?.kind === 'dot' && focus.id === d.id;
            const c = COLOR[d.home];
            return (
              <g key={d.id} style={{ cursor: 'pointer' }}>
                <circle
                  cx={x}
                  cy={y}
                  r={on ? 6 : 4.5}
                  style={{ fill: c, stroke: GT.paperRaised, strokeWidth: 1.5 }}
                />
                <g style={halo}>
                  <T
                    x={x + 9}
                    y={y + 0.5}
                    anchor="start"
                    size={fs}
                    color={GT.ink}
                    weight={on ? 700 : 400}
                  >
                    {d.label}
                  </T>
                </g>
              </g>
            );
          })}
        </svg>
      </div>
      <Reserve
        className="mt-2"
        items={capItems.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
