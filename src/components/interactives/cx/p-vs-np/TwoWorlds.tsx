import { useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  lerp,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import { useDemoFlip, useMerge, type World } from './world';

/**
 * CX-04.1 · two worlds. NP drawn as a box; P sits inside it on the left, the NP-complete
 * problems in a strip on the right, and the problems believed to be neither in between
 * (Ladner 1975: if P ≠ NP such problems must exist). Switching to P = NP sweeps P's right edge
 * across the whole box; every dot turns green as the edge passes it. Colours match the series
 * map: P green, NP blue, NP-complete red.
 */
type Zone = 'p' | 'mid' | 'npc';
type Dot = { label: string; zone: Zone; row: number };

const DOTS: Dot[] = [
  { label: 'sorting', zone: 'p', row: 0 },
  { label: 'shortest route', zone: 'p', row: 1 },
  { label: 'is it prime?', zone: 'p', row: 2 },
  { label: 'factoring', zone: 'mid', row: 0 },
  { label: 'graph isomorphism', zone: 'mid', row: 1 },
  { label: 'Sudoku', zone: 'npc', row: 0 },
  { label: 'salesman', zone: 'npc', row: 1 },
  { label: 'timetabling', zone: 'npc', row: 2 },
  { label: 'protein folding', zone: 'npc', row: 3 },
];

const LAYOUTS = {
  wide: {
    W: 520,
    H: 276,
    np: [8, 8, 504, 260],
    p: [22, 62, 150, 192],
    mid: 186,
    npc: [356, 62, 140, 192],
    fs: 10.5,
    rowH: 38,
    row0: 96,
  },
  compact: {
    W: 360,
    H: 276,
    np: [4, 8, 352, 260],
    p: [12, 62, 104, 192],
    mid: 124,
    npc: [242, 62, 106, 192],
    fs: 9.5,
    rowH: 38,
    row0: 96,
  },
};

export default function TwoWorlds() {
  const [wrapRef, compact] = useCompact(460);
  const L = compact ? LAYOUTS.compact : LAYOUTS.wide;
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<World>('ne');
  const [world, bind] = useHoverPreview(sel);
  const touch = useDemoFlip(inView, reduced, setSel);
  const m = useMerge(world, reduced);

  const [nx, ny, nw, nh] = L.np;
  const [px, py, pw0, ph] = L.p;
  const [cx, cy, cw, ch] = L.npc;
  // P's right edge sweeps from its own box to the right edge of the NP-complete strip
  const pRight = lerp(px + pw0, cx + cw, m);

  const dotX = (d: Dot) =>
    d.zone === 'p' ? px + 12 : d.zone === 'mid' ? L.mid + 6 : cx + (compact ? 8 : 12);
  const dotY = (d: Dot) => L.row0 + d.row * L.rowH + (d.zone === 'mid' ? L.rowH / 2 : 0);
  const inP = (d: Dot) => dotX(d) < pRight - 0.5;
  const easy = DOTS.filter(inP).length;

  const zoneColor = (z: Zone) => (z === 'p' ? GT.opt : z === 'mid' ? GT.a : GT.b);
  const fade = 1 - m;
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  const captions = [
    'If P ≠ NP: the hardest problems in NP stay out of reach, and some problems sit in between.',
    'If P = NP: one fast method would cover all of NP. Every dot is easy, Sudoku and factoring included.',
  ];

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-04.1"
      title="two worlds"
      kicker="which one is ours?"
      hint="hover or tap a world"
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${L.W} ${L.H}`}
          className="block w-full select-none"
          role="img"
          aria-label={
            world === 'eq'
              ? 'If P equals NP, the P region fills all of NP and every problem shown is easy'
              : 'If P is not NP, P sits inside NP, NP-complete problems sit at the far edge, and factoring and graph isomorphism may sit in between'
          }
        >
          {/* NP */}
          <rect
            x={nx}
            y={ny}
            width={nw}
            height={nh}
            style={{ fill: GT.aSoft, stroke: GT.a, strokeWidth: 1.5 }}
          />
          <T x={nx + 10} y={ny + 16} anchor="start" size={L.fs + 2} weight={700} color={GT.a}>
            NP
          </T>
          <T x={nx + 38} y={ny + 16} anchor="start" size={L.fs - 1} color={GT.a}>
            easy to check
          </T>
          {/* NP-complete strip */}
          <rect
            x={cx}
            y={cy}
            width={cw}
            height={ch}
            style={{ fill: GT.bSoft, stroke: GT.b, strokeWidth: 1.5 }}
          />
          <g style={{ opacity: fade }}>
            <T x={cx + 8} y={cy - 10} anchor="start" size={L.fs} weight={700} color={GT.b}>
              NP-complete
            </T>
            <T x={L.mid} y={cy - 10} anchor="start" size={L.fs} weight={700} color={GT.a}>
              in between?
            </T>
          </g>
          {/* P, growing */}
          <rect
            x={px}
            y={py}
            width={pRight - px}
            height={ph}
            style={{ fill: GT.optSoft, stroke: GT.opt, strokeWidth: 1.5 }}
          />
          <T x={px + 8} y={py - 10} anchor="start" size={L.fs + 2} weight={700} color={GT.opt}>
            P
          </T>
          <g style={{ opacity: m }}>
            <T x={px + 22} y={py - 10} anchor="start" size={L.fs + 2} weight={700} color={GT.opt}>
              = NP
            </T>
          </g>
          <T x={px + 8} y={py + ph - 12} anchor="start" size={L.fs - 1} color={GT.opt}>
            easy to solve
          </T>
          {/* the problems */}
          {DOTS.map((d) => {
            const x = dotX(d);
            const y = dotY(d);
            const c = inP(d) ? GT.opt : zoneColor(d.zone);
            return (
              <g key={d.label}>
                <circle cx={x} cy={y} r={4.5} style={{ fill: c }} />
                <g style={halo}>
                  <T x={x + 9} y={y} anchor="start" size={L.fs} color={GT.ink}>
                    {d.label}
                  </T>
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <BigStat
          label="easy problems here"
          value={easy}
          unit={`of ${DOTS.length}`}
          color={easy === DOTS.length ? GT.opt : GT.ink}
        />
        <Seg
          options={[
            { value: 'ne', label: 'P ≠ NP' },
            { value: 'eq', label: 'P = NP' },
          ]}
          value={world}
          onChange={(v) => {
            touch();
            setSel(v);
          }}
          bind={(v) => {
            const b = bind(v);
            return {
              ...b,
              onPointerEnter: (e) => {
                touch();
                b.onPointerEnter(e);
              },
            };
          }}
        />
      </div>
      <Reserve
        items={captions.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={world === 'eq' ? 1 : 0}
      />
    </Specimen>
  );
}
