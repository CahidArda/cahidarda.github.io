import { useState, type ReactNode } from 'react';
import {
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
import { useDemoFlip, useMerge, type World } from './world';

/**
 * CX-04.2 · what would change. Four real tasks, each a small picture that morphs between the
 * two worlds on one shared clock: the padlock opens, the delivery route untangles, the protein
 * chain folds, the proof writes itself. One toggle, hover to preview.
 */
const GW = 160;
const GH = 92;

function Lock({ m }: { m: number }) {
  const lift = 13 * m;
  return (
    <g>
      <path
        d={`M66 ${44 - lift} V${30 - lift} a14 14 0 0 1 28 0 V${44 - lift * 0.2}`}
        style={{ fill: 'none', stroke: GT.ink, strokeWidth: 6, strokeLinecap: 'butt' }}
      />
      <rect x={54} y={42} width={52} height={40} style={{ fill: GT.ink }} />
      <rect x={54} y={42} width={52} height={40} style={{ fill: GT.b, opacity: m }} />
      <circle cx={80} cy={58} r={5} style={{ fill: GT.paperRaised }} />
      <rect x={78} y={60} width={4} height={11} style={{ fill: GT.paperRaised }} />
    </g>
  );
}

// seven stops; a decent tour with a detour (403 units) vs the shortest tour (343, brute force)
const CITIES = [
  [22, 20],
  [70, 12],
  [128, 22],
  [140, 64],
  [96, 82],
  [44, 76],
  [70, 46],
];
const TOUR_OK = [0, 1, 2, 6, 3, 4, 5];
const TOUR_BEST = [0, 1, 2, 3, 4, 6, 5];
const tourPath = (tour: number[]) =>
  tour.map((c, i) => `${i ? 'L' : 'M'}${CITIES[c][0]} ${CITIES[c][1]}`).join('') + 'Z';

function Route({ m }: { m: number }) {
  return (
    <g>
      <path
        d={tourPath(TOUR_OK)}
        style={{
          fill: 'none',
          stroke: GT.ink,
          strokeWidth: 2,
          strokeLinejoin: 'round',
          opacity: 1 - m,
        }}
      />
      <path
        d={tourPath(TOUR_BEST)}
        style={{
          fill: 'none',
          stroke: GT.opt,
          strokeWidth: 2.5,
          strokeLinejoin: 'round',
          opacity: m,
        }}
      />
      {CITIES.map(([x, y], i) => (
        <rect
          key={i}
          x={x - 4}
          y={y - 4}
          width={8}
          height={8}
          style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.5 }}
        />
      ))}
    </g>
  );
}

// a chain of 8 beads (H = water-fearing, dark) on a square lattice: stretched out vs folded so
// the dark beads touch in the middle
const HP = [1, 0, 1, 0, 0, 1, 0, 1];
const OPEN = [
  [20, 46],
  [38, 34],
  [56, 46],
  [74, 34],
  [92, 46],
  [110, 34],
  [128, 46],
  [146, 34],
];
const FOLD = [
  [62, 28],
  [62, 46],
  [80, 46],
  [80, 64],
  [98, 64],
  [98, 46],
  [116, 46],
  [116, 28],
];

function Protein({ m }: { m: number }) {
  const pts = OPEN.map(([x, y], i) => [lerp(x, FOLD[i][0], m), lerp(y + 14, FOLD[i][1] + 10, m)]);
  return (
    <g>
      <path
        d={pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join('')}
        style={{ fill: 'none', stroke: GT.inkSoft, strokeWidth: 2.5, strokeLinejoin: 'round' }}
      />
      {pts.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={6.5}
          style={{ fill: HP[i] ? GT.ink : GT.paperRaised, stroke: GT.ink, strokeWidth: 1.5 }}
        />
      ))}
    </g>
  );
}

function Proof({ m }: { m: number }) {
  const lines = [0.9, 0.7, 0.85, 0.6, 0.75];
  return (
    <g>
      <rect
        x={44}
        y={6}
        width={72}
        height={82}
        style={{ fill: GT.paper, stroke: GT.ink, strokeWidth: 1.5 }}
      />
      <rect x={52} y={14} width={40} height={4} style={{ fill: GT.ink }} />
      {lines.map((f, i) => (
        <rect
          key={i}
          x={52}
          y={26 + i * 9}
          width={Math.max(0, 56 * f * m)}
          height={3}
          style={{ fill: GT.inkSoft }}
        />
      ))}
      <g style={{ opacity: 1 - m }}>
        <T x={80} y={52} size={30} font="display" color={GT.muted}>
          ?
        </T>
      </g>
      <rect x={98} y={74} width={9} height={9} style={{ fill: GT.opt, opacity: m }} />
    </g>
  );
}

const CARDS: {
  title: string;
  tag: string;
  glyph: (m: number) => ReactNode;
  ne: string;
  eq: string;
}[] = [
  {
    title: 'Online banking',
    tag: 'factoring',
    glyph: (m) => <Lock m={m} />,
    ne: 'Safe while factoring stays hard.',
    eq: 'Broken: the secret keys could be computed.',
  },
  {
    title: 'Delivery routes',
    tag: 'salesman',
    glyph: (m) => <Route m={m} />,
    ne: 'Good routes, with no proof they are the best.',
    eq: 'The shortest route, found fast.',
  },
  {
    title: 'Protein folding',
    tag: 'NP-hard in lattice models',
    glyph: (m) => <Protein m={m} />,
    ne: 'The lowest-energy fold means a search.',
    eq: 'The lowest-energy fold, found fast.',
  },
  {
    title: 'Mathematics',
    tag: 'theorem proving',
    glyph: (m) => <Proof m={m} />,
    ne: 'Checking a proof is easy. Finding one is not.',
    eq: 'Any proof short enough to check, found by machine.',
  },
];

export default function WhatChanges() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<World>('ne');
  const [world, bind] = useHoverPreview(sel);
  const touch = useDemoFlip(inView, reduced, setSel, 1400);
  const m = useMerge(world, reduced, 1.1);
  const idx = world === 'eq' ? 1 : 0;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-04.2"
      title="what would change"
      kicker="if a fast method for NP were found"
      hint="hover or tap a world"
    >
      <div className="grid grid-cols-2 gap-3">
        {CARDS.map((c) => (
          <div key={c.title} className="min-w-0 border border-line bg-paper-raised p-3">
            <svg
              viewBox={`0 0 ${GW} ${GH}`}
              className="mx-auto block h-auto w-full max-w-[200px]"
              role="img"
              aria-label={`${c.title}: ${idx ? c.eq : c.ne}`}
            >
              {c.glyph(m)}
            </svg>
            <div className="mt-2 flex items-baseline justify-between gap-2">
              <span className="label truncate text-ink">{c.title}</span>
            </div>
            <Reserve
              className="mt-1"
              active={idx}
              items={[c.ne, c.eq].map((t, i) => (
                <p
                  key={i}
                  className="text-[0.8rem] leading-snug"
                  style={{ color: i ? GT.ink : GT.inkSoft }}
                >
                  {t}
                </p>
              ))}
            />
            <div className="mt-1 font-mono text-[0.62rem] text-muted">{c.tag}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex justify-end">
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
    </Specimen>
  );
}
