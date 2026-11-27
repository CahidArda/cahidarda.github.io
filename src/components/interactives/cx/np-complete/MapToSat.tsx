import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import {
  BORDERS,
  CLAUSES,
  REGIONS,
  clauseTrue,
  type Clause,
  type Colouring,
  type Region,
} from './mapSat';

/* Map on the left (top on phones), its 19 SAT clauses on the right (below). One SVG,
 * one coordinate system per layout. */
const COLS = [GT.a, GT.warn, GT.alt];

const V: Record<string, [number, number]> = {
  P1: [12, 28],
  P2: [104, 18],
  P3: [238, 22],
  P4: [124, 122],
  P5: [178, 118],
  P7: [18, 226],
  P8: [100, 232],
  P9: [240, 232],
  P10: [254, 126],
  P11: [182, 182],
};
const POLY: Record<Region, string[]> = {
  A: ['P1', 'P2', 'P4', 'P8', 'P7'],
  B: ['P2', 'P3', 'P5', 'P4'],
  C: ['P4', 'P5', 'P11', 'P9', 'P8'],
  D: ['P3', 'P10', 'P9', 'P11', 'P5'],
};
const BORDER_PATH: Record<string, string[]> = {
  AB: ['P2', 'P4'],
  AC: ['P4', 'P8'],
  BC: ['P4', 'P5'],
  BD: ['P3', 'P5'],
  CD: ['P5', 'P11', 'P9'],
};
const LABEL: Record<Region, [number, number]> = {
  A: [58, 128],
  B: [170, 60],
  C: [160, 196],
  D: [222, 120],
};

const DEMO: [Region, number | null][] = [
  ['A', 0],
  ['B', 0],
  ['B', 1],
  ['C', 2],
  ['D', 0],
];
const EMPTY: Colouring = { A: null, B: null, C: null, D: null };

export default function MapToSat() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const [col, setCol] = useState<Colouring>(EMPTY);
  const [hoverR, setHoverR] = useState<Region | null>(null);
  const [hoverC, setHoverC] = useState<number | null>(null);
  const timer = useRef<number | null>(null);
  const started = useRef(false);

  const stopDemo = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = null;
  };
  const runDemo = (delay: number) => {
    stopDemo();
    let i = 0;
    const next = () => {
      const [r, c] = DEMO[i];
      setCol((s) => ({ ...s, [r]: c }));
      i++;
      if (i < DEMO.length) timer.current = window.setTimeout(next, 800);
      else timer.current = null;
    };
    timer.current = window.setTimeout(next, delay);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      setCol({ A: 0, B: 1, C: 2, D: 0 });
      return;
    }
    runDemo(700);
  }, [inView, reduced]);
  useEffect(() => stopDemo, []);
  // reduced motion (known only after mount): no demo, show the finished colouring
  useEffect(() => {
    if (!reduced) return;
    stopDemo();
    started.current = true;
    setCol({ A: 0, B: 1, C: 2, D: 0 });
  }, [reduced]);

  const restart = () => {
    started.current = true;
    stopDemo();
    setCol(EMPTY);
    if (reduced) setCol({ A: 0, B: 1, C: 2, D: 0 });
    else runDemo(1000);
  };
  const cycle = (r: Region) => {
    started.current = true;
    stopDemo();
    setCol((s) => ({ ...s, [r]: s[r] === null ? 0 : s[r] === 2 ? null : (s[r] as number) + 1 }));
  };
  const setLamp = (r: Region, c: number) => {
    started.current = true;
    stopDemo();
    setCol((s) => ({ ...s, [r]: s[r] === c ? null : c }));
  };

  const truth = CLAUSES.map((cl) => clauseTrue(cl, col));
  const nTrue = truth.filter(Boolean).length;
  const conflicts = BORDERS.filter(([a, b]) => col[a] !== null && col[a] === col[b]);
  const uncoloured = REGIONS.filter((r) => col[r] === null);
  const capIdx =
    nTrue === CLAUSES.length ? 3 : conflicts.length ? 2 : uncoloured.length === 4 ? 0 : 1;

  // what is highlighted by hovering
  const hc: Clause | null = hoverC === null ? null : CLAUSES[hoverC];
  const regionHot = (r: Region) =>
    hc ? (hc.kind === 'some' ? hc.r === r : hc.a === r || hc.b === r) : hoverR === r;
  const anyHot = hc !== null || hoverR !== null;
  const clauseHot = (cl: Clause, i: number) =>
    hoverC === i ||
    (hoverR !== null &&
      (cl.kind === 'some' ? cl.r === hoverR : cl.a === hoverR || cl.b === hoverR));

  /* ── layout ── */
  const W = compact ? 360 : 640;
  const H = compact ? 516 : 262;
  const MX = compact ? 50 : 6;
  const MY = compact ? 4 : 12;
  const pt = (k: string) => `${V[k][0] + MX},${V[k][1] + MY}`;

  // clause panel geometry
  const someX = compact ? (i: number) => 14 + (i % 2) * 172 : () => 300;
  const someY = compact ? (i: number) => 284 + Math.floor(i / 2) * 32 : (i: number) => 52 + i * 34;
  const SOME_W = compact ? 160 : 112;
  const diffRowY = compact ? (j: number) => 384 + j * 26 : (j: number) => 52 + j * 34;
  const diffColX = compact ? (c: number) => 62 + c * 98 : (c: number) => 470 + c * 56;
  const DIFF_W = compact ? 90 : 52;
  const CH = compact ? 22 : 26;
  const diffLabelX = compact ? 14 : 432;
  const headY1 = compact ? 268 : 30;
  const headY2 = compact ? 354 : 30;

  const chip = (i: number, x: number, yy: number, w: number, content: React.ReactNode) => {
    const ok = truth[i];
    const hot = clauseHot(CLAUSES[i], i);
    return (
      <g
        key={i}
        onPointerEnter={(e) => e.pointerType !== 'touch' && setHoverC(i)}
        onPointerLeave={() => setHoverC(null)}
        style={{ opacity: anyHot && !hot ? 0.35 : 1, transition: 'opacity 150ms' }}
      >
        <rect
          x={x}
          y={yy}
          width={w}
          height={CH}
          style={{
            fill: ok ? GT.optSoft : GT.bSoft,
            stroke: ok ? GT.opt : GT.b,
            strokeWidth: hot ? 2 : 1.2,
            transition: 'fill 200ms, stroke 200ms',
          }}
        />
        {content}
      </g>
    );
  };

  const sq = (x: number, yy: number, c: number, s = 7) => (
    <rect x={x} y={yy - s / 2} width={s} height={s} style={{ fill: COLS[c] }} />
  );

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-05.1"
      title="a map becomes a formula"
      kicker="12 switches · 19 clauses"
      hint="click a country, or one of its three switches"
      footer={<Btn onClick={restart}>restart</Btn>}
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="A map of four countries A to D and the 19 clauses that say it is properly coloured"
        >
          {/* countries */}
          {REGIONS.map((r) => {
            const c = col[r];
            const hot = regionHot(r);
            return (
              <g
                key={r}
                onClick={() => cycle(r)}
                onPointerEnter={(e) => e.pointerType !== 'touch' && setHoverR(r)}
                onPointerLeave={() => setHoverR(null)}
                style={{ cursor: 'pointer' }}
              >
                <polygon
                  points={POLY[r].map(pt).join(' ')}
                  style={{
                    fill: c === null ? GT.paper : COLS[c],
                    fillOpacity: c === null ? 1 : 0.85,
                    stroke: GT.ink,
                    strokeWidth: 1.5,
                    strokeLinejoin: 'round',
                    opacity: anyHot && !hot ? 0.4 : 1,
                    transition: 'fill 220ms, opacity 150ms',
                  }}
                />
              </g>
            );
          })}
          {/* borders where both sides share a colour */}
          {BORDERS.map(([a, b]) => {
            const key = a + b;
            const bad = col[a] !== null && col[a] === col[b];
            const hot = hc?.kind === 'diff' && hc.a === a && hc.b === b;
            if (!bad && !hot) return null;
            return (
              <polyline
                key={key}
                points={BORDER_PATH[key].map(pt).join(' ')}
                style={{
                  fill: 'none',
                  stroke: bad ? GT.b : GT.ink,
                  strokeWidth: 5,
                  strokeLinecap: 'round',
                  pointerEvents: 'none',
                }}
              />
            );
          })}
          {/* labels and the three switches per country */}
          {REGIONS.map((r) => {
            const [lx, ly] = LABEL[r];
            const x0 = lx + MX;
            const y0 = ly + MY;
            return (
              <g
                key={'l' + r}
                onPointerEnter={(e) => e.pointerType !== 'touch' && setHoverR(r)}
                onPointerLeave={() => setHoverR(null)}
              >
                <g style={{ pointerEvents: 'none' }}>
                  <rect
                    x={x0 - 26}
                    y={y0 - 24}
                    width={52}
                    height={44}
                    style={{ fill: GT.paperRaised, opacity: 0.88 }}
                  />
                  <T x={x0} y={y0 - 11} size={17} font="display" color={GT.ink}>
                    {r}
                  </T>
                </g>
                {[0, 1, 2].map((c) => {
                  const on = col[r] === c;
                  return (
                    <rect
                      key={c}
                      x={x0 - 21 + c * 15}
                      y={y0 + 2}
                      width={12}
                      height={12}
                      onClick={() => setLamp(r, c)}
                      style={{
                        fill: on ? COLS[c] : GT.paperRaised,
                        stroke: COLS[c],
                        strokeWidth: 1.8,
                        cursor: 'pointer',
                      }}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* clause panel headings */}
          <T x={someX(0)} y={headY1} anchor="start" size={10} color={GT.ink}>
            {compact ? 'each country gets a colour' : 'gets a colour'}
          </T>
          <T x={diffLabelX} y={headY2} anchor="start" size={10} color={GT.ink}>
            {compact ? 'neighbours not both' : 'neighbours: not both'}
          </T>
          {[0, 1, 2].map((c) => (
            <g key={'h' + c}>
              {sq(diffColX(c) + DIFF_W / 2 - 4.5, headY2 + (compact ? 17 : 12), c, 9)}
            </g>
          ))}

          {/* "some colour" clauses: A■ ∨ A■ ∨ A■ */}
          {CLAUSES.map((cl, i) => {
            if (cl.kind !== 'some') return null;
            const k = REGIONS.indexOf(cl.r);
            const x0 = someX(k);
            const y0 = someY(k);
            const step = compact ? 50 : 34;
            const pad = compact ? 16 : 8;
            return chip(
              i,
              x0,
              y0,
              SOME_W,
              <g style={{ pointerEvents: 'none' }}>
                {[0, 1, 2].map((c) => (
                  <g key={c}>
                    <T
                      x={x0 + pad + c * step}
                      y={y0 + CH / 2 + 1}
                      anchor="start"
                      size={11}
                      color={GT.ink}
                    >
                      {cl.r}
                    </T>
                    {sq(x0 + pad + 9 + c * step, y0 + CH / 2 + 3, c, 6)}
                    {c < 2 && (
                      <T
                        x={x0 + pad + 23 + c * step + (compact ? 7 : 0)}
                        y={y0 + CH / 2 + 1}
                        size={11}
                        color={GT.inkSoft}
                      >
                        ∨
                      </T>
                    )}
                  </g>
                ))}
              </g>,
            );
          })}

          {/* "not both" clauses: one row per border, one column per colour */}
          {BORDERS.map(([a, b], j) => (
            <T key={'r' + j} x={diffLabelX} y={diffRowY(j) + CH / 2 + 1} anchor="start" size={10}>
              {a}–{b}
            </T>
          ))}
          {CLAUSES.map((cl, i) => {
            if (cl.kind !== 'diff') return null;
            const j = BORDERS.findIndex(([a, b]) => a === cl.a && b === cl.b);
            const x0 = diffColX(cl.c);
            const y0 = diffRowY(j);
            return chip(
              i,
              x0,
              y0,
              DIFF_W,
              <g style={{ pointerEvents: 'none' }}>
                <T x={x0 + DIFF_W / 2} y={y0 + CH / 2 + 1} size={11} color={GT.ink}>
                  ¬{cl.a} ∨ ¬{cl.b}
                </T>
              </g>,
            );
          })}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="clauses true"
          value={`${nTrue} / ${CLAUSES.length}`}
          color={nTrue === CLAUSES.length ? GT.opt : GT.ink}
        />
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="the map"
          value={nTrue === CLAUSES.length ? 'coloured' : conflicts.length ? 'clash' : 'unfinished'}
          color={nTrue === CLAUSES.length ? GT.opt : conflicts.length ? GT.b : GT.muted}
        />
      </div>
      <Reserve
        active={capIdx}
        items={[
          'Each country has three switches, one per colour. With every switch off, the four "gets a colour" clauses are false.',
          'A country with no colour leaves its "gets a colour" clause false.',
          'Two neighbours share a colour (red border), so one "not both" clause is false.',
          `A proper colouring, and all ${CLAUSES.length} clauses are true. It works backwards too: any way to make all ${CLAUSES.length} true gives a proper colouring.`,
        ].map((t) => (
          <Caption key={t}>{t}</Caption>
        ))}
      />
    </Specimen>
  );
}
