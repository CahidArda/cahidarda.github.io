/**
 * The series map: the complexity zoo as nested regions. P sits inside NP, NP inside PSPACE,
 * PSPACE inside EXPTIME; BQP (quantum) overlaps P and part of NP; undecidable problems live
 * outside every region. Famous problems are dots; hover (or focus) a dot or a region to read
 * one line about it, click to open its page. A slow tour visits each dot until the reader
 * takes over.
 */
import { useEffect, useState } from 'react';
import { GT, Specimen, useInView, useReducedMotion } from '../gt/shared';

type Item = {
  id: string;
  label: string;
  cls: string;
  line: string;
  slug: string;
  x: number;
  y: number;
  lx?: number;
  ly?: number;
  anchor?: 'start' | 'middle' | 'end';
};

const W = 520;
const H = 440;
const R = {
  exp: { x: 14, y: 34, w: 412, h: 392 },
  pspace: { x: 28, y: 76, w: 340, h: 336 },
  np: { x: 42, y: 118, w: 270, h: 280 },
  npc: { x: 262, y: 118, w: 50, h: 280 },
  p: { x: 56, y: 200, w: 110, h: 180 },
};
const BQP = { cx: 150, cy: 300, rx: 80, ry: 50 };

const DOTS: Item[] = [
  {
    id: 'sort',
    label: 'sorting',
    cls: 'P',
    line: 'Put a million names in order in a blink: n log n steps.',
    slug: 'p',
    x: 82,
    y: 236,
    lx: 90,
    ly: 236,
    anchor: 'start',
  },
  {
    id: 'route',
    label: 'routes',
    cls: 'P',
    line: 'Shortest drive between two towns: Dijkstra settles it fast.',
    slug: 'p',
    x: 82,
    y: 268,
    lx: 90,
    ly: 268,
    anchor: 'start',
  },
  {
    id: 'prime',
    label: 'is it prime?',
    cls: 'P',
    line: 'Testing whether a number is prime is in P (AKS, 2002).',
    slug: 'factoring',
    x: 82,
    y: 368,
    lx: 90,
    ly: 368,
    anchor: 'start',
  },
  {
    id: 'factor',
    label: 'factoring',
    cls: 'NP, and BQP',
    line: 'Splitting a number into primes: no fast classical method known, fast on a big quantum computer.',
    slug: 'factoring',
    x: 214,
    y: 300,
    lx: 214,
    ly: 286,
    anchor: 'middle',
  },
  {
    id: 'sudoku',
    label: 'Sudoku',
    cls: 'NP-complete',
    line: 'Hard to solve, instant to check (for n×n grids).',
    slug: 'np',
    x: 287,
    y: 150,
    lx: 255,
    ly: 150,
    anchor: 'end',
  },
  {
    id: 'tsp',
    label: 'salesman',
    cls: 'NP-complete',
    line: 'The shortest tour through every city: (n−1)!/2 tours to try.',
    slug: 'tsp',
    x: 287,
    y: 196,
    lx: 255,
    ly: 196,
    anchor: 'end',
  },
  {
    id: 'mines',
    label: 'Minesweeper',
    cls: 'NP-complete',
    line: 'Is this board even possible? Kaye, 2000.',
    slug: 'minesweeper',
    x: 287,
    y: 242,
    lx: 255,
    ly: 242,
    anchor: 'end',
  },
  {
    id: 'exams',
    label: 'exam slots',
    cls: 'NP-complete',
    line: 'The fewest exam slots with no clashes is graph colouring.',
    slug: 'exam-timetable',
    x: 287,
    y: 334,
    lx: 255,
    ly: 334,
    anchor: 'end',
  },
  {
    id: 'sat',
    label: '3-SAT',
    cls: 'NP-complete',
    line: 'The first NP-complete problem, and where the hard instances hide.',
    slug: 'phase-transition',
    x: 287,
    y: 380,
    lx: 255,
    ly: 380,
    anchor: 'end',
  },
  {
    id: 'go',
    label: 'Go',
    cls: 'PSPACE-hard',
    line: 'Who wins from this position, on an n×n board?',
    slug: 'beyond-np',
    x: 340,
    y: 394,
    lx: 340,
    ly: 380,
    anchor: 'middle',
  },
  {
    id: 'chess',
    label: 'chess',
    cls: 'EXPTIME-complete',
    line: 'Who wins, on an n×n board? Provably out of reach of any fast algorithm.',
    slug: 'beyond-np',
    x: 397,
    y: 412,
    lx: 397,
    ly: 398,
    anchor: 'middle',
  },
  {
    id: 'halt',
    label: 'halting',
    cls: 'undecidable',
    line: 'Will this program ever stop? No algorithm can always answer.',
    slug: 'halting',
    x: 473,
    y: 412,
    lx: 473,
    ly: 398,
    anchor: 'middle',
  },
  {
    id: 'tiles',
    label: 'tiling',
    cls: 'undecidable',
    line: 'Do these square tiles cover the whole plane? Berger, 1966: no algorithm can always say.',
    slug: 'tiling',
    x: 473,
    y: 312,
    lx: 473,
    ly: 298,
    anchor: 'middle',
  },
  {
    id: 'dioph',
    label: 'integer equations',
    cls: 'undecidable',
    line: 'Does this polynomial equation have a whole-number solution? Hilbert asked in 1900; the answer (1970): no method exists.',
    slug: 'prime-polynomial',
    x: 473,
    y: 212,
    lx: 473,
    ly: 198,
    anchor: 'middle',
  },
];

const REGIONS: { id: string; label: string; line: string; slug: string }[] = [
  {
    id: 'p',
    label: 'P',
    line: 'Solvable fast: the time grows like a polynomial in the input size.',
    slug: 'p',
  },
  {
    id: 'np',
    label: 'NP',
    line: 'Checkable fast: a proposed answer can be verified quickly.',
    slug: 'np',
  },
  {
    id: 'npc',
    label: 'NP-complete',
    line: 'The hardest problems in NP: solve one fast and you solve them all.',
    slug: 'np-complete',
  },
  {
    id: 'pspace',
    label: 'PSPACE',
    line: 'Solvable with a modest amount of memory, however long it takes.',
    slug: 'beyond-np',
  },
  {
    id: 'exp',
    label: 'EXPTIME',
    line: 'Solvable in exponential time. Provably bigger than P.',
    slug: 'beyond-np',
  },
  { id: 'bqp', label: 'BQP', line: 'Solvable fast on a quantum computer.', slug: 'beyond-np' },
  {
    id: 'und',
    label: 'undecidable',
    line: 'No algorithm solves these at all, given any amount of time.',
    slug: 'halting',
  },
  {
    id: 'pnp',
    label: 'P = NP?',
    line: 'Is the inner circle the whole of NP? Nobody knows. One million dollars.',
    slug: 'p-vs-np',
  },
];

const href = (slug: string) => `/articles/how-hard-is-hard/${slug}`;

export default function Zoo() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [sel, setSel] = useState<string>('dot:sudoku');
  const [touched, setTouched] = useState(false);
  const tour = [...DOTS.map((d) => `dot:${d.id}`), 'reg:pnp'];

  useEffect(() => {
    if (reduced || touched || !inView) return;
    const id = setInterval(() => setSel((s) => tour[(tour.indexOf(s) + 1) % tour.length]), 2400);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, touched, inView]);

  const pick = (key: string) => {
    setTouched(true);
    setSel(key);
  };
  const isDot = sel.startsWith('dot:');
  const dot = isDot ? DOTS.find((d) => `dot:${d.id}` === sel) : undefined;
  const reg = !isDot ? REGIONS.find((r) => `reg:${r.id}` === sel) : undefined;
  const regOn = (id: string) => sel === `reg:${id}`;

  const box = (k: keyof typeof R, fill: string, stroke: string, dash?: string) => {
    const r = R[k];
    const on = regOn(k);
    return (
      <rect
        x={r.x}
        y={r.y}
        width={r.w}
        height={r.h}
        rx={14}
        onMouseEnter={() => pick(`reg:${k}`)}
        style={{
          fill,
          stroke: on ? GT.eq : stroke,
          strokeWidth: on ? 2.5 : 1.5,
          strokeDasharray: dash,
          transition: 'stroke 200ms',
        }}
      />
    );
  };
  const regLabel = (x: number, y: number, text: string, id: string, size = 12) => (
    <a
      href={href(REGIONS.find((r) => r.id === id)!.slug)}
      className="cx-link"
      style={{ textDecoration: 'none' }}
    >
      <text
        x={x}
        y={y}
        onMouseEnter={() => pick(`reg:${id}`)}
        style={{
          fill: regOn(id) ? GT.eq : GT.ink,
          fontFamily: GT.mono,
          fontSize: size,
          fontWeight: 700,
          letterSpacing: '0.08em',
          cursor: 'pointer',
        }}
      >
        {text}
      </text>
    </a>
  );

  const cardTitle = dot ? dot.label : reg ? reg.label : '';
  const cardCls = dot ? dot.cls : 'class';
  const cardLine = dot ? dot.line : reg ? reg.line : '';
  const cardSlug = dot ? dot.slug : reg ? reg.slug : 'p';

  return (
    <Specimen
      code="CX-00"
      title="the complexity zoo"
      kicker="14 problems · 6 classes"
      hint="hover a problem or a region · click to open its page"
      rootRef={rootRef}
    >
      <div className="grid gap-4 sm:grid-cols-[minmax(0,36rem)_minmax(0,1fr)] sm:items-center">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="Map of complexity classes with famous problems placed inside"
        >
          {/* undecidable ground */}
          <rect
            x={0}
            y={0}
            width={W}
            height={H}
            rx={0}
            onMouseEnter={() => pick('reg:und')}
            style={{ fill: 'transparent' }}
          />
          {regLabel(436, 54, 'UNDECIDABLE', 'und', 10)}
          {box('exp', GT.paper, GT.lineStrong)}
          {box('pspace', GT.paperRaised, GT.lineStrong)}
          {box('np', GT.aSoft, GT.a)}
          {box('npc', GT.bSoft, GT.b)}
          {box('p', GT.optSoft, GT.opt)}
          <ellipse
            cx={BQP.cx}
            cy={BQP.cy}
            rx={BQP.rx}
            ry={BQP.ry}
            onMouseEnter={() => pick('reg:bqp')}
            style={{
              fill: 'none',
              stroke: regOn('bqp') ? GT.eq : GT.alt,
              strokeWidth: regOn('bqp') ? 2.5 : 1.5,
              strokeDasharray: '5 4',
            }}
          />
          {regLabel(24, 54, 'EXPTIME', 'exp')}
          {regLabel(38, 96, 'PSPACE', 'pspace')}
          {regLabel(52, 138, 'NP', 'np', 14)}
          {regLabel(66, 220, 'P', 'p', 14)}
          {regLabel(100, 343, 'BQP', 'bqp', 10)}
          <g transform="rotate(90 302 258)">{regLabel(246, 262, 'NP-COMPLETE', 'npc', 10)}</g>
          {regLabel(92, 194, 'P = NP ?', 'pnp', 10)}

          {DOTS.map((d) => {
            const on = sel === `dot:${d.id}`;
            return (
              <a
                key={d.id}
                href={href(d.slug)}
                className="cx-link"
                aria-label={`${d.label}: ${d.cls}`}
                style={{ textDecoration: 'none' }}
              >
                <g
                  onMouseEnter={() => pick(`dot:${d.id}`)}
                  onFocus={() => pick(`dot:${d.id}`)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle cx={d.x} cy={d.y} r={11} style={{ fill: 'transparent' }} />
                  {on && (
                    <circle
                      cx={d.x}
                      cy={d.y}
                      r={9}
                      style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
                    />
                  )}
                  <circle cx={d.x} cy={d.y} r={4.5} style={{ fill: on ? GT.eq : GT.ink }} />
                  <text
                    x={d.lx ?? d.x + 8}
                    y={(d.ly ?? d.y) + 0.5}
                    textAnchor={d.anchor ?? 'start'}
                    dominantBaseline="middle"
                    style={{
                      fill: on ? GT.eq : GT.ink,
                      fontFamily: 'var(--font-sans)',
                      fontSize: 11.5,
                      fontWeight: on ? 600 : 400,
                    }}
                  >
                    {d.label}
                  </text>
                </g>
              </a>
            );
          })}
        </svg>

        <a
          href={href(cardSlug)}
          className="block border border-line-strong bg-paper p-4 transition-colors hover:border-ink"
        >
          <span className="label block" style={{ color: GT.eq }}>
            {cardCls}
          </span>
          <span className="mt-1 block font-display text-2xl leading-tight text-ink">
            {cardTitle}
          </span>
          <span className="mt-2 block min-h-[3.6em] text-[0.9rem] leading-snug text-ink-soft">
            {cardLine}
          </span>
          <span className="mt-2 block font-mono text-[0.7rem] text-accent">open the page →</span>
        </a>
      </div>
    </Specimen>
  );
}
