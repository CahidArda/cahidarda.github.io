/**
 * The series map. The hatched outer frame is the undecidable set; inside it the decidable box
 * holds EXPTIME, PSPACE, NP and P nested inside each other, with NP-complete inside NP. Every
 * set's name sits at the top left of its box, and the problems that belong to it (and to no
 * smaller set) sit on the same row as the name.
 *
 * Hover (or focus, or tap) a problem or a class to read one line about it in the card on the
 * right; problems also show their page number. Click to open the page. Hover selects at once and
 * the selection stays when the pointer leaves. A slow tour visits each problem until the reader
 * takes over.
 */
import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react';
import { GT, Specimen, useInView, useReducedMotion } from '../gt/shared';

/** page numbers, as ordered by seriesOrder */
const PAGE: Record<string, string> = {
  growth: '01',
  p: '02',
  np: '03',
  'p-vs-np': '04',
  'np-complete': '05',
  tsp: '06',
  'phase-transition': '07',
  factoring: '08',
  halting: '09',
  unprovable: '10',
  'beyond-np': '11',
  minesweeper: '12',
  'exam-timetable': '13',
  tiling: '14',
  'prime-polynomial': '15',
};

type DotId =
  | 'sort'
  | 'route'
  | 'prime'
  | 'factor'
  | 'sudoku'
  | 'tsp'
  | 'mines'
  | 'exams'
  | 'sat'
  | 'go'
  | 'chess'
  | 'halt'
  | 'tiles'
  | 'dioph';
type RegId = 'dec' | 'exp' | 'pspace' | 'np' | 'npc' | 'p';
type LabelId = RegId | 'und';

const DOTS: { id: DotId; label: string; cls: string; line: string; slug: string }[] = [
  {
    id: 'sort',
    label: 'sorting',
    cls: 'P',
    line: 'Put a million names in order in a blink: n log n steps.',
    slug: 'p',
  },
  {
    id: 'route',
    label: 'routes',
    cls: 'P',
    line: 'Shortest drive between two towns: Dijkstra settles it fast.',
    slug: 'p',
  },
  {
    id: 'prime',
    label: 'is it prime?',
    cls: 'P',
    line: 'Testing whether a number is prime is in P (AKS, 2002).',
    slug: 'factoring',
  },
  {
    id: 'factor',
    label: 'factoring',
    cls: 'NP',
    line: 'Splitting a number into primes: easy to check, no fast classical method known. A big quantum computer would do it fast.',
    slug: 'factoring',
  },
  {
    id: 'sudoku',
    label: 'Sudoku',
    cls: 'NP-complete',
    line: 'Hard to solve, instant to check (for n×n grids).',
    slug: 'np',
  },
  {
    id: 'tsp',
    label: 'salesman',
    cls: 'NP-complete',
    line: 'The shortest tour through every city: (n−1)!/2 tours to try.',
    slug: 'tsp',
  },
  {
    id: 'mines',
    label: 'Minesweeper',
    cls: 'NP-complete',
    line: 'Is this board even possible? Kaye, 2000.',
    slug: 'minesweeper',
  },
  {
    id: 'exams',
    label: 'exam slots',
    cls: 'NP-complete',
    line: 'The fewest exam slots with no clashes is graph colouring.',
    slug: 'exam-timetable',
  },
  {
    id: 'sat',
    label: '3-SAT',
    cls: 'NP-complete',
    line: 'The first NP-complete problem, and where the hard instances hide.',
    slug: 'phase-transition',
  },
  {
    id: 'go',
    label: 'Go',
    cls: 'PSPACE-hard',
    line: 'Who wins from this position, on an n×n board?',
    slug: 'beyond-np',
  },
  {
    id: 'chess',
    label: 'chess',
    cls: 'EXPTIME-complete',
    line: 'Who wins, on an n×n board? Provably out of reach of any fast algorithm.',
    slug: 'beyond-np',
  },
  {
    id: 'halt',
    label: 'halting',
    cls: 'undecidable',
    line: 'Will this program ever stop? No algorithm can always answer.',
    slug: 'halting',
  },
  {
    id: 'tiles',
    label: 'tiling',
    cls: 'undecidable',
    line: 'Do these square tiles cover the whole plane? Berger, 1966: no algorithm can always say.',
    slug: 'tiling',
  },
  {
    id: 'dioph',
    label: 'integer equations',
    cls: 'undecidable',
    line: 'Does this polynomial equation have a whole-number solution? Hilbert asked in 1900; the answer (1970): no method exists.',
    slug: 'prime-polynomial',
  },
];

const REGIONS: Record<LabelId, { label: string; line: string; slug: string }> = {
  und: {
    label: 'UNDECIDABLE',
    line: 'No algorithm solves these at all, given any amount of time. The hatched band: everything outside the decidable box.',
    slug: 'halting',
  },
  dec: {
    label: 'DECIDABLE',
    line: 'Some algorithm always answers, given enough time. Everything inside this box.',
    slug: 'halting',
  },
  exp: {
    label: 'EXPTIME',
    line: 'Solvable in exponential time. Provably bigger than P.',
    slug: 'beyond-np',
  },
  pspace: {
    label: 'PSPACE',
    line: 'Solvable with a modest amount of memory, however long it takes.',
    slug: 'beyond-np',
  },
  np: {
    label: 'NP',
    line: 'Checkable fast: a proposed answer can be verified quickly. Is NP bigger than P? Nobody knows.',
    slug: 'p-vs-np',
  },
  npc: {
    label: 'NP-COMPLETE',
    line: 'The hardest problems in NP: solve one fast and you solve them all.',
    slug: 'np-complete',
  },
  p: {
    label: 'P',
    line: 'Solvable fast: the time grows like a polynomial in the input size.',
    slug: 'p',
  },
};
const REG_ORDER: RegId[] = ['dec', 'exp', 'pspace', 'np', 'p', 'npc'];
const TITLE: Record<LabelId, string> = {
  und: 'undecidable',
  dec: 'decidable',
  exp: 'EXPTIME',
  pspace: 'PSPACE',
  np: 'NP',
  npc: 'NP-complete',
  p: 'P',
};

type Rect = { x: number; y: number; w: number; h: number };
type Pt = { x: number; y: number };

/**
 * One layout for every width. Each set's name sits at the top left of its box, and the problems
 * that belong to that set (and to no smaller one) sit on the same row, right after the name.
 */
const W = 420;
const H = 336;
const FRAME: Rect = { x: 1, y: 1, w: 418, h: 334 };
const REG: Record<RegId, Rect> = {
  dec: { x: 10, y: 28, w: 400, h: 298 },
  exp: { x: 20, y: 54, w: 380, h: 262 },
  pspace: { x: 30, y: 80, w: 360, h: 226 },
  np: { x: 40, y: 106, w: 340, h: 190 },
  p: { x: 50, y: 136, w: 150, h: 150 },
  npc: { x: 210, y: 136, w: 160, h: 150 },
};
const LAB: Record<LabelId, Pt> = {
  und: { x: 12, y: 15 },
  dec: { x: 22, y: 42 },
  exp: { x: 32, y: 68 },
  pspace: { x: 42, y: 94 },
  np: { x: 52, y: 121 },
  p: { x: 62, y: 152 },
  npc: { x: 222, y: 152 },
};
const DOT: Record<DotId, Pt> = {
  halt: { x: 106, y: 15 },
  tiles: { x: 194, y: 15 },
  dioph: { x: 272, y: 15 },
  chess: { x: 112, y: 68 },
  go: { x: 112, y: 94 },
  factor: { x: 112, y: 121 },
  sort: { x: 68, y: 182 },
  route: { x: 68, y: 214 },
  prime: { x: 68, y: 246 },
  sudoku: { x: 228, y: 176 },
  tsp: { x: 228, y: 200 },
  mines: { x: 228, y: 224 },
  exams: { x: 228, y: 248 },
  sat: { x: 228, y: 272 },
};
/** how tall a dot's hover strip is: rows touch, so moving down a column goes dot to dot */
const ROW: Record<DotId, number> = {
  halt: 22,
  tiles: 22,
  dioph: 22,
  chess: 22,
  go: 22,
  factor: 22,
  sort: 32,
  route: 32,
  prime: 32,
  sudoku: 24,
  tsp: 24,
  mines: 24,
  exams: 24,
  sat: 24,
};

const href = (slug: string) => `/articles/how-hard-is-hard/${slug}`;
const NUM = { fontVariantNumeric: 'lining-nums tabular-nums' } as const;
/** a paper halo, so text stays readable over the undecidable hatching */
const HALO: CSSProperties = {
  paintOrder: 'stroke',
  stroke: GT.paper,
  strokeWidth: 4,
  strokeLinejoin: 'round',
};

type Sel = `dot:${DotId}` | `reg:${LabelId}`;
const info = (sel: Sel) => {
  if (sel.startsWith('dot:')) {
    const d = DOTS.find((x) => `dot:${x.id}` === sel)!;
    return { title: d.label, cls: d.cls, line: d.line, slug: d.slug };
  }
  const id = sel.slice(4) as LabelId;
  const r = REGIONS[id];
  return { title: TITLE[id], cls: 'class', line: r.line, slug: r.slug };
};

type Handlers = {
  pick: (key: Sel) => void;
  click: (key: Sel, e: ReactMouseEvent) => void;
};

function ZooMap({
  sel,
  h,
  onPointerType,
}: {
  sel: Sel;
  h: Handlers;
  onPointerType: (t: string) => void;
}) {
  const hatch = `cx-hatch-${useId().replace(/:/g, '')}`;
  const svgRef = useRef<SVGSVGElement>(null);
  // the page tag sits right after the selected problem's name, so it is measured
  const [tag, setTag] = useState<{ x: number; y: number } | null>(null);
  useLayoutEffect(() => {
    const t = sel.startsWith('dot:')
      ? svgRef.current?.querySelector<SVGTextElement>(`[data-key="${sel}"]`)
      : null;
    if (!t || !t.getBBox) return setTag(null);
    const b = t.getBBox();
    if (!b.width) return setTag(null);
    setTag({ x: b.x + b.width + 5, y: b.y + b.height / 2 });
  }, [sel]);

  const page = PAGE[info(sel).slug];
  const on = (k: Sel) => sel === k;
  const link = (key: Sel, slug: string, label: string) => ({
    href: href(slug),
    className: 'cx-link',
    'aria-label': `${label}, page ${PAGE[slug]}`,
    onMouseEnter: () => h.pick(key),
    onFocus: () => h.pick(key),
    onClick: (e: ReactMouseEvent) => h.click(key, e),
    style: { cursor: 'pointer', textDecoration: 'none' },
  });

  const regLabel = (id: LabelId) => {
    const key = `reg:${id}` as Sel;
    const p = LAB[id];
    return (
      <text
        data-key={key}
        data-label={id}
        x={p.x}
        y={p.y}
        dominantBaseline="middle"
        style={{
          fill: on(key) ? GT.eq : GT.ink,
          fontFamily: GT.mono,
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: '0.08em',
          ...(id === 'und' ? HALO : {}),
        }}
      >
        {REGIONS[id].label}
      </text>
    );
  };

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full"
      role="group"
      aria-label="Map of complexity classes with famous problems placed inside"
      onPointerDown={(e) => onPointerType(e.pointerType)}
    >
      <defs>
        <pattern
          id={hatch}
          width={7}
          height={7}
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <line x1={0} y1={0} x2={0} y2={7} style={{ stroke: GT.line, strokeWidth: 2 }} />
        </pattern>
      </defs>

      {/* the outer set: whatever is not in the decidable box is undecidable */}
      <a {...link('reg:und', REGIONS.und.slug, 'undecidable')}>
        <rect
          data-region="frame"
          x={FRAME.x}
          y={FRAME.y}
          width={FRAME.w}
          height={FRAME.h}
          rx={10}
          style={{
            fill: `url(#${hatch})`,
            stroke: on('reg:und') ? GT.eq : GT.lineStrong,
            strokeWidth: on('reg:und') ? 2.5 : 1.5,
          }}
        />
        {regLabel('und')}
      </a>

      {REG_ORDER.map((id) => {
        const r = REG[id];
        const key = `reg:${id}` as Sel;
        const fill = {
          dec: GT.paper,
          exp: GT.paperRaised,
          pspace: GT.paper,
          np: GT.aSoft,
          p: GT.optSoft,
          npc: GT.bSoft,
        }[id];
        const stroke = {
          dec: GT.ink,
          exp: GT.lineStrong,
          pspace: GT.lineStrong,
          np: GT.a,
          p: GT.opt,
          npc: GT.b,
        }[id];
        return (
          <a key={id} {...link(key, REGIONS[id].slug, TITLE[id])}>
            <rect
              data-region={id}
              x={r.x}
              y={r.y}
              width={r.w}
              height={r.h}
              rx={8}
              style={{
                fill,
                stroke: on(key) ? GT.eq : stroke,
                strokeWidth: on(key) ? 2.5 : 1.5,
              }}
            />
            {regLabel(id)}
          </a>
        );
      })}

      {DOTS.map((d) => {
        const key = `dot:${d.id}` as Sel;
        const p = DOT[d.id];
        const o = on(key);
        const rh = ROW[d.id];
        return (
          <a key={d.id} {...link(key, d.slug, `${d.label}: ${d.cls}`)}>
            {/* hover strip: the dot, its name and a little air */}
            <rect
              x={p.x - 9}
              y={p.y - rh / 2}
              width={24 + d.label.length * 7}
              height={rh}
              style={{ fill: 'transparent' }}
            />
            {o && (
              <circle
                cx={p.x}
                cy={p.y}
                r={7.5}
                style={{ fill: 'none', stroke: GT.eq, strokeWidth: 1.75 }}
              />
            )}
            <circle data-dot={d.id} cx={p.x} cy={p.y} r={4} style={{ fill: o ? GT.eq : GT.ink }} />
            <text
              data-key={key}
              data-label={`dot-${d.id}`}
              x={p.x + 11}
              y={p.y + 0.5}
              dominantBaseline="middle"
              style={{
                fill: o ? GT.eq : GT.ink,
                fontFamily: 'var(--font-sans)',
                fontSize: 12.5,
                fontWeight: o ? 600 : 400,
                ...(d.cls === 'undecidable' ? HALO : {}),
              }}
            >
              {d.label}
            </text>
          </a>
        );
      })}

      {/* page tag on the selected problem */}
      {tag && (
        <g data-tag style={{ pointerEvents: 'none' }}>
          <rect x={tag.x} y={tag.y - 6.5} width={19} height={13} rx={2} style={{ fill: GT.eq }} />
          <text
            x={tag.x + 9.5}
            y={tag.y + 0.5}
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fill: GT.paper, fontFamily: GT.mono, fontSize: 9.5, fontWeight: 700, ...NUM }}
          >
            {page}
          </text>
        </g>
      )}
    </svg>
  );
}

export default function Zoo() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [sel, setSel] = useState<Sel>('dot:sudoku');
  const [touched, setTouched] = useState(false);
  const pointer = useRef('mouse');
  const selAtDown = useRef<Sel | null>(null);
  const tour: Sel[] = DOTS.map((d) => `dot:${d.id}` as Sel);

  useEffect(() => {
    if (reduced || touched || !inView) return;
    const id = setInterval(() => setSel((s) => tour[(tour.indexOf(s) + 1) % tour.length]), 2400);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, touched, inView]);

  const h: Handlers = {
    // hover selects at once; leaving the map keeps the last selection
    pick: (key) => {
      setTouched(true);
      setSel(key);
    },
    // on a phone the first tap selects (and shows the card); a second tap opens the page
    click: (key, e) => {
      // compare with the selection before the tap: tapping a link focuses it, which selects it
      if (pointer.current !== 'mouse' && pointer.current !== 'pen' && selAtDown.current !== key) {
        e.preventDefault();
        setTouched(true);
        setSel(key);
      }
    },
  };

  const c = info(sel);
  const page = PAGE[c.slug];

  return (
    <Specimen
      code="CX-00"
      title="the complexity zoo"
      kicker="14 problems on one map"
      hint="hover a problem or a class · click to open its page"
      rootRef={rootRef}
    >
      <div className="@container">
        <div className="grid gap-4 @min-[560px]:grid-cols-[minmax(0,1fr)_minmax(0,12.5rem)]">
          <ZooMap
            sel={sel}
            h={h}
            onPointerType={(t) => {
              pointer.current = t;
              selAtDown.current = sel;
            }}
          />

          <a
            href={href(c.slug)}
            data-card
            className="flex flex-col border border-line-strong bg-paper p-4 transition-colors hover:border-ink"
          >
            <span className="label block" style={{ color: GT.eq, ...NUM }}>
              page {page}
            </span>
            <span className="mt-1 block font-display text-2xl leading-tight text-ink">
              {c.title}
            </span>
            <span className="label mt-1 block text-muted">{c.cls}</span>
            <span className="mt-2 block min-h-[2.8em] @min-[560px]:min-h-[5.5em] text-[0.9rem] leading-snug text-ink-soft">
              {c.line}
            </span>
            <span className="mt-auto block pt-2 font-mono text-[0.7rem] text-accent" style={NUM}>
              open page {page} →
            </span>
          </a>
        </div>
      </div>
    </Specimen>
  );
}
