/**
 * The series map. An outer frame holds all problems. Inside it, one big box holds the decidable
 * ones (some algorithm always answers): EXPTIME, PSPACE, NP and P nested inside each other, with
 * NP-complete as a band on NP's edge. Everything in the frame but outside that box is
 * undecidable, and the undecidable problems sit in their own strip at the bottom.
 *
 * Famous problems are dots. Hover (or focus, or tap) a dot or a region to read one line about it
 * and see its page number; click to open the page. The selection changes only on entering an
 * item (regions after a short dwell, so crossing them on the way to the card does not change
 * it) and stays put when the pointer leaves the map. A slow tour visits each dot until the
 * reader takes over. Phones get their own, taller layout.
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
type LabelId = RegId | 'und' | 'pnp';

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

const REGIONS: Record<LabelId, { label: string; line: string; slug: string; size: number }> = {
  dec: {
    label: 'DECIDABLE',
    line: 'Some algorithm always answers, given enough time. Everything inside this box.',
    slug: 'halting',
    size: 10,
  },
  exp: {
    label: 'EXPTIME',
    line: 'Solvable in exponential time. Provably bigger than P.',
    slug: 'beyond-np',
    size: 11,
  },
  pspace: {
    label: 'PSPACE',
    line: 'Solvable with a modest amount of memory, however long it takes.',
    slug: 'beyond-np',
    size: 11,
  },
  np: {
    label: 'NP',
    line: 'Checkable fast: a proposed answer can be verified quickly.',
    slug: 'np',
    size: 14,
  },
  npc: {
    label: 'NP-COMPLETE',
    line: 'The hardest problems in NP: solve one fast and you solve them all.',
    slug: 'np-complete',
    size: 10,
  },
  p: {
    label: 'P',
    line: 'Solvable fast: the time grows like a polynomial in the input size.',
    slug: 'p',
    size: 14,
  },
  pnp: {
    label: 'P = NP ?',
    line: 'Is the P box the whole of NP? Nobody knows. One million dollars.',
    slug: 'p-vs-np',
    size: 10,
  },
  und: {
    label: 'UNDECIDABLE',
    line: 'No algorithm solves these at all, given any amount of time. Everything outside the decidable box.',
    slug: 'halting',
    size: 10,
  },
};
const REG_ORDER: RegId[] = ['dec', 'exp', 'pspace', 'np', 'p', 'npc'];
const TITLE: Record<LabelId, string> = {
  dec: 'decidable',
  exp: 'EXPTIME',
  pspace: 'PSPACE',
  np: 'NP',
  npc: 'NP-complete',
  p: 'P',
  pnp: 'P = NP ?',
  und: 'undecidable',
};

type Rect = { x: number; y: number; w: number; h: number };
type Pt = { x: number; y: number };
type Layout = {
  W: number;
  H: number;
  frame: Rect;
  reg: Record<RegId, Rect>;
  lab: Record<LabelId, Pt>;
  /** where the one-line gloss after DECIDABLE / UNDECIDABLE starts */
  gloss: number;
  dot: Record<DotId, Pt>;
};

const WIDE: Layout = {
  W: 580,
  H: 500,
  frame: { x: 1, y: 24, w: 578, h: 475 },
  reg: {
    dec: { x: 14, y: 38, w: 552, h: 380 },
    exp: { x: 26, y: 70, w: 528, h: 336 },
    pspace: { x: 38, y: 102, w: 420, h: 290 },
    np: { x: 50, y: 134, w: 320, h: 248 },
    p: { x: 62, y: 200, w: 132, h: 156 },
    npc: { x: 220, y: 134, w: 136, h: 234 },
  },
  lab: {
    dec: { x: 26, y: 58 },
    exp: { x: 38, y: 90 },
    pspace: { x: 50, y: 122 },
    np: { x: 62, y: 156 },
    pnp: { x: 118, y: 155 },
    p: { x: 74, y: 222 },
    npc: { x: 232, y: 156 },
    und: { x: 26, y: 432 },
  },
  gloss: 136,
  dot: {
    factor: { x: 74, y: 182 },
    sort: { x: 80, y: 248 },
    route: { x: 80, y: 280 },
    prime: { x: 80, y: 312 },
    sudoku: { x: 234, y: 190 },
    tsp: { x: 234, y: 226 },
    mines: { x: 234, y: 262 },
    exams: { x: 234, y: 298 },
    sat: { x: 234, y: 334 },
    go: { x: 382, y: 250 },
    chess: { x: 470, y: 250 },
    halt: { x: 34, y: 466 },
    tiles: { x: 190, y: 466 },
    dioph: { x: 330, y: 466 },
  },
};

const NARROW: Layout = {
  W: 360,
  H: 600,
  frame: { x: 1, y: 24, w: 358, h: 575 },
  reg: {
    dec: { x: 12, y: 38, w: 336, h: 490 },
    exp: { x: 22, y: 70, w: 316, h: 446 },
    pspace: { x: 32, y: 102, w: 296, h: 382 },
    np: { x: 42, y: 134, w: 276, h: 314 },
    p: { x: 52, y: 170, w: 150, h: 130 },
    npc: { x: 52, y: 312, w: 256, h: 124 },
  },
  lab: {
    dec: { x: 22, y: 58 },
    exp: { x: 32, y: 90 },
    pspace: { x: 42, y: 122 },
    np: { x: 52, y: 156 },
    pnp: { x: 106, y: 155 },
    p: { x: 62, y: 192 },
    npc: { x: 62, y: 334 },
    und: { x: 22, y: 552 },
  },
  gloss: 124,
  dot: {
    sort: { x: 68, y: 216 },
    route: { x: 68, y: 244 },
    prime: { x: 68, y: 272 },
    factor: { x: 216, y: 236 },
    sudoku: { x: 66, y: 358 },
    tsp: { x: 66, y: 386 },
    mines: { x: 66, y: 414 },
    exams: { x: 190, y: 358 },
    sat: { x: 190, y: 386 },
    go: { x: 56, y: 466 },
    chess: { x: 56, y: 500 },
    halt: { x: 30, y: 580 },
    tiles: { x: 126, y: 580 },
    dioph: { x: 206, y: 580 },
  },
};

const href = (slug: string) => `/articles/how-hard-is-hard/${slug}`;
const NUM = { fontVariantNumeric: 'lining-nums tabular-nums' } as const;
/** a paper halo, so labels stay readable over the undecidable hatching */
const HALO: CSSProperties = {
  paintOrder: 'stroke',
  stroke: GT.paper,
  strokeWidth: 4,
  strokeLinejoin: 'round',
};
const DOT_DWELL = 60; // ms
const REG_DWELL = 220; // ms: crossing a region on the way to the card does not select it

type Sel = `dot:${DotId}` | `reg:${LabelId}`;
const info = (sel: Sel) => {
  if (sel.startsWith('dot:')) {
    const d = DOTS.find((x) => `dot:${x.id}` === sel)!;
    return { title: d.label, cls: d.cls, line: d.line, slug: d.slug };
  }
  const id = sel.slice(4) as LabelId;
  const r = REGIONS[id];
  return {
    title: TITLE[id],
    cls: id === 'pnp' ? 'open question' : 'region',
    line: r.line,
    slug: r.slug,
  };
};

type Handlers = {
  enter: (key: Sel, ms: number) => void;
  focus: (key: Sel) => void;
  click: (key: Sel, e: ReactMouseEvent) => void;
};

/** One map. Two of these exist (wide and narrow); CSS shows the one that fits. */
function ZooMap({
  L,
  sel,
  h,
  onLeave,
  onPointerType,
}: {
  L: Layout;
  sel: Sel;
  h: Handlers;
  onLeave: () => void;
  onPointerType: (t: string) => void;
}) {
  const hatch = `cx-hatch-${useId().replace(/:/g, '')}`;
  const svgRef = useRef<SVGSVGElement>(null);
  // the page tag sits right after the selected item's label, so it is measured
  const [tag, setTag] = useState<{ x: number; y: number } | null>(null);
  useLayoutEffect(() => {
    const t = svgRef.current?.querySelector<SVGTextElement>(`[data-key="${sel}"]`);
    if (!t || !t.getBBox) return setTag(null);
    const b = t.getBBox();
    if (!b.width) return setTag(null);
    setTag({ x: b.x + b.width + 5, y: b.y + b.height / 2 });
  }, [sel, L]);

  const page = PAGE[info(sel).slug];
  const on = (k: Sel) => sel === k;
  const link = (key: Sel, slug: string, label: string) => ({
    href: href(slug),
    className: 'cx-link',
    'aria-label': `${label}, page ${PAGE[slug]}`,
    onMouseEnter: () => h.enter(key, key.startsWith('dot:') ? DOT_DWELL : REG_DWELL),
    onFocus: () => h.focus(key),
    onClick: (e: ReactMouseEvent) => h.click(key, e),
    style: { cursor: 'pointer', textDecoration: 'none' },
  });

  const regLabel = (id: LabelId) => {
    const key = `reg:${id}` as Sel;
    const p = L.lab[id];
    return (
      <text
        data-key={key}
        data-label={id}
        x={p.x}
        y={p.y}
        style={{
          fill: on(key) ? GT.eq : GT.ink,
          fontFamily: GT.mono,
          fontSize: REGIONS[id].size,
          fontWeight: 700,
          letterSpacing: '0.08em',
          ...(id === 'und' ? HALO : {}),
        }}
      >
        {REGIONS[id].label}
      </text>
    );
  };
  const gloss = (y: number, text: string, halo: boolean) => (
    <text
      x={L.gloss}
      y={y}
      data-gloss
      style={{
        fill: GT.inkSoft,
        fontFamily: 'var(--font-sans)',
        fontSize: 11,
        pointerEvents: 'none',
        ...(halo ? HALO : {}),
      }}
    >
      {text}
    </text>
  );

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${L.W} ${L.H}`}
      className="block w-full"
      role="group"
      aria-label="Map of complexity classes with famous problems placed inside"
      onMouseLeave={onLeave}
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
      <text
        x={L.frame.x + 1}
        y={14}
        data-label="all"
        style={{
          fill: GT.muted,
          fontFamily: GT.mono,
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: '0.08em',
        }}
      >
        ALL PROBLEMS
      </text>

      {/* all problems: whatever is not in the decidable box is undecidable */}
      <a {...link('reg:und', REGIONS.und.slug, 'undecidable')}>
        <rect
          data-region="frame"
          x={L.frame.x}
          y={L.frame.y}
          width={L.frame.w}
          height={L.frame.h}
          style={{
            fill: `url(#${hatch})`,
            stroke: on('reg:und') ? GT.eq : GT.lineStrong,
            strokeWidth: on('reg:und') ? 2.5 : 1.5,
            transition: 'stroke 200ms',
          }}
        />
        {regLabel('und')}
      </a>
      {gloss(L.lab.und.y, 'no algorithm can always answer', true)}

      {REG_ORDER.map((id) => {
        const r = L.reg[id];
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
              rx={10}
              style={{
                fill,
                stroke: on(key) ? GT.eq : stroke,
                strokeWidth: on(key) ? 2.5 : 1.5,
                transition: 'stroke 200ms',
              }}
            />
            {regLabel(id)}
          </a>
        );
      })}
      {gloss(L.lab.dec.y, 'some algorithm always answers', false)}
      <a {...link('reg:pnp', REGIONS.pnp.slug, 'P = NP?')}>{regLabel('pnp')}</a>

      {DOTS.map((d) => {
        const key = `dot:${d.id}` as Sel;
        const p = L.dot[d.id];
        const o = on(key);
        return (
          <a key={d.id} {...link(key, d.slug, `${d.label}: ${d.cls}`)}>
            <circle cx={p.x} cy={p.y} r={11} style={{ fill: 'transparent' }} />
            {o && (
              <circle
                cx={p.x}
                cy={p.y}
                r={8.5}
                style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
              />
            )}
            <circle
              data-dot={d.id}
              cx={p.x}
              cy={p.y}
              r={4.5}
              style={{ fill: o ? GT.eq : GT.ink }}
            />
            <text
              data-key={key}
              data-label={`dot-${d.id}`}
              x={p.x + 13}
              y={p.y + 0.5}
              dominantBaseline="middle"
              style={{
                fill: o ? GT.eq : GT.ink,
                fontFamily: 'var(--font-sans)',
                fontSize: 11.5,
                fontWeight: o ? 600 : 400,
                ...(d.cls === 'undecidable' ? HALO : {}),
              }}
            >
              {d.label}
            </text>
          </a>
        );
      })}

      {/* page tag on the selected item */}
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
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pointer = useRef('mouse');
  const selAtDown = useRef<Sel | null>(null);
  const tour: Sel[] = [...DOTS.map((d) => `dot:${d.id}` as Sel), 'reg:pnp'];

  useEffect(() => {
    if (reduced || touched || !inView) return;
    const id = setInterval(() => setSel((s) => tour[(tour.indexOf(s) + 1) % tour.length]), 2400);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, touched, inView]);
  useEffect(
    () => () => {
      if (pending.current) clearTimeout(pending.current);
    },
    [],
  );

  const cancel = () => {
    if (pending.current) clearTimeout(pending.current);
    pending.current = null;
  };
  const pick = (key: Sel) => {
    cancel();
    setTouched(true);
    setSel(key);
  };
  const h: Handlers = {
    enter: (key, ms) => {
      setTouched(true);
      cancel();
      pending.current = setTimeout(() => pick(key), ms);
    },
    focus: pick,
    // on a phone the first tap selects (and shows the card); a second tap opens the page
    click: (key, e) => {
      // compare with the selection before the tap: tapping a link focuses it, which selects it
      if (pointer.current !== 'mouse' && pointer.current !== 'pen' && selAtDown.current !== key) {
        e.preventDefault();
        pick(key);
      }
    },
  };
  const maps = {
    sel,
    h,
    onLeave: cancel,
    onPointerType: (t: string) => {
      pointer.current = t;
      selAtDown.current = sel;
    },
  };

  const c = info(sel);
  const page = PAGE[c.slug];

  return (
    <Specimen
      code="CX-00"
      title="the complexity zoo"
      kicker="14 problems on one map"
      hint="hover a problem or a region · click to open its page"
      rootRef={rootRef}
    >
      {/* side by side only when the figure is wide enough for the map to stay large */}
      <div className="@container">
        <div className="grid gap-4 @min-[760px]:grid-cols-[minmax(0,1fr)_minmax(0,17rem)] @min-[760px]:items-center">
          <div>
            <div className="hidden @min-[480px]:block">
              <ZooMap L={WIDE} {...maps} />
            </div>
            <div className="@min-[480px]:hidden">
              <ZooMap L={NARROW} {...maps} />
            </div>
          </div>

          <a
            href={href(c.slug)}
            data-card
            className="block border border-line-strong bg-paper p-4 transition-colors hover:border-ink"
          >
            <span className="label block" style={{ color: GT.eq, ...NUM }}>
              page {page} · {c.cls}
            </span>
            <span className="mt-1 block font-display text-2xl leading-tight text-ink">
              {c.title}
            </span>
            <span className="mt-2 block min-h-[4.2em] text-[0.9rem] @min-[760px]:min-h-[5.5em] leading-snug text-ink-soft">
              {c.line}
            </span>
            <span className="mt-2 block font-mono text-[0.7rem] text-accent" style={NUM}>
              open page {page} →
            </span>
          </a>
        </div>
      </div>
    </Specimen>
  );
}
