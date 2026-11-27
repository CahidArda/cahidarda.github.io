/**
 * CX-13.1 "build the timetable". Twelve made-up courses; a line joins two courses that share a
 * student. Each course gets an exam slot (a colour) so that no line joins two of the same
 * colour. Left: the clash graph. Right: the timetable, one column per course, one row per slot,
 * filled as the algorithm goes. Tabs: greedy in A-to-Z order (5 slots) vs DSatur (4, the
 * fewest possible). After the demo, drag a filled cell to another row: clashes light up red.
 */
import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import {
  COURSES,
  EXAMS,
  FULL,
  chromatic,
  clashes,
  dsatur,
  edges,
  greedy,
  slotsUsed,
} from './graph';

type Tab = 'greedy' | 'dsatur';
const RUNS: Record<Tab, { order: number[]; col: number[] }> = {
  greedy: greedy(
    EXAMS,
    COURSES.map((_, i) => i),
  ),
  dsatur: dsatur(EXAMS),
};
const TABS: { value: Tab; label: string }[] = [
  { value: 'greedy', label: 'greedy, A to Z' },
  { value: 'dsatur', label: 'DSatur' },
];
const CAPTION: Record<Tab, string> = {
  greedy:
    'Greedy takes the courses in list order and gives each the first slot none of its neighbours has. Fast and simple, and here it needs 5 slots.',
  dsatur:
    'DSatur (Brélaz, 1979) picks next the course whose neighbours already block the most different slots. It starts with Statistics and fits everything in 4.',
};
const BEST = chromatic(EXAMS); // 4
const EDGES = edges(EXAMS);
const SLOTS = 5;
const SLOT_COLORS = [GT.a, GT.opt, GT.warn, GT.alt, GT.inkSoft];

// node positions (graph space 320 x 270), from a seeded spring layout
const POS: Record<string, [number, number]> = {
  Art: [30, 99],
  Bio: [290, 32],
  Calc: [224, 162],
  Chem: [290, 119],
  Econ: [158, 83],
  Hist: [68, 60],
  Law: [148, 30],
  Ling: [58, 240],
  Phil: [49, 167],
  Phys: [264, 198],
  Prog: [156, 202],
  Stat: [225, 106],
};
const NR = 17;

// timeline
const T0 = 0.4;
const STEP = 0.5;
const T_END = T0 + COURSES.length * STEP + 0.2;

type State = { t: number; user: number[] | null };
const EMPTY: State = { t: 0, user: null };
const POSTER: State = { t: T_END, user: null };

function view(tab: Tab, s: State) {
  const run = RUNS[tab];
  const k = Math.floor((s.t - T0) / STEP) + 1; // nodes coloured so far
  const col = s.user ? s.user : COURSES.map((_, v) => (run.order.indexOf(v) < k ? run.col[v] : -1));
  const cur = s.user || k > COURSES.length || k < 1 ? -1 : run.order[k - 1];
  return { col, cur, used: slotsUsed(col), clash: clashes(EXAMS, col) };
}

export default function Timetable() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact<HTMLDivElement>(600);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [sel, setSel] = useState<Tab>('greedy');
  const [tab, bindTab] = useHoverPreview(sel);
  const [states, setStates] = useState<Record<Tab, State>>({ greedy: EMPTY, dsatur: POSTER });
  const [wait, setWait] = useState(0);
  const started = useRef(false);
  const [drag, setDrag] = useState<{ v: number; y: number } | null>(null);

  useEffect(() => {
    if (reduced) setStates({ greedy: POSTER, dsatur: POSTER });
  }, [reduced]);
  useEffect(() => {
    if (inView && !started.current && !reduced) {
      started.current = true;
      setWait(0.3);
    }
  }, [inView, reduced]);

  const cur = states[sel];
  useFrame(
    (dt) => {
      if (wait > 0) return setWait((w) => Math.max(0, w - dt));
      setStates((s) => ({ ...s, [sel]: { ...s[sel], t: Math.min(T_END, s[sel].t + dt) } }));
    },
    started.current && !reduced && inView && !cur.user && cur.t < T_END,
  );

  const replay = (t: Tab, pause: number) => {
    setStates((s) => ({ ...s, [t]: reduced ? POSTER : EMPTY }));
    if (!reduced) setWait(pause);
  };
  const choose = (t: Tab) => {
    if (t === sel) return;
    setSel(t);
    replay(t, 0.4);
  };

  const v = view(tab, states[tab]);

  // layout: graph and table side by side, or stacked on phones
  const W = compact ? 340 : 690;
  const H = compact ? 560 : 290;
  const GX = compact ? 10 : 0;
  const GY = compact ? 4 : 10;
  const TX = compact ? 46 : 396; // first course column
  const TY = compact ? 318 : 40; // first slot row
  const CW = 24;
  const RH = compact ? 44 : 46;
  const rowY = (s: number) => TY + s * RH;
  const colX = (c: number) => TX + c * CW;
  const node = (i: number) => {
    const [x, y] = POS[COURSES[i]];
    return { x: GX + x, y: GY + y };
  };

  const takeOver = () => {
    if (states[sel].user) return states[sel].user!;
    const col = view(sel, states[sel]).col;
    setStates((s) => ({ ...s, [sel]: { ...s[sel], user: col } }));
    return col;
  };
  const yToSlot = (y: number) => clamp(Math.round((y - TY - (RH - 6) / 2) / RH), 0, SLOTS - 1);

  return (
    <Specimen
      code="CX-13.1"
      title="build the timetable"
      kicker="12 courses · illustrative"
      rootRef={rootRef}
      hint="drag a coloured cell to another slot"
      footer={<Btn onClick={() => replay(sel, 1)}>restart</Btn>}
    >
      <div ref={boxRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          role="img"
          aria-label={`Clash graph of 12 courses and the exam timetable. ${v.used} slots used, ${v.clash} clashes.`}
          onPointerMove={(e) => {
            if (!drag) return;
            const p = svgPoint(svgRef.current!, e.clientX, e.clientY);
            setDrag({ v: drag.v, y: clamp(p.y, TY, rowY(SLOTS - 1) + RH - 6) });
          }}
          onPointerUp={() => {
            if (!drag) return;
            const col = [...(states[sel].user ?? view(sel, states[sel]).col)];
            col[drag.v] = yToSlot(drag.y);
            setStates((s) => ({ ...s, [sel]: { ...s[sel], user: col } }));
            setDrag(null);
          }}
          onPointerCancel={() => setDrag(null)}
        >
          {/* clash graph */}
          {EDGES.map(([a, b]) => {
            const A = node(a);
            const B = node(b);
            const bad = v.col[a] >= 0 && v.col[a] === v.col[b];
            const blocking =
              v.cur >= 0 && (a === v.cur || b === v.cur) && v.col[a === v.cur ? b : a] >= 0;
            return (
              <line
                key={`${a}-${b}`}
                x1={A.x}
                y1={A.y}
                x2={B.x}
                y2={B.y}
                style={{
                  stroke: bad ? GT.b : blocking ? GT.ink : GT.lineStrong,
                  strokeWidth: bad ? 3 : blocking ? 1.8 : 1.1,
                }}
              />
            );
          })}
          {COURSES.map((c, i) => {
            const p = node(i);
            const s = v.col[i];
            const bad = s >= 0 && EXAMS.adj[i].some((u) => v.col[u] === s);
            return (
              <g key={c}>
                {v.cur === i && (
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={NR + 5}
                    style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.5 }}
                  />
                )}
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={NR}
                  style={{
                    fill: s >= 0 ? SLOT_COLORS[s] : GT.paperRaised,
                    stroke: bad ? GT.b : s >= 0 ? SLOT_COLORS[s] : GT.lineStrong,
                    strokeWidth: bad ? 3 : 1.2,
                    transition: 'fill 200ms, stroke 200ms',
                  }}
                />
                <T x={p.x} y={p.y + 0.5} size={9.5} weight={600} color={s >= 0 ? GT.paper : GT.ink}>
                  {c}
                </T>
              </g>
            );
          })}

          {/* timetable: one column per course, one row per slot */}
          {COURSES.map((c, i) => (
            <T
              key={c}
              x={colX(i) + CW / 2}
              y={TY - 10}
              size={8}
              color={v.cur === i ? GT.ink : GT.muted}
              weight={v.cur === i ? 700 : 400}
            >
              {c.slice(0, 4)}
            </T>
          ))}
          {Array.from({ length: SLOTS }, (_, s) => (
            <g key={s}>
              <T
                x={TX - 6}
                y={rowY(s) + (RH - 6) / 2}
                anchor="end"
                size={9}
                color={SLOT_COLORS[s]}
                weight={600}
              >
                {`slot ${s + 1}`}
              </T>
              {COURSES.map((_, i) => (
                <rect
                  key={i}
                  x={colX(i) + 1}
                  y={rowY(s)}
                  width={CW - 2}
                  height={RH - 6}
                  style={{ fill: GT.paper, stroke: GT.line, strokeWidth: 1 }}
                />
              ))}
            </g>
          ))}
          {COURSES.map((c, i) => {
            const s = v.col[i];
            if (s < 0) return null;
            const dragging = drag?.v === i;
            const y = dragging ? drag!.y - (RH - 6) / 2 : rowY(s);
            const ds = dragging ? yToSlot(drag!.y) : s;
            const bad = EXAMS.adj[i].some((u) => v.col[u] === s);
            return (
              <rect
                key={c}
                x={colX(i) + 1}
                y={clamp(y, TY, rowY(SLOTS - 1))}
                width={CW - 2}
                height={RH - 6}
                onPointerDown={(e) => {
                  e.preventDefault();
                  (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture?.(
                    e.pointerId,
                  );
                  takeOver();
                  setDrag({ v: i, y: rowY(s) + (RH - 6) / 2 });
                }}
                style={{
                  fill: SLOT_COLORS[ds],
                  stroke: bad && !dragging ? GT.b : GT.paper,
                  strokeWidth: bad && !dragging ? 3 : 1,
                  cursor: dragging ? 'grabbing' : 'grab',
                  transition: dragging ? 'none' : 'fill 200ms',
                }}
              >
                <title>{FULL[c]}</title>
              </rect>
            );
          })}
        </svg>
      </div>
      <div className="mt-4 grid gap-x-8 gap-y-4 md:grid-cols-[auto_minmax(0,1fr)] md:items-start">
        <div className="grid grid-cols-3 gap-4 md:gap-6">
          <BigStat label="slots used" value={v.used} color={v.used > BEST ? GT.warn : GT.ink} />
          <BigStat label="fewest possible" value={BEST} />
          <BigStat label="clashes" value={v.clash} color={v.clash ? GT.b : GT.ink} />
        </div>
        <div className="grid gap-3">
          <Seg options={TABS} value={tab} onChange={choose} bind={bindTab} />
          <Reserve
            items={TABS.map((t) => (
              <p key={t.value} className="text-[0.85rem] leading-snug text-ink-soft">
                {CAPTION[t.value]}
              </p>
            ))}
            active={tab === 'greedy' ? 0 : 1}
          />
        </div>
      </div>
    </Specimen>
  );
}
