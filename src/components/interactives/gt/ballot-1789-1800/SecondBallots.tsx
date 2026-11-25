import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Specimen,
  T,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../shared';
import { RECIPIENTS, STATES_1789, electors, type Rec } from './data';
import { useCompact } from './useCompact';

type Mode = 'state' | 'name';

const colorOf = (r: Rec) =>
  r === 'Adams' ? GT.a : r === 'Clinton' ? GT.b : r === 'Jay' ? GT.alt : GT.warn;

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const r2 = (x: number) => Math.round(x * 100) / 100;

export default function SecondBallots() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact();
  const list = useMemo(() => electors(), []);
  const [sel, setSel] = useState<Mode>('state');
  const [mode, bindMode] = useHoverPreview(sel);
  const [t, setT] = useState(0); // 0 = by state, 1 = by name
  const [hover, setHover] = useState<number | null>(null);
  const touched = useRef(false);
  const demo = useRef(false);

  const target = mode === 'name' ? 1 : 0;
  useFrame(
    (dt) =>
      setT((x) => {
        const d = target - x;
        const step = dt / 1.6;
        return Math.abs(d) <= step ? target : x + Math.sign(d) * step;
      }),
    !reduced && t !== target,
  );
  useEffect(() => {
    if (reduced) setT(target);
  }, [reduced, target]);

  useEffect(() => {
    if (!inView || demo.current) return;
    demo.current = true;
    const id = setTimeout(() => !touched.current && setSel('name'), 1600);
    return () => clearTimeout(id);
  }, [inView]);

  // geometry
  const W = compact ? 380 : 640;
  const H = compact ? 300 : 236;
  const s = compact ? 8 : 13; // square
  const g = compact ? 1.6 : 2.4; // gap
  const p = s + g;

  // by state: one column per state, stacked from the bottom
  const colW = (W - 20) / STATES_1789.length;
  const base = H - 40;
  const stateXY = (e: (typeof list)[number]) => {
    const cx = 10 + colW * (e.state + 0.5);
    const twoCols = compact;
    const col = twoCols ? e.inState % 2 : 0;
    const row = twoCols ? Math.floor(e.inState / 2) : e.inState;
    const x = twoCols ? cx - p + col * p + g / 2 : cx - s / 2;
    return { x, y: base - (row + 1) * p };
  };
  // by name: one row per recipient
  const labelW = compact ? 92 : 132;
  const rowH = compact ? 21 : 19;
  const top = 14;
  const perRow = compact ? 17 : 40;
  const nameRowIdx = useMemo(() => {
    const m: Record<string, number> = {};
    RECIPIENTS.forEach((r, i) => (m[r.id] = i));
    return m;
  }, []);
  const nameXY = (e: (typeof list)[number]) => {
    const r = nameRowIdx[e.rec];
    const wrap = Math.floor(e.inRec / perRow);
    const col = e.inRec % perRow;
    // Adams may wrap on phones: give his row two lines
    const extra = compact && r > 0 ? rowH * 0.55 : 0;
    return { x: labelW + col * p, y: top + r * rowH + wrap * p + extra };
  };

  const h = hover !== null ? list[hover] : null;
  const adams = list.filter((e) => e.rec === 'Adams').length;

  const caption = h
    ? `${STATES_1789[h.state].name} elector: Washington + ${RECIPIENTS[nameRowIdx[h.rec]].name}. ${STATES_1789[h.state].note}`
    : mode === 'name'
      ? 'The same 69 second votes, sorted by name. Adams got 34, under half. The other 35 were spread over ten other men.'
      : 'Each square is one elector’s second vote, grouped by state. Every one of them also wrote Washington. Hover a square for its state.';

  return (
    <Specimen
      code="EX-01.2"
      title="the 69 second votes"
      kicker="1789 · official tally"
      rootRef={rootRef}
      hint="switch the sorting; hover a square"
      footer="source: National Archives, 1789 results"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Seg
          options={[
            { value: 'state', label: 'by state' },
            { value: 'name', label: 'by name' },
          ]}
          value={mode}
          onChange={(m) => {
            touched.current = true;
            setSel(m);
          }}
          bind={bindMode}
          label="sort"
        />
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <BigStat label="Washington" value="69" unit="/ 69" sub="every elector" />
        <BigStat label="Adams" value={adams} color={GT.a} sub="vice president" />
        <BigStat
          label="elsewhere"
          value={69 - adams}
          color={GT.warn}
          sub="ten other names"
          align="right"
        />
      </div>

      <div ref={boxRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="mt-3 block w-full select-none"
          role="img"
          aria-label="69 electors' second votes: Adams 34, Jay 9, Harrison 6, Rutledge 6, Hancock 4, Clinton 3, Huntington 2, Milton 2, Armstrong 1, Lincoln 1, Telfair 1."
          onMouseLeave={() => setHover(null)}
        >
          {/* state labels fade out, name labels fade in */}
          <g style={{ opacity: 1 - t }}>
            {STATES_1789.map((st, i) => {
              const cx = 10 + colW * (i + 0.5);
              const n = Object.values(st.second).reduce((x, y) => x + (y ?? 0), 0);
              const on = h ? h.state === i : false;
              return (
                <g key={st.id}>
                  <line
                    x1={cx - colW / 2 + 3}
                    y1={base + 3}
                    x2={cx + colW / 2 - 3}
                    y2={base + 3}
                    style={{ stroke: GT.lineStrong }}
                  />
                  <T
                    x={cx}
                    y={base + 15}
                    size={compact ? 10 : 11}
                    color={on ? GT.ink : GT.inkSoft}
                    weight={on ? 700 : 500}
                  >
                    {st.id}
                  </T>
                  <T x={cx} y={base + 29} size={9.5}>
                    {n}
                  </T>
                </g>
              );
            })}
          </g>
          <g style={{ opacity: t }}>
            {RECIPIENTS.map((r, i) => {
              const extra = compact && i > 0 ? rowH * 0.55 : 0;
              const y = top + i * rowH + extra + s / 2;
              const on = h ? h.rec === r.id : false;
              return (
                <g key={r.id}>
                  <T
                    x={labelW - 8}
                    y={y}
                    anchor="end"
                    size={compact ? 10 : 11}
                    color={on ? GT.ink : GT.inkSoft}
                    weight={on ? 700 : 500}
                  >
                    {compact ? r.name.split(' ').slice(-1)[0] : r.name}
                  </T>
                  <T
                    x={labelW + Math.min(r.votes, perRow) * p + 6}
                    y={y}
                    anchor="start"
                    size={11}
                    color={colorOf(r.id)}
                    weight={700}
                  >
                    {r.votes}
                  </T>
                </g>
              );
            })}
          </g>

          {list.map((e) => {
            const a = stateXY(e);
            const b = nameXY(e);
            // stagger: squares leave in order, so the move reads as a flow
            const d = (e.i / list.length) * 0.45;
            const u = ease(Math.min(1, Math.max(0, (t - d) / 0.55)));
            const x = r2(a.x + (b.x - a.x) * u);
            const y = r2(a.y + (b.y - a.y) * u);
            const dim = h && !(h.state === e.state || h.rec === e.rec);
            return (
              <rect
                key={e.i}
                x={x}
                y={y}
                width={s}
                height={s}
                onMouseEnter={() => setHover(e.i)}
                style={{
                  fill: colorOf(e.rec),
                  opacity: dim ? 0.25 : 1,
                  stroke: hover === e.i ? GT.ink : 'none',
                  strokeWidth: 1.5,
                  cursor: 'pointer',
                }}
              />
            );
          })}
        </svg>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.a}>Adams</Chip>
        <Chip color={GT.alt}>Jay</Chip>
        <Chip color={GT.b}>Clinton (Anti-Federalist)</Chip>
        <Chip color={GT.warn}>someone else</Chip>
      </div>
      <Caption>{caption}</Caption>
    </Specimen>
  );
}
