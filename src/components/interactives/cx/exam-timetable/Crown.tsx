/**
 * CX-13.2 "the worst case". A crown graph: n exams on top (A), n below (B); every A clashes
 * with every B except its own partner. Two slots always suffice (all A, then all B). Greedy
 * visiting partners in turn (A1, B1, A2, B2, ...) is pushed to a new slot at every pair and
 * ends with n. Same algorithm, same graph, only the order differs.
 */
import { useEffect, useState } from 'react';
import {
  BigStat,
  GT,
  Seg,
  Slider,
  Specimen,
  T,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import { crown, edges, greedy, rowsFirst, slotsUsed, zigzag } from './graph';

type Order = 'zigzag' | 'rows';
const ORDERS: { value: Order; label: string }[] = [
  { value: 'zigzag', label: 'partners in turn' },
  { value: 'rows', label: 'all A first' },
];
const COLORS = [GT.a, GT.opt, GT.warn, GT.alt, GT.inkSoft, GT.eq];
const T0 = 0.35;
const STEP = 0.32;
const NMAX = 6;
const NMIN = 3;

export default function Crown() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact<HTMLDivElement>(520);
  const [n, setN] = useState(4);
  const [sel, setSel] = useState<Order>('zigzag');
  const [order, bindOrder, previewing] = useHoverPreview(sel);
  const [t, setT] = useState(0);
  const [started, setStarted] = useState(false);

  const g = crown(n);
  const run = greedy(g, order === 'zigzag' ? zigzag(n) : rowsFirst(n));
  const tEnd = T0 + 2 * n * STEP + 0.2;
  // a previewed tab shows its finished colouring
  const tt = previewing || reduced ? tEnd : t;
  const k = Math.floor((tt - T0) / STEP) + 1;
  const col = run.col.map((c, v) => (run.order.indexOf(v) < k ? c : -1));
  const cur = k >= 1 && k <= 2 * n ? run.order[k - 1] : -1;
  const used = slotsUsed(col);

  useEffect(() => {
    if (inView && !started) setStarted(true);
  }, [inView, started]);
  useFrame((dt) => setT((x) => Math.min(tEnd, x + dt)), started && inView && !reduced && t < tEnd);

  const W = compact ? 340 : 560;
  const H = compact ? 250 : 230;
  const yA = compact ? 58 : 56;
  const yB = compact ? 198 : 180;
  const R = compact ? 16 : 18;
  const xs = (i: number) => {
    const span = W - 2 * (compact ? 34 : 70);
    return (compact ? 34 : 70) + (n === 1 ? span / 2 : (i * span) / (n - 1));
  };
  const pos = (v: number) => (v < n ? { x: xs(v), y: yA } : { x: xs(v - n), y: yB });
  const label = (v: number) => (v < n ? `A${v + 1}` : `B${v - n + 1}`);

  return (
    <Specimen
      code="CX-13.2"
      title="the worst case"
      kicker="crown graph · greedy"
      rootRef={rootRef}
      hint="pick an order, slide the number of pairs"
      footer="every A clashes with every B except its partner"
    >
      <div ref={boxRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`Crown graph with ${n} pairs, greedy ${order === 'zigzag' ? 'partners in turn' : 'all A first'}: ${used} slots used, 2 needed.`}
        >
          {edges(g).map(([a, b]) => {
            const A = pos(a);
            const B = pos(b);
            const blocking = cur >= 0 && (a === cur || b === cur) && col[a === cur ? b : a] >= 0;
            return (
              <line
                key={`${a}-${b}`}
                x1={A.x}
                y1={A.y}
                x2={B.x}
                y2={B.y}
                style={{
                  stroke: blocking ? GT.ink : GT.lineStrong,
                  strokeWidth: blocking ? 1.8 : 1,
                }}
              />
            );
          })}
          {Array.from({ length: 2 * n }, (_, v) => {
            const p = pos(v);
            const c = col[v];
            const step = run.order.indexOf(v) + 1;
            return (
              <g key={v}>
                {cur === v && (
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={R + 5}
                    style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.5 }}
                  />
                )}
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={R}
                  style={{
                    fill: c >= 0 ? COLORS[c] : GT.paperRaised,
                    stroke: c >= 0 ? COLORS[c] : GT.lineStrong,
                    strokeWidth: 1.2,
                    transition: 'fill 180ms, stroke 180ms',
                  }}
                />
                <T x={p.x} y={p.y + 0.5} size={10} weight={600} color={c >= 0 ? GT.paper : GT.ink}>
                  {label(v)}
                </T>
                {/* visiting order, outside the node, away from the edges */}
                <T x={p.x} y={v < n ? p.y - R - 10 : p.y + R + 11} size={9} color={GT.muted}>
                  {`#${step}`}
                </T>
                {c >= 0 && (
                  <T
                    x={p.x + R + 3}
                    y={v < n ? p.y - R + 2 : p.y + R - 2}
                    anchor="start"
                    size={8.5}
                    color={COLORS[c]}
                    weight={600}
                  >
                    {`s${c + 1}`}
                  </T>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-4 grid items-end gap-x-8 gap-y-4 md:grid-cols-[auto_minmax(0,1fr)]">
        <div className="grid grid-cols-2 gap-6">
          <BigStat
            label="slots used"
            value={used}
            color={used > 2 ? GT.warn : GT.ink}
            sub="by greedy"
          />
          <BigStat label="fewest possible" value={2} sub="A in one, B in another" />
        </div>
        <div className="grid gap-3">
          <Seg
            options={ORDERS}
            value={order}
            onChange={(o) => {
              setSel(o);
              setT(0);
            }}
            bind={bindOrder}
          />
          <Slider
            label="pairs"
            value={n}
            min={NMIN}
            max={NMAX}
            onChange={(x) => {
              setN(x);
              setT(0);
            }}
          />
        </div>
      </div>
      <Reserve
        className="mt-3"
        items={ORDERS.map((o) => (
          <p key={o.value} className="text-[0.85rem] leading-snug text-ink-soft">
            {o.value === 'zigzag'
              ? `Greedy gives each exam the first free slot. Taking partners in turn, every new pair clashes with all earlier slots, so ${n} pairs need ${n} slots.`
              : 'Taking all A first, the A exams share slot 1 (no two A clash) and every B fits in slot 2. Two slots, whatever the size.'}
          </p>
        ))}
        active={order === 'zigzag' ? 0 : 1}
      />
    </Specimen>
  );
}
