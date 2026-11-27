import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  lerp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import { nearestNeighbour, optimalTour, tourLength, twoOptRun, type Move, type Pt } from './model';

/* One SVG coordinate system: 1 unit = 1 km. Twelve cities (a fixed, seeded layout on
 * which nearest neighbour ends 24% above the best tour and 2-opt brings it to 4%). */
const W = 480;
const H = 340;
const PAD = 22;
const START: Pt[] = [
  [380, 172],
  [294, 181],
  [63, 226],
  [177, 292],
  [203, 93],
  [145, 227],
  [127, 138],
  [161, 46],
  [380, 84],
  [232, 170],
  [262, 84],
  [403, 280],
].map(([x, y]) => ({ x, y }));

const NN_SECONDS = 2.1; // the pen draws the whole greedy tour in this time, at constant speed
const MOVE_SECONDS = 0.85; // one 2-opt swap

type Anim =
  | { kind: 'empty' }
  | { kind: 'nn'; order: number[]; s: number; total: number }
  | { kind: 'opt'; from: number[]; moves: Move[]; k: number; u: number }
  | { kind: 'idle'; order: number[] };

const smooth = (t: number) => t * t * (3 - 2 * t);

function finalOrder(a: Anim): number[] | null {
  if (a.kind === 'empty') return null;
  if (a.kind === 'nn' || a.kind === 'idle') return a.order;
  return a.moves.length ? a.moves[a.moves.length - 1].next : a.from;
}

/** The order on screen right now (during a swap: the order before it). */
function currentOrder(a: Anim): number[] | null {
  if (a.kind === 'empty') return null;
  if (a.kind === 'nn' || a.kind === 'idle') return a.order;
  return a.k === 0 ? a.from : a.moves[a.k - 1].next;
}

const edgeKey = (i: number, j: number) => (i < j ? `${i}-${j}` : `${j}-${i}`);

export default function DrawTour() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [pts, setPts] = useState<Pt[]>(START);
  const ptsRef = useRef(pts);
  ptsRef.current = pts;
  const anim = useRef<Anim>({ kind: 'empty' });
  const [, setTick] = useState(0);
  const rerender = () => setTick((x) => (x + 1) % 1e6);
  const [drag, setDrag] = useState<number | null>(null);
  const started = useRef(false);
  const pending = useRef<number | null>(null);

  const opt = useMemo(() => optimalTour(pts), [pts]);

  const startNN = (p: Pt[]) => {
    const order = nearestNeighbour(p);
    anim.current = { kind: 'nn', order, s: 0, total: tourLength(p, order) };
    rerender();
  };
  const startOpt = (p: Pt[], from: number[]) => {
    const moves = twoOptRun(p, from);
    anim.current = moves.length
      ? { kind: 'opt', from, moves, k: 0, u: 0 }
      : { kind: 'idle', order: from };
    rerender();
  };
  /** Jump to the end of everything (reduced motion). */
  const poster = (p: Pt[]) => {
    const nn = nearestNeighbour(p);
    const moves = twoOptRun(p, nn);
    anim.current = { kind: 'idle', order: moves.length ? moves[moves.length - 1].next : nn };
    rerender();
  };

  // auto-play once, shortly after the figure scrolls in
  useEffect(() => {
    if (!inView || started.current) return;
    if (reduced) {
      started.current = true;
      poster(ptsRef.current);
      return;
    }
    const id = window.setTimeout(() => {
      if (started.current) return;
      started.current = true;
      startNN(ptsRef.current);
    }, 700);
    return () => window.clearTimeout(id);
  }, [inView, reduced]);

  useEffect(
    () => () => {
      if (pending.current) window.clearTimeout(pending.current);
    },
    [],
  );

  // one clock drives the pen and every swap
  const running = inView && !reduced && (anim.current.kind === 'nn' || anim.current.kind === 'opt');
  useFrame((dt) => {
    const a = anim.current;
    if (a.kind === 'nn') {
      a.s += (a.total / NN_SECONDS) * dt;
      if (a.s >= a.total) {
        // carry straight on into 2-opt, no pause
        startOpt(ptsRef.current, a.order);
        return;
      }
    } else if (a.kind === 'opt') {
      a.u += dt / MOVE_SECONDS;
      while (a.u >= 1 && a.kind === 'opt') {
        a.u -= 1;
        a.k += 1;
        if (a.k >= a.moves.length) {
          anim.current = { kind: 'idle', order: a.moves[a.moves.length - 1].next };
          break;
        }
      }
    }
    rerender();
  }, running);

  const restart = () => {
    started.current = true;
    if (pending.current) window.clearTimeout(pending.current);
    setPts(START);
    ptsRef.current = START;
    anim.current = { kind: 'empty' };
    rerender();
    if (reduced) {
      poster(START);
      return;
    }
    pending.current = window.setTimeout(() => startNN(START), 900);
  };

  /* ── dragging ── */
  const onDown = (i: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    started.current = true;
    if (pending.current) window.clearTimeout(pending.current);
    const ord = finalOrder(anim.current) ?? nearestNeighbour(ptsRef.current);
    anim.current = { kind: 'idle', order: ord };
    setDrag(i);
  };
  const onMove = (e: React.PointerEvent) => {
    if (drag === null || !svgRef.current) return;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    const next = ptsRef.current.slice();
    next[drag] = { x: clamp(p.x, PAD, W - PAD), y: clamp(p.y, PAD, H - PAD) };
    ptsRef.current = next;
    setPts(next);
  };
  const onUp = () => {
    if (drag === null) return;
    setDrag(null);
    const ord = finalOrder(anim.current) ?? nearestNeighbour(ptsRef.current);
    if (reduced) {
      const moves = twoOptRun(ptsRef.current, ord);
      anim.current = { kind: 'idle', order: moves.length ? moves[moves.length - 1].next : ord };
      rerender();
    } else startOpt(ptsRef.current, ord);
  };

  /* ── what to draw ── */
  const a = anim.current;
  const cur = currentOrder(a);
  const P = (i: number) => pts[i];
  let tourLen: number | null = null;
  const solid: [number, number][] = []; // roads drawn in full
  let pen: Pt | null = null;
  let leaving: [number, number][] = [];
  let arriving: { from: Pt; to: Pt; p: number }[] = [];
  let removedFade = 1;

  if (a.kind === 'nn') {
    // the pen draws the greedy tour at constant speed
    let left = a.s;
    const n = a.order.length;
    for (let i = 0; i < n; i++) {
      const p = P(a.order[i]);
      const q = P(a.order[(i + 1) % n]);
      const d = Math.hypot(q.x - p.x, q.y - p.y);
      if (left >= d) {
        solid.push([a.order[i], a.order[(i + 1) % n]]);
        left -= d;
      } else {
        const t = d > 0 ? left / d : 0;
        pen = { x: lerp(p.x, q.x, t), y: lerp(p.y, q.y, t) };
        arriving = [{ from: p, to: pen, p: 1 }];
        break;
      }
    }
    tourLen = a.s;
  } else if (a.kind === 'opt' && cur) {
    const m = a.moves[a.k];
    const drop = new Set([edgeKey(m.a, m.b), edgeKey(m.c, m.d)]);
    for (let i = 0; i < cur.length; i++) {
      const x = cur[i];
      const y = cur[(i + 1) % cur.length];
      if (!drop.has(edgeKey(x, y))) solid.push([x, y]);
    }
    leaving = [
      [m.a, m.b],
      [m.c, m.d],
    ];
    removedFade = 1 - smooth(clamp(a.u / 0.45, 0, 1));
    const grow = smooth(clamp((a.u - 0.25) / 0.75, 0, 1));
    arriving = [
      { from: P(m.a), to: P(m.c), p: grow },
      { from: P(m.b), to: P(m.d), p: grow },
    ];
    const before = tourLength(pts, cur);
    tourLen = lerp(before, before - m.gain, smooth(clamp(a.u, 0, 1)));
  } else if (cur) {
    for (let i = 0; i < cur.length; i++) solid.push([cur[i], cur[(i + 1) % cur.length]]);
    tourLen = tourLength(pts, cur);
  }

  const gap = tourLen !== null && a.kind !== 'nn' ? (tourLen / opt.length - 1) * 100 : null;
  const done = a.kind === 'idle';
  const isBest = done && gap !== null && gap < 0.05;

  const captionIdx =
    drag !== null
      ? 4
      : a.kind === 'empty'
        ? 0
        : a.kind === 'nn'
          ? 1
          : a.kind === 'opt'
            ? 2
            : isBest
              ? 5
              : 3;
  const captions = [
    'Twelve cities. The dashed green loop is the shortest round trip, found by checking them all.',
    'Nearest neighbour: from the start, always drive to the closest city not yet visited.',
    '2-opt: two roads cross, so drop them and reconnect the four cities the other way. The loop gets shorter.',
    'No swap of two roads helps any more. Close to the best, but not quite: drag a city and watch it re-fix.',
    'Drag a city. When you let go, 2-opt untangles whatever crossings you made.',
    'No swap helps any more, and this time 2-opt found the best tour exactly.',
  ];

  const k = compact ? 1.35 : 1;
  const r = 5.5 * k;
  const stroke = (c: string, w: number, extra: React.CSSProperties = {}) => ({
    stroke: c,
    strokeWidth: w,
    strokeLinecap: 'round' as const,
    fill: 'none',
    ...extra,
  });
  const optPath = opt.order.map((i, j) => `${j ? 'L' : 'M'}${P(i).x} ${P(i).y}`).join(' ') + ' Z';
  const fmt = (v: number) => Math.round(v).toLocaleString('en-US');

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-06.1"
      title="draw a tour"
      kicker="12 cities · 1 unit = 1 km"
      hint="drag any city"
      footer={<Btn onClick={restart}>restart</Btn>}
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          style={{ touchAction: drag !== null ? 'none' : 'pan-y' }}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          role="img"
          aria-label="Twelve cities with a round trip through them; the shortest round trip is shown dashed"
        >
          {/* the best tour, as a ghost under everything */}
          <path d={optPath} style={stroke(GT.opt, 1.6, { strokeDasharray: '5 5', opacity: 0.9 })} />

          {/* roads being dropped by the current swap */}
          {leaving.map(([i, j]) => (
            <line
              key={'l' + i + '-' + j}
              x1={P(i).x}
              y1={P(i).y}
              x2={P(j).x}
              y2={P(j).y}
              style={stroke(GT.b, 3, { opacity: removedFade })}
            />
          ))}
          {solid.map(([i, j]) => (
            <line
              key={'s' + edgeKey(i, j)}
              x1={P(i).x}
              y1={P(i).y}
              x2={P(j).x}
              y2={P(j).y}
              style={stroke(GT.a, 2.6)}
            />
          ))}
          {arriving.map((e, i) => (
            <line
              key={'a' + i}
              x1={e.from.x}
              y1={e.from.y}
              x2={lerp(e.from.x, e.to.x, e.p)}
              y2={lerp(e.from.y, e.to.y, e.p)}
              style={stroke(GT.a, 2.6)}
            />
          ))}
          {pen && <circle cx={pen.x} cy={pen.y} r={3.5 * k} style={{ fill: GT.a }} />}

          {/* cities */}
          {pts.map((p, i) => (
            <g
              key={i}
              onPointerDown={onDown(i)}
              style={{ cursor: drag === i ? 'grabbing' : 'grab', touchAction: 'none' }}
            >
              <circle cx={p.x} cy={p.y} r={18} style={{ fill: 'transparent' }} />
              <circle
                cx={p.x}
                cy={p.y}
                r={drag === i ? r + 2 : r}
                style={{
                  fill: i === 0 ? GT.ink : GT.paperRaised,
                  stroke: GT.ink,
                  strokeWidth: 1.8,
                }}
              />
            </g>
          ))}
          <g style={{ pointerEvents: 'none' }}>
            <T x={P(0).x + 10 * k} y={P(0).y - 11 * k} anchor="start" size={10 * k} color={GT.ink}>
              start
            </T>
          </g>
          {/* scale bar */}
          <g style={{ pointerEvents: 'none' }}>
            <line x1={PAD} y1={H - 10} x2={PAD + 100} y2={H - 10} style={stroke(GT.muted, 1.2)} />
            <T x={PAD + 106} y={H - 10} anchor="start" size={9 * k}>
              100 km
            </T>
          </g>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="your tour"
          value={tourLen === null ? '·' : fmt(tourLen)}
          unit="km"
          color={GT.a}
        />
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="best possible"
          value={fmt(opt.length)}
          unit="km"
          color={GT.opt}
        />
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="above best"
          value={gap === null ? '·' : `${gap < 0.05 ? '0' : gap.toFixed(1)}`}
          unit="%"
          color={GT.ink}
        />
      </div>
      <Reserve
        items={captions.map((c) => (
          <Caption key={c}>{c}</Caption>
        ))}
        active={captionIdx}
      />
    </Specimen>
  );
}
