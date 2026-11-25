import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Specimen,
  T,
  mulberry32,
  useFrame,
  useInView,
  useReducedMotion,
  useTween,
} from '../shared';
import {
  DRIVERS,
  START,
  average,
  edgeLoads,
  edgeTimes,
  equilibrium,
  routeTimes,
  step,
  type Flow,
  type Move,
  type Route,
} from './model';
import { useBoost } from './useBoost';

/* ── geometry (one SVG coordinate system) ── */
const W = 520;
const H = 300;
const P = {
  A: { x: 52, y: 150 },
  C: { x: 260, y: 46 },
  D: { x: 260, y: 254 },
  B: { x: 468, y: 150 },
};
type EdgeKey = 'AC' | 'CB' | 'AD' | 'DB' | 'CD';
const EDGES: Record<EdgeKey, [keyof typeof P, keyof typeof P]> = {
  AC: ['A', 'C'],
  CB: ['C', 'B'],
  AD: ['A', 'D'],
  DB: ['D', 'B'],
  CD: ['C', 'D'],
};
const CX = 260;
const CY = 150;
const PATHS: Record<Route, EdgeKey[]> = {
  top: ['AC', 'CB'],
  bot: ['AD', 'DB'],
  zig: ['AC', 'CD', 'DB'],
};
export const ROUTE_COLOR: Record<Route, string> = { top: GT.a, bot: GT.b, zig: GT.alt };
export const ROUTE_NAME: Record<Route, string> = { top: 'A-C-B', bot: 'A-D-B', zig: 'A-C-D-B' };

const CAR = 100; // drivers per drawn car
const SEC_PER_MIN = 0.05; // animation seconds per minute of travel

type Car = { route: Route; seg: number; u: number };

function initialCars(f: Flow): Car[] {
  const rnd = mulberry32(7);
  const cars: Car[] = [];
  (['top', 'bot', 'zig'] as Route[]).forEach((r) => {
    const n = Math.round(f[r] / CAR);
    for (let i = 0; i < n; i++) {
      const k = (i + rnd() * 0.6) / n; // spread along the whole trip
      const segs = PATHS[r];
      const t = edgeTimes(f);
      const total = segs.reduce((s, e) => s + Math.max(t[e], 5), 0);
      let acc = k * total;
      let seg = 0;
      while (seg < segs.length - 1 && acc > Math.max(t[segs[seg]], 5)) {
        acc -= Math.max(t[segs[seg]], 5);
        seg++;
      }
      cars.push({ route: r, seg, u: Math.min(0.999, acc / Math.max(t[segs[seg]], 5)) });
    }
  });
  return cars;
}

function carXY(c: Car) {
  const e = PATHS[c.route][c.seg];
  const [a, b] = EDGES[e];
  const p = P[a];
  const q = P[b];
  const dx = q.x - p.x;
  const dy = q.y - p.y;
  const len = Math.hypot(dx, dy);
  // two lanes: the zig-zag drives on the inner lane, the others on the outer lane
  const mx = (p.x + q.x) / 2 - CX;
  const my = (p.y + q.y) / 2 - CY;
  let nx = -dy / len;
  let ny = dx / len;
  if (nx * mx + ny * my < 0) {
    nx = -nx;
    ny = -ny;
  }
  const off = e === 'CD' ? 0 : c.route === 'zig' ? -5 : 5;
  return { x: p.x + dx * c.u + nx * off, y: p.y + dy * c.u + ny * off };
}

/** Re-point `n` cars from one route to another, preferring cars on a shared road. */
function reassign(cars: Car[], from: Route, to: Route, n: number): Car[] {
  const next = cars.map((c) => ({ ...c }));
  let left = n;
  const shared = (c: Car) => {
    const e = PATHS[from][c.seg];
    return PATHS[to].indexOf(e);
  };
  for (const pass of [0, 1]) {
    for (const c of next) {
      if (!left) break;
      if (c.route !== from) continue;
      const j = shared(c);
      if (pass === 0 && j >= 0) {
        c.route = to;
        c.seg = j;
        left--;
      } else if (pass === 1) {
        c.route = to;
        c.seg = 0;
        c.u = 0;
        left--;
      }
    }
  }
  return next;
}

const fmt = (v: number) =>
  Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1);

export default function NewRoad() {
  const reduced = useReducedMotion();
  const [svgRef, k] = useBoost(W);
  const [rootRef, inView] = useInView();
  const [open, setOpen] = useState(false);
  const [flow, setFlow] = useState<Flow>(START);
  const [move, setMove] = useState<Move | null>(null);
  const [settled, setSettled] = useState(true);
  const carsRef = useRef<Car[]>(initialCars(START));
  const [, setTick] = useState(0);
  const touched = useRef(false);
  const flowRef = useRef(flow);
  flowRef.current = flow;

  const avg = useTween(average(flow), reduced, 8);
  const rt = routeTimes(flow);
  const et = edgeTimes(flow);
  const load = edgeLoads(flow);

  // best-response dynamics: one batch of drivers switches at a time
  useEffect(() => {
    if (reduced) {
      const eq = equilibrium(flowRef.current, open);
      setFlow(eq);
      carsRef.current = initialCars(eq);
      setMove(null);
      setSettled(true);
      setTick((t) => t + 1);
      return;
    }
    setSettled(false);
    const id = setInterval(() => {
      const s = step(flowRef.current, open);
      if (!s) {
        setSettled(true);
        clearInterval(id);
        return;
      }
      const n = Math.round((flowRef.current[s.move.from] - s.flow[s.move.from]) / CAR);
      carsRef.current = reassign(carsRef.current, s.move.from, s.move.to, n);
      flowRef.current = s.flow;
      setFlow(s.flow);
      setMove(s.move);
    }, 380);
    return () => clearInterval(id);
  }, [open, reduced]);

  // cars drive: time on a road is proportional to its travel time
  useFrame((dt) => {
    const t = edgeTimes(flowRef.current);
    for (const c of carsRef.current) {
      const e = PATHS[c.route][c.seg];
      c.u += dt / (Math.max(t[e], 5) * SEC_PER_MIN);
      while (c.u >= 1) {
        c.u -= 1;
        c.seg = (c.seg + 1) % PATHS[c.route].length;
      }
    }
    setTick((x) => (x + 1) % 1e6);
  }, inView && !reduced);

  // auto-play: open the shortcut once, shortly after the figure scrolls in
  useEffect(() => {
    if (!inView || reduced || touched.current) return;
    const id = setTimeout(() => {
      if (!touched.current) {
        touched.current = true;
        setOpen(true);
      }
    }, 1600);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  const toggle = () => {
    touched.current = true;
    setOpen((o) => !o);
  };
  const reset = () => {
    touched.current = true;
    setOpen(false);
    setFlow(START);
    flowRef.current = START;
    carsRef.current = initialCars(START);
    setMove(null);
  };

  const width = (n: number) => 4 + (n / DRIVERS) * 16;
  const atEq = settled;
  let caption: string;
  if (move && !atEq) {
    caption = `${move.from === 'zig' ? 'Shortcut closed: ' : ''}200 drivers leave ${ROUTE_NAME[move.from]} (${fmt(move.before)} min) for ${ROUTE_NAME[move.to]}: ${fmt(move.after)} min for them.`;
  } else if (open) {
    caption =
      flow.zig === DRIVERS
        ? 'Everyone takes the shortcut. Leaving it would cost 85 min, so no one does. All 4000 drivers now spend 80 min.'
        : 'Drivers are switching to the shortcut.';
  } else {
    caption =
      'Two routes, 2000 drivers on each. A driver who switches alone would arrive later than 65 min, so no one moves.';
  }

  const edgeLabel = (e: EdgeKey) => {
    const [a, b] = EDGES[e];
    const mx = (P[a].x + P[b].x) / 2;
    const my = (P[a].y + P[b].y) / 2;
    const up = e === 'AC' || e === 'CB';
    const dx = e === 'AC' || e === 'AD' ? -1 : 1;
    const x = mx + dx * (30 + 34 * (k - 1));
    const y = my + (up ? -1 : 1) * (26 + 18 * (k - 1));
    const variable = e === 'AC' || e === 'DB';
    return (
      <g key={e}>
        <T x={x} y={y - 7 * k} size={15 * k} font="display" color={GT.ink}>
          {fmt(et[e])} min
        </T>
        <T x={x} y={y + 9 * k} size={9 * k}>
          {variable ? `x/100, x = ${load[e]}` : 'fixed 45'}
        </T>
      </g>
    );
  };

  const road = (e: EdgeKey) => {
    const [a, b] = EDGES[e];
    return (
      <line
        key={e}
        x1={P[a].x}
        y1={P[a].y}
        x2={P[b].x}
        y2={P[b].y}
        style={{
          stroke: GT.lineStrong,
          strokeWidth: width(load[e]),
          strokeLinecap: 'butt',
          opacity: 0.55,
          transition: 'stroke-width 300ms',
        }}
      />
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-05.1"
      title="a new road"
      kicker="4000 drivers, A to B"
      hint="click the dashed road, or the switch, to open the shortcut"
      footer="Easley & Kleinberg, fig. 8.1-8.2"
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          role="switch"
          aria-checked={open}
          onClick={toggle}
          className="inline-flex items-center gap-2 border border-line-strong px-2.5 py-1 font-mono text-[0.72rem] text-ink hover:border-ink"
        >
          <span
            className="relative inline-block h-3.5 w-7 border border-ink"
            style={{ background: open ? GT.alt : 'transparent', transition: 'background 200ms' }}
          >
            <span
              className="absolute top-0 h-full w-3 bg-ink"
              style={{ left: open ? 'calc(100% - 0.75rem)' : 0, transition: 'left 200ms' }}
            />
          </span>
          shortcut C→D: {open ? 'open' : 'closed'}
        </button>
        <Btn onClick={reset}>reset</Btn>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        role="img"
        aria-label="Road network from A to B through C or D, with an optional shortcut from C to D"
      >
        {(['AC', 'CB', 'AD', 'DB'] as EdgeKey[]).map(road)}
        {/* the shortcut */}
        <g onClick={toggle} style={{ cursor: 'pointer' }} role="presentation">
          <line
            x1={P.C.x}
            y1={P.C.y}
            x2={P.D.x}
            y2={P.D.y}
            style={{ stroke: 'transparent', strokeWidth: 30 }}
          />
          {open ? (
            <line
              x1={P.C.x}
              y1={P.C.y}
              x2={P.D.x}
              y2={P.D.y}
              style={{ stroke: GT.alt, strokeWidth: width(load.CD), opacity: 0.35 }}
            />
          ) : (
            <>
              <line
                x1={P.C.x}
                y1={P.C.y + 20}
                x2={P.D.x}
                y2={P.D.y - 20}
                style={{ stroke: GT.muted, strokeWidth: 2, strokeDasharray: '5 5' }}
              />
              <g transform={`translate(${P.C.x} ${H / 2})`}>
                <rect
                  x={-34}
                  y={-9}
                  width={68}
                  height={18}
                  style={{ fill: GT.paperRaised, stroke: GT.muted }}
                />
                <T x={0} y={0.5} size={9 * k}>
                  closed
                </T>
              </g>
            </>
          )}
          {open && (
            <g transform={`translate(${P.C.x + 18} ${H / 2})`}>
              <T x={0} y={-7 * k} size={15 * k} font="display" color={GT.alt} anchor="start">
                0 min
              </T>
              <T x={0} y={9 * k} size={9 * k} anchor="start">
                {`x = ${load.CD} cars`}
              </T>
            </g>
          )}
        </g>

        {/* direction chevrons */}
        {(['AC', 'CB', 'AD', 'DB', ...(open ? ['CD'] : [])] as EdgeKey[]).map((e) => {
          const [a, b] = EDGES[e];
          const mx = (P[a].x + P[b].x) / 2;
          const my = (P[a].y + P[b].y) / 2;
          const ang = (Math.atan2(P[b].y - P[a].y, P[b].x - P[a].x) * 180) / Math.PI;
          return (
            <path
              key={'ch' + e}
              d="M-4 -5 L3 0 L-4 5"
              transform={`translate(${mx} ${my}) rotate(${ang})`}
              style={{ fill: 'none', stroke: GT.inkSoft, strokeWidth: 1.5, opacity: 0.6 }}
            />
          );
        })}

        {/* cars, drawn under the towns */}
        {carsRef.current.map((c, i) => {
          const p = carXY(c);
          return (
            <rect
              key={i}
              x={p.x - 3}
              y={p.y - 3}
              width={6}
              height={6}
              style={{ fill: ROUTE_COLOR[c.route] }}
            />
          );
        })}

        {(['AC', 'CB', 'AD', 'DB'] as EdgeKey[]).map(edgeLabel)}

        {(Object.keys(P) as (keyof typeof P)[]).map((k) => (
          <g key={k}>
            <rect
              x={P[k].x - 16}
              y={P[k].y - 16}
              width={32}
              height={32}
              style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.5 }}
            />
            <T x={P[k].x} y={P[k].y + 1} size={17} font="display" color={GT.ink}>
              {k}
            </T>
          </g>
        ))}
        <T x={P.A.x} y={P.A.y + 18 + 11 * k} size={9 * k}>
          start
        </T>
        <T x={P.B.x} y={P.B.y + 18 + 11 * k} size={9 * k}>
          finish
        </T>
      </svg>

      <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-[auto_1fr] sm:items-end">
        <BigStat
          size="lg"
          label="everyone's commute"
          value={fmt(avg)}
          unit="min"
          color={open && flow.zig > 0 ? GT.eq : GT.ink}
          sub={`without the shortcut: 65 min`}
        />
        <div className="grid gap-1 sm:justify-self-end">
          {(['top', 'bot', 'zig'] as Route[]).map((r) => (
            <div
              key={r}
              className="flex items-baseline justify-between gap-4 font-mono text-[0.72rem]"
            >
              <Chip color={ROUTE_COLOR[r]}>{ROUTE_NAME[r]}</Chip>
              <span className="tabular-nums text-ink-soft">
                {r === 'zig' && !open ? (
                  'closed'
                ) : (
                  <>
                    {flow[r].toLocaleString('en-US')} drivers · {fmt(rt[r])} min
                  </>
                )}
              </span>
            </div>
          ))}
        </div>
      </div>
      <Caption>{caption}</Caption>
    </Specimen>
  );
}
