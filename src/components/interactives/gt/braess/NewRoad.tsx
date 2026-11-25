import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
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

/* ── geometry (one SVG coordinate system) ──
 * C and D sit close together so the C-D link reads as what it is in the book's example:
 * a very short, very fast bridge whose travel time is about 0 however many cars use it. */
const W = 520;
const H = 166;
const P = {
  A: { x: 52, y: 82 },
  C: { x: 260, y: 40 },
  D: { x: 260, y: 124 },
  B: { x: 468, y: 82 },
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
const CY = 82;
const BOX = 16; // half-size of a town box
const PATHS: Record<Route, EdgeKey[]> = {
  top: ['AC', 'CB'],
  bot: ['AD', 'DB'],
  zig: ['AC', 'CD', 'DB'],
};
const ROUTE_COLOR: Record<Route, string> = { top: GT.a, bot: GT.b, zig: GT.alt };
const ROUTE_NAME: Record<Route, string> = { top: 'A-C-B', bot: 'A-D-B', zig: 'A-C-D-B' };

const CAR = 100; // drivers per drawn car
const SEC_PER_MIN = 0.05; // animation seconds per minute of travel
/** Seconds a car spends on a road: proportional to its travel time; the bridge is a quick hop. */
const dur = (e: EdgeKey, minutes: number) =>
  e === 'CD' ? 0.18 : Math.max(minutes, 5) * SEC_PER_MIN;

/** A car drives `route`; if its driver has switched but is mid-road, `next` applies from the next trip. */
type Car = { route: Route; seg: number; u: number; next?: Route };

function initialCars(f: Flow): Car[] {
  const rnd = mulberry32(7);
  const t = edgeTimes(f);
  const cars: Car[] = [];
  (['top', 'bot', 'zig'] as Route[]).forEach((r) => {
    const n = Math.round(f[r] / CAR);
    const segs = PATHS[r];
    const total = segs.reduce((s, e) => s + dur(e, t[e]), 0);
    for (let i = 0; i < n; i++) {
      let acc = ((i + rnd() * 0.6) / n) * total; // spread along the whole trip
      let seg = 0;
      while (seg < segs.length - 1 && acc > dur(segs[seg], t[segs[seg]])) {
        acc -= dur(segs[seg], t[segs[seg]]);
        seg++;
      }
      cars.push({ route: r, seg, u: Math.min(0.999, acc / dur(segs[seg], t[segs[seg]])) });
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

/**
 * `n` drivers switch from one route to another. A car on a road both routes share just
 * changes colour and keeps driving; a car on the closing bridge rolls onto the road leaving
 * C or D; any other car finishes its trip and takes the new route from A. No car jumps.
 */
function reassign(cars: Car[], from: Route, to: Route, n: number): Car[] {
  const next = cars.map((c) => ({ ...c }));
  let left = n;
  const plan = (c: Car) => c.next ?? c.route;
  // pass 0: cars on a shared road (switch now), pass 1: everyone else
  for (const pass of [0, 1]) {
    for (const c of next) {
      if (!left) break;
      if (plan(c) !== from) continue;
      const e = PATHS[c.route][c.seg];
      const j = PATHS[to].indexOf(e);
      if (pass === 0) {
        if (c.route === to) {
          c.next = undefined; // was about to leave `to`; stays
          left--;
        } else if (j >= 0) {
          c.route = to;
          c.seg = j;
          c.next = undefined;
          left--;
        }
      } else {
        if (e === 'CD') {
          // off the short bridge: onto C-B (top) or D-B (bottom), the second road of both
          c.route = to;
          c.seg = 1;
          c.u = 0;
          c.next = undefined;
        } else {
          c.next = to;
        }
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
    setMove(null);
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
    }, 420);
    return () => clearInterval(id);
  }, [open, reduced]);

  // cars drive: time on a road is proportional to its travel time
  useFrame((dt) => {
    const t = edgeTimes(flowRef.current);
    for (const c of carsRef.current) {
      let left = dt;
      // carry leftover time across road ends so a car never pauses at a town
      for (let guard = 0; guard < 4 && left > 0; guard++) {
        const e = PATHS[c.route][c.seg];
        const d = dur(e, t[e]);
        const need = (1 - c.u) * d;
        if (left < need) {
          c.u += left / d;
          left = 0;
        } else {
          left -= need;
          c.u = 0;
          c.seg += 1;
          if (c.seg >= PATHS[c.route].length) {
            c.seg = 0;
            if (c.next) {
              c.route = c.next;
              c.next = undefined;
            }
          }
        }
      }
    }
    setTick((x) => (x + 1) % 1e6);
  }, inView && !reduced);

  // auto-play: open the bridge once, shortly after the figure scrolls in
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

  const width = (n: number) => 4 + (n / DRIVERS) * 14;
  let caption: string;
  if (move && !settled) {
    caption = `${move.from === 'zig' ? 'Bridge closed. ' : ''}200 drivers switch from ${ROUTE_NAME[move.from]} to ${ROUTE_NAME[move.to]}: their trip goes from ${fmt(move.before)} to ${fmt(move.after)} min.`;
  } else if (open && flow.zig === DRIVERS) {
    caption =
      'All 4000 drivers take the bridge: 40 + 0 + 40 = 80 min. Leaving it would take 85, so no one does.';
  } else if (open) {
    caption = 'The bridge is open. Drivers start to switch.';
  } else {
    caption =
      'Two routes, 2000 drivers on each, 65 min each. Switching alone would be slower, so no one moves.';
  }

  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  const edgeLabel = (e: EdgeKey) => {
    const [a, b] = EDGES[e];
    const mx = (P[a].x + P[b].x) / 2;
    const my = (P[a].y + P[b].y) / 2;
    const up = e === 'AC' || e === 'CB';
    const dx = e === 'AC' || e === 'AD' ? -1 : 1;
    const x = mx + dx * (30 + 30 * (k - 1));
    const y = my + (up ? -1 : 1) * (26 + 14 * (k - 1));
    const variable = e === 'AC' || e === 'DB';
    return (
      <g key={e} style={halo}>
        <T x={x} y={y - 7 * k} size={15 * k} font="display" color={GT.ink}>
          {fmt(et[e])} min
        </T>
        <T x={x} y={y + 9 * k} size={9 * k}>
          {variable ? `${load[e]} cars ÷ 100` : 'always 45'}
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

  // the bridge: two rails between the C and D boxes
  const by0 = P.C.y + BOX;
  const by1 = P.D.y - BOX;
  const kl = Math.min(k, 1.2);

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-05.1"
      title="a new road"
      kicker="4000 drivers, A to B"
      hint="click the bridge or the switch to open or close it"
      footer="Easley & Kleinberg, fig. 8.1-8.2"
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        role="img"
        aria-label="Road network from A to B through C or D, with a short bridge from C to D that takes about 0 minutes"
      >
        {(['AC', 'CB', 'AD', 'DB'] as EdgeKey[]).map(road)}

        {/* direction chevrons on the four roads */}
        {(['AC', 'CB', 'AD', 'DB'] as EdgeKey[]).map((e) => {
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

        {/* the bridge */}
        <g onClick={toggle} style={{ cursor: 'pointer' }} role="presentation">
          <rect
            x={P.C.x - 22}
            y={by0}
            width={44}
            height={by1 - by0}
            style={{ fill: 'transparent' }}
          />
          {[-6, 6].map((dx) => (
            <line
              key={dx}
              x1={P.C.x + dx}
              y1={by0}
              x2={P.C.x + dx}
              y2={by1}
              style={{
                stroke: open ? GT.alt : GT.muted,
                strokeWidth: open ? 2.5 : 1.5,
                strokeDasharray: open ? undefined : '4 4',
              }}
            />
          ))}
          {open &&
            [0.25, 0.5, 0.75].map((f) => (
              <line
                key={f}
                x1={P.C.x - 6}
                x2={P.C.x + 6}
                y1={by0 + f * (by1 - by0)}
                y2={by0 + f * (by1 - by0)}
                style={{ stroke: GT.alt, strokeWidth: 1, opacity: 0.5 }}
              />
            ))}
        </g>

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

        {/* bridge label */}
        <g style={halo}>
          {open ? (
            <>
              <T
                x={P.C.x + 15}
                y={CY - 6 * kl}
                size={15 * kl}
                font="display"
                color={GT.alt}
                anchor="start"
              >
                ≈ 0 min
              </T>
              <T x={P.C.x + 15} y={CY + 9 * kl} size={9 * kl} anchor="start" color={GT.alt}>
                any traffic
              </T>
            </>
          ) : (
            <T x={P.C.x + 15} y={CY} size={9 * kl} anchor="start">
              bridge closed
            </T>
          )}
        </g>

        {(Object.keys(P) as (keyof typeof P)[]).map((key) => (
          <g key={key}>
            <rect
              x={P[key].x - BOX}
              y={P[key].y - BOX}
              width={BOX * 2}
              height={BOX * 2}
              style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.5 }}
            />
            <T x={P[key].x} y={P[key].y + 1} size={17} font="display" color={GT.ink}>
              {key}
            </T>
          </g>
        ))}
      </svg>

      <div className="mt-4 flex flex-wrap items-center gap-3">
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
          <span className="inline-block w-[18ch] whitespace-nowrap text-left">
            bridge C→D: {open ? 'open' : 'closed'}
          </span>
        </button>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-[auto_1fr] sm:items-end">
        <BigStat
          size="lg"
          label="everyone's commute"
          value={fmt(avg)}
          unit="min"
          color={open && flow.zig > 0 ? GT.eq : GT.ink}
          sub="before the bridge: 65 min"
        />
        <div className="grid grid-cols-[auto_auto_auto] gap-x-4 gap-y-1 font-mono text-[0.72rem] tabular-nums sm:justify-self-end">
          <span className="text-muted">route</span>
          <span className="text-right text-muted">drivers</span>
          <span className="text-right text-muted">trip</span>
          {(['top', 'bot', 'zig'] as Route[]).map((r) => (
            <div key={r} className="contents">
              <span className="inline-flex items-center gap-1.5 text-ink-soft">
                <span className="inline-block h-2.5 w-2.5" style={{ background: ROUTE_COLOR[r] }} />
                {ROUTE_NAME[r]}
              </span>
              <span className="min-w-[5ch] text-right text-ink-soft">
                {r === 'zig' && !open && flow.zig === 0 ? '·' : flow[r].toLocaleString('en-US')}
              </span>
              <span className="min-w-[6ch] text-right text-ink">
                {r === 'zig' && !open ? 'closed' : `${fmt(rt[r])} min`}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="min-h-[3.6rem]">
        <Caption>{caption}</Caption>
      </div>
    </Specimen>
  );
}
