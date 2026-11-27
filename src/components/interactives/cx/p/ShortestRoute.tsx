import { useEffect, useRef, useState } from 'react';
import { Reserve } from '../../gt/Reserve';
import {
  Btn,
  GT,
  Slider,
  Specimen,
  T,
  clamp,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { ROUTES, makeMap, type MapData } from './model';

/**
 * CX-02.1 "shortest route". A k × k town. Dijkstra's algorithm settles crossings in order of
 * distance from A: drawn as a wave that runs along the roads of the shortest-route tree at
 * constant speed, from one clock. It stops when B is settled and the route lights up along
 * visible roads. Counters: Dijkstra's road checks, and the same number of whole routes tried
 * by a brute-force search, out of all the routes it would have to compare (OEIS A007764).
 * The chart beside the map shows both totals for every town size.
 */
const KS = [3, 4, 5, 6, 7, 8, 9, 10];
const MAPS: Record<number, MapData> = Object.fromEntries(KS.map((k) => [k, makeMap(k)]));
const WAVE_S = 2.6; // A to B, any town size
const PATH_S = 0.35; // the route fades in after
const r2 = (v: number) => Math.round(v * 100) / 100;
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n: number) =>
  String(n)
    .split('')
    .map((c) => SUP[Number(c)])
    .join('');

/** Whole numbers with commas, or "4.1 × 10¹⁹" once they get long. */
export function fmtBig(v: bigint | number): string {
  const n = typeof v === 'bigint' ? v : BigInt(Math.round(v));
  if (n < 1_000_000_000n) return n.toLocaleString('en-US');
  const s = n.toString();
  const e = s.length - 1;
  return `${s[0]}.${s[1]} × 10${sup(e)}`;
}

/** Checks made once the wave has reached distance r: every road of every settled crossing but B. */
function checksAt(m: MapData, r: number) {
  let c = 0;
  const B = m.k * m.k - 1;
  for (let u = 0; u < B; u++) if (m.settled[u] && m.dist[u] <= r) c += m.adj[u].length;
  return c;
}

function sentence(k: number) {
  const routes = ROUTES[k];
  const checks = MAPS[k].checks;
  const ratio = Number(routes) / checks;
  return ratio >= 1
    ? `Trying every route means comparing ${fmtBig(routes)} routes: ${fmtBig(Math.round(ratio))} times Dijkstra's ${checks} checks, even counting a whole route as one check.`
    : `On a town this small, the ${routes} routes are fewer than Dijkstra's ${checks} checks. It does not stay that way.`;
}

type Box = { x: number; y: number; s: number };

function Town({ m, r, pathOp, box }: { m: MapData; r: number; pathOp: number; box: Box }) {
  const P = (i: number) => ({
    x: r2(box.x + m.pts[i].x * box.s),
    y: r2(box.y + m.pts[i].y * box.s),
  });
  const B = m.k * m.k - 1;
  const nodeR = m.k > 7 ? 3.2 : 4.2;
  const pathD = m.path.map((i, j) => `${j ? 'L' : 'M'}${P(i).x},${P(i).y}`).join('');
  return (
    <g>
      {/* roads */}
      {m.edges.map(([u, v]) => (
        <line
          key={`${u}-${v}`}
          x1={P(u).x}
          y1={P(u).y}
          x2={P(v).x}
          y2={P(v).y}
          style={{ stroke: GT.lineStrong, strokeWidth: 2.5, strokeLinecap: 'round' }}
        />
      ))}
      {/* the wave: each tree road fills from its parent at the speed of the front */}
      {m.parent.map((p, v) => {
        if (p < 0 || !m.settled[v]) return null;
        const f = clamp((r - m.dist[p]) / (m.dist[v] - m.dist[p]), 0, 1);
        if (f <= 0) return null;
        const a = P(p);
        const b = P(v);
        return (
          <line
            key={v}
            x1={a.x}
            y1={a.y}
            x2={r2(a.x + (b.x - a.x) * f)}
            y2={r2(a.y + (b.y - a.y) * f)}
            style={{ stroke: GT.opt, strokeWidth: 3.5, strokeLinecap: 'round' }}
          />
        );
      })}
      {pathOp > 0 && (
        <path
          d={pathD}
          style={{
            fill: 'none',
            stroke: GT.eq,
            strokeWidth: 5.5,
            strokeLinecap: 'round',
            strokeLinejoin: 'round',
            opacity: r2(pathOp),
          }}
        />
      )}
      {/* crossings */}
      {m.pts.map((_, i) => {
        const on = m.settled[i] && m.dist[i] <= r;
        const end = i === 0 || i === B;
        const p = P(i);
        return (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={end ? nodeR + 2.5 : nodeR}
            style={{
              fill: end ? GT.ink : on ? GT.opt : GT.paperRaised,
              stroke: end ? GT.ink : on ? GT.opt : GT.lineStrong,
              strokeWidth: 1.5,
            }}
          />
        );
      })}
      <T x={P(0).x} y={P(0).y} size={9} color={GT.paperRaised} weight={700}>
        A
      </T>
      <T x={P(B).x} y={P(B).y} size={9} color={GT.paperRaised} weight={700}>
        B
      </T>
    </g>
  );
}

type Plot = { x0: number; x1: number; y0: number; y1: number };
const LY = 20; // log10 top of the chart

function Counts({ k, plot, fs }: { k: number; plot: Plot; fs: number }) {
  const { x0, x1, y0, y1 } = plot;
  const sx = (kk: number) => x0 + ((kk - 3) / 7) * (x1 - x0);
  const sy = (l: number) => y1 - (l / LY) * (y1 - y0);
  const routes = KS.map((kk) => ({ kk, l: Math.log10(Number(ROUTES[kk])) }));
  const dij = KS.map((kk) => ({ kk, l: Math.log10(MAPS[kk].checks) }));
  const line = (pts: { kk: number; l: number }[]) =>
    pts.map((p, i) => `${i ? 'L' : 'M'}${r2(sx(p.kk))},${r2(sy(p.l))}`).join('');
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };
  return (
    <g>
      {[0, 5, 10, 15, 20].map((l) => (
        <g key={l}>
          <line x1={x0} x2={x1} y1={sy(l)} y2={sy(l)} style={{ stroke: GT.line }} />
          <T x={x0 - 5} y={sy(l)} anchor="end" size={fs - 1}>
            {l === 0 ? '1' : `10${sup(l)}`}
          </T>
        </g>
      ))}
      {KS.map((kk) => (
        <T key={kk} x={sx(kk)} y={y1 + 12} size={fs - 1} color={kk === k ? GT.ink : GT.muted}>
          {kk}
        </T>
      ))}
      <T x={(x0 + x1) / 2} y={y1 + 27} size={fs} color={GT.inkSoft}>
        crossings per side
      </T>
      <T x={x0 - 5} y={y0 - 12} anchor="start" size={fs} color={GT.inkSoft}>
        things to check
      </T>
      <line x1={x0} x2={x1} y1={y1} y2={y1} style={{ stroke: GT.lineStrong }} />
      {/* where the reader's town sits */}
      <line
        x1={sx(k)}
        x2={sx(k)}
        y1={y0}
        y2={y1}
        style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
      />
      <path d={line(routes)} style={{ fill: 'none', stroke: GT.b, strokeWidth: 2.2 }} />
      <path d={line(dij)} style={{ fill: 'none', stroke: GT.opt, strokeWidth: 2.2 }} />
      {[...routes.map((p) => ({ ...p, c: GT.b })), ...dij.map((p) => ({ ...p, c: GT.opt }))].map(
        (p, i) => (
          <circle
            key={i}
            cx={r2(sx(p.kk))}
            cy={r2(sy(p.l))}
            r={p.kk === k ? 4.5 : 2.5}
            style={{ fill: p.c, stroke: GT.paperRaised, strokeWidth: p.kk === k ? 1.5 : 0 }}
          />
        ),
      )}
      <g style={halo}>
        <T x={sx(6.4)} y={sy(9.5)} anchor="end" size={fs + 1} color={GT.b} weight={700}>
          every route
        </T>
        <T x={sx(9.9)} y={sy(2.5) - 12} anchor="end" size={fs + 1} color={GT.opt} weight={700}>
          Dijkstra
        </T>
      </g>
    </g>
  );
}

export default function ShortestRoute() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [k, setK] = useState(6);
  const m = MAPS[k];
  const total = WAVE_S + PATH_S;
  const [e, setE] = useState(0); // seconds on the clock
  const eRef = useRef(0);
  const [running, setRunning] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const restart = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    eRef.current = 0;
    setE(0);
    setRunning(false);
    if (reduced) {
      eRef.current = total;
      setE(total);
      return;
    }
    if (delay) timer.current = setTimeout(() => setRunning(true), delay);
    else setRunning(true);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    restart(400);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  useFrame((dt) => {
    const v = Math.min(total, eRef.current + Math.max(0, dt));
    eRef.current = v;
    setE(v);
    if (v >= total) setRunning(false);
  }, running && inView);

  const r = m.dB * clamp(e / WAVE_S, 0, 1);
  const pathOp = clamp((e - WAVE_S) / PATH_S, 0, 1);
  const done = e >= WAVE_S;
  const checks = done ? m.checks : checksAt(m, r);
  const routes = ROUTES[k];

  const chooseK = (v: number) => {
    setK(v);
    started.current = true;
    restart(0);
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-02.1"
      title="shortest route"
      kicker="Dijkstra vs trying every route"
      hint="drag the town size; replay to watch again"
      footer="routes: OEIS A007764"
    >
      <div className="hidden sm:block">
        <svg
          viewBox="0 0 640 320"
          className="block w-full"
          role="img"
          aria-label="A town map with Dijkstra's algorithm spreading from A, and a chart of how many things each method must check"
        >
          <Town m={m} r={r} pathOp={pathOp} box={{ x: 24, y: 20, s: 280 }} />
          <Counts k={k} plot={{ x0: 392, x1: 620, y0: 34, y1: 270 }} fs={10} />
        </svg>
      </div>
      <div className="sm:hidden">
        <svg
          viewBox="0 0 360 560"
          className="block w-full"
          role="img"
          aria-label="A town map with Dijkstra's algorithm spreading from A, and a chart of how many things each method must check"
        >
          <Town m={m} r={r} pathOp={pathOp} box={{ x: 30, y: 14, s: 300 }} />
          <Counts k={k} plot={{ x0: 52, x1: 340, y0: 376, y1: 514 }} fs={10} />
        </svg>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-3">
        <div className="min-w-0">
          <div className="label" style={{ color: GT.opt }}>
            Dijkstra
          </div>
          <div
            className="font-display text-4xl leading-none tabular-nums lining-nums"
            style={{ color: GT.opt }}
          >
            {checks}
            <span className="ml-1.5 font-mono text-sm text-muted">checks</span>
          </div>
          <Reserve
            className="mt-1 font-mono text-[0.68rem] text-muted"
            active={done ? 1 : 0}
            items={['settling crossings, nearest first', 'done: shortest route found']}
          />
        </div>
        <div className="min-w-0">
          <div className="label" style={{ color: GT.b }}>
            every route
          </div>
          <div
            className="font-display text-4xl leading-none tabular-nums lining-nums"
            style={{ color: GT.b }}
          >
            {fmtBig(Math.min(checks, Number(routes)))}
          </div>
          <div className="mt-1 font-mono text-[0.68rem] text-muted">tried, of {fmtBig(routes)}</div>
        </div>
      </div>
      <div className="mt-4 grid items-end gap-x-8 gap-y-3 sm:grid-cols-[minmax(0,1fr)_auto]">
        <div className="max-w-sm">
          <Slider
            label="town size"
            value={k}
            min={3}
            max={10}
            onChange={chooseK}
            format={(v) => `${v} × ${v} crossings`}
          />
        </div>
        <div className="flex justify-end">
          <Btn primary onClick={() => restart(1000)}>
            replay
          </Btn>
        </div>
      </div>
      {/* every town size's sentence stacked in one cell, so the figure never changes height */}
      <Reserve
        className="mt-3 text-[0.85rem] leading-snug text-ink-soft"
        active={KS.indexOf(k)}
        items={KS.map((kk) => (
          <p key={kk}>{sentence(kk)}</p>
        ))}
      />
    </Specimen>
  );
}
