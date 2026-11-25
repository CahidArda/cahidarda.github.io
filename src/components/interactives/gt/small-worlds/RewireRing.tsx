import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Slider,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { adjacency, avgPathLength, clustering, graphAt, makePlan, shortestPath } from './model';

/**
 * Watts-Strogatz on a ring of 72 people, each friends with the 2 nearest on each side.
 * Slide p (log scale): each lattice link is rewired to a random person with probability p.
 */
/** Round to 0.01 so server and client trig agree (avoids hydration mismatches). */
const r2 = (v: number) => Math.round(v * 100) / 100;
const N = 72;
const K = 4;
const SEED = 18;
const plan = makePlan(N, K, SEED);
const adj0 = adjacency(N, graphAt(plan, 0));
const L0 = avgPathLength(adj0);
const C0 = clustering(adj0);
const TOTAL = (N * K) / 2;

/** L(p)/L(0) and C(p)/C(0), averaged over 40 seeded worlds (precomputed with model.curves). */
const CURVE: [number, number, number][] = [
  [0.001, 0.978, 0.995],
  [0.00133, 0.972, 0.994],
  [0.00178, 0.965, 0.992],
  [0.00237, 0.96, 0.992],
  [0.00316, 0.948, 0.99],
  [0.00422, 0.942, 0.988],
  [0.00562, 0.908, 0.983],
  [0.0075, 0.887, 0.979],
  [0.01, 0.866, 0.975],
  [0.0133, 0.833, 0.968],
  [0.0178, 0.791, 0.958],
  [0.0237, 0.742, 0.943],
  [0.0316, 0.695, 0.927],
  [0.0422, 0.646, 0.907],
  [0.0562, 0.588, 0.871],
  [0.075, 0.545, 0.833],
  [0.1, 0.493, 0.767],
  [0.133, 0.456, 0.7],
  [0.178, 0.425, 0.609],
  [0.237, 0.398, 0.509],
  [0.316, 0.379, 0.404],
  [0.422, 0.361, 0.289],
  [0.562, 0.351, 0.19],
  [0.75, 0.342, 0.114],
  [1, 0.339, 0.089],
];

const pOf = (v: number) => (v <= 0 ? 0 : Math.pow(10, -3 + (3 * v) / 100));
const DEMO_V = 49; // p = 0.03

const RS = 360;
const R = 150;
const pos = Array.from({ length: N }, (_, i) => {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / N;
  return { x: r2(RS / 2 + R * Math.cos(a)), y: r2(RS / 2 + R * Math.sin(a)) };
});

/* chart geometry */
const CW = 300;
const CH = 230;
const CM = { l: 34, r: 10, t: 14, b: 34 };
const cx = (p: number) => CM.l + ((Math.log10(Math.max(p, 0.001)) + 3) / 3) * (CW - CM.l - CM.r);
const cy = (v: number) => CM.t + (1 - v / 1.05) * (CH - CM.t - CM.b);
const curvePath = (j: 1 | 2) =>
  CURVE.map((r, i) => `${i ? 'L' : 'M'}${cx(r[0]).toFixed(1)},${cy(r[j]).toFixed(1)}`).join('');

export default function RewireRing() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [v, setV] = useState(0);
  const vRef = useRef(0);
  vRef.current = v;
  const [ends, setEnds] = useState<[number, number]>([0, 36]);
  const nextEnd = useRef<0 | 1>(0);
  const [demo, setDemo] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current || !inView) return;
    started.current = true;
    if (reduced) setV(DEMO_V);
    else setDemo(true);
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      const nx = Math.min(DEMO_V, vRef.current + dt * (DEMO_V / 3.2));
      setV(nx);
      if (nx >= DEMO_V) setDemo(false);
    },
    demo && inView && !reduced,
  );

  const p = pOf(v);
  const g = useMemo(() => {
    const edges = graphAt(plan, p);
    const adj = adjacency(N, edges);
    return {
      edges,
      adj,
      L: avgPathLength(adj),
      C: clustering(adj),
      rewired: edges.filter((e) => e.rewired).length,
    };
  }, [p]);
  const path = useMemo(() => shortestPath(g.adj, ends[0], ends[1]), [g, ends]);
  const onPath = new Set(path);
  const pathEdges = new Set<string>();
  for (let i = 1; i < path.length; i++) {
    const a = Math.min(path[i - 1], path[i]);
    const b = Math.max(path[i - 1], path[i]);
    pathEdges.add(`${a}-${b}`);
  }

  const click = (i: number) => {
    setDemo(false);
    setEnds((e) => {
      const n: [number, number] = [...e] as [number, number];
      n[nextEnd.current] = i;
      if (n[0] === n[1]) return e;
      nextEnd.current = nextEnd.current === 0 ? 1 : 0;
      return n;
    });
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-07.1"
      title="a few random links"
      kicker={`${N} people · ${TOTAL} friendships`}
      hint="slide p; click two people to trace the shortest chain"
      footer={
        <span className="inline-flex flex-wrap gap-x-3 gap-y-1">
          <Chip color={GT.eq}>random link</Chip>
          <Chip color={GT.a}>shortest chain</Chip>
        </span>
      }
    >
      <div className="mb-4 grid grid-cols-3 gap-3">
        <BigStat
          label="chain A→B"
          value={path.length ? path.length - 1 : '∞'}
          unit="steps"
          color={GT.a}
          sub="click two people"
        />
        <BigStat
          label="avg distance L"
          value={g.L.toFixed(1)}
          color={GT.ink}
          sub={`${Math.round((g.L / L0) * 100)}% of the ring's ${L0.toFixed(1)}`}
        />
        <BigStat
          label="clustering C"
          value={g.C.toFixed(2)}
          color={GT.opt}
          sub={`${Math.round((g.C / C0) * 100)}% of the ring's ${C0.toFixed(2)}`}
        />
      </div>
      <div className="grid grid-cols-1 items-center gap-4 sm:grid-cols-[1.25fr_1fr]">
        <svg
          viewBox={`0 0 ${RS} ${RS}`}
          className="mx-auto block w-full max-w-[420px]"
          role="img"
          aria-label={`Ring of ${N} people with ${g.rewired} rewired links; the chain between the two chosen people is ${path.length - 1} steps`}
        >
          {g.edges.map((e, i) => {
            const a = Math.min(e.a, e.b);
            const b = Math.max(e.a, e.b);
            const hot = pathEdges.has(`${a}-${b}`);
            if (hot) return null;
            return (
              <line
                key={i}
                x1={pos[e.a].x}
                y1={pos[e.a].y}
                x2={pos[e.b].x}
                y2={pos[e.b].y}
                style={{
                  stroke: e.rewired ? GT.eq : GT.lineStrong,
                  strokeWidth: e.rewired ? 1.4 : 1,
                  opacity: e.rewired ? 0.85 : 1,
                }}
              />
            );
          })}
          {path.slice(1).map((b, i) => {
            const a = path[i];
            return (
              <line
                key={`p${i}`}
                x1={pos[a].x}
                y1={pos[a].y}
                x2={pos[b].x}
                y2={pos[b].y}
                style={{ stroke: GT.a, strokeWidth: 3, strokeLinecap: 'round' }}
              />
            );
          })}
          {pos.map((q, i) => {
            const isEnd = i === ends[0] || i === ends[1];
            return (
              <g key={i} onClick={() => click(i)} style={{ cursor: 'pointer' }}>
                <circle cx={q.x} cy={q.y} r={9} style={{ fill: 'transparent' }} />
                <circle
                  cx={q.x}
                  cy={q.y}
                  r={isEnd ? 6 : 3.6}
                  style={{
                    fill: onPath.has(i) ? GT.a : GT.paperRaised,
                    stroke: onPath.has(i) ? GT.a : GT.ink,
                    strokeWidth: 1.2,
                  }}
                />
              </g>
            );
          })}
          {ends.map((i, k) => {
            const q = pos[i];
            const dx = q.x - RS / 2;
            const dy = q.y - RS / 2;
            const d = Math.sqrt(dx * dx + dy * dy) || 1;
            return (
              <T
                key={k}
                x={q.x + (dx / d) * 16}
                y={q.y + (dy / d) * 14}
                size={12}
                color={GT.a}
                weight={700}
              >
                {k === 0 ? 'A' : 'B'}
              </T>
            );
          })}
        </svg>
        <div className="min-w-0">
          <svg
            viewBox={`0 0 ${CW} ${CH}`}
            className="block w-full"
            role="img"
            aria-label="Average distance and clustering, relative to the ring, as p grows"
          >
            {[0, 0.5, 1].map((t) => (
              <g key={t}>
                <line x1={CM.l} x2={CW - CM.r} y1={cy(t)} y2={cy(t)} style={{ stroke: GT.line }} />
                <T x={CM.l - 5} y={cy(t)} anchor="end" size={9}>
                  {t === 1 ? '100%' : t === 0.5 ? '50%' : '0'}
                </T>
              </g>
            ))}
            {[0.001, 0.01, 0.1, 1].map((t) => (
              <g key={t}>
                <line x1={cx(t)} x2={cx(t)} y1={CM.t} y2={CH - CM.b} style={{ stroke: GT.line }} />
                <T x={cx(t)} y={CH - CM.b + 11} size={9}>
                  {t}
                </T>
              </g>
            ))}
            <T x={(CM.l + CW - CM.r) / 2} y={CH - 6} size={9.5} color={GT.inkSoft}>
              p, chance a link is rewired
            </T>
            <path d={curvePath(2)} style={{ fill: 'none', stroke: GT.opt, strokeWidth: 2 }} />
            <path d={curvePath(1)} style={{ fill: 'none', stroke: GT.ink, strokeWidth: 2 }} />
            <T x={cx(0.0013)} y={cy(1.0) - 8} anchor="start" size={9.5} color={GT.opt}>
              clustering
            </T>
            <T x={cx(0.2)} y={cy(0.43) - 10} anchor="middle" size={9.5} color={GT.ink}>
              distance
            </T>
            {p > 0 && (
              <>
                <line
                  x1={cx(p)}
                  x2={cx(p)}
                  y1={CM.t}
                  y2={CH - CM.b}
                  strokeDasharray="3 3"
                  style={{ stroke: GT.eq }}
                />
                <circle cx={cx(p)} cy={cy(g.L / L0)} r={4.5} style={{ fill: GT.ink }} />
                <circle cx={cx(p)} cy={cy(g.C / C0)} r={4.5} style={{ fill: GT.opt }} />
              </>
            )}
          </svg>
          <p className="mt-1 font-mono text-[0.62rem] text-muted">
            lines: average of 40 rings · dots: this ring
          </p>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-1 items-end gap-3 sm:grid-cols-[1fr_auto]">
        <Slider
          label="p, chance a link is rewired"
          value={v}
          min={0}
          max={100}
          step={1}
          onChange={(x) => {
            setDemo(false);
            setV(x);
          }}
          format={() =>
            `${g.rewired} of ${TOTAL} rewired · p = ` +
            (p === 0 ? '0' : p < 0.01 ? p.toFixed(4) : p < 0.1 ? p.toFixed(3) : p.toFixed(2))
          }
        />
        <div className="flex gap-2">
          <Btn
            primary
            onClick={() => {
              if (reduced) {
                setV(DEMO_V);
                return;
              }
              setV(0);
              setDemo(true);
            }}
          >
            replay
          </Btn>
          <Btn
            onClick={() => {
              setDemo(false);
              setV(0);
            }}
          >
            reset
          </Btn>
        </div>
      </div>
      <Caption>
        A handful of random links cuts the average distance almost in half, while nearly every
        friend of a friend is still a friend.
      </Caption>
    </Specimen>
  );
}
