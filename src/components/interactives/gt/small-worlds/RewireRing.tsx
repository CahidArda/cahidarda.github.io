import { useEffect, useMemo, useRef, useState } from 'react';
import {
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
import { adjacency, graphAt, makePlan, shortestPath, stepAt, sweep, type Step } from './model';

/**
 * Watts-Strogatz on a ring of 72 people, each friends with the 2 nearest on each side.
 * Slide p (log scale): each lattice link is rewired to a random person with probability p.
 * The ring, the chain A to B and the right-hand chart all come from the same graph: the chart
 * is this ring's own distance and clustering, so its dot always sits on its line.
 */
/** Round to 0.01 so server and client trig agree (avoids hydration mismatches). */
const r2 = (v: number) => Math.round(v * 100) / 100;
const N = 72;
const K = 4;
const SEED = 18;
const plan = makePlan(N, K, SEED);
const STEPS = sweep(plan);
const TOTAL = (N * K) / 2;
const START: [number, number] = [0, 36];

const pOf = (v: number) => (v <= 0 ? 0 : Math.pow(10, -3 + (3 * v) / 100));
const DEMO_V = 49; // p = 0.03
/** Steps between two people on the plain ring (2 friends per side). */
const ringSteps = (a: number, b: number) => {
  const d = Math.abs(a - b);
  return Math.ceil(Math.min(d, N - d) / (K / 2));
};

const RS = 360;
const R = 150;
const pos = Array.from({ length: N }, (_, i) => {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / N;
  return { x: r2(RS / 2 + R * Math.cos(a)), y: r2(RS / 2 + R * Math.sin(a)) };
});

/* chart geometry */
const CW = 300;
const CH = 250;
const CM = { l: 34, r: 10, t: 14, b: 34 };
const cx = (p: number) => CM.l + ((Math.log10(Math.max(p, 0.001)) + 3) / 3) * (CW - CM.l - CM.r);
const cy = (v: number) => CM.t + (1 - v / 1.05) * (CH - CM.t - CM.b);
/** Exact step curve: flat until a link is rewired, then a drop. */
const stepPath = (key: 'L' | 'C') => {
  let d = `M${cx(0).toFixed(1)},${cy(STEPS[0][key]).toFixed(1)}`;
  for (const s of STEPS.slice(1)) d += `H${cx(s.p).toFixed(1)}V${cy(s[key]).toFixed(1)}`;
  return d + `H${cx(1).toFixed(1)}`;
};
const PATH_L = stepPath('L');
const PATH_C = stepPath('C');
const fmtP = (p: number) =>
  p === 0 ? '0' : p < 0.01 ? p.toFixed(4) : p < 0.1 ? p.toFixed(3) : p.toFixed(2);

type Phase = 'idle' | 'wait' | 'play';

export default function RewireRing() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [v, setV] = useState(0);
  const vRef = useRef(0);
  vRef.current = v;
  const [ends, setEnds] = useState<[number, number]>(START);
  const nextEnd = useRef<0 | 1>(0);
  const [phase, setPhase] = useState<Phase>('idle');
  const started = useRef(false);

  /** Back to the plain ring, then (after a beat) slide p up to 0.03. */
  const restart = () => {
    setEnds(START);
    nextEnd.current = 0;
    if (reduced) {
      setV(DEMO_V);
      setPhase('idle');
      return;
    }
    setV(0);
    setPhase('wait');
  };

  useEffect(() => {
    if (started.current || !inView) return;
    started.current = true;
    restart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);

  useEffect(() => {
    if (phase !== 'wait') return;
    const id = setTimeout(() => setPhase('play'), 1000);
    return () => clearTimeout(id);
  }, [phase]);

  useFrame(
    (dt) => {
      const nx = Math.min(DEMO_V, vRef.current + dt * (DEMO_V / 3.2));
      setV(nx);
      if (nx >= DEMO_V) setPhase('idle');
    },
    phase === 'play' && inView && !reduced,
  );

  const p = pOf(v);
  const g = useMemo(() => {
    const edges = graphAt(plan, p);
    return { edges, adj: adjacency(N, edges), rewired: edges.filter((e) => e.rewired).length };
  }, [p]);
  // The chain is a shortest path in exactly the graph drawn below, from A (ends[0]) to B.
  const path = useMemo(() => shortestPath(g.adj, ends[0], ends[1]), [g, ends]);
  const here: Step = stepAt(STEPS, p);
  const onPath = new Set(path);
  const pathEdges = new Set<string>();
  for (let i = 1; i < path.length; i++) {
    const a = Math.min(path[i - 1], path[i]);
    const b = Math.max(path[i - 1], path[i]);
    pathEdges.add(`${a}-${b}`);
  }

  const click = (i: number) => {
    setPhase('idle');
    setEnds((e) => {
      const n: [number, number] = [...e] as [number, number];
      n[nextEnd.current] = i;
      if (n[0] === n[1]) return e;
      nextEnd.current = nextEnd.current === 0 ? 1 : 0;
      return n;
    });
  };

  /*
   * Marker labels on the chart. Both curves only fall as p grows, so the space above-right of
   * the higher dot and below-left of the lower dot is always empty: the labels go there, and
   * slide along the edge instead of flipping sides when they would leave the plot.
   */
  const cHigh = here.C >= here.L;
  const cText = `clustering ${Math.round(here.C * 100)}%`;
  const lText = `distance ${Math.round(here.L * 100)}%`;
  const wOf = (t: string) => t.length * 10.5 * 0.62;
  const upText = cHigh ? cText : lText;
  const downText = cHigh ? lText : cText;
  const upX = Math.min(cx(p) + 8, CW - CM.r - wOf(upText));
  const downX = Math.max(cx(p) - 8, 2 + wOf(downText));
  const upY = cy(Math.max(here.C, here.L)) - 11;
  const downY = cy(Math.min(here.C, here.L)) + 12;
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-07.1"
      title="a few random links"
      kicker={`${N} people · ${TOTAL} friendships`}
      hint="tap two people to move A and B"
      footer={
        <span className="inline-flex flex-wrap gap-x-3 gap-y-1">
          <Chip color={GT.eq}>random link</Chip>
          <Chip color={GT.a}>shortest chain from A to B</Chip>
        </span>
      }
    >
      <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-[1.25fr_1fr]">
        <div className="min-w-0">
          <svg
            viewBox={`0 0 ${RS} ${RS}`}
            className="mx-auto block w-full max-w-[420px]"
            role="img"
            aria-label={`Ring of ${N} people with ${g.rewired} random links; the chain from A to B is ${path.length - 1} steps`}
          >
            {/* the chain: a soft band under the links it uses, so each link keeps its own colour */}
            {path.slice(1).map((b, i) => {
              const a = path[i];
              return (
                <line
                  key={`h${i}`}
                  data-chain={`${a}-${b}`}
                  x1={pos[a].x}
                  y1={pos[a].y}
                  x2={pos[b].x}
                  y2={pos[b].y}
                  style={{ stroke: GT.a, strokeWidth: 8, strokeLinecap: 'round', opacity: 0.3 }}
                />
              );
            })}
            {g.edges.map((e, i) => {
              const a = Math.min(e.a, e.b);
              const b = Math.max(e.a, e.b);
              const hot = pathEdges.has(`${a}-${b}`);
              return (
                <line
                  key={i}
                  data-link={`${e.a}-${e.b}`}
                  x1={pos[e.a].x}
                  y1={pos[e.a].y}
                  x2={pos[e.b].x}
                  y2={pos[e.b].y}
                  style={{
                    stroke: e.rewired ? GT.eq : hot ? GT.a : GT.lineStrong,
                    strokeWidth: hot ? 2.6 : e.rewired ? 1.6 : 1,
                  }}
                />
              );
            })}
            {pos.map((q, i) => {
              const isEnd = i === ends[0] || i === ends[1];
              return (
                <g key={i} onClick={() => click(i)} style={{ cursor: 'pointer' }}>
                  <circle cx={q.x} cy={q.y} r={6.5} style={{ fill: 'transparent' }} />
                  <circle
                    data-node={i}
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
            <g style={halo}>
              {ends.map((i, k) => {
                const q = pos[i];
                const dx = q.x - RS / 2;
                const dy = q.y - RS / 2;
                const d = Math.sqrt(dx * dx + dy * dy) || 1;
                return (
                  <T
                    key={k}
                    x={r2(q.x + (dx / d) * 16)}
                    y={r2(q.y + (dy / d) * 14)}
                    size={12}
                    color={GT.a}
                    weight={700}
                  >
                    {k === 0 ? 'A' : 'B'}
                  </T>
                );
              })}
            </g>
          </svg>
          <p className="mt-1 text-center font-mono text-[0.72rem] text-ink-soft tabular-nums">
            <span className="font-bold text-gt-a">
              A to B: {path.length ? `${path.length - 1} steps` : 'no chain'}
            </span>{' '}
            · {ringSteps(ends[0], ends[1])} on the plain ring
          </p>
        </div>
        <div className="min-w-0">
          <svg
            viewBox={`0 0 ${CW} ${CH}`}
            className="mx-auto block w-full max-w-[420px]"
            role="img"
            aria-label={`This ring's average distance is ${Math.round(here.L * 100)}% and its clustering ${Math.round(here.C * 100)}% of the plain ring`}
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
            <path d={PATH_C} style={{ fill: 'none', stroke: GT.opt, strokeWidth: 2 }} />
            <path d={PATH_L} style={{ fill: 'none', stroke: GT.ink, strokeWidth: 2 }} />
            <line
              x1={cx(p)}
              x2={cx(p)}
              y1={CM.t}
              y2={CH - CM.b}
              strokeDasharray="3 3"
              style={{ stroke: GT.muted }}
            />
            <circle cx={cx(p)} cy={cy(here.L)} r={4.5} style={{ fill: GT.ink }} />
            <circle cx={cx(p)} cy={cy(here.C)} r={4.5} style={{ fill: GT.opt }} />
            <g style={halo}>
              <T
                x={upX}
                y={upY}
                anchor="start"
                size={10.5}
                weight={700}
                color={cHigh ? GT.opt : GT.ink}
              >
                {upText}
              </T>
              <T
                x={downX}
                y={downY}
                anchor="end"
                size={10.5}
                weight={700}
                color={cHigh ? GT.ink : GT.opt}
              >
                {downText}
              </T>
            </g>
          </svg>
          <p className="mt-1 text-center font-mono text-[0.72rem] text-muted">
            this ring, as % of the plain ring
          </p>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-[1fr_auto] items-end gap-3">
        <Slider
          label="rewiring p"
          value={v}
          min={0}
          max={100}
          step={1}
          onChange={(x) => {
            setPhase('idle');
            setV(x);
          }}
          format={() => `p = ${fmtP(p)} · ${g.rewired} of ${TOTAL} links`}
        />
        <Btn primary onClick={restart}>
          restart
        </Btn>
      </div>
      <Caption>
        A handful of random links cuts the distance almost in half, while nearly every friend of a
        friend is still a friend.
      </Caption>
    </Specimen>
  );
}
