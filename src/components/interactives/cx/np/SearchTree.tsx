import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Slider,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import {
  NUMS,
  N_MAX,
  N_MIN,
  TARGETS,
  dfsWalk,
  guessPath,
  nodeAt,
  solutionLeaf,
  type Geom,
} from './subset';

/**
 * CX-03.2 · the search tree. Subset sum with n numbers: every root-to-leaf path picks a
 * subset (left = skip, right = take), so there are 2^n leaves. The searcher (ink) walks the
 * tree depth first, leaf by leaf. The lucky guesser (accent) goes straight down the right
 * branch, then the checker adds up what it took. Both move one edge (or one addition) per
 * tick of the same clock; the clock rate is set so the search lasts a few seconds.
 */
const DURATION: Record<number, number> = {
  3: 4,
  4: 5,
  5: 6.5,
  6: 7.5,
  7: 8.5,
  8: 9,
  9: 9.5,
  10: 10,
};

export default function SearchTree() {
  const [wrapRef, compact] = useCompact(460);
  const W = compact ? 360 : 520;
  const H = compact ? 290 : 280;
  const LX = compact ? 40 : 54; // the column of numbers on the left
  const geom: Geom = { left: LX + 6, right: W - 6, top: 18, bottom: H - 46 };

  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [n, setN] = useState(5);
  const [t, setT] = useState(0);
  const [running, setRunning] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tRef = useRef(0);

  const xs = NUMS.slice(0, n);
  const target = TARGETS[n];
  const leaf = useMemo(() => solutionLeaf(n), [n]);
  const take = (d: number) => ((leaf >> (n - 1 - d)) & 1) === 1;
  const taken = xs.map((_, d) => take(d));
  const k = taken.filter(Boolean).length;
  const walk = useMemo(() => dfsWalk(geom, n, leaf), [n, leaf, W, H]);
  const gp = useMemo(() => guessPath(geom, n, leaf), [n, leaf, W, H]);
  const edges = walk.pts.length - 1;
  const rate = edges / DURATION[n]; // edges per second, shared by both
  const T_END = DURATION[n];
  const T_GUESS = (n + k) / rate;

  const finish = () => {
    tRef.current = T_END;
    setT(T_END);
    setRunning(false);
  };
  const restart = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    if (reduced) {
      finish();
      return;
    }
    tRef.current = 0;
    setT(0);
    setRunning(false);
    timer.current = setTimeout(() => setRunning(true), delay);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    restart(600);
  }, [inView, reduced]);
  // the reduced-motion hook settles one render after mount: jump to the poster then
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    finish();
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  // a new n restarts the race at once
  const firstN = useRef(true);
  useEffect(() => {
    if (firstN.current) {
      firstN.current = false;
      return;
    }
    restart(250);
  }, [n]);

  useFrame((dt) => {
    const v = Math.min(T_END, tRef.current + Math.max(0, dt)); // the first rAF can report dt < 0
    tRef.current = v;
    setT(v);
    if (v >= T_END) setRunning(false);
  }, running && inView);

  // ── searcher ──
  const prog = Math.min(edges, t * rate); // edges walked, continuous
  const sIdx = Math.floor(prog);
  const sPos =
    sIdx >= edges
      ? walk.pts[edges]
      : {
          x: walk.pts[sIdx].x + (walk.pts[sIdx + 1].x - walk.pts[sIdx].x) * (prog - sIdx),
          y: walk.pts[sIdx].y + (walk.pts[sIdx + 1].y - walk.pts[sIdx].y) * (prog - sIdx),
        };
  let sPath = `M${walk.pts[0].x.toFixed(1)} ${walk.pts[0].y.toFixed(1)}`;
  for (let i = 1; i <= sIdx; i++)
    sPath += `L${walk.pts[i].x.toFixed(1)} ${walk.pts[i].y.toFixed(1)}`;
  sPath += `L${sPos.x.toFixed(2)} ${sPos.y.toFixed(2)}`;
  const leafPtIdx = walk.leafPt;
  const tried = leafPtIdx.filter((i) => i !== undefined && i <= prog + 1e-9).length;
  const searchDone = prog >= edges;

  // ── lucky guesser: n moves down, then k additions ──
  const gSteps = Math.min(n + k, t * rate);
  const gMove = Math.min(n, gSteps);
  const gi = Math.floor(gMove);
  const gPos =
    gi >= n
      ? gp.pts[n]
      : {
          x: gp.pts[gi].x + (gp.pts[gi + 1].x - gp.pts[gi].x) * (gMove - gi),
          y: gp.pts[gi].y + (gp.pts[gi + 1].y - gp.pts[gi].y) * (gMove - gi),
        };
  let gPath = `M${gp.pts[0].x.toFixed(1)} ${gp.pts[0].y.toFixed(1)}`;
  for (let i = 1; i <= gi && i <= n; i++)
    gPath += `L${gp.pts[i].x.toFixed(1)} ${gp.pts[i].y.toFixed(1)}`;
  gPath += `L${gPos.x.toFixed(2)} ${gPos.y.toFixed(2)}`;
  const adds = Math.max(0, Math.floor(gSteps - n + 1e-9)); // additions done
  const takenOrder = xs.map((_, d) => d).filter((d) => taken[d]);
  const added = new Set(takenOrder.slice(0, adds));
  const sum = takenOrder.slice(0, adds).reduce((a, d) => a + xs[d], 0);
  const checked = t >= T_GUESS - 1e-9 && t > 0;

  // all edges of the tree, faint
  let tree = '';
  for (let d = 0; d < n; d++)
    for (let j = 0; j < 1 << d; j++) {
      const p = nodeAt(geom, n, d, j);
      for (const c of [2 * j, 2 * j + 1]) {
        const q = nodeAt(geom, n, d + 1, c);
        tree += `M${p.x.toFixed(1)} ${p.y.toFixed(1)}L${q.x.toFixed(1)} ${q.y.toFixed(1)}`;
      }
    }

  const L = 1 << n;
  const leafW = (geom.right - geom.left) / L;
  const sol = nodeAt(geom, n, n, leaf);
  const lvlY = (d: number) => (nodeAt(geom, n, d, 0).y + nodeAt(geom, n, d + 1, 0).y) / 2;
  const fs = compact ? 10 : 10.5;
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  const captions = [
    'The guesser walks straight to the right subset. The searcher has to try them in order.',
    `Guessed and checked in ${n + k} steps: ${n} moves down, ${k} additions. The search goes on.`,
    `The search reached the answer after ${tried.toLocaleString('en-US')} of ${L.toLocaleString('en-US')} subsets. Add one number and the tree doubles.`,
  ];
  const capIdx = searchDone && t > 0 ? 2 : checked ? 1 : 0;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-03.2"
      title="the search tree"
      kicker={`subset sum · n = ${n}`}
      hint="drag n: one more number doubles the leaves"
      footer={
        <Btn primary onClick={() => restart(1000)}>
          replay
        </Btn>
      }
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`A binary tree of all ${L} subsets of ${n} numbers; a searcher walks it leaf by leaf while a lucky guesser goes straight to the subset that sums to ${target}`}
        >
          {/* the numbers being decided, one per level */}
          <T x={LX - 4} y={6} anchor="end" size={fs - 1}>
            take?
          </T>
          {xs.map((x, d) => {
            const on = checked ? added.has(d) || taken[d] : gMove > d + 0.5 && taken[d];
            const adding = added.has(d);
            return (
              <T
                key={d}
                x={LX - 4}
                y={lvlY(d)}
                anchor="end"
                size={fs + 0.5}
                weight={on ? 700 : 400}
                color={adding ? GT.opt : on ? GT.eq : GT.inkSoft}
              >
                {x}
              </T>
            );
          })}
          <path d={tree} style={{ fill: 'none', stroke: GT.line, strokeWidth: 1 }} />
          {/* skip / take, on the root's two edges */}
          <g style={halo}>
            <T
              x={(geom.left + geom.right) / 2 - (geom.right - geom.left) / 8 - 6}
              y={lvlY(0) - 4}
              anchor="end"
              size={fs - 1}
            >
              skip
            </T>
            <T
              x={(geom.left + geom.right) / 2 + (geom.right - geom.left) / 8 + 6}
              y={lvlY(0) - 4}
              anchor="start"
              size={fs - 1}
            >
              take
            </T>
          </g>
          {/* the searcher's walk so far */}
          <path
            d={sPath}
            style={{
              fill: 'none',
              stroke: GT.ink,
              strokeWidth: 1.6,
              strokeLinejoin: 'round',
              opacity: 0.75,
            }}
          />
          {/* leaves */}
          {Array.from({ length: L }, (_, j) => {
            const p = nodeAt(geom, n, n, j);
            const was = leafPtIdx[j] !== undefined && leafPtIdx[j] <= prog + 1e-9;
            if (j === leaf) return null;
            return L <= 64 ? (
              <circle
                key={j}
                cx={p.x}
                cy={p.y}
                r={L <= 16 ? 3.5 : 2.5}
                style={{ fill: was ? GT.ink : GT.paperRaised, stroke: GT.muted, strokeWidth: 1 }}
              />
            ) : (
              <rect
                key={j}
                x={p.x - Math.max(0.5, leafW * 0.4)}
                y={p.y}
                width={Math.max(1, leafW * 0.8)}
                height={8}
                style={{ fill: was ? GT.ink : GT.line }}
              />
            );
          })}
          {/* the lucky guess */}
          <path
            d={gPath}
            style={{
              fill: 'none',
              stroke: GT.eq,
              strokeWidth: 3,
              strokeLinecap: 'round',
              strokeLinejoin: 'round',
            }}
          />
          <circle
            cx={sol.x}
            cy={sol.y}
            r={5}
            style={{
              fill: checked ? GT.opt : GT.paperRaised,
              stroke: checked ? GT.opt : GT.eq,
              strokeWidth: 2,
            }}
          />
          {/* tokens */}
          {!searchDone && t > 0 && (
            <circle cx={sPos.x} cy={sPos.y} r={4} style={{ fill: GT.ink }} />
          )}
          {gMove < n && t > 0 && <circle cx={gPos.x} cy={gPos.y} r={5} style={{ fill: GT.eq }} />}
          {/* the check: the sum under the leaf */}
          <g style={halo}>
            <T
              x={Math.min(geom.right - 2, Math.max(geom.left + 2, sol.x))}
              y={geom.bottom + 22}
              anchor={sol.x > geom.right - 60 ? 'end' : sol.x < geom.left + 60 ? 'start' : 'middle'}
              size={fs + 0.5}
              weight={700}
              color={checked ? GT.opt : GT.eq}
            >
              {adds > 0
                ? `${takenOrder
                    .slice(0, adds)
                    .map((d) => xs[d])
                    .join(' + ')} = ${sum}${checked ? ' ✓' : ''}`
                : ''}
            </T>
          </g>
          <T x={geom.left} y={H - 6} anchor="start" size={fs - 1}>
            {`${L.toLocaleString('en-US')} leaves, one per subset`}
          </T>
          <T x={geom.right} y={H - 6} anchor="end" size={fs - 1}>
            {`target ${target}`}
          </T>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <BigStat
          label="search: subsets tried"
          value={tried.toLocaleString('en-US')}
          unit={`of ${L.toLocaleString('en-US')}`}
          color={GT.ink}
        />
        <BigStat
          label="lucky guess + check"
          value={Math.floor(gSteps)}
          unit="steps"
          color={checked ? GT.opt : GT.eq}
        />
      </div>
      <div className="mt-4 max-w-xs">
        <Slider
          label="how many numbers"
          value={n}
          min={N_MIN}
          max={N_MAX}
          onChange={setN}
          format={(v) => `n = ${v}`}
        />
      </div>
      <Reserve
        items={captions.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
