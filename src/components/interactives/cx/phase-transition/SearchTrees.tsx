import { useEffect, useRef, useState } from 'react';
import { Btn, GT, Specimen, T, useFrame, useInView, useReducedMotion } from '../../gt/shared';
import { dpll, randomFormula, type Node } from './sat';

/**
 * Three random 3-SAT puzzles with 50 variables, solved by the same DPLL solver at the same
 * pace: one node of the search tree every 1/RATE seconds. Loose (2 clauses per variable),
 * at the threshold (4.3) and tight (7). Each instance is the one closest to the median
 * solver effort among 101 seeded draws at its ratio (see sat.ts and the page text).
 *
 * All three trees share one scale, so the size of a tree is the work it took. A faint ghost
 * of each finished tree is the server-rendered poster; the solver draws over it.
 */
const N = 50;
const RATE = 55; // nodes per second, the same for all three
const CASES = [
  { key: 'loose', alpha: 2, seed: 779, name: 'loose' },
  { key: 'mid', alpha: 4.3, seed: 830, name: 'threshold' },
  { key: 'tight', alpha: 7, seed: 780, name: 'tight' },
] as const;

type Laid = {
  nodes: Node[];
  x: number[]; // in leaf units, 0..leaves-1
  leaves: number;
  depth: number;
  sat: boolean;
  conflicts: number;
};

function layout(alpha: number, seed: number): Laid {
  const r = dpll(N, randomFormula(N, Math.round(alpha * N), seed), true);
  const nodes = r.nodes!;
  const kids: number[][] = nodes.map(() => []);
  nodes.forEach((nd, i) => nd.parent >= 0 && kids[nd.parent].push(i));
  const x = new Array(nodes.length).fill(0);
  let leaf = 0;
  const place = (i: number) => {
    if (kids[i].length === 0) {
      x[i] = leaf++;
      return;
    }
    kids[i].forEach(place);
    x[i] = kids[i].reduce((s, k) => s + x[k], 0) / kids[i].length;
  };
  place(0);
  return {
    nodes,
    x,
    leaves: leaf,
    depth: Math.max(...nodes.map((d) => d.depth)),
    sat: r.sat,
    conflicts: nodes.filter((d) => d.end === 'conflict').length,
  };
}

const TREES = CASES.map((c) => layout(c.alpha, c.seed));
const LONGEST = Math.max(...TREES.map((t) => t.nodes.length)) / RATE;

type Box = { x0: number; x1: number; y0: number }; // where one tree is drawn
type Geo = { W: number; H: number; boxes: Box[]; dx: number; dy: number };

/** Wide: three columns. Narrow: the threshold tree on top, the other two below it. */
const WIDE: Geo = (() => {
  const dx = 4.6;
  const dy = 7;
  return {
    W: 640,
    H: 312,
    dx,
    dy,
    boxes: [
      { x0: 10, x1: 120, y0: 80 },
      { x0: 130, x1: 490, y0: 80 },
      { x0: 500, x1: 630, y0: 80 },
    ],
  };
})();
const NARROW: Geo = {
  W: 360,
  H: 470,
  dx: 4.4,
  dy: 6.4,
  boxes: [
    { x0: 0, x1: 150, y0: 266 },
    { x0: 0, x1: 360, y0: 72 },
    { x0: 160, x1: 360, y0: 266 },
  ],
};

const fmt = (n: number) => n.toLocaleString('en-US');

export default function SearchTrees() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  // t = seconds of solver time shown; null = not started (the ghost poster only)
  const [t, setT] = useState<number | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (reduced) {
      setT(LONGEST + 1);
      return;
    }
    if (!inView || started.current) return;
    const id = setTimeout(() => {
      started.current = true;
      setT(0);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      setT((v) => (v === null ? v : Math.min(LONGEST + 0.5, v + dt)));
    },
    inView && !reduced && t !== null && t < LONGEST + 0.5,
  );

  // replay: empty trees for 0.9 s (t runs from -0.9 to 0), then the solvers start again
  const replay = () => {
    started.current = true;
    if (!reduced) setT(-0.9);
  };

  const draw = (g: Geo, cls: string) => (
    <svg
      viewBox={`0 0 ${g.W} ${g.H}`}
      className={cls}
      role="img"
      aria-label="Three search trees: loose puzzles need one straight path, threshold puzzles a large bushy tree, tight puzzles a small tree"
    >
      {TREES.map((tr, k) => {
        const b = g.boxes[k];
        const c = CASES[k];
        const mid = (b.x0 + b.x1) / 2;
        const X = (i: number) => mid + (tr.x[i] - (tr.leaves - 1) / 2) * g.dx;
        const Y = (i: number) => b.y0 + tr.nodes[i].depth * g.dy;
        const shown = t === null ? 0 : t < 0 ? 0 : t * RATE; // nodes revealed (fractional)
        const done = shown >= tr.nodes.length;
        const count = Math.min(tr.nodes.length, Math.floor(shown) + (t !== null && t >= 0 ? 1 : 0));
        const ghost = 0.13;
        const isMid = k === 1;
        const lastY = b.y0 + tr.depth * g.dy;
        return (
          <g key={c.key}>
            {/* column label and live count, directly above its tree */}
            <T x={mid} y={b.y0 - 62} size={10.5} weight={600} color={isMid ? GT.eq : GT.ink}>
              {c.name}
            </T>
            <T x={mid} y={b.y0 - 48} size={9}>
              {`${c.alpha} clauses per variable`}
            </T>
            <T x={mid} y={b.y0 - 26} size={24} font="display" color={GT.ink}>
              {fmt(t === null ? tr.nodes.length : count)}
            </T>
            <T x={mid} y={b.y0 - 9} size={9}>
              solver steps
            </T>

            {/* ghost of the finished tree */}
            <g style={{ opacity: ghost }}>
              {tr.nodes.map((nd, i) =>
                i === 0 ? null : (
                  <line
                    key={i}
                    x1={X(nd.parent)}
                    y1={Y(nd.parent)}
                    x2={X(i)}
                    y2={Y(i)}
                    style={{ stroke: GT.ink, strokeWidth: 1 }}
                  />
                ),
              )}
            </g>

            {/* the solver's progress */}
            {t !== null &&
              tr.nodes.map((nd, i) => {
                if (i === 0) return null;
                const f = Math.max(0, Math.min(1, shown - (i - 1)));
                if (f <= 0) return null;
                const x1 = X(nd.parent);
                const y1 = Y(nd.parent);
                return (
                  <line
                    key={'p' + i}
                    x1={x1}
                    y1={y1}
                    x2={x1 + (X(i) - x1) * f}
                    y2={y1 + (Y(i) - y1) * f}
                    style={{ stroke: isMid ? GT.eq : GT.ink, strokeWidth: 1.3 }}
                  />
                );
              })}
            {tr.nodes.map((nd, i) => {
              const on = t === null || shown >= i;
              if (!on || nd.end === 'open') return null;
              if (nd.end === 'sat')
                return (
                  <circle
                    key={'s' + i}
                    cx={X(i)}
                    cy={Y(i)}
                    r={4}
                    style={{
                      fill: GT.opt,
                      opacity: t === null ? 0.35 : 1,
                    }}
                  />
                );
              return (
                <circle
                  key={'c' + i}
                  cx={X(i)}
                  cy={Y(i)}
                  r={1.7}
                  style={{ fill: GT.b, opacity: t === null ? 0.3 : 0.9 }}
                />
              );
            })}
            <circle cx={X(0)} cy={Y(0)} r={2.5} style={{ fill: GT.ink }} />

            {/* verdict under the tree, once the solver gets there */}
            <g style={{ opacity: t === null || done ? 1 : 0, transition: 'opacity 250ms' }}>
              <T x={mid} y={lastY + 16} size={9.5} color={tr.sat ? GT.opt : GT.b}>
                {tr.sat ? 'solution found' : 'no solution exists'}
              </T>
            </g>
          </g>
        );
      })}
    </svg>
  );

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-07.2"
      title="the search tree"
      kicker="50 variables, same solver, same speed"
      hint={
        <span className="inline-flex items-center gap-3">
          <Btn onClick={replay}>replay</Btn>
          <span>
            <span style={{ color: GT.b }}>●</span> dead end
            <span className="ml-3" style={{ color: GT.opt }}>
              ●
            </span>{' '}
            solution
          </span>
        </span>
      }
      footer="typical draws, seeded"
    >
      {draw(WIDE, 'hidden w-full sm:block')}
      {draw(NARROW, 'block w-full sm:hidden')}
    </Specimen>
  );
}
