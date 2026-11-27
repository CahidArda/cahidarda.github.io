import { useEffect, useRef, useState } from 'react';
import { Reserve } from '../../gt/Reserve';
import {
  BigStat,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { ALPHAS, RUNS } from './data';

/**
 * The seeded experiment behind the page (data.ts, generated from sat.ts): random 3-SAT with
 * 20, 50 and 100 variables, clauses per variable from 2 to 7. Top: the share of puzzles that
 * have a solution. Bottom: the median work for a DPLL solver (log scale). One cursor runs
 * through both. On view it glides from 2 to the threshold, then it is the reader's to drag.
 */
const NS = [20, 50, 100] as const;
const A0 = 2;
const A1 = 7;
const THRESH = 4.27; // Mertens, Mezard, Zecchina 2006: 4.267
const L_LO = Math.log10(6);
const L_HI = Math.log10(9000);
const DEMO_END = THRESH;
const DEMO_S = 2.2;

/** Height (share solvable) at which each top-panel curve gets its name, staggered. */
const LABEL_AT: Record<number, number> = { 20: 0.42, 50: 0.24, 100: 0.07 };
const COLOR: Record<number, string> = { 20: GT.muted, 50: GT.a, 100: GT.eq };

type Geo = {
  W: number;
  H: number;
  l: number;
  r: number;
  top: [number, number];
  bot: [number, number];
  narrow: boolean;
};
const WIDE: Geo = { W: 640, H: 392, l: 44, r: 92, top: [26, 168], bot: [200, 350], narrow: false };
const NARROW: Geo = { W: 360, H: 404, l: 36, r: 56, top: [26, 170], bot: [204, 356], narrow: true };

/** Value of a series at any alpha by straight-line interpolation (in the space it is drawn in). */
function at(series: number[], a: number, log = false) {
  const f = clamp((a - A0) / 0.1, 0, ALPHAS.length - 1);
  const i = Math.min(ALPHAS.length - 2, Math.floor(f));
  const u = f - i;
  const v0 = log ? Math.log10(series[i]) : series[i];
  const v1 = log ? Math.log10(series[i + 1]) : series[i + 1];
  const v = v0 + (v1 - v0) * u;
  return log ? 10 ** v : v;
}

/** First alpha where a falling series drops below `level`. */
function crossAt(series: number[], level: number) {
  for (let i = 1; i < series.length; i++)
    if (series[i - 1] >= level && series[i] < level)
      return ALPHAS[i - 1] + ((series[i - 1] - level) / (series[i - 1] - series[i])) * 0.1;
  return A1;
}

const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const r2 = (v: number) => Math.round(v * 100) / 100;

export default function Cliff() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [a, setA] = useState(A0);
  const demo = useRef<number | null>(null);
  const touched = useRef(false);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (reduced) {
      if (!touched.current) setA(DEMO_END);
      return;
    }
    if (!inView || touched.current) return;
    const id = setTimeout(() => {
      if (touched.current) return;
      touched.current = true;
      demo.current = 0;
      setTick((x) => x + 1);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      if (demo.current === null) return;
      demo.current += dt;
      const k = clamp(demo.current / DEMO_S, 0, 1);
      setA(A0 + (DEMO_END - A0) * ease(k));
      if (k >= 1) {
        demo.current = null;
        setTick((x) => x + 1);
      }
    },
    inView && !reduced && demo.current !== null,
  );

  const dragging = useRef(false);
  const fromPointer = (svg: SVGSVGElement, g: Geo, cx: number, cy: number) => {
    const p = svgPoint(svg, cx, cy);
    let v = A0 + ((p.x - g.l) / (g.W - g.l - g.r)) * (A1 - A0);
    v = clamp(v, A0, A1);
    if (Math.abs(v - THRESH) < 0.08) v = THRESH;
    demo.current = null;
    touched.current = true;
    setA(v);
  };

  const p100 = at(RUNS[100].p, a);
  const m100 = at(RUNS[100].med, a, true);
  const regime = a < 3.9 ? 0 : a <= 4.7 ? 1 : 2;

  const chart = (g: Geo, cls: string) => {
    const X0 = g.l;
    const X1 = g.W - g.r;
    const sx = (v: number) => r2(X0 + ((v - A0) / (A1 - A0)) * (X1 - X0));
    const syP = (p: number) => r2(g.top[1] - p * (g.top[1] - g.top[0]));
    const syL = (m: number) =>
      r2(g.bot[1] - ((Math.log10(m) - L_LO) / (L_HI - L_LO)) * (g.bot[1] - g.bot[0]));
    const fs = g.narrow ? 9 : 10;
    const line = (n: number, kind: 'p' | 'med') =>
      ALPHAS.map(
        (al, i) =>
          `${i ? 'L' : 'M'}${sx(al)} ${kind === 'p' ? syP(RUNS[n].p[i]) : syL(RUNS[n].med[i])}`,
      ).join('');
    const cx = sx(a);
    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={`${cls} touch-none select-none`}
        style={{ cursor: 'ew-resize' }}
        role="img"
        aria-label="Share of solvable random 3-SAT puzzles drops from 100% to 0% around 4.27 clauses per variable, while solver effort peaks there"
        onPointerDown={(e) => {
          dragging.current = true;
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
          fromPointer(e.currentTarget, g, e.clientX, e.clientY);
        }}
        onPointerMove={(e) =>
          dragging.current && fromPointer(e.currentTarget, g, e.clientX, e.clientY)
        }
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        {/* panel titles */}
        <T x={X0} y={g.top[0] - 14} anchor="start" size={fs} color={GT.inkSoft}>
          {g.narrow ? 'have a solution' : 'puzzles that have a solution'}
        </T>
        <T x={X0} y={g.bot[0] - 14} anchor="start" size={fs} color={GT.inkSoft}>
          solver work (median steps, log scale)
        </T>

        {/* grids */}
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1={X0} x2={X1} y1={syP(p)} y2={syP(p)} style={{ stroke: GT.line }} />
            <T x={X0 - 6} y={syP(p)} anchor="end" size={fs - 1}>
              {`${p * 100}%`}
            </T>
          </g>
        ))}
        {[10, 100, 1000].map((m) => (
          <g key={m}>
            <line x1={X0} x2={X1} y1={syL(m)} y2={syL(m)} style={{ stroke: GT.line }} />
            <T x={X0 - 6} y={syL(m)} anchor="end" size={fs - 1}>
              {m.toLocaleString('en-US')}
            </T>
          </g>
        ))}
        <line x1={X0} x2={X1} y1={g.bot[1]} y2={g.bot[1]} style={{ stroke: GT.lineStrong }} />
        {[2, 3, 4, 5, 6, 7].map((v) => (
          <g key={v}>
            <line
              x1={sx(v)}
              x2={sx(v)}
              y1={g.bot[1]}
              y2={g.bot[1] + 4}
              style={{ stroke: GT.lineStrong }}
            />
            <T x={sx(v)} y={g.bot[1] + 14} size={fs - 1}>
              {v}
            </T>
          </g>
        ))}
        <T x={X0} y={g.bot[1] + 30} anchor="start" size={fs - 1}>
          clauses per variable →
        </T>

        {/* the predicted threshold for very large puzzles */}
        <line
          x1={sx(THRESH)}
          x2={sx(THRESH)}
          y1={g.top[0] - 4}
          y2={g.bot[1]}
          style={{ stroke: GT.eq, strokeWidth: 1, strokeDasharray: '3 4', opacity: 0.7 }}
        />
        <T x={sx(THRESH) + 4} y={g.top[0] - 14} anchor="start" size={fs} color={GT.eq}>
          4.27
        </T>

        {/* curves */}
        {NS.map((n) => (
          <g key={n}>
            <path
              d={line(n, 'p')}
              style={{ fill: 'none', stroke: COLOR[n], strokeWidth: n === 100 ? 2.4 : 1.8 }}
            />
            <path
              d={line(n, 'med')}
              style={{ fill: 'none', stroke: COLOR[n], strokeWidth: n === 100 ? 2.4 : 1.8 }}
            />
          </g>
        ))}
        {/* direct labels at the right end of the work curves */}
        {NS.map((n) => (
          <T
            key={n}
            x={X1 + 6}
            y={syL(RUNS[n].med[ALPHAS.length - 1])}
            anchor="start"
            size={fs}
            color={COLOR[n]}
            weight={n === 100 ? 600 : 400}
          >
            {g.narrow ? `n=${n}` : `${n} variables`}
          </T>
        ))}
        {/* top panel: the same names, short, just right of where each curve gets low */}
        {NS.map((n) => {
          const lv = LABEL_AT[n];
          const al = crossAt(RUNS[n].p, lv);
          return (
            <T
              key={n}
              x={sx(al) + 6}
              y={syP(lv) - 2}
              anchor="start"
              size={fs}
              color={COLOR[n]}
              weight={n === 100 ? 600 : 400}
            >
              {String(n)}
            </T>
          );
        })}

        {/* cursor */}
        <line
          x1={cx}
          x2={cx}
          y1={g.top[0] - 2}
          y2={g.bot[1]}
          style={{ stroke: GT.ink, strokeWidth: 1 }}
        />
        {NS.map((n) => (
          <g key={n}>
            <circle
              cx={cx}
              cy={syP(at(RUNS[n].p, a))}
              r={n === 100 ? 4.5 : 3.5}
              style={{ fill: COLOR[n], stroke: GT.paperRaised, strokeWidth: 1.5 }}
            />
            <circle
              cx={cx}
              cy={syL(at(RUNS[n].med, a, true))}
              r={n === 100 ? 4.5 : 3.5}
              style={{ fill: COLOR[n], stroke: GT.paperRaised, strokeWidth: 1.5 }}
            />
          </g>
        ))}
      </svg>
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-07.1"
      title="the cliff"
      kicker="random 3-SAT, seeded run"
      hint="drag along the chart · snaps to 4.27"
      footer="readouts: 100 variables"
    >
      {chart(WIDE, 'hidden w-full sm:block')}
      {chart(NARROW, 'block w-full sm:hidden')}
      <div className="mt-4 grid grid-cols-3 gap-4">
        <BigStat label="clauses / variable" value={a.toFixed(2)} size="sm" />
        <BigStat label="solvable" color={GT.eq} value={`${Math.round(p100 * 100)}%`} size="sm" />
        <BigStat
          label="median steps"
          color={GT.eq}
          value={Math.round(m100).toLocaleString('en-US')}
          size="sm"
        />
      </div>
      <Reserve
        className="mt-3 text-[0.85rem] leading-snug text-ink-soft"
        active={regime}
        items={[
          'Loose: few rules per variable. Almost every puzzle has a solution, and the solver walks straight to one.',
          'The threshold: about half the puzzles are solvable, and telling which is which takes the most work.',
          'Tight: so many rules that contradictions turn up early. The solver proves "no solution" far faster than at the threshold.',
        ]}
      />
    </Specimen>
  );
}
