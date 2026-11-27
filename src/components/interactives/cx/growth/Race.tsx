import { Fragment, useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import { Stat } from './Stat';
import { COLOR } from './palette';
import {
  Btn,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import {
  CURVES,
  LOG_LIFETIME,
  LOG_UNIVERSE,
  fmtSteps,
  fmtTime,
  logSeconds,
  type CurveId,
} from './model';

/**
 * CX-01.1 "the race". Time to finish against input size n, on a log time axis, for n, n log n,
 * n², n³, 2ⁿ and n! at a billion steps a second. One cursor; dots sit on the curves because both
 * come from the same function. On view the cursor sweeps n from 5 to 60 on one eased clock, then
 * the reader drags (snaps to multiples of 5). Readouts use the whole n under the cursor.
 */

const N_MAX = 100;
const LY0 = -9; // 1 ns
const LY1 = 19; // top of the chart, log10 seconds
const START = 5;
const END = 60;
const DEMO_MS = 3200;
const r2 = (v: number) => Math.round(v * 100) / 100;
const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;

type Geo = { W: number; H: number; L: number; R: number; Tp: number; B: number; fs: number };
const WIDE: Geo = { W: 640, H: 360, L: 50, R: 584, Tp: 14, B: 322, fs: 10 };
const NARROW: Geo = { W: 360, H: 380, L: 48, R: 294, Tp: 14, B: 338, fs: 10 };

const TICKS: [number, string][] = [
  [-9, '1 ns'],
  [-6, '1 μs'],
  [-3, '1 ms'],
  [Math.log10(3600), '1 hour'],
  [Math.log10(365.25 * 86400), '1 year'],
];
// on phones the two high lines are labelled right of n!, which crosses them at n = 20 and 27 (its label sits just past 27)
const REFS: [number, string, string][] = [
  [0, '1 second', '1 second'],
  [LOG_LIFETIME, 'a lifetime (80 years)', 'a lifetime'],
  [LOG_UNIVERSE, 'age of the universe', 'age of the universe'],
];

/** Where the curve leaves the top of the chart (n), or null if it never does. */
function exitN(id: CurveId): number | null {
  const c = CURVES.find((q) => q.id === id)!;
  let lo = 1;
  let hi = N_MAX;
  if (logSeconds(c.logSteps(hi)) < LY1) return null;
  for (let i = 0; i < 50; i++) {
    const m = (lo + hi) / 2;
    if (logSeconds(c.logSteps(m)) < LY1) lo = m;
    else hi = m;
  }
  return lo;
}
const EXITS: Partial<Record<CurveId, number | null>> = {
  exp: exitN('exp'),
  fact: exitN('fact'),
};

function Chart({
  g,
  n,
  onDown,
}: {
  g: Geo;
  n: number;
  onDown: (e: RPointerEvent<SVGSVGElement>) => void;
}) {
  const { W, H, L, R, Tp, B, fs } = g;
  const sx = (v: number) => L + (v / N_MAX) * (R - L);
  const sy = (ls: number) => B - ((ls - LY0) / (LY1 - LY0)) * (B - Tp);
  const narrow = W < 500;

  const paths = CURVES.map((c) => {
    const n0 = c.id === 'nlogn' ? 2 : 1;
    let d = '';
    for (let v = n0; v <= N_MAX + 1e-9; v += 0.25) {
      let ls = logSeconds(c.logSteps(v));
      if (ls > LY1) {
        // stop exactly at the top edge
        const e = EXITS[c.id];
        if (e != null) d += `L${sx(e).toFixed(1)},${sy(LY1).toFixed(1)}`;
        break;
      }
      ls = Math.max(ls, LY0);
      d += `${d ? 'L' : 'M'}${sx(v).toFixed(1)},${sy(ls).toFixed(1)}`;
    }
    return { c, d };
  });

  // labels for the curves that reach n = 100: in the right gutter, pushed apart if crowded
  const gutter = CURVES.filter((c) => EXITS[c.id] == null).map((c) => ({
    c,
    y: sy(logSeconds(c.logSteps(N_MAX))),
  }));
  gutter.sort((a, b) => b.y - a.y);
  const gap = fs + 2;
  for (let i = 1; i < gutter.length; i++)
    if (gutter[i - 1].y - gutter[i].y < gap) gutter[i].y = gutter[i - 1].y - gap;

  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };
  const xc = r2(sx(n));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full touch-none select-none"
      role="img"
      aria-label="Time to finish against input size, for six growth rates"
      onPointerDown={onDown}
      style={{ cursor: 'ew-resize' }}
    >
      {/* grid */}
      {TICKS.map(([ly, lab]) => (
        <g key={lab}>
          <line x1={L} x2={R} y1={sy(ly)} y2={sy(ly)} style={{ stroke: GT.line }} />
          <T x={L - 5} y={sy(ly)} anchor="end" size={fs - 1}>
            {lab}
          </T>
        </g>
      ))}
      {[0, 20, 40, 60, 80, 100].map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={B} y2={B + 4} style={{ stroke: GT.lineStrong }} />
          <T x={sx(v)} y={B + 13} size={fs - 1}>
            {v}
          </T>
        </g>
      ))}
      <T x={(L + R) / 2} y={H - 6} size={fs} color={GT.inkSoft}>
        n, the size of the input
      </T>
      <line x1={L} x2={L} y1={Tp} y2={B} style={{ stroke: GT.lineStrong }} />
      <line x1={L} x2={R} y1={B} y2={B} style={{ stroke: GT.lineStrong }} />

      {/* reference lines, labelled on the left where the curves are still low */}
      {REFS.map(([ly, wide, short]) => (
        <g key={wide}>
          <line
            x1={L}
            x2={R}
            y1={sy(ly)}
            y2={sy(ly)}
            style={{ stroke: GT.inkSoft, strokeDasharray: '4 4', strokeWidth: 1 }}
          />
          <g style={halo}>
            <T
              x={narrow && ly > 5 ? sx(ly > 12 ? 40 : 24) : L + 5}
              y={sy(ly) - 7}
              anchor="start"
              size={fs}
              color={GT.inkSoft}
            >
              {narrow ? short : wide}
            </T>
          </g>
        </g>
      ))}

      {/* curves */}
      {paths.map(({ c, d }) => (
        <path
          key={c.id}
          d={d}
          style={{
            fill: 'none',
            stroke: COLOR[c.id],
            strokeWidth: c.poly ? 2 : 2.6,
            strokeLinejoin: 'round',
          }}
        />
      ))}

      {/* direct labels */}
      {gutter.map(({ c, y }) => (
        <T key={c.id} x={R + 6} y={y} anchor="start" size={fs + 1} color={COLOR[c.id]} weight={600}>
          {c.label}
        </T>
      ))}
      {(['fact', 'exp'] as const).map((id) => {
        const e = EXITS[id]!;
        return (
          <g key={id} style={halo}>
            {/* right of where each line leaves the top; 2ⁿ a little lower, under its own line */}
            <T
              x={sx(e) + 7}
              y={id === 'fact' ? Tp + 8 : Tp + 22}
              anchor="start"
              size={fs + 2}
              color={COLOR[id]}
              weight={700}
            >
              {CURVES.find((c) => c.id === id)!.label}
            </T>
          </g>
        );
      })}

      {/* cursor */}
      <line
        x1={xc}
        x2={xc}
        y1={Tp}
        y2={B}
        style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
      />
      {CURVES.map((c) => {
        const ls = logSeconds(c.logSteps(Math.max(n, c.id === 'nlogn' ? 2 : 1)));
        if (ls > LY1)
          return (
            <path key={c.id} d={`M${xc},${Tp - 1} l-4,7 h8 z`} style={{ fill: COLOR[c.id] }} />
          );
        return (
          <circle
            key={c.id}
            cx={xc}
            cy={r2(sy(Math.max(LY0, ls)))}
            r={3.6}
            style={{ fill: COLOR[c.id], stroke: GT.paperRaised, strokeWidth: 1.2 }}
          />
        );
      })}
      <rect
        x={xc - 7}
        y={B - 7}
        width={14}
        height={14}
        style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 2 }}
      />
    </svg>
  );
}

export default function Race() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [n, setN] = useState(START); // continuous, what is drawn
  const target = useRef(START);
  const demo = useRef<{ t0: number } | null>(null);
  const touched = useRef(false);
  const [running, setRunning] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const svgs = useRef<SVGSVGElement | null>(null);
  const dragging = useRef(false);

  const startDemo = () => {
    target.current = END;
    demo.current = { t0: performance.now() };
    setRunning(true);
  };

  // auto-play once, when first seen
  useEffect(() => {
    if (!inView || touched.current) return;
    touched.current = true;
    if (reduced) {
      target.current = END;
      setN(END);
      return;
    }
    timer.current = setTimeout(startDemo, 500);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // one clock: the scripted sweep, or a short glide to the snapped target after a drag
  useFrame((dt) => {
    if (demo.current) {
      const k = clamp((performance.now() - demo.current.t0) / DEMO_MS, 0, 1);
      setN(START + (END - START) * ease(k));
      if (k >= 1) {
        demo.current = null;
        setRunning(false);
      }
      return;
    }
    setN((v) => {
      const d = target.current - v;
      if (Math.abs(d) < 0.01) {
        setRunning(false);
        return target.current;
      }
      return v + d * clamp(dt * 14, 0, 1);
    });
  }, running && inView);

  const take = () => {
    touched.current = true;
    demo.current = null;
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };
  const goTo = (v: number) => {
    target.current = v;
    if (reduced) setN(v);
    else setRunning(true);
  };
  const fromPointer = (e: RPointerEvent) => {
    const svg = svgs.current;
    if (!svg) return;
    const p = svgPoint(svg, e.clientX, e.clientY);
    const vb = svg.viewBox.baseVal;
    const g = vb.width < 500 ? NARROW : WIDE;
    const raw = ((p.x - g.L) / (g.R - g.L)) * N_MAX;
    goTo(clamp(Math.round(raw / 5) * 5, 5, N_MAX));
  };
  const onDown = (e: RPointerEvent<SVGSVGElement>) => {
    take();
    svgs.current = e.currentTarget;
    dragging.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    fromPointer(e);
  };
  const replay = () => {
    take();
    if (reduced) {
      goTo(END);
      return;
    }
    demo.current = null;
    target.current = START;
    setN(START);
    setRunning(false);
    timer.current = setTimeout(startDemo, 1000);
  };

  const whole = Math.round(n);
  const rows = CURVES.map((c) => {
    const ls = c.logSteps(Math.max(whole, 1));
    return { c, ls, t: fmtTime(logSeconds(ls)) };
  });

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-01.1"
      title="the race"
      kicker="a billion steps a second"
      hint="drag across the chart to change n; it snaps to fives"
      footer="Sipser, §7.1"
    >
      <div
        tabIndex={0}
        role="slider"
        aria-label="input size n"
        aria-valuemin={5}
        aria-valuemax={N_MAX}
        aria-valuenow={whole}
        onPointerMove={(e) => dragging.current && fromPointer(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        onKeyDown={(e) => {
          const step =
            e.key === 'ArrowRight' || e.key === 'ArrowUp'
              ? 5
              : e.key === 'ArrowLeft' || e.key === 'ArrowDown'
                ? -5
                : 0;
          if (!step) return;
          e.preventDefault();
          take();
          goTo(clamp(Math.round(target.current / 5) * 5 + step, 5, N_MAX));
        }}
        style={{ outline: 'none' }}
      >
        <div className="hidden sm:block">
          <Chart g={WIDE} n={n} onDown={onDown} />
        </div>
        <div className="sm:hidden">
          <Chart g={NARROW} n={n} onDown={onDown} />
        </div>
      </div>

      <div className="mt-3 flex items-end justify-between gap-4 border-t border-line pt-3">
        <div className="font-display text-3xl leading-none tabular-nums lining-nums">
          <span className="mr-2 font-mono text-sm text-muted">n =</span>
          {whole}
        </div>
        <Btn primary onClick={replay}>
          replay
        </Btn>
      </div>
      <div className="mt-4 hidden grid-cols-3 gap-x-6 gap-y-4 sm:grid">
        {rows.map(({ c, t, ls }) => (
          <Stat
            key={c.id}
            label={c.label}
            value={t.value}
            unit={t.unit}
            color={COLOR[c.id]}
            sub={`${fmtSteps(ls)} steps`}
          />
        ))}
      </div>
      {/* phones: one row per growth rate, so long durations never wrap */}
      <div className="mt-3 grid grid-cols-[auto_minmax(0,1fr)_auto] items-baseline gap-x-3 gap-y-1.5 sm:hidden">
        <span />
        <span className="label">time</span>
        <span className="label text-right">steps</span>
        {rows.map(({ c, t, ls }) => (
          <Fragment key={c.id}>
            <span className="font-mono text-[0.8rem] font-semibold" style={{ color: COLOR[c.id] }}>
              {c.label}
            </span>
            <span className="whitespace-nowrap" style={{ color: COLOR[c.id] }}>
              <span className="font-display text-xl tabular-nums lining-nums">{t.value}</span>
              <span className="ml-1 font-mono text-[0.68rem] text-muted">{t.unit}</span>
            </span>
            <span className="text-right font-mono text-[0.68rem] text-muted tabular-nums">
              {fmtSteps(ls)}
            </span>
          </Fragment>
        ))}
      </div>
    </Specimen>
  );
}
