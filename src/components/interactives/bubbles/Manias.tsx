import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import {
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../gt/shared';
import { useCompact } from '../gt/useCompact';
import { MANIAS, M0, M1, TODAY_M, valueAt, type Mania } from './manias';

/**
 * BX-01 "the same shape". Each mania's price, as a share of its own peak, against months from
 * that peak. On view one cursor sweeps from the left edge to the right on one eased clock and
 * every line is drawn up to it; a series with no data past "today" stops there with a marker.
 * Afterwards the reader drags the cursor. Dots and readouts are computed from the same points
 * the lines are drawn from (valueAt), so a dot always sits on its line. Hover a readout to
 * bring that line forward.
 */

type Geo = { W: number; H: number; L: number; R: number; Tp: number; B: number; fs: number };
const WIDE: Geo = { W: 640, H: 340, L: 44, R: 600, Tp: 16, B: 300, fs: 10 };
const NARROW: Geo = { W: 360, H: 330, L: 38, R: 344, Tp: 16, B: 292, fs: 10 };
const SWEEP_MS = 4200;
const HOLD_MS = 700;
const BACK_MS = 1300;
const TOTAL_MS = SWEEP_MS + HOLD_MS + BACK_MS;
const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const r2 = (v: number) => Math.round(v * 100) / 100;

function pathUpTo(m: Mania, upto: number, X: (v: number) => number, Y: (v: number) => number) {
  let d = '';
  const pts = m.points;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (p.m > upto) {
      if (i > 0) d += `L${r2(X(upto))} ${r2(Y(valueAt(m, upto)!))}`;
      break;
    }
    d += `${i ? 'L' : 'M'}${r2(X(p.m))} ${r2(Y(p.v))}`;
  }
  return d;
}

export default function Manias() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const g = compact ? NARROW : WIDE;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const X = (m: number) => g.L + ((m - M0) / (M1 - M0)) * (g.R - g.L);
  const Y = (v: number) => g.B - (v / 110) * (g.B - g.Tp);

  // SSR / reduced motion / rest: everything drawn, cursor on "today"
  const [cur, setCur] = useState(TODAY_M);
  const [drawn, setDrawn] = useState(M1);
  const [playing, setPlaying] = useState(false);
  const clock = useRef(0);
  const started = useRef(false);
  const dragging = useRef(false);
  const [focus, setFocus] = useState<string | null>(null);

  const play = () => {
    if (reduced) {
      setCur(TODAY_M);
      setDrawn(M1);
      return;
    }
    clock.current = 0;
    setDrawn(M0);
    setCur(M0);
    setPlaying(true);
  };
  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) return;
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);

  // one clock (ms): sweep to the right edge, hold, then glide back to "today"
  useFrame((dt) => {
    clock.current = Math.min(TOTAL_MS, clock.current + dt * 1000);
    const t = clock.current;
    if (t <= SWEEP_MS) {
      const m = M0 + ease(t / SWEEP_MS) * (M1 - M0);
      setDrawn(m);
      setCur(m);
    } else {
      setDrawn(M1);
      const u = Math.max(0, (t - SWEEP_MS - HOLD_MS) / BACK_MS);
      setCur(M1 + ease(Math.min(1, u)) * (TODAY_M - M1));
    }
    if (t >= TOTAL_MS) setPlaying(false);
  }, playing && inView);

  const fromPointer = (e: RPointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    const m = clamp(((p.x - g.L) / (g.R - g.L)) * (M1 - M0) + M0, M0, M1);
    setCur(Math.abs(m) < 0.8 ? 0 : Math.abs(m - TODAY_M) < 0.8 ? TODAY_M : m);
  };

  const curM = Math.round(cur);
  const pct = (id: string) => {
    const m = MANIAS.find((x) => x.id === id);
    const v = m ? valueAt(m, cur) : null;
    return v === null ? null : Math.round(v);
  };
  let caption: string;
  if (Math.abs(cur - TODAY_M) < 0.05)
    caption = `Today, seven weeks after its peak, Nvidia is at ${pct('ai')}% of it. At the same point after their peaks, Cisco was at ${pct('fiber')}% and the railways at ${pct('rail')}%.`;
  else if (curM === 0 && Math.abs(cur) < 0.05) caption = 'At the peak, every line is at 100%.';
  else
    caption = `${Math.abs(curM)} month${Math.abs(curM) === 1 ? '' : 's'} ${cur < 0 ? 'before' : 'after'} the peak.`;
  const fs = g.fs;
  const today = MANIAS.find((m) => m.today);

  return (
    <Specimen
      rootRef={rootRef}
      code="BX-01"
      title="the same shape"
      kicker="price as % of each peak"
      hint="drag across the chart; hover a name to bring its line forward"
      footer="sources at the end of the post"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${g.W} ${g.H}`}
          className="block w-full touch-none select-none"
          role="img"
          aria-label="Price of each mania as a percentage of its peak, aligned at the peak"
          style={{ cursor: 'ew-resize' }}
          onPointerDown={(e) => {
            setPlaying(false);
            setDrawn(M1);
            dragging.current = true;
            (e.target as Element).setPointerCapture?.(e.pointerId);
            fromPointer(e);
          }}
          onPointerMove={(e) => dragging.current && fromPointer(e)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          {/* grid */}
          {[0, 25, 50, 75, 100].map((v) => (
            <g key={v}>
              <line
                x1={g.L}
                x2={g.R}
                y1={Y(v)}
                y2={Y(v)}
                style={{
                  stroke: v === 100 ? GT.lineStrong : GT.line,
                  strokeDasharray: v === 100 ? '4 3' : undefined,
                }}
              />
              <T x={g.L - 6} y={Y(v)} anchor="end" size={fs - 1}>
                {`${v}%`}
              </T>
            </g>
          ))}
          {[-36, -24, -12, 0, 12, 24, 36, 48, 60]
            .filter((m) => m >= M0 && m <= M1)
            .map((m) => (
              <g key={m}>
                <line x1={X(m)} x2={X(m)} y1={g.B} y2={g.B + 4} style={{ stroke: GT.lineStrong }} />
                <T
                  x={X(m)}
                  y={g.B + 14}
                  size={fs - 1}
                  color={m === 0 ? GT.ink : GT.muted}
                  weight={m === 0 ? 700 : 400}
                >
                  {m === 0 ? 'peak' : `${m > 0 ? '+' : ''}${m / 12} yr`}
                </T>
              </g>
            ))}
          <line x1={X(0)} x2={X(0)} y1={g.Tp} y2={g.B} style={{ stroke: GT.lineStrong }} />

          {/* lines; the focused one is drawn last and thickest */}
          {[...MANIAS]
            .sort((a, b) => (a.id === focus ? 1 : 0) - (b.id === focus ? 1 : 0))
            .map((m) => {
              const dim = focus !== null && focus !== m.id;
              return (
                <path
                  key={m.id}
                  d={pathUpTo(m, drawn, X, Y)}
                  style={{
                    fill: 'none',
                    stroke: m.color,
                    strokeWidth: m.today ? 3 : focus === m.id ? 3 : 2,
                    strokeLinejoin: 'round',
                    opacity: dim ? 0.18 : 1,
                    transition: 'opacity 200ms',
                  }}
                />
              );
            })}

          {/* sparse series: a small mark on every real data point */}
          {MANIAS.filter((m) => m.sparse).map((m) =>
            m.points
              .filter((p) => p.m <= drawn && p.m >= M0 && p.m <= M1)
              .map((p, i) => (
                <rect
                  key={`${m.id}-${i}`}
                  x={r2(X(p.m) - 2.5)}
                  y={r2(Y(p.v) - 2.5)}
                  width={5}
                  height={5}
                  style={{ fill: m.color, opacity: focus && focus !== m.id ? 0.18 : 1 }}
                />
              )),
          )}

          {/* "today" marker at the end of the live series */}
          {today &&
            drawn >= today.points[today.points.length - 1].m &&
            (() => {
              const last = today.points[today.points.length - 1];
              return (
                <g>
                  <circle cx={X(last.m)} cy={Y(last.v)} r={5} style={{ fill: today.color }} />
                  <g
                    style={{
                      stroke: GT.paperRaised,
                      strokeWidth: 5,
                      paintOrder: 'stroke',
                      strokeLinejoin: 'round',
                    }}
                  >
                    <T
                      x={X(last.m) + 8}
                      y={Y(last.v) + 1}
                      anchor="start"
                      size={fs}
                      color={today.color}
                      weight={700}
                    >
                      {today.todayLabel ?? 'today'}
                    </T>
                  </g>
                </g>
              );
            })()}

          {/* cursor */}
          <line
            x1={r2(X(cur))}
            x2={r2(X(cur))}
            y1={g.Tp}
            y2={g.B}
            style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
          />
          {MANIAS.map((m) => {
            const v = valueAt(m, cur);
            if (v === null || cur > drawn + 1e-6) return null;
            return (
              <circle
                key={m.id}
                cx={r2(X(cur))}
                cy={r2(Y(v))}
                r={3.5}
                style={{
                  fill: GT.paperRaised,
                  stroke: m.color,
                  strokeWidth: 2,
                  opacity: focus && focus !== m.id ? 0.25 : 1,
                }}
              />
            );
          })}
        </svg>
      </div>

      {/* readouts double as the legend */}
      <div className="mt-3 grid grid-cols-3 gap-x-4 gap-y-3">
        {MANIAS.map((m) => {
          const v = cur <= drawn + 1e-6 ? valueAt(m, cur) : null;
          return (
            <button
              key={m.id}
              type="button"
              className="min-w-0 text-left"
              onPointerEnter={() => setFocus(m.id)}
              onPointerLeave={() => setFocus(null)}
              onFocus={() => setFocus(m.id)}
              onBlur={() => setFocus(null)}
              style={{ opacity: focus && focus !== m.id ? 0.45 : 1, transition: 'opacity 200ms' }}
            >
              <div
                className="flex items-center gap-1.5 font-mono text-[0.72rem] font-semibold"
                style={{ color: m.color }}
              >
                <span className="inline-block h-[3px] w-3" style={{ background: m.color }} />
                {m.label}
              </div>
              <div
                className="mt-0.5 whitespace-nowrap font-display text-2xl leading-none tabular-nums lining-nums"
                style={{ color: v === null ? GT.muted : m.color }}
              >
                {v === null ? 'no data' : `${Math.round(v)}%`}
              </div>
              <div className="mt-1 whitespace-nowrap font-mono text-[0.66rem] text-muted">
                {m.peakLabel}
              </div>
            </button>
          );
        })}
      </div>
      <div className="mt-3 flex items-start justify-between gap-4">
        <Caption>{caption}</Caption>
        <div className="mt-3 shrink-0">
          <Btn primary onClick={play}>
            replay
          </Btn>
        </div>
      </div>
    </Specimen>
  );
}
