import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  BigStat,
  GT,
  Specimen,
  T,
  clamp,
  lerp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { useCompact } from '../../gt/useCompact';
import { tourCount } from './model';

/* Number of round trips (n-1)!/2 against n, on a log scale. At a billion tours a second,
 * 10^9 tours take one second, so the time lines sit on the same axis. */
const N0 = 3;
const N1 = 30;
const Y1 = 32; // top of the log10 axis
const RATE = 1e9;
const SEC = { day: 86400, year: 365.25 * 86400 };
const UNIVERSE_YEARS = 13.8e9;
const LINES = [
  { log: 9, wide: '1 second', narrow: '1 s' },
  { log: Math.log10(RATE * SEC.day), wide: '1 day', narrow: '1 day' },
  { log: Math.log10(RATE * SEC.year), wide: '1 year', narrow: '1 yr' },
  {
    log: Math.log10(RATE * SEC.year * UNIVERSE_YEARS),
    wide: 'age of the universe',
    narrow: 'universe',
  },
];
const log10Tours = (n: number) => Math.log10(tourCount(n));
/** 10 to a power, with a real raised exponent (mono fonts lack superscript digits). */
const Pow = ({ e }: { e: number }) => (
  <>
    10<sup className="text-[0.55em]">{e}</sup>
  </>
);

/** "43,589,145,600" up to 10^12, then "6.1 × 10¹⁶". */
function fmtCount(x: number): ReactNode {
  if (x < 1e12) return Math.round(x).toLocaleString('en-US');
  const e = Math.floor(Math.log10(x));
  return (
    <>
      {(x / 10 ** e).toFixed(1)} × <Pow e={e} />
    </>
  );
}

/** Seconds as [value, unit], always short enough for one readout. */
function fmtTime(s: number): [ReactNode, string] {
  const one = (v: number) => (v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString('en-US'));
  if (s < 1e-6) return [one(s * 1e9), 'ns'];
  if (s < 1e-3) return [one(s * 1e6), 'µs'];
  if (s < 1) return [one(s * 1e3), 'ms'];
  if (s < 120) return [one(s), 's'];
  if (s < 7200) return [one(s / 60), 'min'];
  if (s < 2 * SEC.day) return [one(s / 3600), 'hours'];
  const y = s / SEC.year;
  if (y < 1) return [one(s / SEC.day), 'days'];
  if (y < 1e3) return [one(y), 'years'];
  if (y < 1e6) return [one(y / 1e3), 'thousand years'];
  if (y < 1e9) return [one(y / 1e6), 'million years'];
  const u = y / UNIVERSE_YEARS;
  if (u < 1) return [one(y / 1e9), 'billion years'];
  if (u < 1000) return [one(u), '× age of universe'];
  const e = Math.floor(Math.log10(u));
  return [
    <>
      {(u / 10 ** e).toFixed(1)} × <Pow e={e} />
    </>,
    '× age of universe',
  ];
}

const ease = (t: number) => t * t * (3 - 2 * t);

export default function TourCount() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [n, setN] = useState(4);
  const nRef = useRef(4);
  const glide = useRef<{ from: number; to: number; t: number; dur: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const started = useRef(false);
  const [, setTick] = useState(0);

  const W = compact ? 360 : 560;
  const H = compact ? 300 : 300;
  const L = compact ? 40 : 52;
  const R = compact ? 300 : 440;
  const TOP = 16;
  const BOT = H - 34;
  const x = (v: number) => L + ((v - N0) / (N1 - N0)) * (R - L);
  const y = (lg: number) => BOT - (lg / Y1) * (BOT - TOP);
  const nOf = (px: number) => clamp(N0 + ((px - L) / (R - L)) * (N1 - N0), N0, N1);

  const setNow = (v: number) => {
    nRef.current = v;
    setN(v);
  };

  // auto-demo: start at 4 cities and glide to 15 once the figure scrolls in
  useEffect(() => {
    if (!inView || started.current) return;
    if (reduced) {
      started.current = true;
      setNow(15);
      return;
    }
    const id = window.setTimeout(() => {
      if (started.current) return;
      started.current = true;
      glide.current = { from: nRef.current, to: 15, t: 0, dur: 2.2 };
      setTick((x) => x + 1);
    }, 600);
    return () => window.clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      const g = glide.current;
      if (!g) return;
      g.t += dt / g.dur;
      if (g.t >= 1) {
        glide.current = null;
        setNow(g.to);
        return;
      }
      setNow(lerp(g.from, g.to, ease(g.t)));
    },
    inView && !reduced && glide.current !== null,
  );

  const pointerN = (e: React.PointerEvent) => {
    if (!svgRef.current) return nRef.current;
    return nOf(svgPoint(svgRef.current, e.clientX, e.clientY).x);
  };
  const onDown = (e: React.PointerEvent) => {
    started.current = true;
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    glide.current = null;
    setDragging(true);
    setNow(pointerN(e));
  };
  const onMove = (e: React.PointerEvent) => {
    if (dragging) setNow(pointerN(e));
  };
  const onUp = () => {
    if (!dragging) return;
    setDragging(false);
    const to = Math.round(nRef.current);
    if (reduced) setNow(to);
    else {
      glide.current = { from: nRef.current, to, t: 0, dur: 0.18 };
      setTick((x) => x + 1);
    }
  };
  const onKey = (e: React.KeyboardEvent) => {
    const d =
      e.key === 'ArrowRight' || e.key === 'ArrowUp'
        ? 1
        : e.key === 'ArrowLeft' || e.key === 'ArrowDown'
          ? -1
          : 0;
    if (!d) return;
    e.preventDefault();
    started.current = true;
    glide.current = null;
    setNow(clamp(Math.round(nRef.current) + d, N0, N1));
  };

  // the curve: one point per whole number of cities; the cursor dot interpolates along it
  const pts: [number, number][] = [];
  for (let k = N0; k <= N1; k++) pts.push([x(k), y(log10Tours(k))]);
  const lo = Math.floor(n);
  const hi = Math.min(N1, lo + 1);
  const f = n - lo;
  const dotLog = lerp(log10Tours(lo), log10Tours(hi), f);
  const cx = x(n);
  const cy = y(dotLog);
  const path = (arr: [number, number][]) =>
    arr.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(2)} ${py.toFixed(2)}`).join(' ');
  const done = pts.filter((_, i) => N0 + i <= n);
  const doneLine = [...done, [cx, cy] as [number, number]];

  const nInt = Math.round(n);
  const tours = tourCount(nInt);
  const [tv, tu] = fmtTime(tours / RATE);
  const fs = compact ? 9.5 : 10;

  const yTicks = [0, 10, 20, 30];
  const xTicks = compact ? [5, 10, 15, 20, 25, 30] : [5, 10, 15, 20, 25, 30];

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-06.2"
      title="how many tours"
      kicker="checking a billion tours a second"
      hint="drag across the chart"
      footer="tours = (n − 1)! / 2"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none outline-none"
          style={{ touchAction: 'pan-y', cursor: dragging ? 'grabbing' : 'ew-resize' }}
          tabIndex={0}
          onKeyDown={onKey}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          role="slider"
          aria-label="Number of cities"
          aria-valuemin={N0}
          aria-valuemax={N1}
          aria-valuenow={nInt}
        >
          {/* axes */}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={L} x2={R} y1={y(t)} y2={y(t)} style={{ stroke: GT.line, strokeWidth: 1 }} />
              <T x={L - 6} y={y(t)} anchor="end" size={fs}>
                {t === 0 ? (
                  '1'
                ) : (
                  <>
                    10
                    <tspan dy={-5} style={{ fontSize: fs * 0.72 }}>
                      {t}
                    </tspan>
                  </>
                )}
              </T>
            </g>
          ))}
          {xTicks.map((t) => (
            <T key={t} x={x(t)} y={BOT + 14} size={fs}>
              {t}
            </T>
          ))}
          <T x={(L + R) / 2} y={BOT + 28} size={fs}>
            cities
          </T>
          <line x1={L} x2={R} y1={BOT} y2={BOT} style={{ stroke: GT.lineStrong, strokeWidth: 1 }} />

          {/* time lines, labelled where they end */}
          {LINES.map((l) => (
            <g key={l.wide}>
              <line
                x1={L}
                x2={R}
                y1={y(l.log)}
                y2={y(l.log)}
                style={{ stroke: GT.warn, strokeWidth: 1.2, strokeDasharray: '4 3' }}
              />
              <T x={R + 6} y={y(l.log)} anchor="start" size={fs} color={GT.warn}>
                {compact ? l.narrow : l.wide}
              </T>
            </g>
          ))}

          {/* the curve: faint ahead of the cursor, solid behind it */}
          <path
            d={path(pts)}
            style={{ fill: 'none', stroke: GT.a, strokeWidth: 2, opacity: 0.28 }}
          />
          <path d={path(doneLine)} style={{ fill: 'none', stroke: GT.a, strokeWidth: 2.6 }} />

          {/* cursor */}
          <line
            x1={cx}
            x2={cx}
            y1={TOP}
            y2={BOT}
            style={{ stroke: GT.ink, strokeWidth: 1, opacity: 0.5 }}
          />
          <circle
            cx={cx}
            cy={cy}
            r={5.5}
            style={{ fill: GT.a, stroke: GT.paperRaised, strokeWidth: 2 }}
          />
          <g>
            <rect x={cx - 15} y={TOP - 2} width={30} height={16} style={{ fill: GT.ink }} />
            <T x={cx} y={TOP + 6} size={10} color={GT.paperRaised}>
              {nInt}
            </T>
          </g>
          {/* hit area over the plot */}
          <rect
            x={L - 10}
            y={TOP - 4}
            width={R - L + 20}
            height={BOT - TOP + 20}
            style={{ fill: 'transparent' }}
          />
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-[4.5rem_1fr] gap-x-6 gap-y-3 sm:grid-cols-[5rem_17rem_1fr]">
        <BigStat size={compact ? 'sm' : 'md'} label="cities" value={nInt} color={GT.ink} />
        <BigStat
          size={compact ? 'sm' : 'md'}
          label="round trips"
          value={<span className="whitespace-nowrap">{fmtCount(tours)}</span>}
          color={GT.a}
        />
        <div className="col-span-2 sm:col-span-1">
          <BigStat
            size={compact ? 'sm' : 'md'}
            label="to check them all"
            value={<span className="whitespace-nowrap">{tv}</span>}
            unit={tu}
            color={GT.warn}
          />
        </div>
      </div>
    </Specimen>
  );
}
