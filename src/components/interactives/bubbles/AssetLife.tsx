import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import {
  BigStat,
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
import { useBoost } from '../gt/braess/useBoost';
import { ASSETS } from './data';

/**
 * BX-03 "what the crash leaves behind". One bar per boom: how many years the thing it built
 * stays useful. A cursor sweeps the years after the crash on one eased clock; each bar shows
 * "still working" until the cursor passes its end, then greys out with "written off".
 * Bars and cursor come from the same scale, so the cursor always meets the bar ends exactly.
 */

const W = 600;
const L = 16;
const R = 572;
const MAX = 30; // years shown
const TOP = 34;
const ROW = 58;
const BAR = 14;
const H = TOP + ASSETS.length * ROW + 34;
const px = (y: number) => L + (clamp(y, 0, MAX) / MAX) * (R - L);
const r2 = (v: number) => Math.round(v * 100) / 100;
const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const SWEEP_MS = 5200;
const SNAPS = ASSETS.map((a) => a.life).filter((v) => v < MAX);

export default function AssetLife() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [svgRef, k] = useBoost(W);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const clock = useRef(0);
  const touched = useRef(false);
  const dragging = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!inView || touched.current) return;
    if (reduced) {
      setT(MAX);
      touched.current = true;
      return;
    }
    timer.current = setTimeout(() => setPlaying(true), 700);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [inView, reduced]);

  useFrame((dt) => {
    clock.current = Math.min(1, clock.current + (dt * 1000) / SWEEP_MS);
    setT(ease(clock.current) * MAX);
    if (clock.current >= 1) {
      setPlaying(false);
      touched.current = true;
    }
  }, playing && inView);

  const take = () => {
    touched.current = true;
    setPlaying(false);
    if (timer.current) clearTimeout(timer.current);
  };
  const restart = () => {
    take();
    if (reduced) {
      setT(MAX);
      return;
    }
    setT(0);
    clock.current = 0;
    timer.current = setTimeout(() => setPlaying(true), 900);
  };
  const snap = (v: number) => {
    for (const s of SNAPS) if (Math.abs(v - s) < 0.6) return s;
    return v;
  };
  const fromPointer = (e: RPointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    setT(snap(clamp(((p.x - L) / (R - L)) * MAX, 0, MAX)));
  };

  const year = Math.round(t);
  const alive = ASSETS.filter((a) => t < a.life);
  const dead = ASSETS.filter((a) => t >= a.life);
  let status: string;
  if (t < 0.5) status = 'The crash. Every boom leaves its hardware behind.';
  else if (dead.length === 0)
    status = `Year ${year}: everything built in the booms still works.`;
  else
    status = `Year ${year}: ${dead.map((a) => a.short).join(' and ')} ${
      dead.length > 1 ? 'are' : 'is'
    } written off. ${alive.map((a) => a.short).join(', ')} still work${alive.length === 1 ? 's' : ''}.`;

  const cx = r2(px(t));

  return (
    <Specimen
      rootRef={rootRef}
      code="BX-03"
      title="what the crash leaves behind"
      kicker="useful life after the boom"
      hint="drag across the chart to move through the years"
      footer="sources below the figure"
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full touch-none select-none"
        role="img"
        aria-label="Bars showing how many years the assets built in each boom stay useful"
        style={{ cursor: 'ew-resize' }}
        onPointerDown={(e) => {
          take();
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          fromPointer(e);
        }}
        onPointerMove={(e) => dragging.current && fromPointer(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        {/* year grid */}
        {[0, 5, 10, 15, 20, 25, 30].map((y) => (
          <g key={y}>
            <line
              x1={px(y)}
              x2={px(y)}
              y1={TOP - 6}
              y2={H - 26}
              style={{ stroke: GT.line, strokeDasharray: y === 0 ? undefined : '2 4' }}
            />
            <T x={px(y)} y={H - 14} size={9 * k} anchor={y === 0 ? 'start' : y === 30 ? 'end' : 'middle'}>
              {y === 0 ? 'crash' : `${y} yr`}
            </T>
          </g>
        ))}

        {ASSETS.map((a, i) => {
          const y0 = TOP + i * ROW;
          const by = y0 + 24;
          const isDead = t >= a.life;
          const end = px(a.life);
          const forever = a.life > MAX;
          const col = isDead ? GT.muted : a.color;
          return (
            <g key={a.id}>
              <T x={L} y={y0 + 8} anchor="start" size={11 * k} color={GT.ink} weight={600}>
                {a.label}
              </T>
              <T x={R} y={y0 + 8} anchor="end" size={9 * k} color={GT.muted}>
                {a.built}
              </T>
              {/* the bar: full life, outlined */}
              <rect
                x={L}
                y={by}
                width={r2(end - L)}
                height={BAR}
                style={{
                  fill: isDead ? 'transparent' : a.soft,
                  stroke: col,
                  strokeWidth: 1.5,
                  transition: 'fill 200ms, stroke 200ms',
                }}
              />
              {/* the years already used */}
              <rect
                x={L}
                y={by}
                width={r2(Math.max(0, Math.min(cx, end) - L))}
                height={BAR}
                style={{ fill: col, transition: 'fill 200ms' }}
              />
              {forever && (
                <path
                  d={`M${R - 1} ${by - 3} L${R + 9} ${by + BAR / 2} L${R - 1} ${by + BAR + 3} z`}
                  style={{ fill: col }}
                />
              )}
              <T
                x={L}
                y={by + BAR + 11 * k}
                anchor="start"
                size={9.5 * k}
                color={isDead ? GT.b : GT.inkSoft}
                weight={isDead ? 700 : 400}
              >
                {isDead ? `written off · ${a.lifeLabel}` : a.lifeLabel}
              </T>
            </g>
          );
        })}

        {/* the cursor */}
        <line
          x1={cx}
          x2={cx}
          y1={TOP - 10}
          y2={H - 26}
          style={{ stroke: GT.ink, strokeWidth: 1.25 }}
        />
        <rect
          x={r2(cx - 5)}
          y={TOP - 16}
          width={10}
          height={10}
          style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 2 }}
        />
      </svg>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-8">
          <BigStat size="lg" label="years after the crash" value={year} />
          <BigStat
            label="still working"
            value={`${alive.length} of ${ASSETS.length}`}
            color={dead.length ? GT.b : GT.ink}
          />
        </div>
        <Btn primary onClick={restart}>
          restart
        </Btn>
      </div>
      <div className="min-h-[3.2rem]">
        <Caption>{status}</Caption>
      </div>
    </Specimen>
  );
}
