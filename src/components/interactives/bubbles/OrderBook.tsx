import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../gt/shared';
import { useBoost } from '../gt/braess/useBoost';
import { BOEING, BOOM } from './data';

const AIR = BOOM.air.color;

/**
 * BX-04 "the order book can shrink". Boeing commercial net orders per year (bars, red when
 * negative) with deliveries as a tick on each bar. On view the bars grow in year by year on
 * one continuous clock; hover (or tap) a year to read it. The bar heights and the hover
 * readout come from the same row.
 */

const W = 600;
const H = 300;
const L = 46;
const R = 588;
const TOP = 18;
const BOT = 262;
const V0 = -1100;
const V1 = 1700;
const y = (v: number) => TOP + ((V1 - v) / (V1 - V0)) * (BOT - TOP);
const n = BOEING.length;
const slot = (R - L) / n;
const bw = slot * 0.56;
const GROW_MS = 3000;
const r2 = (v: number) => Math.round(v * 100) / 100;

export default function OrderBook() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [svgRef, k] = useBoost(W);
  const [clk, setClk] = useState(1); // SSR / reduced: fully drawn
  const clock = useRef(1);
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);
  const [hover, setHover] = useState<number | null>(null);

  const play = () => {
    if (reduced) return;
    clock.current = 0;
    setClk(0);
    setPlaying(true);
  };
  useEffect(() => {
    if (!inView || started.current || reduced) return;
    started.current = true;
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);
  useFrame((dt) => {
    clock.current = Math.min(1, clock.current + (dt * 1000) / GROW_MS);
    setClk(clock.current);
    if (clock.current >= 1) setPlaying(false);
  }, playing && inView);

  // each bar grows over its own share of the clock, one after another
  const grow = (i: number) => {
    const a = (i / n) * 0.85;
    const u = Math.max(0, Math.min(1, (clk - a) / 0.15));
    return 1 - (1 - u) * (1 - u);
  };
  const sel = hover ?? BOEING.findIndex((r) => r.year === 2020);
  const row = BOEING[sel];

  return (
    <Specimen
      rootRef={rootRef}
      code="BX-04"
      title="the order book can shrink"
      kicker="Boeing commercial airplanes"
      hint="hover a year to read it"
      footer="Boeing annual results; Reuters"
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full select-none"
        role="img"
        aria-label="Boeing net orders and deliveries per year, 2015 to 2025"
        onPointerLeave={() => setHover(null)}
      >
        {[-1000, -500, 0, 500, 1000, 1500].map((v) => (
          <g key={v}>
            <line
              x1={L}
              x2={R}
              y1={y(v)}
              y2={y(v)}
              style={{ stroke: v === 0 ? GT.ink : GT.line, strokeWidth: v === 0 ? 1.25 : 1 }}
            />
            <T x={L - 6} y={y(v)} anchor="end" size={9 * k}>
              {v === 0 ? '0' : v.toLocaleString('en-US')}
            </T>
          </g>
        ))}
        <T x={L - 6} y={TOP - 8} anchor="end" size={9 * k}>
          planes
        </T>

        {BOEING.map((r, i) => {
          const cx = L + slot * (i + 0.5);
          const g = grow(i);
          const v = r.net * g;
          const neg = r.net < 0;
          const on = i === sel;
          const y0 = y(0);
          const yv = y(v);
          return (
            <g
              key={r.year}
              onPointerEnter={() => setHover(i)}
              onClick={() => setHover(i)}
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={L + slot * i}
                y={TOP}
                width={slot}
                height={BOT - TOP + 24}
                style={{ fill: 'transparent' }}
              />
              <rect
                x={r2(cx - bw / 2)}
                y={r2(Math.min(y0, yv))}
                width={r2(bw)}
                height={r2(Math.abs(yv - y0))}
                style={{
                  fill: neg ? GT.b : AIR,
                  opacity: on ? 1 : 0.72,
                  transition: 'opacity 150ms',
                }}
              />
              {/* deliveries: an ink tick at its height */}
              {g >= 1 && (
                <line
                  x1={r2(cx - bw / 2 - 4)}
                  x2={r2(cx + bw / 2 + 4)}
                  y1={y(r.delivered)}
                  y2={y(r.delivered)}
                  style={{ stroke: GT.ink, strokeWidth: 2.5 }}
                />
              )}
              <T
                x={cx}
                y={BOT + 14}
                size={9 * k}
                color={on ? GT.ink : GT.muted}
                weight={on ? 700 : 400}
              >
                {k > 1.2 ? `’${String(r.year).slice(2)}` : r.year}
              </T>
            </g>
          );
        })}
        {/* direct labels */}
        <g
          style={{
            stroke: GT.paperRaised,
            strokeWidth: 5,
            paintOrder: 'stroke',
            strokeLinejoin: 'round',
          }}
        >
          <T
            x={L + slot * 0.5 - bw / 2}
            y={y(BOEING[0].net) - 18}
            anchor="start"
            size={9.5 * k}
            color={AIR}
            weight={600}
          >
            net new orders
          </T>
          <T
            x={L + slot * 3.5 + bw / 2 + 6}
            y={y(BOEING[3].delivered) + 1}
            anchor="start"
            size={9.5 * k}
            color={GT.ink}
            weight={600}
          >
            delivered
          </T>
        </g>
      </svg>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-8">
          <BigStat
            size="lg"
            label={`${row.year} net orders`}
            value={row.net.toLocaleString('en-US')}
            color={row.net < 0 ? GT.b : AIR}
          />
          <BigStat label="delivered" value={row.delivered.toLocaleString('en-US')} />
        </div>
        <Btn primary onClick={play}>
          replay
        </Btn>
      </div>
      <div className="min-h-[3.2rem]">
        <Caption>{row.note}</Caption>
      </div>
    </Specimen>
  );
}
