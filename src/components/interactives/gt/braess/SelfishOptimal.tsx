import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { useBoost } from './useBoost';

/**
 * Pigou's example. One unit of traffic, two roads from S to T. The fixed road always takes
 * 60 min; the other takes x * 60 min when a fraction x of the drivers use it.
 * Average trip C(x) = x * x + (1 - x) * 1 hours: minimum 3/4 h at x = 1/2, but the only
 * equilibrium is x = 1 (average 1 h). Ratio 4/3.
 */
const cost = (x: number) => (x * x + (1 - x)) * 60;

const W = 520;
const H = 384;
// road diagram band
const S = { x: 40, y: 62 };
const Tn = { x: 480, y: 62 };
// plot band
const PL = 58;
const PR = 496;
const PT = 150;
const PB = 318;
const Y0 = 30;
const Y1 = 66;
const px = (x: number) => PL + x * (PR - PL);
const py = (m: number) => PB - ((m - Y0) / (Y1 - Y0)) * (PB - PT);

export default function SelfishOptimal() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [x, setX] = useState(0.5);
  const xRef = useRef(x);
  xRef.current = x;
  const [demo, setDemo] = useState(false);
  const touched = useRef(false);
  const [svgRef, k] = useBoost(W);
  const dragging = useRef(false);

  // auto-play: start at the optimum, then let fixed-road drivers drift over
  useEffect(() => {
    if (!inView || touched.current) return;
    if (reduced) {
      setX(1);
      touched.current = true;
      return;
    }
    const id = setTimeout(() => {
      if (!touched.current) setDemo(true);
    }, 1400);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame((dt) => {
    const n = Math.min(1, xRef.current + dt * 0.13);
    xRef.current = n;
    setX(n);
    if (n >= 1) {
      setDemo(false);
      touched.current = true;
    }
  }, demo && inView);

  const take = () => {
    touched.current = true;
    setDemo(false);
  };
  const fromPointer = (e: RPointerEvent) => {
    if (!svgRef.current) return;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    setX(clamp((p.x - PL) / (PR - PL), 0, 1));
  };

  const xs = Math.round(x * 100) / 100;
  const c = cost(xs);
  const gain = 60 - 60 * xs;
  let status: string;
  if (xs >= 0.995)
    status =
      'Everyone is on the x-road and it takes 60 min, the same as the fixed road. No one gains by switching: this is the equilibrium.';
  else if (Math.abs(xs - 0.5) < 0.005)
    status = `The best possible average, 45 min. But the x-road takes 30 min, so each fixed-road driver gains 30 min by switching.`;
  else
    status = `The x-road takes ${Math.round(60 * xs)} min, so a fixed-road driver gains ${Math.round(gain)} min by switching. Drivers keep switching until the x-road is full.`;

  // the average-cost curve
  let curve = '';
  for (let i = 0; i <= 100; i++) {
    const u = i / 100;
    curve += `${i ? 'L' : 'M'}${px(u).toFixed(1)} ${py(cost(u)).toFixed(1)}`;
  }
  const wFixed = 2 + (1 - xs) * 14;
  const wVar = 2 + xs * 14;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-05.2"
      title="selfish vs optimal"
      kicker="Pigou's two roads"
      hint="drag across the chart to choose how traffic splits"
      footer="Roughgarden & Tardos, 2002"
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full touch-none select-none"
        role="img"
        aria-label="Two roads and a chart of average travel time against the share of drivers on the variable road"
        onPointerDown={(e) => {
          take();
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          fromPointer(e);
        }}
        onPointerMove={(e) => dragging.current && fromPointer(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        style={{ cursor: 'ew-resize' }}
      >
        {/* two roads */}
        <path
          d={`M${S.x + 16} ${S.y} C 150 ${S.y - 58}, 370 ${S.y - 58}, ${Tn.x - 16} ${Tn.y}`}
          style={{ fill: 'none', stroke: GT.b, strokeWidth: wFixed, opacity: 0.8 }}
        />
        <path
          d={`M${S.x + 16} ${S.y} C 150 ${S.y + 58}, 370 ${S.y + 58}, ${Tn.x - 16} ${Tn.y}`}
          style={{ fill: 'none', stroke: GT.a, strokeWidth: wVar, opacity: 0.8 }}
        />
        <T x={260} y={10} size={10 * k} color={GT.b}>
          {`fixed road · 60 min · ${Math.round((1 - xs) * 100)}% of drivers`}
        </T>
        <T x={260} y={118} size={10 * k} color={GT.a}>
          {`x-road · 60x = ${Math.round(xs * 60)} min · ${Math.round(xs * 100)}% of drivers`}
        </T>
        {[
          ['S', S],
          ['T', Tn],
        ].map(([k, p]) => {
          const q = p as { x: number; y: number };
          return (
            <g key={k as string}>
              <rect
                x={q.x - 15}
                y={q.y - 15}
                width={30}
                height={30}
                style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.5 }}
              />
              <T x={q.x} y={q.y + 1} size={16} font="display" color={GT.ink}>
                {k as string}
              </T>
            </g>
          );
        })}

        {/* chart axes */}
        <line x1={PL} y1={PB} x2={PR} y2={PB} style={{ stroke: GT.lineStrong }} />
        <line x1={PL} y1={PT} x2={PL} y2={PB} style={{ stroke: GT.lineStrong }} />
        {[30, 45, 60].map((m) => (
          <g key={m}>
            <line x1={PL} x2={PR} y1={py(m)} y2={py(m)} style={{ stroke: GT.line }} />
            <T x={PL - 6} y={py(m)} anchor="end" size={9 * k}>
              {m}
            </T>
          </g>
        ))}
        <T x={PL - 6} y={PT - 10 * k} anchor="end" size={9 * k}>
          min
        </T>
        {[0, 0.25, 0.5, 0.75, 1].map((u) => (
          <T key={u} x={px(u)} y={PB + 12 * k} size={9 * k}>
            {`${u * 100}%`}
          </T>
        ))}
        <T x={(PL + PR) / 2} y={PB + 30 * k} size={9 * k}>
          share of drivers on the x-road
        </T>

        {/* per-road times */}
        <line
          x1={px(0)}
          y1={py(60)}
          x2={px(1)}
          y2={py(60)}
          style={{ stroke: GT.b, strokeWidth: 1.5, strokeDasharray: '5 4' }}
        />
        <line
          x1={px(0.5)}
          y1={py(30)}
          x2={px(1)}
          y2={py(60)}
          style={{ stroke: GT.a, strokeWidth: 1.5, strokeDasharray: '5 4' }}
        />
        {/* the average */}
        <path d={curve} style={{ fill: 'none', stroke: GT.ink, strokeWidth: 2.5 }} />

        {/* landmarks */}
        <circle cx={px(0.5)} cy={py(45)} r={6} style={{ fill: GT.opt }} />
        <T x={px(0.5)} y={py(45) + 17 * k} size={10 * k} color={GT.opt} weight={700}>
          best: 45 min
        </T>
        <circle cx={px(1)} cy={py(60)} r={6} style={{ fill: GT.eq }} />
        <T x={px(1) - 8} y={py(60) - 16 * k} anchor="end" size={10 * k} color={GT.eq} weight={700}>
          selfish: 60 min
        </T>

        {/* the reader's split */}
        <line
          x1={px(xs)}
          x2={px(xs)}
          y1={PT - 4}
          y2={PB}
          style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
        />
        <g
          tabIndex={0}
          role="slider"
          aria-label="share of drivers on the x-road"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(xs * 100)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
              take();
              setX((v) => clamp(v + 0.05, 0, 1));
              e.preventDefault();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
              take();
              setX((v) => clamp(v - 0.05, 0, 1));
              e.preventDefault();
            }
          }}
          style={{ outline: 'none' }}
        >
          <rect
            x={px(xs) - 8}
            y={py(c) - 8}
            width={16}
            height={16}
            style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 2.5 }}
          />
        </g>
      </svg>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.ink}>average trip</Chip>
        <Chip color={GT.b}>fixed road</Chip>
        <Chip color={GT.a}>x-road</Chip>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-[auto_auto_1fr] sm:items-end">
        <BigStat
          size="lg"
          label="average trip"
          value={Math.round(c * 10) / 10}
          unit="min"
          color={xs >= 0.995 ? GT.eq : Math.abs(xs - 0.5) < 0.005 ? GT.opt : GT.ink}
        />
        <BigStat label="selfish ÷ best" value="4/3" sub="60 min ÷ 45 min" />
        <div className="col-span-2 flex gap-2 sm:col-span-1 sm:justify-self-end">
          <Btn
            onClick={() => {
              take();
              setX(0.5);
            }}
          >
            best split
          </Btn>
          <Btn
            onClick={() => {
              take();
              setX(1);
            }}
          >
            selfish split
          </Btn>
        </div>
      </div>
      <Caption>{status}</Caption>
    </Specimen>
  );
}
