import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Specimen,
  T,
  svgPoint,
  useInView,
  useReducedMotion,
} from '../shared';
import { SIM, UNHAPPY } from './curve';
import { pct, prefersReduced, useSharedT } from './sharedT';

// Final share of alike neighbours (y) against what residents asked for (x), from the
// precomputed runs in curve.ts. The diagonal is "what people asked for".

const W = 600;
const H = 320;
const P = { l: 44, r: 14, t: 16, b: 46 };
const TMAX = 0.9;
const x = (t: number) => P.l + (t / TMAX) * (W - P.l - P.r);
const y = (v: number) => P.t + (1 - v) * (H - P.t - P.b);
const at = (t: number) => SIM[Math.max(0, Math.min(SIM.length - 1, Math.round(t * 100)))];
const BREAK = UNHAPPY.findIndex((u) => u > 0) / 100; // first t that never settles

const path =
  SIM.map((v, k) => (k === 0 ? `M${x(0)},${y(v)}` : `H${x(k / 100)}V${y(v)}`)).join('') +
  `H${x(TMAX)}`;

export default function TippingCurve() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [t, setT] = useSharedT();
  const [demo, setDemo] = useState<number | null>(null);
  const started = useRef(false);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // auto-play once: sweep the marker across, then settle back on the shared t
  useEffect(() => {
    if (reduced || prefersReduced() || started.current || !inView) return;
    started.current = true;
    let raf = 0;
    const t0 = performance.now();
    const D = 3800;
    const tick = (now: number) => {
      const u = (now - t0) / D;
      if (u >= 1) {
        setDemo(null);
        return;
      }
      setDemo(0.8 * (u < 0.5 ? u * 2 : 2 - u * 2));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      setDemo(null);
    };
  }, [reduced, inView]);

  const cur = demo ?? t;
  const got = at(cur);
  const churn = cur >= BREAK - 1e-9;

  const onPtr = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    if (e.type === 'pointermove' && e.buttons !== 1) return;
    setDemo(null);
    started.current = true;
    const p = svgPoint(svgRef.current, e.clientX, e.clientY);
    const v =
      Math.round(Math.max(0, Math.min(TMAX, ((p.x - P.l) / (W - P.l - P.r)) * TMAX)) * 100) / 100;
    setT(v);
  };

  return (
    <Specimen
      code="GT-09.3"
      title="the tipping curve"
      kicker="40 × 40 city · 4 runs per point"
      rootRef={rootRef}
      hint="drag across the chart to pick a bar"
      footer={`t = ${pct(cur)}`}
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block w-full touch-none select-none"
        style={{ cursor: 'ew-resize' }}
        onPointerDown={onPtr}
        onPointerMove={onPtr}
        role="img"
        aria-label="Final share of alike neighbours against the share residents asked for"
      >
        {/* never-settles zone */}
        <rect
          x={x(BREAK)}
          y={P.t}
          width={x(TMAX) - x(BREAK)}
          height={H - P.t - P.b}
          style={{ fill: GT.warn, opacity: 0.12 }}
        />
        <T x={(x(BREAK) + x(TMAX)) / 2} y={y(0.2)} size={11} color={GT.warn}>
          never
        </T>
        <T x={(x(BREAK) + x(TMAX)) / 2} y={y(0.2) + 14} size={11} color={GT.warn}>
          settles
        </T>

        {/* grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={P.l}
              x2={W - P.r}
              y1={y(v)}
              y2={y(v)}
              style={{ stroke: GT.line, strokeDasharray: v === 0 ? undefined : '2 4' }}
            />
            <T x={P.l - 8} y={y(v)} anchor="end" size={13}>
              {pct(v)}
            </T>
          </g>
        ))}
        {[0, 0.2, 0.4, 0.6, 0.8].map((v) => (
          <T key={v} x={x(v)} y={H - P.b + 14} size={13}>
            {pct(v)}
          </T>
        ))}
        <T x={(P.l + W - P.r) / 2} y={H - 8} size={13} color={GT.inkSoft}>
          share alike each resident asks for →
        </T>

        {/* what people asked for: y = t */}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(TMAX)}
          y2={y(TMAX)}
          style={{ stroke: GT.inkSoft, strokeWidth: 1.4, strokeDasharray: '6 5' }}
        />
        <T x={x(0.56)} y={y(0.5)} anchor="start" size={13} color={GT.inkSoft}>
          what they asked for
        </T>

        {/* what they got */}
        <path d={path} style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.6 }} />
        <T x={x(0.04)} y={y(0.64)} anchor="start" size={13} color={GT.eq}>
          what they got
        </T>

        {/* marker */}
        <line
          x1={x(cur)}
          x2={x(cur)}
          y1={P.t}
          y2={H - P.b}
          style={{ stroke: GT.ink, strokeWidth: 1, opacity: 0.5 }}
        />
        <line
          x1={x(cur)}
          x2={x(cur)}
          y1={y(cur)}
          y2={y(got)}
          style={{ stroke: GT.eq, strokeWidth: 4, opacity: 0.35 }}
        />
        <circle
          cx={x(cur)}
          cy={y(cur)}
          r={5}
          style={{ fill: GT.paperRaised, stroke: GT.inkSoft, strokeWidth: 2 }}
        />
        <circle cx={x(cur)} cy={y(got)} r={6} style={{ fill: churn ? GT.warn : GT.eq }} />
        {!churn && got - cur > 0.08 && (
          <T
            x={x(cur) + 10}
            y={(y(cur) + y(got)) / 2}
            anchor="start"
            size={14}
            color={GT.eq}
            weight={600}
          >
            +{Math.round((got - cur) * 100)} pts
          </T>
        )}
      </svg>

      <div className="mt-3 flex flex-wrap items-end gap-x-10 gap-y-3 border-t border-line pt-4 sm:pl-[7.3%]">
        <BigStat label="asked for" value={pct(cur)} color={GT.inkSoft} sub="at least this alike" />
        <BigStat
          label="got"
          value={pct(got)}
          size="lg"
          color={churn ? GT.warn : GT.eq}
          sub={churn ? 'never settles' : 'once everyone settles'}
        />
      </div>
      <Caption>
        Past about {pct(BREAK)} some residents never find a spot, the moving never stops, and the
        city stays mixed.
      </Caption>
    </Specimen>
  );
}
