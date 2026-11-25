import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { PAY } from './pd';
import { useCompact } from '../useCompact';

/*
 * Grim trigger. Each round, the game goes on with probability delta. Both cooperate
 * until someone defects; after that, both defect forever.
 *   cooperate forever:  R + dR + d^2 R + ...          = R / (1 - d)
 *   cheat now:          T + dP + d^2 P + ...          = T + dP / (1 - d)
 * Multiply by (1 - d) to read both as an average per round:
 *   cooperate: R        cheat: (1 - d) T + d P
 * Cooperation holds when R >= (1 - d) T + d P, i.e. d >= (T - R) / (T - P).
 */
const { T: TT, R, P } = PAY;
const D_STAR = (TT - R) / (TT - P); // = 1/2

// chart: plot area on the left, direct line labels in the right margin.
// Phones get a narrower viewBox so the labels stay legible instead of shrinking.
const H = 280;
const CX0 = 40;
const CY0 = 236;
const CY1 = 28;
const py = (v: number) => CY0 - (v / 5) * (CY0 - CY1);
const cheatAvg = (d: number) => (1 - d) * TT + d * P;

const SWEEP: [number, number][] = [
  [0, 0.2],
  [0.6, 0.2],
  [3.2, 0.9],
  [4.6, 0.75],
];
function sweepAt(t: number) {
  for (let i = 1; i < SWEEP.length; i++) {
    const [t0, v0] = SWEEP[i - 1];
    const [t1, v1] = SWEEP[i];
    if (t <= t1) {
      const u = (t - t0) / (t1 - t0);
      const e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      return v0 + (v1 - v0) * e;
    }
  }
  return SWEEP[SWEEP.length - 1][1];
}

export default function Shadow() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [d, setD] = useState(0.2);
  const [auto, setAuto] = useState(true);
  const t = useRef(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const [boxRef, compact] = useCompact();
  const W = compact ? 380 : 700;
  const CX1 = W - 124;
  const px = (v: number) => CX0 + v * (CX1 - CX0);
  const dragging = useRef(false);

  useEffect(() => {
    if (reduced) {
      setAuto(false);
      setD(0.75);
    }
  }, [reduced]);

  useFrame(
    (dt) => {
      t.current += dt;
      setD(sweepAt(t.current));
      if (t.current > SWEEP[SWEEP.length - 1][0]) setAuto(false);
    },
    auto && inView && !reduced,
  );

  const fromPointer = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const pt = svgPoint(svgRef.current, e.clientX, e.clientY);
    setD(clamp((pt.x - CX0) / (CX1 - CX0), 0, 0.95));
  };

  const coop = R;
  const cheat = cheatAvg(d);
  const holds = d >= D_STAR;
  const rounds = 1 / (1 - d);

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.2"
      title="the shadow of the future"
      kicker="grim trigger"
      hint="drag along the chart (or use ← →) to change the chance of another round"
    >
      <div ref={boxRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          style={{ cursor: 'ew-resize' }}
          role="img"
          aria-label="Per-round value of cooperating and of cheating once as the chance of another round grows"
          onPointerDown={(e) => {
            setAuto(false);
            dragging.current = true;
            (e.target as Element).setPointerCapture?.(e.pointerId);
            fromPointer(e);
          }}
          onPointerMove={(e) => dragging.current && fromPointer(e)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
          tabIndex={0}
          onKeyDown={(e) => {
            const k = e.key === 'ArrowRight' ? 0.05 : e.key === 'ArrowLeft' ? -0.05 : 0;
            if (!k) return;
            e.preventDefault();
            setAuto(false);
            setD((v) => clamp(Math.round((v + k) * 20) / 20, 0, 0.95));
          }}
        >
          {/* where cooperation holds */}
          <rect
            x={px(D_STAR)}
            y={CY1}
            width={CX1 - px(D_STAR)}
            height={CY0 - CY1}
            style={{ fill: GT.optSoft, opacity: 0.8 }}
          />
          <T x={(px(D_STAR) + CX1) / 2} y={CY1 + 14} size={11} color={GT.opt} weight={600}>
            cooperation holds
          </T>
          {[0, 1, 3, 5].map((v) => (
            <g key={v}>
              <line
                x1={CX0}
                y1={py(v)}
                x2={CX1}
                y2={py(v)}
                style={{ stroke: GT.line, strokeWidth: 0.6 }}
              />
              <T x={CX0 - 8} y={py(v)} anchor="end" size={11}>
                {v}
              </T>
            </g>
          ))}
          <line x1={CX0} y1={CY0} x2={CX1} y2={CY0} style={{ stroke: GT.lineStrong }} />
          <line x1={CX0} y1={CY0} x2={CX0} y2={CY1} style={{ stroke: GT.lineStrong }} />
          {[0, 0.5, 1].map((v) => (
            <T key={v} x={px(v)} y={CY0 + 14} size={11}>
              {v === 0.5 ? '½' : v}
            </T>
          ))}
          <T x={px(0.5)} y={CY0 + 32} size={11}>
            {'δ = chance of another round'}
          </T>
          <T x={CX0 - 8} y={CY1 - 16} anchor="start" size={10}>
            points per round
          </T>

          {/* the two lines, labelled where they end */}
          <line
            x1={px(0)}
            y1={py(R)}
            x2={px(1)}
            y2={py(R)}
            style={{ stroke: GT.a, strokeWidth: 3 }}
          />
          <line
            x1={px(0)}
            y1={py(TT)}
            x2={px(1)}
            y2={py(P)}
            style={{ stroke: GT.b, strokeWidth: 3 }}
          />
          <T x={CX1 + 10} y={py(R) - 7} anchor="start" size={11} color={GT.a} weight={600}>
            keep cooperating
          </T>
          <T x={CX1 + 10} y={py(R) + 8} anchor="start" size={10} color={GT.a}>
            {`${R} every round`}
          </T>
          <T x={CX1 + 10} y={py(P) - 7} anchor="start" size={11} color={GT.b} weight={600}>
            cheat once
          </T>
          <T x={CX1 + 10} y={py(P) + 8} anchor="start" size={10} color={GT.b}>
            {`${TT} once, then ${P}`}
          </T>
          <circle
            cx={px(D_STAR)}
            cy={py(R)}
            r={6}
            style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.5 }}
          />
          <T x={px(D_STAR) + 10} y={py(R) - 14} anchor="start" size={11} color={GT.eq} weight={600}>
            {'δ* = ½'}
          </T>

          {/* the reader's delta */}
          <line
            x1={px(d)}
            y1={CY1}
            x2={px(d)}
            y2={CY0}
            style={{ stroke: GT.ink, strokeWidth: 1.5 }}
          />
          <circle
            cx={px(d)}
            cy={py(coop)}
            r={5}
            style={{ fill: GT.a, stroke: GT.paperRaised, strokeWidth: 2 }}
          />
          <circle
            cx={px(d)}
            cy={py(cheat)}
            r={5}
            style={{ fill: GT.b, stroke: GT.paperRaised, strokeWidth: 2 }}
          />
          <rect x={px(d) - 6} y={CY0 - 6} width={12} height={12} style={{ fill: GT.ink }} />
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <BigStat
          label="another round"
          value={`${Math.round(d * 100)}%`}
          sub={`≈ ${rounds.toFixed(1)} rounds`}
        />
        <BigStat label="cooperate" value={coop.toFixed(2)} color={GT.a} sub="per round" />
        <BigStat
          label="cheat once"
          value={cheat.toFixed(2)}
          color={GT.b}
          sub={holds ? 'not worth it' : 'cheating pays'}
        />
      </div>
      <Caption>
        Cheating wins one big round and loses every round after it. Once another round is likely
        enough, those lost rounds outweigh the windfall.
      </Caption>
    </Specimen>
  );
}
