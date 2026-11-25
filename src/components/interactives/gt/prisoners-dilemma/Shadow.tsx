import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
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
import { PAY } from './pd';

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

const W = 520;
const H = 350;
// streams (top)
const N = 12;
const SX0 = 112;
const SQ = 30;
const SG = 4;
const SY = [30, 72];
// chart (bottom)
const CX0 = 60;
const CX1 = 488;
const CY0 = 312;
const CY1 = 128;
const px = (d: number) => CX0 + d * (CX1 - CX0);
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

  const stream = (row: 0 | 1) =>
    Array.from({ length: N }, (_, k) => {
      const v = row === 0 ? R : k === 0 ? TT : P;
      const isD = row === 1;
      const w = Math.pow(d, k);
      const x = SX0 + k * (SQ + SG);
      return (
        <g key={k} style={{ opacity: Math.max(0.04, w) }}>
          <rect
            x={x}
            y={SY[row]}
            width={SQ}
            height={SQ}
            style={{ fill: isD ? (k === 0 ? GT.b : GT.bSoft) : GT.a, stroke: isD ? GT.b : GT.a }}
          />
          <T
            x={x + SQ / 2}
            y={SY[row] + SQ / 2 + 1}
            size={14}
            font="display"
            color={isD && k > 0 ? GT.b : GT.paper}
          >
            {v}
          </T>
        </g>
      );
    });

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.2"
      title="the shadow of the future"
      kicker="grim trigger"
      hint="drag along the chart to change the chance of another round"
      footer={`threshold δ* = ${D_STAR}`}
    >
      <div className="flex flex-wrap gap-x-10 gap-y-4">
        <BigStat
          label="another round"
          value={`${Math.round(d * 100)}%`}
          sub={`about ${rounds.toFixed(1)} rounds on average`}
        />
        <BigStat label="keep cooperating" value={coop.toFixed(2)} unit="/round" color={GT.a} />
        <BigStat
          label="cheat once"
          value={cheat.toFixed(2)}
          unit="/round"
          color={GT.b}
          sub={holds ? 'not worth it' : 'cheating pays'}
        />
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="mt-3 block w-full touch-none select-none"
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
      >
        {/* the two futures */}
        <T x={SX0 - 10} y={SY[0] + SQ / 2} anchor="end" size={11} color={GT.a}>
          cooperate
        </T>
        <T x={SX0 - 10} y={SY[1] + SQ / 2} anchor="end" size={11} color={GT.b}>
          cheat once
        </T>
        {stream(0)}
        {stream(1)}
        <T x={SX0} y={SY[1] + SQ + 14} anchor="start" size={11}>
          {'round 1, 2, 3 … faded by the chance it is ever played'}
        </T>

        {/* chart */}
        <rect
          x={px(D_STAR)}
          y={CY1}
          width={CX1 - px(D_STAR)}
          height={CY0 - CY1}
          style={{ fill: GT.optSoft, opacity: 0.8 }}
        />
        <T x={(px(D_STAR) + CX1) / 2} y={CY1 + 12} size={11} color={GT.opt}>
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
          <T key={v} x={px(v)} y={CY0 + 13} size={11}>
            {v === 0.5 ? '½' : v}
          </T>
        ))}
        <T x={px(0.5)} y={CY0 + 30} size={11}>
          {'δ = chance of another round'}
        </T>
        <g transform={`translate(${CX0 - 30} ${(CY0 + CY1) / 2}) rotate(-90)`}>
          <T x={0} y={0} size={11}>
            payoff per round
          </T>
        </g>

        <line
          x1={px(0)}
          y1={py(R)}
          x2={px(1)}
          y2={py(R)}
          style={{ stroke: GT.a, strokeWidth: 2.5 }}
        />
        <line
          x1={px(0)}
          y1={py(TT)}
          x2={px(1)}
          y2={py(P)}
          style={{ stroke: GT.b, strokeWidth: 2.5 }}
        />
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

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.a}>cooperate every round: R = 3</Chip>
        <Chip color={GT.b}>defect once (T = 5), punished forever after (P = 1)</Chip>
      </div>
      <Caption>
        Cheating wins one big round and loses every round after it. When the future is likely, those
        lost rounds outweigh the windfall.
      </Caption>
    </Specimen>
  );
}
