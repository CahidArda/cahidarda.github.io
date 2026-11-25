import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Slider,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { Reserve } from './Reserve';

/*
 * Matching Pennies. Blue (row) wins +1 when the coins match, red (column) wins +1 when
 * they differ. q = probability red shows Heads; p = probability blue shows Heads.
 *   E_blue[Heads] = q - (1 - q) = 2q - 1      E_blue[Tails] = 1 - 2q
 * Blue's best reply: Heads if q > 1/2, Tails if q < 1/2, anything at q = 1/2.
 * Red's best reply (wants a mismatch): Tails if p > 1/2, Heads if p < 1/2.
 */
const VW = 260;
const VH = 236;

// panel A: payoff vs q
const ax = (q: number) => 34 + 180 * q;
const ay = (v: number) => 112 - 84 * v;
// panel B: the unit square, x = q, y = p
const L = 44;
const S = 180;
const bx = (q: number) => L + S * q;
const by = (p: number) => 22 + S * (1 - p);

const blueBR = (q: number) => (q > 0.5 ? 1 : q < 0.5 ? 0 : 0.5);
const pct = (v: number) => `${Math.round(v * 100)}%`;
const sgn = (v: number) =>
  v < -0.005 ? `−${Math.abs(v).toFixed(2)}` : `+${Math.abs(v).toFixed(2)}`;

// Auto-play: sweep red's mix across, then settle on the equilibrium 1/2.
const SWEEP: [number, number][] = [
  [0, 0.3],
  [1.2, 0.1],
  [3.4, 0.9],
  [5.0, 0.5],
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

export default function MixedWay() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [q, setQ] = useState(0.3);
  const [auto, setAuto] = useState(true);
  const tRef = useRef(0);
  const aRef = useRef<SVGSVGElement>(null);
  const bRef = useRef<SVGSVGElement>(null);
  const drag = useRef<null | 'a' | 'b'>(null);

  useEffect(() => {
    if (reduced) {
      setAuto(false);
      setQ(0.5);
    }
  }, [reduced]);

  useFrame(
    (dt) => {
      tRef.current += dt;
      setQ(sweepAt(tRef.current));
      if (tRef.current > SWEEP[SWEEP.length - 1][0]) setAuto(false);
    },
    auto && inView && !reduced,
  );

  const fromPointer = (which: 'a' | 'b', e: React.PointerEvent) => {
    const svg = which === 'a' ? aRef.current : bRef.current;
    if (!svg) return;
    const pt = svgPoint(svg, e.clientX, e.clientY);
    const v = which === 'a' ? (pt.x - 34) / 180 : (pt.x - L) / S;
    let nq = clamp(v, 0, 1);
    if (Math.abs(nq - 0.5) < 0.025) nq = 0.5; // snap to the equilibrium
    setQ(nq);
  };
  const bind = (which: 'a' | 'b') => ({
    onPointerDown: (e: React.PointerEvent) => {
      setAuto(false);
      drag.current = which;
      (e.target as Element).setPointerCapture?.(e.pointerId);
      fromPointer(which, e);
    },
    onPointerMove: (e: React.PointerEvent) => {
      if (drag.current === which) fromPointer(which, e);
    },
    onPointerUp: () => (drag.current = null),
    onPointerCancel: () => (drag.current = null),
  });

  const eH = 2 * q - 1;
  const eT = 1 - 2 * q;
  const best = Math.abs(q - 0.5) < 0.005 ? 'either' : q > 0.5 ? 'Heads' : 'Tails';
  const p = blueBR(q);
  const atEq = best === 'either';

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-01.2"
      title="the mixed way out"
      kicker="matching pennies"
      hint="drag either chart to change how often red shows Heads"
      footer="equilibrium: both play Heads 50%"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {/* ── A: blue's expected payoff from each move ── */}
        <figure className="m-0">
          <div className="label mb-1">blue's payoff from each move</div>
          <svg
            ref={aRef}
            viewBox={`0 0 ${VW} ${VH}`}
            className="block w-full touch-none select-none"
            style={{ cursor: 'ew-resize' }}
            role="img"
            aria-label="Expected payoff of Heads and Tails as red's mix changes"
            {...bind('a')}
          >
            {/* axes */}
            <line x1={ax(0)} y1={ay(0)} x2={ax(1)} y2={ay(0)} style={{ stroke: GT.line }} />
            <line x1={ax(0)} y1={ay(1)} x2={ax(0)} y2={ay(-1)} style={{ stroke: GT.lineStrong }} />
            {[1, 0, -1].map((v) => (
              <T key={v} x={ax(0) - 6} y={ay(v)} anchor="end" size={9}>
                {v === -1 ? '−1' : v === 1 ? '+1' : '0'}
              </T>
            ))}
            {[0, 0.5, 1].map((v) => (
              <T key={v} x={ax(v)} y={ay(-1) + 14} size={9}>
                {v === 0.5 ? '½' : v}
              </T>
            ))}
            <T x={ax(0.5)} y={ay(-1) + 28} size={9}>
              red's P(Heads)
            </T>
            {/* best-reply envelope */}
            <polyline
              points={`${ax(0)},${ay(1)} ${ax(0.5)},${ay(0)} ${ax(1)},${ay(1)}`}
              style={{ fill: 'none', stroke: GT.a, strokeWidth: 7, opacity: 0.18 }}
            />
            {/* the two lines */}
            <line
              x1={ax(0)}
              y1={ay(-1)}
              x2={ax(1)}
              y2={ay(1)}
              style={{ stroke: GT.a, strokeWidth: 2 }}
            />
            <line
              x1={ax(0)}
              y1={ay(1)}
              x2={ax(1)}
              y2={ay(-1)}
              style={{ stroke: GT.a, strokeWidth: 2, strokeDasharray: '5 4' }}
            />
            <T x={ax(1) + 6} y={ay(1)} anchor="start" size={10} color={GT.a}>
              Heads
            </T>
            <T x={ax(1) + 6} y={ay(-1)} anchor="start" size={10} color={GT.a}>
              Tails
            </T>
            {/* crossing */}
            <circle
              cx={ax(0.5)}
              cy={ay(0)}
              r={5}
              style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
            />
            {/* red's current mix */}
            <line
              x1={ax(q)}
              y1={ay(1) - 4}
              x2={ax(q)}
              y2={ay(-1)}
              style={{ stroke: GT.b, strokeWidth: 1.5 }}
            />
            <circle cx={ax(q)} cy={ay(eH)} r={4.5} style={{ fill: GT.a }} />
            <circle
              cx={ax(q)}
              cy={ay(eT)}
              r={4.5}
              style={{ fill: GT.paperRaised, stroke: GT.a, strokeWidth: 2 }}
            />
            <rect x={ax(q) - 9} y={ay(1) - 14} width={18} height={10} style={{ fill: GT.b }} />
          </svg>
        </figure>

        {/* ── B: both best-response curves ── */}
        <figure className="m-0">
          <div className="label mb-1">both best replies</div>
          <svg
            ref={bRef}
            viewBox={`0 0 ${VW} ${VH}`}
            className="block w-full touch-none select-none"
            style={{ cursor: 'ew-resize' }}
            role="img"
            aria-label="Best-response curves of both players cross only at one half, one half"
            {...bind('b')}
          >
            <rect
              x={L}
              y={by(1)}
              width={S}
              height={S}
              style={{ fill: 'none', stroke: GT.lineStrong }}
            />
            <line
              x1={bx(0.5)}
              y1={by(1)}
              x2={bx(0.5)}
              y2={by(0)}
              style={{ stroke: GT.line, strokeDasharray: '2 3' }}
            />
            <line
              x1={bx(0)}
              y1={by(0.5)}
              x2={bx(1)}
              y2={by(0.5)}
              style={{ stroke: GT.line, strokeDasharray: '2 3' }}
            />
            {/* red's best reply to p: q = 1 below 1/2, q = 0 above */}
            <polyline
              points={`${bx(1)},${by(0)} ${bx(1)},${by(0.5)} ${bx(0)},${by(0.5)} ${bx(0)},${by(1)}`}
              style={{ fill: 'none', stroke: GT.b, strokeWidth: 3, strokeLinejoin: 'miter' }}
            />
            {/* blue's best reply to q: p = 0 left of 1/2, p = 1 right */}
            <polyline
              points={`${bx(0)},${by(0)} ${bx(0.5)},${by(0)} ${bx(0.5)},${by(1)} ${bx(1)},${by(1)}`}
              style={{ fill: 'none', stroke: GT.a, strokeWidth: 3, strokeLinejoin: 'miter' }}
            />
            <circle
              cx={bx(0.5)}
              cy={by(0.5)}
              r={atEq ? 11 : 8}
              style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.5, transition: 'r 200ms' }}
            />
            <T
              x={bx(0.5) + 14}
              y={by(0.5) - 12}
              anchor="start"
              size={10}
              color={GT.eq}
              weight={600}
            >
              Nash (½, ½)
            </T>
            {/* blue's reply to the current q */}
            <circle
              cx={bx(q)}
              cy={by(p)}
              r={6}
              style={{ fill: GT.a, stroke: GT.paperRaised, strokeWidth: 2 }}
            />
            <rect x={bx(q) - 1} y={by(0) + 3} width={2} height={7} style={{ fill: GT.b }} />
            {/* axis labels */}
            {[0, 1].map((v) => (
              <T key={'x' + v} x={bx(v)} y={by(0) + 16} size={9}>
                {v}
              </T>
            ))}
            <T x={bx(0.5)} y={by(0) + 16} size={9}>
              ½
            </T>
            <T x={bx(0.5)} y={by(0) + 30} size={9} color={GT.b}>
              red's P(Heads)
            </T>
            {[0, 1].map((v) => (
              <T key={'y' + v} x={L - 8} y={by(v)} anchor="end" size={9}>
                {v}
              </T>
            ))}
            <g transform={`translate(${L - 24} ${by(0.5)}) rotate(-90)`}>
              <T x={0} y={0} size={9} color={GT.a}>
                blue's P(Heads)
              </T>
            </g>
          </svg>
        </figure>
      </div>

      <div className="mt-4 grid items-end gap-x-8 gap-y-4 sm:grid-cols-[12rem_minmax(0,1fr)]">
        <div>
          <BigStat label="blue's best reply" value={best} color={atEq ? GT.eq : GT.a} />
          <Reserve
            className="mt-1 font-mono text-[0.68rem] text-muted"
            items={[
              <span key="e">indifferent: any mix</span>,
              <span key="g">{`earns ${sgn(Math.max(eH, eT))} a round`}</span>,
            ]}
            active={atEq ? 0 : 1}
          />
        </div>
        <div>
          <Slider
            label="red's P(Heads)"
            value={q}
            min={0}
            max={1}
            step={0.01}
            color={GT.b}
            onChange={(v) => {
              setAuto(false);
              setQ(Math.abs(v - 0.5) < 0.015 ? 0.5 : v);
            }}
            format={pct}
          />
        </div>
      </div>
      <Caption>
        Any lean by red gives blue a sure edge. Only at 50% is blue indifferent, so both mixing
        50:50 is the one equilibrium.
      </Caption>
    </Specimen>
  );
}
