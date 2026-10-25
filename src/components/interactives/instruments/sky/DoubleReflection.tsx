/**
 * Plate 3: why the arc is read double. Two plane mirrors; turn the upper one by θ against the
 * lower one and the light leaving the pair is bent by exactly 2θ from the light arriving,
 * whatever angle it arrived at. Tilt the whole pair (a ship rolling) and the ray hits the
 * mirrors at new places, yet leaves in the same direction: the sun stays on the horizon.
 */
import { useRef, useState } from 'react';
import { C, Controls, InkText, Plate, Readout, Slider, arcPath, useMeasuredWidth } from '../shared';
import { reflect, rot, type V2 } from './sextantGeom';

const RAD = Math.PI / 180;
const NB0: V2 = { x: Math.SQRT1_2, y: Math.SQRT1_2 }; // lower mirror's normal at no tilt
const MA = 0.26; // half length of the upper (index) mirror
const MB = 0.3; // half length of the lower (horizon) mirror

function solve(theta: number, tilt: number) {
  const nB = rot(NB0, tilt);
  const nA = rot(nB, -theta);
  const A = rot({ x: 0, y: 1 }, tilt);
  const d1 = rot({ x: 1, y: 0 }, -2 * theta); // sunlight, fixed in the world
  const d2 = reflect(d1, nA);
  // meet the lower mirror's line: P = A + a d2, (P · nB) = 0
  const den = d2.x * nB.x + d2.y * nB.y;
  const a = -(A.x * nB.x + A.y * nB.y) / den;
  const P = { x: A.x + a * d2.x, y: A.y + a * d2.y };
  const d3 = reflect(d2, nB);
  const tB = { x: -nB.y, y: nB.x };
  const along = P.x * tB.x + P.y * tB.y;
  // the sunlight must reach the silvered face of the upper mirror, not its back
  const front = d1.x * nA.x + d1.y * nA.y > 0;
  return { nA, nB, A, d1, d2, P, d3, hits: front && Math.abs(along) <= MB + 1e-6 && a > 0 };
}

const deg = (v: V2) => Math.atan2(v.y, v.x) / RAD;

export default function DoubleReflection() {
  const [theta, setTheta] = useState(15);
  const [tilt, setTilt] = useState(0);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const narrow = W < 520;
  const H = narrow ? 330 : 360;
  const S = narrow ? Math.min(H * 0.42, W * 0.36) : H * 0.44;
  const ox = W * (narrow ? 0.42 : 0.42);
  const oy = H * 0.8;
  const P2 = (v: V2) => ({ x: ox + v.x * S, y: oy - v.y * S });
  const sv = solve(theta, tilt);
  const { A, P, d1, d3, nA, nB } = sv;
  const mirror = (c: V2, n0: V2, half: number, backSign: number) => {
    const n = { x: n0.x * backSign, y: n0.y * backSign };
    const t = { x: -n.y, y: n.x };
    const p = P2({ x: c.x + t.x * half, y: c.y + t.y * half });
    const q = P2({ x: c.x - t.x * half, y: c.y - t.y * half });
    // a hatched back on the side away from the light
    const bp = P2({ x: c.x + t.x * half - n.x * 0.035, y: c.y + t.y * half - n.y * 0.035 });
    const bq = P2({ x: c.x - t.x * half - n.x * 0.035, y: c.y - t.y * half - n.y * 0.035 });
    return { p, q, back: `M ${p.x} ${p.y} L ${bp.x} ${bp.y} L ${bq.x} ${bq.y} L ${q.x} ${q.y} Z` };
  };
  const mA = mirror(A, nA, MA, -1);
  const mB = mirror({ x: 0, y: 0 }, nB, MB, 1);
  const a = P2(A);
  const p = P2(P);
  const src = P2({ x: A.x - d1.x * 0.78, y: A.y - d1.y * 0.78 });
  const outLen = (W - 18 - p.x) / S;
  const out = P2({ x: P.x + d3.x * outLen, y: P.y + d3.y * outLen });
  // the 2θ construction near the exit: a dashed copy of the incoming sunlight
  const k = P2({ x: P.x + d3.x * outLen * 0.62, y: P.y + d3.y * outLen * 0.62 });
  const ghost = { x: k.x + d1.x * S * 0.6, y: k.y - d1.y * S * 0.6 };
  const r2 = S * 0.42;
  const dev = deg(d3) - deg(d1);
  // θ shown at the upper mirror: a dashed line parallel to the lower mirror
  const tBn = { x: -nB.y, y: nB.x };
  const par1 = P2({ x: A.x + tBn.x * 0.36, y: A.y + tBn.y * 0.36 });
  const par2 = P2({ x: A.x - tBn.x * 0.36, y: A.y - tBn.y * 0.36 });
  const angB = deg(tBn);
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };

  // drag the upper mirror to turn it
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragging = useRef(false);
  const onMove = (e: React.PointerEvent) => {
    if (!dragging.current || !svgRef.current) return;
    const r = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const y = ((e.clientY - r.top) / r.height) * H;
    const ang = Math.atan2(-(y - a.y), x - a.x) / RAD; // direction from the mirror centre
    // the mirror's own direction is tA = rot(tB, -θ); pick the nearer end
    let th = angB - ang;
    while (th > 90) th -= 180;
    while (th < -90) th += 180;
    setTheta(Math.max(0, Math.min(40, Math.round(th * 2) / 2)));
  };

  return (
    <Plate
      n={3}
      title="Turn the mirror by θ, the light turns by 2θ"
      kicker="drag the upper mirror"
      caption={
        <>
          The upper mirror is the index mirror on the arm; the lower one is the horizon mirror.
          Whatever you do with the tilt, the light that leaves is bent by twice the angle between the
          mirrors. So the arc only needs to be a sixth of a circle to read 120°, and a rolling deck
          moves the ray around on the mirrors without moving the sun off the horizon.
        </>
      }
    >
      <div ref={ref}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          style={{ display: 'block', touchAction: 'pan-y' }}
          role="img"
          aria-label={`Two mirrors at ${theta} degrees; the light is deviated by ${(2 * theta).toFixed(0)} degrees`}
          onPointerMove={onMove}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          {/* sunlight, a few parallel rays with the one that is traced in red */}
          {[-0.09, 0.09].map((o) => {
            const n = { x: -d1.y * o, y: d1.x * o };
            const s1 = P2({ x: A.x - d1.x * 0.78 + n.x, y: A.y - d1.y * 0.78 + n.y });
            const s2 = P2({ x: A.x - d1.x * 0.3 + n.x, y: A.y - d1.y * 0.3 + n.y });
            return <line key={o} x1={s1.x} y1={s1.y} x2={s2.x} y2={s2.y} style={{ stroke: C.red, strokeWidth: 0.7, opacity: 0.45 }} />;
          })}
          <path d={`M ${src.x} ${src.y} L ${a.x} ${a.y} L ${p.x} ${p.y} L ${out.x} ${out.y}`} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.5 }} />
          <circle cx={src.x} cy={src.y} r={9} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.2 }} />
          <InkText x={src.x + 6} y={src.y - 14} size={12} role="soft">sun</InkText>
          {/* horizon: a dashed line straight ahead, for reference */}
          <line x1={8} y1={p.y} x2={W - 8} y2={p.y} style={{ stroke: C.sky, strokeWidth: 0.8, strokeDasharray: '5 4', opacity: tilt === 0 ? 0 : 0.8 }} />
          {/* mirrors */}
          <path d={mB.back} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 0.8 }} />
          <line x1={mB.p.x} y1={mB.p.y} x2={mB.q.x} y2={mB.q.y} style={{ ...ink, strokeWidth: 2.4 }} />
          <path d={mA.back} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 0.8 }} />
          <line x1={par1.x} y1={par1.y} x2={par2.x} y2={par2.y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '4 3' }} />
          <line
            x1={mA.p.x}
            y1={mA.p.y}
            x2={mA.q.x}
            y2={mA.q.y}
            style={{ ...ink, strokeWidth: 2.4, cursor: 'grab' }}
          />
          {/* a fat invisible handle so it is easy to grab */}
          <line
            x1={mA.p.x}
            y1={mA.p.y}
            x2={mA.q.x}
            y2={mA.q.y}
            style={{ stroke: 'transparent', strokeWidth: 26, cursor: 'grab', touchAction: 'none' }}
            onPointerDown={(e) => {
              dragging.current = true;
              (e.target as Element).setPointerCapture(e.pointerId);
            }}
          />
          <circle cx={a.x} cy={a.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          {theta > 0.4 && (
            <>
              <path d={arcPath(a.x, a.y, S * 0.42, angB - theta, angB)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }} />
              {(() => {
                const q = { x: a.x + Math.cos((angB - theta / 2) * RAD) * (S * 0.42 + 14), y: a.y - Math.sin((angB - theta / 2) * RAD) * (S * 0.42 + 14) };
                return <InkText x={q.x} y={q.y} anchor="middle" baseline="central" size={14}>θ</InkText>;
              })()}
            </>
          )}
          {/* 2θ at the exit */}
          <line x1={k.x} y1={k.y} x2={ghost.x} y2={ghost.y} style={{ stroke: C.red, strokeWidth: 0.9, strokeDasharray: '4 3' }} />
          {theta > 0.4 && (
            <>
              <path d={arcPath(k.x, k.y, r2, deg(d1), deg(d3))} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.1 }} />
              {(() => {
                const m = (deg(d1) + deg(d3)) / 2;
                const q = { x: k.x + Math.cos(m * RAD) * (r2 + 16), y: k.y - Math.sin(m * RAD) * (r2 + 16) };
                return <InkText x={q.x} y={q.y} anchor="middle" baseline="central" size={14} role="red">2θ</InkText>;
              })()}
            </>
          )}
          {/* the eye */}
          <path d={`M ${out.x - 14} ${out.y} q 7 -7 14 0 q -7 7 -14 0 Z`} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          <circle cx={out.x - 7} cy={out.y} r={2.2} style={{ fill: C.ink }} />
          <InkText x={mA.q.x + 10} y={mA.q.y + 4} size={12} role="soft">index mirror</InkText>
          <InkText x={mB.q.x - 6} y={mB.q.y + 18} anchor="middle" size={12} role="soft">horizon mirror</InkText>
          {!sv.hits && (
            <InkText x={W / 2} y={H - 8} anchor="middle" size={12} role="red">
              a real mirror would miss this ray, but the angle still holds
            </InkText>
          )}
        </svg>
      </div>
      <Controls>
        <Slider label="mirror angle θ" value={theta} min={0} max={40} step={0.5} onChange={setTheta} format={(v) => `${v.toFixed(1)}°`} />
        <Slider label="tilt (roll)" value={tilt} min={-12} max={12} step={0.5} onChange={setTilt} format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}°`} width="7rem" />
        <Readout label="light turned by">
          <span style={{ color: C.red }}>{dev.toFixed(1)}°</span> = 2 × {theta.toFixed(1)}°
        </Readout>
      </Controls>
    </Plate>
  );
}
