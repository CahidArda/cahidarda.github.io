/**
 * Plate: why the arc is read double. The world is fixed: the sun, its light, and the horizon
 * never move. The reader turns the index mirror by θ against the horizon mirror; the light
 * leaving the horizon mirror swings by twice that, and when it runs along the telescope the
 * sun's image sits on the horizon. Rolling the whole instrument turns both mirrors together:
 * the outgoing light keeps the same angle D = 2θ to the sunlight, so the image stays on the
 * horizon. Geometry lives in ./doubleGeom.ts (swept by the verification script).
 */
import { useRef, useState } from 'react';
import { C, Controls, Eq, InkText, Plate, Slider, Sym, V, Working, arcPath, useMeasuredWidth, type Role } from '../shared';
import { ROLL_MAX, ROLL_MIN, SUN_ALT, THETA_MAX, THETA_MIN, alignedTheta, drScene, type Label } from './doubleGeom';

const RAD = Math.PI / 180;
const f1 = (v: number) => `${v.toFixed(1)}°`;
const clampT = (v: number) => Math.max(THETA_MIN, Math.min(THETA_MAX, Math.round(v * 10) / 10));

export default function DoubleReflection() {
  const [theta, setTheta] = useState(14);
  const [roll, setRoll] = useState(0);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const s = drScene(theta, roll, W);
  const { g, H, sun, sunR, idx, hor, pivot, P, start, out, eye, telFrom } = s;
  const lab = Object.fromEntries(s.labels.map((l) => [l.id, l])) as Record<string, Label>;
  const T = (l: Label, role: Role = 'soft') => (
    <InkText x={l.x} y={l.y} anchor={l.anchor} baseline={l.baseline ?? 'alphabetic'} size={l.size} role={role}>
      {l.text}
    </InkText>
  );
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const aligned = g.aligned;
  const img = g.imageVsHorizon;

  // drag the index mirror to turn it
  const svgRef = useRef<SVGSVGElement | null>(null);
  const grab = useRef<{ a0: number; t0: number } | null>(null);
  const pointerAng = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const y = ((e.clientY - r.top) / r.height) * H;
    return Math.atan2(-(y - pivot.y), x - pivot.x) / RAD;
  };
  const onMove = (e: React.PointerEvent) => {
    if (!grab.current || !svgRef.current) return;
    let d = pointerAng(e) - grab.current.a0;
    while (d > 90) d -= 180;
    while (d < -90) d += 180;
    setTheta(clampT(grab.current.t0 - d)); // clockwise turn = bigger θ
  };
  const stop = () => (grab.current = null);

  return (
    <Plate
      n={3}
      title="Two mirrors double the angle"
      kicker="drag the index mirror"
      caption={
        <>
          The sun and the horizon stay where they are; only the instrument moves. Turn the index mirror
          by θ and the light leaving the horizon mirror swings by twice as much. When it runs along the
          telescope, the sun&apos;s image sits on the horizon and the arc reads 2θ, the sun&apos;s
          height h. Roll the whole instrument and both mirrors turn together: the light still leaves at
          2θ to the sunlight, so the image stays on the horizon.
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
          aria-label={`Index mirror turned ${theta.toFixed(1)} degrees; the light leaves turned by ${g.D.toFixed(1)} degrees; the sun's image is ${Math.abs(img).toFixed(1)} degrees ${img >= 0 ? 'above' : 'below'} the horizon`}
          onPointerMove={onMove}
          onPointerUp={stop}
          onPointerCancel={stop}
        >
          {/* sunlight: a fixed beam of parallel rays */}
          {s.beam.map(([a, b], i) => (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={{ stroke: C.red, strokeWidth: 0.7, opacity: 0.4 }} />
          ))}
          <circle cx={sun.x} cy={sun.y} r={sunR} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.3 }} />
          {T(lab.sun)}
          {/* the horizon: light from it is level; a dashed reference through the horizon mirror */}
          <line x1={s.horizonLine[0].x} y1={s.horizonLine[0].y} x2={s.horizonLine[1].x} y2={s.horizonLine[1].y}
            style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '5 4' }} />
          {/* telescope: rolls with the instrument */}
          <line x1={telFrom.x} y1={telFrom.y} x2={eye.x} y2={eye.y} style={{ stroke: C.ink, strokeWidth: 5, opacity: 0.18 }} />
          {/* the traced ray: sun → index mirror → horizon mirror → out */}
          <path d={`M ${start.x} ${start.y} L ${s.Q.x} ${s.Q.y} L ${P.x} ${P.y} L ${out.x} ${out.y}`}
            style={{ fill: 'none', stroke: C.red, strokeWidth: 1.6, strokeLinejoin: 'round' }} />
          {/* h: the sun's height above the horizon (fixed) */}
          <path d={arcPath(s.hArc.cx, s.hArc.cy, s.hArc.r, s.hArc.from, s.hArc.to)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.3 }} />
          {T(lab.h, 'sky')}
          {/* D: angle between the arriving sunlight (dashed copy) and the light leaving */}
          <line x1={s.ghost[0].x} y1={s.ghost[0].y} x2={s.ghost[1].x} y2={s.ghost[1].y} style={{ stroke: C.red, strokeWidth: 0.9, strokeDasharray: '3 3' }} />
          <path d={arcPath(s.dArc.cx, s.dArc.cy, s.dArc.r, s.dArc.from, s.dArc.to)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.4 }} />
          {T(lab.D, 'red')}
          {/* mirrors */}
          <line x1={hor[0].x} y1={hor[0].y} x2={hor[1].x} y2={hor[1].y} style={{ ...ink, strokeWidth: 3 }} />
          <line x1={s.parallel[0].x} y1={s.parallel[0].y} x2={s.parallel[1].x} y2={s.parallel[1].y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '4 3' }} />
          <line x1={idx[0].x} y1={idx[0].y} x2={idx[1].x} y2={idx[1].y} style={{ ...ink, stroke: C.brass, strokeWidth: 3 }} />
          <line x1={s.idxExt[0].x} y1={s.idxExt[0].y} x2={s.idxExt[1].x} y2={s.idxExt[1].y} style={{ stroke: C.brass, strokeWidth: 0.8, strokeDasharray: '4 3' }} />
          <path d={arcPath(s.thetaArc.cx, s.thetaArc.cy, s.thetaArc.r, s.thetaArc.from, s.thetaArc.to)} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.4 }} />
          {T(lab.theta, 'brass')}
          <line
            x1={idx[0].x} y1={idx[0].y} x2={idx[1].x} y2={idx[1].y}
            style={{ stroke: 'transparent', strokeWidth: 28, cursor: 'grab', touchAction: 'none' }}
            onPointerDown={(e) => {
              grab.current = { a0: pointerAng(e), t0: theta };
              (e.target as Element).setPointerCapture(e.pointerId);
            }}
          />
          <circle cx={pivot.x} cy={pivot.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          {T(lab.idx)}
          {T(lab.hor)}
          {/* the eye at the telescope */}
          <path d={`M ${eye.x - 8} ${eye.y} q 8 -8 16 0 q -8 8 -16 0 Z`} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          <circle cx={eye.x} cy={eye.y} r={2.2} style={{ fill: C.ink }} />
          {T(lab.eye)}
        </svg>
      </div>
      <Controls>
        <Slider label="index mirror turned θ" value={theta} min={THETA_MIN} max={THETA_MAX} step={0.1} onChange={setTheta} format={f1} />
        <Slider label="roll the instrument" value={roll} min={ROLL_MIN} max={ROLL_MAX} step={0.5} onChange={setRoll}
          format={(v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}°`} width="7rem" />
        <button type="button" className="in-seg" onClick={() => setTheta(alignedTheta)}
          style={{ padding: '0.35rem 0.7rem', fontFamily: 'var(--font-mono)', fontSize: '0.66rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: C.ink, background: 'transparent' }}>
          bring the sun down
        </button>
      </Controls>
      <Working label="Angle between the rays is twice the angle between the mirrors">
        <Eq>
          <Sym role="red">D</Sym> = 2 × <Sym role="brass">θ</Sym>
        </Eq>
        <Eq>
          <V role="red">{f1(g.D)}</V> = 2 × <V role="brass">{f1(theta)}</V>
        </Eq>
        <Eq>
          {aligned ? (
            <>
              on the horizon: <Sym role="sky">h</Sym> = <Sym role="red">D</Sym> = <V role="sky">{f1(SUN_ALT)}</V>
            </>
          ) : (
            <>
              sun&apos;s image <V role="red">{f1(Math.abs(img))}</V> {img > 0 ? 'above' : 'below'} the horizon
              (<Sym role="red">D</Sym> {img > 0 ? '<' : '>'} <Sym role="sky">h</Sym> = <V role="sky">{f1(SUN_ALT)}</V>)
            </>
          )}
        </Eq>
      </Working>
    </Plate>
  );
}
