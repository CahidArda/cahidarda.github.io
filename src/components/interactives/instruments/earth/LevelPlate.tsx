/**
 * Earth, Plate 1: the plumb level. Two modes share one piece of physics: a frame whose
 * plumb line hangs from its apex and only crosses the centre mark when the base is level.
 *  - A-frame: tilt the beam; the string crosses the crossbar at x = h·tan θ from centre.
 *  - Cord (havayi terazi): a brass triangle hangs on a cord between two graduated poles.
 *    Slide the far end of the cord until the bob sits on the mark, then read the far pole.
 * The bob swings and settles (damped pendulum) after each change; reduced motion snaps it.
 */
import { useEffect, useRef, useState } from 'react';
import {
  C,
  Controls,
  DimLine,
  HatchDefs,
  InkText,
  Plate,
  Readout,
  Segmented,
  Slider,
  useHoverPreview,
  useMeasuredWidth,
  useReducedMotion,
} from '../shared';

type Mode = 'aframe' | 'cord';
const RAD = Math.PI / 180;

/** Damped swing of the bob (radians from vertical), kicked whenever the frame tilts. */
function useSwing(tilt: number, reduced: boolean) {
  const [phi, setPhi] = useState(0);
  const st = useRef({ phi: 0, w: 0, last: tilt, raf: 0 });
  useEffect(() => {
    const s = st.current;
    const d = tilt - s.last;
    s.last = tilt;
    if (reduced) {
      s.phi = 0;
      s.w = 0;
      setPhi(0);
      return;
    }
    // the apex moves with the frame, the bob lags behind: a kick opposite to the tilt
    s.w += -d * 0.9;
    if (s.raf) return;
    let prev = 0;
    const step = (now: number) => {
      const dt = prev ? Math.min((now - prev) / 1000, 0.05) : 0.016;
      prev = now;
      const k = 26; // g / l for a drawing-sized pendulum
      s.w += (-k * Math.sin(s.phi) - 2.2 * s.w) * dt;
      s.phi += s.w * dt;
      setPhi(s.phi);
      if (Math.abs(s.phi) < 0.0004 && Math.abs(s.w) < 0.002) {
        s.phi = 0;
        s.w = 0;
        setPhi(0);
        s.raf = 0;
        return;
      }
      s.raf = requestAnimationFrame(step);
    };
    s.raf = requestAnimationFrame(step);
  }, [tilt, reduced]);
  useEffect(() => () => cancelAnimationFrame(st.current.raf), []);
  return phi;
}

const fmtSigned = (v: number, d = 1) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`;

/** Intersection of the vertical-ish string (p, dir) with segment a→b, as a parameter on a→b. */
function hitParam(
  p: { x: number; y: number },
  d: { x: number; y: number },
  a: { x: number; y: number },
  b: { x: number; y: number },
) {
  const ex = b.x - a.x;
  const ey = b.y - a.y;
  const den = d.x * ey - d.y * ex;
  if (Math.abs(den) < 1e-9) return 0.5;
  return (d.x * (a.y - p.y) - d.y * (a.x - p.x)) / -den;
}

function Bob({ x, y, s = 1 }: { x: number; y: number; s?: number }) {
  // a brass plumb bob: a cap, a body, a point (the point is the reference)
  const w = 7 * s;
  const h = 18 * s;
  return (
    <g>
      <path
        d={`M ${x - w * 0.35} ${y} L ${x + w * 0.35} ${y} L ${x + w} ${y + h * 0.35} Q ${x + w * 0.9} ${y + h * 0.65} ${x} ${y + h} Q ${x - w * 0.9} ${y + h * 0.65} ${x - w} ${y + h * 0.35} Z`}
        style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }}
      />
      <path
        d={`M ${x - w} ${y + h * 0.35} L ${x + w} ${y + h * 0.35}`}
        style={{ stroke: C.soft, strokeWidth: 0.7 }}
      />
    </g>
  );
}

function AFrame({ W, theta, phi }: { W: number; theta: number; phi: number }) {
  const narrow = W < 520;
  const k = Math.min(1, W / 470);
  const Hs = narrow ? 300 : 330;
  const cx = W / 2;
  const gy = Hs - 70;
  const Hf = 215 * k + (narrow ? 25 : 0);
  const L = 118 * k;
  const h = Hf * 0.55; // apex to crossbar
  const t = theta * RAD;
  const P = (x: number, y: number) => ({
    x: cx + x * Math.cos(t) - y * Math.sin(t),
    y: gy - (x * Math.sin(t) + y * Math.cos(t)),
  });
  const beamHalf = Math.min(W * 0.44, 270);
  const beam = [P(-beamHalf, 0), P(beamHalf, 0), P(beamHalf, -20), P(-beamHalf, -20)];
  const apex = P(0, Hf);
  const fl = P(-L, 0);
  const fr = P(L, 0);
  const cb = h / Hf; // crossbar sits at this fraction down the legs
  const cl = P(-L * cb, Hf - h);
  const cr = P(L * cb, Hf - h);
  const mark = P(0, Hf - h);
  const strLen = Hf - 34 * k;
  const dir = { x: Math.sin(phi), y: Math.cos(phi) };
  const bob = { x: apex.x + dir.x * strLen, y: apex.y + dir.y * strLen };
  const u = hitParam(apex, dir, cl, cr);
  const hit = { x: cl.x + (cr.x - cl.x) * u, y: cl.y + (cr.y - cl.y) * u };
  // crossbar graduations
  const ticks: string[] = [];
  for (let i = -6; i <= 6; i++) {
    if (i === 0) continue;
    const xx = (i * (L * cb)) / 7;
    const a = P(xx, Hf - h);
    const b = P(xx, Hf - h + (i % 2 === 0 ? 7 : 4));
    ticks.push(`M ${a.x} ${a.y} L ${b.x} ${b.y}`);
  }
  const m1 = P(0, Hf - h - 6);
  const m2 = P(0, Hf - h + 11);
  const level = Math.abs(theta) < 0.15;
  const off = Math.tan(t) * h;
  const legW = 4.5 * k + 2;
  const leg = (f: { x: number; y: number }) => {
    const dx = f.x - apex.x;
    const dy = f.y - apex.y;
    const n = Math.hypot(dx, dy);
    const nx = (-dy / n) * legW;
    const ny = (dx / n) * legW;
    return `M ${apex.x + nx} ${apex.y + ny} L ${f.x + nx} ${f.y + ny} L ${f.x - nx} ${f.y - ny} L ${apex.x - nx} ${apex.y - ny} Z`;
  };
  return (
    <svg viewBox={`0 0 ${W} ${Hs}`} width={W} height={Hs} style={{ display: 'block' }}>
      <HatchDefs id="lv-hatch" gap={6} />
      {/* true horizontal, for reference */}
      <line
        x1={cx - beamHalf - 10}
        y1={gy}
        x2={cx + beamHalf + 10}
        y2={gy}
        style={{ stroke: C.faint, strokeWidth: 0.8, strokeDasharray: '4 4' }}
      />
      {/* label sits above the dashed line on the side where the beam dips below it */}
      <InkText
        x={theta >= 0 ? cx - beamHalf - 6 : cx + beamHalf + 6}
        y={gy - 7}
        anchor={theta >= 0 ? 'start' : 'end'}
        size={11}
        role="soft"
      >
        true level
      </InkText>
      {/* the beam (a stone course in section) */}
      <path
        d={`M ${beam.map((p) => `${p.x} ${p.y}`).join(' L ')} Z`}
        style={{ fill: 'url(#lv-hatch)', stroke: C.ink, strokeWidth: 1 }}
      />
      {/* frame */}
      <path d={leg(fl)} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
      <path d={leg(fr)} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
      <path
        d={`M ${cl.x} ${cl.y} L ${cr.x} ${cr.y}`}
        style={{ stroke: C.ink, strokeWidth: 5 * k + 2, strokeLinecap: 'butt' }}
      />
      <path
        d={`M ${cl.x} ${cl.y} L ${cr.x} ${cr.y}`}
        style={{ stroke: C.paper, strokeWidth: 5 * k, strokeLinecap: 'butt' }}
      />
      <path d={ticks.join(' ')} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
      <line x1={m1.x} y1={m1.y} x2={m2.x} y2={m2.y} style={{ stroke: C.red, strokeWidth: 1.6 }} />
      <circle cx={apex.x} cy={apex.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
      {/* string and bob */}
      <line
        x1={apex.x}
        y1={apex.y}
        x2={bob.x}
        y2={bob.y}
        style={{ stroke: C.ink, strokeWidth: 0.9 }}
      />
      <circle cx={hit.x} cy={hit.y} r={2.4} style={{ fill: level ? C.red : C.ink }} />
      <Bob x={bob.x} y={bob.y} s={0.85 + 0.15 * k} />
      {/* offset dimension, along the crossbar */}
      {Math.abs(theta) >= 0.6 && (
        <DimLine
          x1={mark.x}
          y1={mark.y}
          x2={P(off, Hf - h).x}
          y2={P(off, Hf - h).y}
          offset={theta > 0 ? 26 : -26}
        />
      )}
      {Math.abs(theta) >= 0.6 &&
        (() => {
          const q = P(off + (theta > 0 ? 10 : -10), Hf - h + (theta > 0 ? 26 : -26));
          return (
            <InkText x={q.x} y={q.y} anchor={theta > 0 ? 'start' : 'end'} baseline="central" size={14}>
              x
            </InkText>
          );
        })()}
      {/* apex-to-crossbar dimension */}
      {(() => {
        const a = P(-L * 0.75 - 26, Hf);
        const b = P(-L * 0.75 - 26, Hf - h);
        return (
          <g>
            <DimLine x1={b.x} y1={b.y} x2={a.x} y2={a.y} />
            <InkText x={(a.x + b.x) / 2 - 8} y={(a.y + b.y) / 2} anchor="end" baseline="central" size={14}>
              h
            </InkText>
          </g>
        );
      })()}
      <InkText x={apex.x + 10} y={apex.y - 8} size={12}>
        apex
      </InkText>
      <InkText
        x={cx}
        y={Hs - 22}
        anchor="middle"
        size={narrow ? 12 : 13}
        role={level ? 'red' : 'soft'}
      >
        {level
          ? 'level: the string falls on the mark'
          : `tilted ${Math.abs(theta).toFixed(1)}°: the string misses the mark`}
      </InkText>
    </svg>
  );
}

/* ── cord mode: the havayi terazi between two poles ── */

const SPAN = 6; // m between poles
const NEAR = 1.2; // cord tied at this mark on the near pole (m)
const DROP = 0.3; // the ground falls this much between the poles (m)

function Cord({ W, far, phi }: { W: number; far: number; phi: number }) {
  const narrow = W < 520;
  const Hs = narrow ? 320 : 340;
  const S = Math.min(W * 0.4, 250);
  const xL = W / 2 - S;
  const xR = W / 2 + S;
  const ky = (Hs - 110) / 2.1; // px per metre, vertical (exaggerated)
  const gL = Hs - 48;
  const gR = gL + DROP * ky;
  const yAt = (pole: 'L' | 'R', m: number) => (pole === 'L' ? gL : gR) - m * ky;
  const a = { x: xL, y: yAt('L', NEAR) };
  const b = { x: xR, y: yAt('R', far) };
  const tiltDrawn = Math.atan2(a.y - b.y, b.x - a.x); // rad, + = far end higher
  const tiltReal = Math.atan2(far - (NEAR + DROP), SPAN); // the true cord slope
  // triangle hung at the cord midpoint, apex down
  const side = narrow ? 92 : 116;
  const th = (side * Math.sqrt(3)) / 2;
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const ux = Math.cos(tiltDrawn);
  const uy = -Math.sin(tiltDrawn); // along the cord, screen space
  const nx = -uy;
  const ny = ux; // "down" off the cord, perpendicular
  const drop = 10; // hooks
  const top = { x: mid.x + nx * drop, y: mid.y + ny * drop };
  const h1 = { x: top.x - ux * side * 0.5, y: top.y - uy * side * 0.5 };
  const h2 = { x: top.x + ux * side * 0.5, y: top.y + uy * side * 0.5 };
  const tip = { x: top.x + nx * th, y: top.y + ny * th };
  const dir = { x: Math.sin(phi), y: Math.cos(phi) };
  const strLen = th + 6;
  const bob = { x: top.x + dir.x * strLen, y: top.y + dir.y * strLen };
  const err = far - (NEAR + DROP);
  const level = Math.abs(err) < 0.006;
  // pole graduations
  const pole = (x: number, g: number, side: 'L' | 'R') => {
    const p: string[] = [];
    const lab: React.ReactNode[] = [];
    for (let i = 0; i <= 20; i++) {
      const y = g - (i / 10) * ky;
      const w = i % 5 === 0 ? 9 : 5;
      p.push(side === 'L' ? `M ${x} ${y} L ${x - w} ${y}` : `M ${x} ${y} L ${x + w} ${y}`);
      if (i % 5 === 0 && i > 0)
        lab.push(
          <InkText
            key={i}
            x={side === 'L' ? x - 12 : x + 12}
            y={y}
            kind="num"
            size={10}
            anchor={side === 'L' ? 'end' : 'start'}
            baseline="central"
            role="soft"
          >
            {(i / 10).toFixed(1)}
          </InkText>,
        );
    }
    return (
      <g>
        <rect
          x={x - 3}
          y={g - 2.05 * ky}
          width={6}
          height={2.05 * ky}
          style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }}
        />
        <path d={p.join(' ')} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
        {lab}
      </g>
    );
  };
  const ground = `M ${xL - 40} ${gL - 3} L ${xL} ${gL} C ${xL + S * 0.6} ${gL + 4}, ${xR - S * 0.7} ${gR - 6}, ${xR} ${gR} L ${xR + 40} ${gR + 2}`;
  return (
    <svg viewBox={`0 0 ${W} ${Hs}`} width={W} height={Hs} style={{ display: 'block' }}>
      <HatchDefs id="cd-hatch" gap={6} />
      <path
        d={`${ground} L ${xR + 40} ${Hs} L ${xL - 40} ${Hs} Z`}
        style={{ fill: 'url(#cd-hatch)', stroke: 'none' }}
      />
      <path d={ground} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      {pole(xL, gL, 'L')}
      {pole(xR, gR, 'R')}
      {/* level reference through the near tie point */}
      <line
        x1={xL}
        y1={a.y}
        x2={xR + 14}
        y2={a.y}
        style={{ stroke: C.faint, strokeWidth: 0.8, strokeDasharray: '4 4' }}
      />
      {/* the cord */}
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={{ stroke: C.ink, strokeWidth: 1.2 }} />
      <circle cx={a.x} cy={a.y} r={3} style={{ fill: C.ink }} />
      <circle cx={b.x} cy={b.y} r={3.4} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.5 }} />
      {/* hooks + triangle */}
      <path
        d={`M ${h1.x - nx * drop} ${h1.y - ny * drop} L ${h1.x} ${h1.y} M ${h2.x - nx * drop} ${h2.y - ny * drop} L ${h2.x} ${h2.y}`}
        style={{ stroke: C.soft, strokeWidth: 0.9 }}
      />
      <path
        d={`M ${h1.x} ${h1.y} L ${h2.x} ${h2.y} L ${tip.x} ${tip.y} Z`}
        style={{ fill: 'none', stroke: C.brass, strokeWidth: 5, strokeLinejoin: 'round' }}
      />
      <path
        d={`M ${h1.x} ${h1.y} L ${h2.x} ${h2.y} L ${tip.x} ${tip.y} Z`}
        style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.8, strokeLinejoin: 'round' }}
      />
      <line
        x1={tip.x - ux * 9}
        y1={tip.y - uy * 9}
        x2={tip.x + ux * 9}
        y2={tip.y + uy * 9}
        style={{ stroke: C.red, strokeWidth: 1.6 }}
      />
      <line x1={top.x} y1={top.y} x2={bob.x} y2={bob.y} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
      <Bob x={bob.x} y={bob.y - 2} s={0.7} />
      {/* detail circle: the tip and the string at five times size */}
      {(() => {
        const r = narrow ? 34 : 40;
        const dx = W / 2 + S * 0.64;
        const cordY = a.y + ((b.y - a.y) * (dx - xL)) / (xR - xL);
        const dc = { x: dx, y: cordY + r + 18 };
        const off = Math.max(-r * 0.7, Math.min(r * 0.7, Math.sin(tiltDrawn) * th * 2));
        const mx = dc.x + off;
        const bar = r * 0.75;
        const sx = Math.sin(phi) * r;
        return (
          <g>
            <circle cx={tip.x} cy={tip.y} r={9} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.7, strokeDasharray: '2 2' }} />
            <line x1={tip.x + 9} y1={tip.y - 3} x2={dc.x - r} y2={dc.y + 6} style={{ stroke: C.soft, strokeWidth: 0.6, strokeDasharray: '2 2' }} />
            <circle cx={dc.x} cy={dc.y} r={r} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 0.9 }} />
            <clipPath id="cd-detail">
              <circle cx={dc.x} cy={dc.y} r={r - 1} />
            </clipPath>
            <g clipPath="url(#cd-detail)">
            <g transform={`rotate(${-tiltDrawn / RAD} ${mx} ${dc.y + 4})`}>
              <path d={`M ${mx - bar * 0.5} ${dc.y - r} L ${mx} ${dc.y + 4} L ${mx + bar * 0.5} ${dc.y - r}`} style={{ fill: 'none', stroke: C.brass, strokeWidth: 6, strokeLinejoin: 'round' }} />
              <line x1={mx - 10} y1={dc.y + 8} x2={mx + 10} y2={dc.y + 8} style={{ stroke: C.red, strokeWidth: 1.8 }} />
            </g>
            <line x1={dc.x - sx} y1={dc.y - r + 2} x2={dc.x + sx} y2={dc.y + r - 2} style={{ stroke: C.ink, strokeWidth: 1 }} />
            </g>
            <InkText x={dc.x} y={dc.y + r + 14} anchor="middle" size={11} role="soft">
              the tip, enlarged
            </InkText>
          </g>
        );
      })()}
      {/* labels */}
      <InkText x={xL - 6} y={gL - 2.05 * ky - 8} anchor="middle" size={12} role="soft">
        near pole
      </InkText>
      <InkText x={xR + 6} y={gR - 2.05 * ky - 8} anchor="middle" size={12} role="soft">
        far pole
      </InkText>
      <InkText
        x={W / 2}
        y={Hs - 14}
        anchor="middle"
        size={narrow ? 12 : 13}
        role={level ? 'red' : 'soft'}
      >
        {level
          ? `level: the ground falls ${(DROP * 100).toFixed(0)} cm between the poles`
          : `cord ${err > 0 ? 'rises' : 'falls'} ${Math.abs(tiltReal / RAD).toFixed(2)}°: keep sliding`}
      </InkText>
    </svg>
  );
}

export default function LevelPlate() {
  const [sel, setSel] = useState<Mode>('aframe');
  const [mode, bindMode] = useHoverPreview(sel);
  const [theta, setTheta] = useState(4);
  const [far, setFar] = useState(1.7);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const reduced = useReducedMotion();
  const phiA = useSwing(mode === 'aframe' ? theta * RAD : 0, reduced);
  const phiC = useSwing(mode === 'cord' ? Math.atan2(far - NEAR - DROP, SPAN) * 4 : 0, reduced);
  const tanT = Math.tan(theta * RAD);
  const seked = Math.abs(tanT) < 1e-3 ? '∞' : (7 / Math.abs(tanT)).toFixed(0);
  return (
    <Plate
      n={1}
      title={mode === 'aframe' ? 'The A-frame level' : 'The havayi terazi on a cord'}
      kicker={mode === 'aframe' ? 'tilt the beam' : 'slide the far end'}
      caption={
        mode === 'aframe' ? (
          <>
            Gravity keeps the string vertical while the frame tilts with the stone. The line from
            the apex to the centre mark is square to the feet, so the two agree only when the
            feet are level. A marked crossbar turns the miss into a slope reading.
          </>
        ) : (
          <>
            The same triangle, hung on a cord stretched between two poles. Move the far end
            until the bob sits on the mark, then read the far pole: the difference from the
            near pole is how much the ground falls. Vertical scale exaggerated; the triangle is
            drawn large.
          </>
        )
      }
    >
      <Controls>
        <Segmented
          label="Instrument"
          options={[
            { value: 'aframe', label: 'A-frame' },
            { value: 'cord', label: 'Terazi on a cord' },
          ]}
          value={sel}
          onChange={setSel}
          preview={bindMode}
        />
      </Controls>
      <div ref={ref} style={{ marginTop: '0.6rem' }}>
        {mode === 'aframe' ? (
          <AFrame W={W} theta={theta} phi={phiA} />
        ) : (
          <Cord W={W} far={far} phi={phiC} />
        )}
      </div>
      {mode === 'aframe' ? (
        <Controls>
          <Slider
            label="tilt θ"
            value={theta}
            min={-10}
            max={10}
            step={0.1}
            onChange={setTheta}
            format={(v) => `${fmtSigned(v)}°`}
          />
          <Readout label="slope">{`${Math.abs(tanT * 1000).toFixed(0)} mm per m`}</Readout>
          <Readout label="offset, h = 50 cm">{`${(Math.abs(tanT) * 500).toFixed(0)} mm`}</Readout>
          <Readout label="seked">{`${seked} palms per cubit`}</Readout>
        </Controls>
      ) : (
        <Controls>
          <Slider
            label="far end"
            value={far}
            min={1.3}
            max={1.9}
            step={0.01}
            onChange={setFar}
            format={(v) => `${v.toFixed(2)} m`}
          />
          <Readout label="near pole">{`${NEAR.toFixed(2)} m`}</Readout>
          <Readout label="difference">{`${((far - NEAR) * 100).toFixed(0)} cm`}</Readout>
        </Controls>
      )}
    </Plate>
  );
}
