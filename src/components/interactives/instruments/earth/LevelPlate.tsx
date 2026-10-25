/**
 * Earth, Plate 1: the plumb level. Two modes share one piece of physics: a frame whose
 * plumb line hangs from its apex and only crosses the centre mark when the base is level.
 *  - A-frame: tilt the beam; the string crosses the crossbar at x = h·tan θ from centre.
 *  - Cord (havayi terazi): a brass triangle hangs on a cord between two graduated poles.
 *    Slide the far end of the cord until the bob sits on the mark, then read the far pole.
 * The bob swings and settles (damped pendulum) after each change; reduced motion snaps it.
 */
import { useEffect, useRef, useState } from 'react';
import { AFRAME_H_CM, CORD, RAD, aframeLayout, cordLayout, hitParam } from './geometry';
import {
  C,
  Controls,
  DimLine,
  HatchDefs,
  InkText,
  Plate,
  Segmented,
  Slider,
  Sym,
  V,
  Working,
  Eq,
  arcPath,
  useHoverPreview,
  useMeasuredWidth,
  useReducedMotion,
} from '../shared';

type Mode = 'aframe' | 'cord';

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
  const G = aframeLayout(W, theta);
  const { P, apex, cl, cr, mark, Hf, h, L, s: k, cx, gy, beamHalf, narrow } = G;
  const strLen = Hf - 34 * k;
  const dir = { x: Math.sin(phi), y: Math.cos(phi) };
  const bob = { x: apex.x + dir.x * strLen, y: apex.y + dir.y * strLen };
  // where the (possibly still swinging) string crosses the crossbar; at rest this is G.hit
  const u = hitParam(apex, dir, cl, cr);
  const hit = { x: cl.x + (cr.x - cl.x) * u, y: cl.y + (cr.y - cl.y) * u };
  const ticks: string[] = [];
  for (let i = -6; i <= 6; i++) {
    if (i === 0) continue;
    const xx = (i * (L * (h / Hf))) / 7;
    const a = P(xx, Hf - h);
    const b = P(xx, Hf - h + (i % 2 === 0 ? 7 : 4));
    ticks.push(`M ${a.x} ${a.y} L ${b.x} ${b.y}`);
  }
  const m1 = P(0, Hf - h - 6);
  const m2 = P(0, Hf - h + 11);
  const legW = 4.5 * k + 2;
  const leg = (f: { x: number; y: number }) => {
    const dx = f.x - apex.x;
    const dy = f.y - apex.y;
    const n = Math.hypot(dx, dy);
    const nx = (-dy / n) * legW;
    const ny = (dx / n) * legW;
    return `M ${apex.x + nx} ${apex.y + ny} L ${f.x + nx} ${f.y + ny} L ${f.x - nx} ${f.y - ny} L ${apex.x - nx} ${apex.y - ny} Z`;
  };
  const showX = Math.abs(theta) >= 0.6;
  return (
    <svg viewBox={`0 0 ${W} ${G.Hs}`} width="100%" style={{ display: 'block', height: 'auto' }}>
      <HatchDefs id="lv-hatch" gap={6} />
      {/* true horizontal, for reference */}
      <line
        x1={cx - beamHalf - 10}
        y1={gy}
        x2={cx + beamHalf + 10}
        y2={gy}
        style={{ stroke: C.faint, strokeWidth: 0.8, strokeDasharray: '4 4' }}
      />
      <InkText x={G.levelLabel.x} y={G.levelLabel.y} anchor={G.levelLabel.anchor} size={narrow ? 11 : 12} role="soft">
        {G.levelText}
      </InkText>
      <path
        d={`M ${G.beam.map((p) => `${p.x} ${p.y}`).join(' L ')} Z`}
        style={{ fill: 'url(#lv-hatch)', stroke: C.ink, strokeWidth: 1 }}
      />
      {/* θ between true level and the beam */}
      {showX && (
        <g>
          <path d={arcPath(cx, gy, G.thetaR, G.thetaFrom, G.thetaTo)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.3 }} />
          <InkText x={G.thetaLabel.x} y={G.thetaLabel.y} anchor={G.thetaLabel.anchor} size={14} role="sky">
            θ
          </InkText>
        </g>
      )}
      <path d={leg(P(-L, 0))} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
      <path d={leg(P(L, 0))} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
      <path d={`M ${cl.x} ${cl.y} L ${cr.x} ${cr.y}`} style={{ stroke: C.ink, strokeWidth: 5 * k + 2, strokeLinecap: 'butt' }} />
      <path d={`M ${cl.x} ${cl.y} L ${cr.x} ${cr.y}`} style={{ stroke: C.paper, strokeWidth: 5 * k, strokeLinecap: 'butt' }} />
      <path d={ticks.join(' ')} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
      <line x1={m1.x} y1={m1.y} x2={m2.x} y2={m2.y} style={{ stroke: C.ink, strokeWidth: 1.6 }} />
      <circle cx={apex.x} cy={apex.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
      {/* apex-to-crossbar dimension, h */}
      <DimLine x1={G.hDim.b.x} y1={G.hDim.b.y} x2={G.hDim.a.x} y2={G.hDim.a.y} role="brass" />
      <InkText x={G.hLabel.x} y={G.hLabel.y} anchor="end" baseline="central" size={14} role="brass">
        h
      </InkText>
      {/* the miss x, from the centre mark to where the string crosses (its rest position) */}
      {showX && (
        <g>
          <DimLine x1={mark.x} y1={mark.y} x2={G.hit.x} y2={G.hit.y} offset={G.dimOffset} role="red" />
          <InkText x={G.xLabelPt.x} y={G.xLabelPt.y} anchor="middle" baseline="central" size={14} role="red">
            x
          </InkText>
        </g>
      )}
      <line x1={apex.x} y1={apex.y} x2={bob.x} y2={bob.y} style={{ stroke: C.ink, strokeWidth: 0.9 }} />
      <circle cx={hit.x} cy={hit.y} r={2.6} style={{ fill: C.red }} />
      <Bob x={bob.x} y={bob.y} s={0.85 + 0.15 * k} />
      <InkText x={apex.x + 10} y={apex.y - 8} size={12}>
        apex
      </InkText>
      <InkText x={G.status.x} y={G.status.y} anchor="middle" size={narrow ? 12 : 13} role={G.level ? 'red' : 'soft'}>
        {G.statusText}
      </InkText>
    </svg>
  );
}

/* ── cord mode: the havayi terazi between two poles ── */

const { SPAN, NEAR, DROP } = CORD;

function Cord({ W, far, phi }: { W: number; far: number; phi: number }) {
  const G = cordLayout(W, far);
  const { narrow, Hs, xL, xR, ky, gL, gR, a, b, tiltDrawn, th, ux, uy, nx, ny, drop, top, h1, h2, tip } = G;
  const dir = { x: Math.sin(phi), y: Math.cos(phi) };
  const strLen = th + 6;
  const bob = { x: top.x + dir.x * strLen, y: top.y + dir.y * strLen };
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
        <rect x={x - 3} y={G.poleTop(g)} width={6} height={2.05 * ky} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
        <path d={p.join(' ')} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
        {lab}
      </g>
    );
  };
  const ground = G.groundPath;
  const { r, c: dc } = G.detail;
  // inside the detail circle: the tip and mark, magnified, with the vertical string. The
  // string-to-mark offset in the detail is the drawn offset times the magnification.
  // the string's x at the tip's height (it swings about `top`), relative to the tip, magnified
  const strAtTip = top.x + dir.x * ((tip.y - top.y) / Math.max(0.2, dir.y));
  const off = Math.max(-r * 0.8, Math.min(r * 0.8, (tip.x - strAtTip) * G.detail.mag));
  const bar = r * 0.75;
  return (
    <svg viewBox={`0 0 ${W} ${Hs}`} width="100%" style={{ display: 'block', height: 'auto' }}>
      <HatchDefs id="cd-hatch" gap={6} />
      <path d={`${ground} L ${xR + 40} ${Hs} L ${xL - 40} ${Hs} Z`} style={{ fill: 'url(#cd-hatch)', stroke: 'none' }} />
      <path d={ground} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      {pole(xL, gL, 'L')}
      {pole(xR, gR, 'R')}
      <line x1={xL} y1={a.y} x2={xR + 14} y2={a.y} style={{ stroke: C.faint, strokeWidth: 0.8, strokeDasharray: '4 4' }} />
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={{ stroke: C.ink, strokeWidth: 1.2 }} />
      <circle cx={a.x} cy={a.y} r={3.4} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.6 }} />
      <circle cx={b.x} cy={b.y} r={3.4} style={{ fill: C.paper, stroke: C.brass, strokeWidth: 1.6 }} />
      <path
        d={`M ${h1.x - nx * drop} ${h1.y - ny * drop} L ${h1.x} ${h1.y} M ${h2.x - nx * drop} ${h2.y - ny * drop} L ${h2.x} ${h2.y}`}
        style={{ stroke: C.soft, strokeWidth: 0.9 }}
      />
      <path d={`M ${h1.x} ${h1.y} L ${h2.x} ${h2.y} L ${tip.x} ${tip.y} Z`} style={{ fill: 'none', stroke: C.brass, strokeWidth: 5, strokeLinejoin: 'round' }} />
      <path d={`M ${h1.x} ${h1.y} L ${h2.x} ${h2.y} L ${tip.x} ${tip.y} Z`} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.8, strokeLinejoin: 'round' }} />
      <line x1={tip.x - ux * 9} y1={tip.y - uy * 9} x2={tip.x + ux * 9} y2={tip.y + uy * 9} style={{ stroke: C.red, strokeWidth: 1.6 }} />
      <line x1={top.x} y1={top.y} x2={bob.x} y2={bob.y} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
      <Bob x={bob.x} y={bob.y - 2} s={0.7} />
      <g>
        <circle cx={tip.x} cy={tip.y} r={9} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.7, strokeDasharray: '2 2' }} />
        <line x1={tip.x + 9} y1={tip.y} x2={dc.x - r} y2={dc.y} style={{ stroke: C.soft, strokeWidth: 0.6, strokeDasharray: '2 2' }} />
        <circle cx={dc.x} cy={dc.y} r={r} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 0.9 }} />
        <clipPath id="cd-detail">
          <circle cx={dc.x} cy={dc.y} r={r - 1} />
        </clipPath>
        <g clipPath="url(#cd-detail)">
          <g transform={`rotate(${-tiltDrawn / RAD} ${dc.x + off} ${dc.y + 4})`}>
            <path
              d={`M ${dc.x + off - bar * 0.5} ${dc.y - r} L ${dc.x + off} ${dc.y + 4} L ${dc.x + off + bar * 0.5} ${dc.y - r}`}
              style={{ fill: 'none', stroke: C.brass, strokeWidth: 6, strokeLinejoin: 'round' }}
            />
            <line x1={dc.x + off - 10} y1={dc.y + 8} x2={dc.x + off + 10} y2={dc.y + 8} style={{ stroke: C.red, strokeWidth: 1.8 }} />
          </g>
          <line x1={dc.x} y1={dc.y - r} x2={dc.x} y2={dc.y + r} style={{ stroke: C.ink, strokeWidth: 1 }} />
        </g>
        <InkText x={dc.x} y={dc.y + r + 14} anchor="middle" size={11} role="soft">
          the tip, enlarged
        </InkText>
      </g>
      <InkText x={xL} y={G.poleTop(gL) - 8} anchor="middle" size={12} role="sky">
        near pole
      </InkText>
      <InkText x={xR} y={G.poleTop(gR) - 8} anchor="middle" size={12} role="brass">
        far pole
      </InkText>
      <InkText x={G.status.x} y={G.status.y} anchor="middle" size={narrow ? 12 : 13} role={G.level ? 'red' : 'soft'}>
        {G.statusText}
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
  const xmm = AFRAME_H_CM * 10 * Math.abs(tanT);
  const seked = Math.abs(tanT) < 1e-3 ? null : 7 / Math.abs(tanT);
  const farMax = 1.9;
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
        <>
          <Controls>
            <Slider
              label="tilt of the stone, θ"
              value={theta}
              min={-10}
              max={10}
              step={0.1}
              onChange={setTheta}
              format={(v) => `${fmtSigned(v)}°`}
            />
          </Controls>
          <Working label="How far the string misses the mark">
            <Eq>
              <Sym role="red">x</Sym> = <Sym role="brass">h</Sym> · tan <Sym role="sky">θ</Sym>
            </Eq>
            <Eq>
              <V role="red">{`${xmm.toFixed(0)} mm`}</V> = <V role="brass">{`${AFRAME_H_CM} cm`}</V> · tan{' '}
              <V role="sky">{`${Math.abs(theta).toFixed(1)}°`}</V>
            </Eq>
            <Eq>
              slope <V>{`${Math.abs(tanT * 1000).toFixed(0)} mm`}</V> per metre, or in Egyptian terms{' '}
              {seked === null ? (
                <>flat</>
              ) : (
                <>
                  a seked of <V>{`${seked >= 100 ? seked.toFixed(0) : seked.toFixed(1)} palms`}</V> run per cubit of rise
                </>
              )}
            </Eq>
          </Working>
        </>
      ) : (
        <>
          <Controls>
            <Slider
              label="cord height at the far pole"
              value={far}
              min={CORD.MIN}
              max={farMax}
              step={0.01}
              onChange={setFar}
              format={(v) => `${v.toFixed(2)} m`}
            />
          </Controls>
          <Working label="Height difference between the poles">
            <Eq>
              fall of the ground = <Sym role="brass">far reading</Sym> − <Sym role="sky">near reading</Sym>
            </Eq>
            <Eq>
              <V role={Math.abs(far - NEAR - DROP) < 0.006 ? 'red' : 'ink'}>{`${((far - NEAR) * 100).toFixed(0)} cm`}</V> ={' '}
              <V role="brass">{`${far.toFixed(2)} m`}</V> − <V role="sky">{`${NEAR.toFixed(2)} m`}</V>
              {Math.abs(far - NEAR - DROP) < 0.006 ? ' (cord level, so this is the answer)' : ' (cord not level yet)'}
            </Eq>
          </Working>
        </>
      )}
    </Plate>
  );
}
