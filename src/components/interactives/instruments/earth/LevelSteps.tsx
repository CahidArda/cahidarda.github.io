/**
 * Episode 1, step by step: vertical and level from a weight on a string.
 *   1 True a wall with a plumb line · 2 Stand the A-frame on the road · 3 Read the miss x ·
 *   4 Work out the slope · 5 Level a long run with the havayi terazi on a cord · 6 Where it fails.
 * The A-frame stands on the site's real surfaces (the world sets θ; the reader moves the frame).
 * Drawings: ./geometry.ts (aframeLayout, cordLayout) and ./levelGeom.ts (wall, numbers).
 * The bob swings and settles (damped pendulum) after each move; reduced motion snaps it.
 */
import { useEffect, useRef, useState } from 'react';
import { AFRAME_H_CM, CORD, RAD, aframeLayout, cordLayout, hitParam, swingKick, swingStep } from './geometry';
import {
  C,
  DimLine,
  Eq,
  HatchDefs,
  InkText,
  Known,
  Segmented,
  Slider,
  StepPlate,
  Sym,
  Working,
  arcPath,
  useMeasuredWidth,
  useReducedMotion,
  type StepDef,
} from '../shared';
import { SiteRef } from './SiteRef';
import {
  CORD_LEVEL_TOL,
  LEAN,
  ROAD_RUN_PER_RISE,
  SURFACES,
  cordFall,
  cordLevel,
  missMm,
  seked,
  slopeFromMiss,
  wallScene,
  crossbarScene,
  type SurfaceId,
} from './levelGeom';

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
    // the apex jumps sideways with the frame and the bob lags behind it (see swingKick)
    s.w = swingKick(s.w, d);
    if (s.raf) return;
    let prev = 0;
    const step = (now: number) => {
      const dt = prev ? Math.min((now - prev) / 1000, 0.05) : 0.016;
      prev = now;
      const n = swingStep(s.phi, s.w, dt);
      s.phi = n.phi;
      s.w = n.w;
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

const { NEAR, DROP } = CORD;

function Cord({ W, far, phi }: { W: number; far: number; phi: number }) {
  const G = cordLayout(W, far);
  const { narrow, Hs, xL, xR, ky, gL, gR, a, b, tiltDrawn, th, ux, uy, nx, ny, drop, top, h1, h2, tip } = G;
  const dir = { x: Math.sin(phi), y: Math.cos(phi) };
  const strLen = th + 6;
  const bob = { x: top.x + dir.x * strLen, y: top.y + dir.y * strLen };
  const pole = (x: number, g: number, side: 'L' | 'R') => {
    const p: string[] = [];
    const lab: React.ReactNode[] = [];
    for (let i = 0; i <= 23; i++) {
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
        <rect x={x - 3} y={G.poleTop(g)} width={6} height={2.35 * ky} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
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


/* ── step 1: truing a wall with a plumb line ── */

function WallView({ lean }: { lean: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = wallScene(W, lean);
  const id = `wall-h-${W}`;
  const courses = [0.25, 0.5, 0.75];
  const at = (f: number) => ({ x: g.faceTop.x + (g.faceFoot.x - g.faceTop.x) * f, y: g.top + (g.foot - g.top) * f });
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={g.labels.status.text}>
        <HatchDefs id={id} gap={6} />
        <path
          d={`M ${g.wallBack} ${g.top} L ${g.faceTop.x} ${g.faceTop.y} L ${g.faceFoot.x} ${g.faceFoot.y} L ${g.wallBack} ${g.foot} Z`}
          style={{ fill: `url(#${id})`, stroke: C.ink, strokeWidth: 1.1 }}
        />
        {courses.map((f) => (
          <line key={f} x1={g.wallBack} y1={at(f).y} x2={at(f).x} y2={at(f).y} style={{ stroke: C.soft, strokeWidth: 0.7 }} />
        ))}
        <line x1={g.wallBack - 12} y1={g.foot} x2={W - 10} y2={g.foot} style={{ stroke: C.ink, strokeWidth: 1.1 }} />
        {/* the bracket on the top edge and the line, truly vertical */}
        <path d={`M ${g.faceTop.x - 10} ${g.top - 8} H ${g.lineX + 4} V ${g.top - 2}`} style={{ fill: 'none', stroke: C.ink, strokeWidth: 2 }} />
        <line x1={g.lineX} y1={g.top - 2} x2={g.lineX} y2={g.foot - 18} style={{ stroke: C.red, strokeWidth: 1 }} />
        <Bob x={g.lineX} y={g.foot - 20} s={0.8} />
        {/* the two gaps */}
        <DimLine x1={g.faceTop.x} y1={g.top + 6} x2={g.lineX} y2={g.top + 6} role="sky" />
        <DimLine x1={g.faceFoot.x} y1={g.foot - 10} x2={g.lineX} y2={g.foot - 10} role={g.plumb ? 'sky' : 'red'} />
        <InkText x={g.labels.top.x} y={g.labels.top.y} kind="num" size={11} role="sky">{g.labels.top.text}</InkText>
        <InkText x={g.labels.foot.x} y={g.labels.foot.y} kind="num" size={11} role={g.plumb ? 'sky' : 'red'}>{g.labels.foot.text}</InkText>
        <InkText x={14} y={18} kind="label" size={11} role="soft">{`wall face, side view, gaps ×${g.EX}`}</InkText>
        <InkText x={g.labels.status.x} y={g.labels.status.y} anchor="middle" size={12} role={g.plumb ? 'red' : 'soft'}>
          {g.labels.status.text}
        </InkText>
      </svg>
    </div>
  );
}

/* ── step 3: the crossbar, enlarged ── */

function CrossbarView({ theta }: { theta: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = crossbarScene(W, theta);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The string crosses the crossbar ${g.read} from the centre mark`}>
        <InkText x={14} y={18} kind="label" size={11} role="soft">the crossbar, enlarged (cm)</InkText>
        <rect x={g.cx - g.half - 8} y={g.y} width={2 * g.half + 16} height={18} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
        {g.ticks.map((t, i) => (
          <g key={i}>
            <line x1={t.x} y1={g.y} x2={t.x} y2={g.y + (t.major ? 10 : 5)} style={{ stroke: C.ink, strokeWidth: t.major ? 1 : 0.6 }} />
            {t.label && (
              <InkText x={t.x} y={g.y + 32} kind="num" anchor="middle" size={10} role="soft">
                {t.label}
              </InkText>
            )}
          </g>
        ))}
        <line x1={g.cx} y1={g.y - 6} x2={g.cx} y2={g.y + 18} style={{ stroke: C.ink, strokeWidth: 2 }} />
        <line x1={g.sx} y1={8 + 22} x2={g.sx} y2={g.y + 30} style={{ stroke: C.ink, strokeWidth: 1 }} />
        <circle cx={g.sx} cy={g.y} r={3} style={{ fill: C.red }} />
        {Math.abs(g.sx - g.cx) > 4 && <DimLine x1={g.cx} y1={g.y + 50} x2={g.sx} y2={g.y + 50} role="red" />}
        <InkText x={g.readLabel.x} y={g.readLabel.y} anchor="middle" kind="num" size={13} role="red">
          {g.readLabel.text}
        </InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">
          {Math.abs(theta) < 0.05 ? 'the string is on the centre mark' : 'the string hangs on the downhill side'}
        </InkText>
      </svg>
    </div>
  );
}

/* ── the plate ── */

const ROAD = SURFACES.find((s) => s.id === 'road')!;
const FAR_START = 2.25;

export default function LevelSteps() {
  const [step, setStep] = useState(0);
  const [lean, setLean] = useState(LEAN.START);
  const [surf, setSurf] = useState<SurfaceId>('hill');
  const [far, setFar] = useState(FAR_START);
  const reduced = useReducedMotion();
  const theta = SURFACES.find((s) => s.id === surf)!.theta;
  const phi = useSwing(step >= 1 && step <= 3 ? theta * RAD : 0, reduced);
  const plumb = Math.abs(lean) <= LEAN.TOL;
  const onRoad = surf === 'road';
  const level = cordLevel(far);
  const xmm = missMm(ROAD.theta);
  const thetaRead = slopeFromMiss(Math.round(xmm * 2) / 2); // read to half a millimetre
  const runPerRise = 1 / Math.tan(thetaRead * RAD);
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'plumb',
      title: 'True a wall',
      text: (
        <>
          Gravity gives one direction for free: down. Hang a weight from a bracket on the top of the wall and it shows a
          truly vertical line. If the gap to the face is the same at the top and the foot, the wall is plumb.
        </>
      ),
      controls: (
        <Slider label="lean of the wall face" value={lean} min={LEAN.MIN} max={LEAN.MAX} step={LEAN.STEP} onChange={setLean} format={(v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)} mm/m`} />
      ),
      gated: true,
      done: plumb,
      hint: 'Tap the stone until the two gaps match',
    },
    {
      id: 'stand',
      title: 'Stand the A-frame',
      text: (
        <>
          Level has no direction of its own. The A-frame borrows one: its apex-to-centre line is square to its feet, and the
          string hangs straight down. Carry it from the hilltop down the road toward the quay.
        </>
      ),
      controls: (
        <Segmented
          label="where the frame stands"
          options={SURFACES.map((s) => ({ value: s.id, label: s.name }))}
          value={surf}
          onChange={setSurf}
        />
      ),
      gated: true,
      done: onRoad,
      hint: 'Stand it on the road',
    },
    {
      id: 'read',
      title: 'Read the miss',
      text: (
        <>
          On the road the string no longer falls on the centre mark. Read how far off it crosses the marked crossbar: that
          distance <Sym role="red">x</Sym> is the measurement.
        </>
      ),
    },
    {
      id: 'slope',
      title: 'Work out the slope',
      text: (
        <>
          The apex is <Sym role="brass">h</Sym> = {AFRAME_H_CM} cm above the crossbar, so the slope angle is arctan(x / h). Builders
          wrote it as a ratio: 1 in {ROAD_RUN_PER_RISE.toFixed(0)}, or as the Egyptian seked, palms of run per cubit of rise.
        </>
      ),
    },
    {
      id: 'cord',
      title: 'Level a longer run',
      text: (
        <>
          A frame spans a metre. To carry level over 6 m, hang the havayi terazi on a cord between two graduated poles and
          slide the far end until the bob sits on the mark. The cord is then level; read both poles.
        </>
      ),
      controls: (
        <Slider label="cord height at the far pole" value={far} min={CORD.MIN} max={CORD.MAX} step={0.01} onChange={setFar} format={(v) => `${v.toFixed(2)} m`} />
      ),
      gated: true,
      done: level,
      hint: 'Slide until the bob sits on the mark',
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Leapfrog the poles down the road and the falls add up to the whole drop from the precinct to the quay. But a level
          only says how much higher. It never says how far: the length of the road, M to A, is still unknown.
        </>
      ),
    },
  ];
  const view = (i: number) => {
    if (i === 0) return <WallView lean={lean} />;
    if (i === 1) return <Frame theta={theta} phi={phi} />;
    if (i === 2) return <CrossbarView theta={theta} />;
    if (i === 3) return <Frame theta={theta} phi={phi} />;
    return <CordFrame far={i === 5 ? CORD.NEAR + CORD.DROP : far} />;
  };
  const hiOf = (i: number) =>
    (i === 0 ? ['minaret'] : i <= 3 ? ['slope'] : i === 4 ? ['slope', 'fall'] : ['fall', 'line']) as ('minaret' | 'slope' | 'fall' | 'line')[];
  return (
    <StepPlate
      n={1}
      wide
      title="Vertical and level from a plumb line"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="The instrument"
      reference={(i) => (
        <SiteRef
          hi={hiOf(i)}
          best={i === 5 ? 'known: up and down · unknown: how far' : i >= 3 ? 'known: the slope of the road' : 'known: which way is down'}
          label="The site: a minaret on its hill, a road down to the quay, the strait and the far shore, in section and plan"
        />
      )}
      referenceLabel="The site (same in every plate)"
      onSkip={() => {
        setLean(0);
        setSurf('road');
        setFar(CORD.NEAR + CORD.DROP);
      }}
      equation={
        <Working label="Slope from the A-frame, fall from the cord">
          <Eq>
            <Sym role="red">x</Sym> = <Sym role="brass">h</Sym> · tan <Sym role="sky">θ</Sym>, so <Sym role="sky">θ</Sym> = arctan(
            <Sym role="red">x</Sym> / <Sym role="brass">h</Sym>)
          </Eq>
          <Eq>
            <Known when={kn(3)} role="sky">{`${thetaRead.toFixed(1)}°`}</Known> = arctan(<Known when={kn(2) && onRoad} role="red">{`${(Math.round(xmm * 2) / 2).toFixed(1)} mm`}</Known> /{' '}
            <Known when={kn(1)} role="brass">{`${AFRAME_H_CM * 10} mm`}</Known>)
          </Eq>
          <Eq>
            1 in <Known when={kn(3)}>{runPerRise.toFixed(1)}</Known> · seked <Known when={kn(3)}>{`${seked(thetaRead).toFixed(0)} palms`}</Known> per cubit
          </Eq>
          <Eq>
            fall = <Sym role="brass">far</Sym> − <Sym role="sky">near</Sym> = <Known when={kn(4) && level} role="brass">{`${far.toFixed(2)} m`}</Known> −{' '}
            <Known when={kn(4)} role="sky">{`${CORD.NEAR.toFixed(2)} m`}</Known> = <Known when={kn(4) && level} role="red">{`${(cordFall(far) * 100).toFixed(0)} cm`}</Known>
          </Eq>
        </Working>
      }
      caption={
        <>
          The reference on the right is a composite of Istanbul, drawn to scale: a Süleymaniye minaret (76 m with its finial)
          on a 40 m hill, a road falling 1 in 8 to a quay, and a strait as wide as the Bosphorus at its narrows (about 700 m).
          The real minaret and the real narrows are kilometres apart. Every plate in this post measures this same piece of
          ground. A-frame and cord drawings are not to scale; the cord counts as level within {CORD_LEVEL_TOL * 1000} mm.
        </>
      }
    />
  );
}

function Frame({ theta, phi }: { theta: number; phi: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  return <div ref={ref}><AFrame W={W} theta={theta} phi={phi} /></div>;
}
function CordFrame({ far }: { far: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  return <div ref={ref}><Cord W={W} far={far} phi={0} /></div>;
}
