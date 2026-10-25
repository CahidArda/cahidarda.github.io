/**
 * The 3D sextant plate. The reader moves the index arm; the deck tips the whole instrument a
 * little in its own plane. The sun and the sea horizon stay fixed in the world. Every ray, the
 * eyepiece inset, the status and the worked equation come from ONE function, `sextantView`
 * (./sextantView.ts), checked against an independent 3D model of the two mirrors.
 */
import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import type * as THREEns from 'three';
import {
  ArcScale,
  BlueprintCanvas,
  C,
  Controls,
  Eq,
  InkText,
  Plate,
  Slider,
  Sym,
  V,
  Working,
  polar,
} from '../shared';
import {
  ARC_FROM,
  ARC_TO,
  H,
  NH_DEG,
  TEL_FROM,
  TEL_TO,
  armAngle,
  pol,
  type V2,
} from './sextantGeom';
import { FIELD_R, SUN_ALT, SUN_SD, sextantView, tiltAt, type SextantView } from './sextantView';

export { SUN_ALT };
const Z_SUN = 0.085; // height of the reflected path above the frame (silvered half)
const Z_HOR = 0.155; // the direct horizon ray (clear half)
const Z_AXIS = 0.12; // the telescope's axis: the line between the two halves

const fmtDeg = (v: number) => {
  const d = Math.floor(v + 1e-9);
  const m = Math.round((v - d) * 60);
  return m === 60 ? `${d + 1}°00′` : `${d}°${String(m).padStart(2, '0')}′`;
};
const sgn = (v: number, dp: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(dp)}°`;

/* ───────────── poster: the same instrument, face on, set to the sun ───────────── */

function Poster() {
  const W = 640;
  const Hh = 400;
  const S = 300;
  const ox = W * 0.56;
  const oy = Hh * 0.33;
  const P = (v: V2) => ({ x: ox + v.x * S, y: oy - v.y * S });
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const reading = SUN_ALT;
  const a = armAngle(reading);
  const v = sextantView(reading, 0);
  const line = (pts: V2[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${P(p).x} ${P(p).y}`).join(' ');
  const bar = (deg: number) => {
    const p = polar(ox, oy, S * 0.88, deg);
    return `M ${ox} ${oy} L ${p.x} ${p.y}`;
  };
  const hE = P(H);
  const tA = P({ x: TEL_FROM, y: H.y });
  const tB = P({ x: TEL_TO, y: H.y });
  const arm = polar(ox, oy, S * 1.06, a);
  const mHalf = 0.07 * S;
  const mDir = ((NH_DEG + 90) * Math.PI) / 180;
  const iDir = ((NH_DEG - reading / 2 + 90) * Math.PI) / 180;
  const iHalf = 0.11 * S;
  const sun = P(v.sun);
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`}>
      <ArcScale
        cx={ox}
        cy={oy}
        r={S * 0.98}
        from={-120}
        to={-60}
        major={5}
        minor={1}
        labelEvery={10}
        value={(x) => -2 * (x + 60)}
        inward
        tick={9}
        size={10}
      />
      <path
        d={`${bar(-120)} ${bar(-60)}`}
        style={{ ...ink, strokeWidth: 6, stroke: C.soft, opacity: 0.35 }}
      />
      <path d={`${bar(-120)} ${bar(-60)}`} style={ink} />
      <path d={`M ${ox} ${oy} L ${arm.x} ${arm.y}`} style={{ ...ink, strokeWidth: 2.2 }} />
      <circle cx={ox} cy={oy} r={7} style={{ ...ink, fill: C.paper }} />
      <line
        x1={ox - Math.cos(iDir) * iHalf}
        y1={oy + Math.sin(iDir) * iHalf}
        x2={ox + Math.cos(iDir) * iHalf}
        y2={oy - Math.sin(iDir) * iHalf}
        style={{ ...ink, stroke: C.brass, strokeWidth: 3.2 }}
      />
      <line
        x1={hE.x - Math.cos(mDir) * mHalf}
        y1={hE.y + Math.sin(mDir) * mHalf}
        x2={hE.x + Math.cos(mDir) * mHalf}
        y2={hE.y - Math.sin(mDir) * mHalf}
        style={{ ...ink, strokeWidth: 3.2 }}
      />
      <rect
        x={tA.x}
        y={tA.y - 9}
        width={tB.x - tA.x}
        height={18}
        style={{ ...ink, fill: C.paper }}
      />
      <path
        d={line(v.blue)}
        style={{ fill: 'none', stroke: C.sky, strokeWidth: 1, strokeDasharray: '5 4' }}
      />
      <path d={line(v.red)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.4 }} />
      <circle
        cx={sun.x}
        cy={sun.y}
        r={11}
        style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.2 }}
      />
      <InkText x={16} y={P(v.blue[0]).y - 8} size={13} role="sky">
        horizon
      </InkText>
      <InkText x={sun.x + 16} y={sun.y + 4} size={13} role="red">
        sun
      </InkText>
    </svg>
  );
}

/* ───────────── the eyepiece: horizon (clear half) and sun (silvered half) ───────────── */

const EYE_R = 50; // px, the field stop
const EYE_K = EYE_R / FIELD_R; // px per degree: the sun's disc is drawn at its true size

type EyeRefs = {
  hor: SVGGElement | null;
  sun: SVGGElement | null;
  up: SVGPathElement | null;
  down: SVGPathElement | null;
};

/** Place the eyepiece marks for one view. Used by the first render and by every 3D frame. */
function eyeLayout(v: SextantView) {
  const horY = -v.horizonEl * EYE_K;
  const sunY = v.sunEl === null ? null : -v.sunEl * EYE_K;
  // no image: the light passes above or below the field (gap > 0: the image would be higher)
  const show = v.inField && sunY !== null;
  const arrow = show ? 0 : v.gap > 0 ? 1 : -1;
  return { horY, sunY: show ? sunY : 0, show, arrow };
}
function applyEye(r: EyeRefs, v: SextantView) {
  const L = eyeLayout(v);
  r.hor?.setAttribute('transform', `translate(0 ${L.horY.toFixed(2)})`);
  if (r.sun) {
    r.sun.setAttribute('transform', `translate(0 ${L.sunY.toFixed(2)})`);
    r.sun.style.display = L.show ? '' : 'none';
  }
  if (r.up) r.up.style.display = L.arrow > 0 ? '' : 'none';
  if (r.down) r.down.style.display = L.arrow < 0 ? '' : 'none';
}

function Eyepiece({ view, refs }: { view: SextantView; refs: React.RefObject<EyeRefs> }) {
  const R = EYE_R;
  const L = eyeLayout(view);
  const sx = R * 0.45;
  return (
    <svg viewBox="-60 -60 120 120" style={{ width: '100%', display: 'block' }} aria-hidden="true">
      <defs>
        <clipPath id="sx-eye">
          <circle r={R} />
        </clipPath>
      </defs>
      <circle r={R + 4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
      <g clipPath="url(#sx-eye)">
        <rect x={0} y={-R} width={R} height={2 * R} style={{ fill: C.deep }} />
        <g
          ref={(el) => void (refs.current.hor = el)}
          transform={`translate(0 ${L.horY.toFixed(2)})`}
        >
          {/* the sea, seen straight through the clear half; its line carried faintly across */}
          <rect x={-R} y={0} width={R} height={R * 3} style={{ fill: C.sky, opacity: 0.22 }} />
          <line x1={-R} y1={0} x2={0} y2={0} style={{ stroke: C.sky, strokeWidth: 1.6 }} />
          <line
            x1={0}
            y1={0}
            x2={R}
            y2={0}
            style={{ stroke: C.sky, strokeWidth: 0.9, strokeDasharray: '3 3' }}
          />
        </g>
        <g
          ref={(el) => void (refs.current.sun = el)}
          transform={`translate(0 ${L.sunY.toFixed(2)})`}
          style={{ display: L.show ? undefined : 'none' }}
        >
          <circle
            cx={sx}
            cy={0}
            r={SUN_SD * EYE_K}
            style={{ fill: C.red, stroke: C.paper, strokeWidth: 0.8 }}
          />
        </g>
      </g>
      <path
        ref={(el) => void (refs.current.up = el)}
        d={`M ${sx - 6} ${-R + 16} L ${sx} ${-R + 8} L ${sx + 6} ${-R + 16}`}
        style={{
          fill: 'none',
          stroke: C.red,
          strokeWidth: 1.6,
          display: L.arrow > 0 ? undefined : 'none',
        }}
      />
      <path
        ref={(el) => void (refs.current.down = el)}
        d={`M ${sx - 6} ${R - 16} L ${sx} ${R - 8} L ${sx + 6} ${R - 16}`}
        style={{
          fill: 'none',
          stroke: C.red,
          strokeWidth: 1.6,
          display: L.arrow < 0 ? undefined : 'none',
        }}
      />
      <line x1={0} y1={-R} x2={0} y2={R} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
      <circle r={R} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1 }} />
    </svg>
  );
}

/* ───────────── text on the limb, as a theme-tracking texture ───────────── */

function numeral(T: typeof THREEns, text: string, mat: THREEns.MeshBasicMaterial, h: number) {
  const cv = document.createElement('canvas');
  cv.width = 128;
  cv.height = 64;
  const g = cv.getContext('2d')!;
  g.fillStyle = '#fff';
  g.font = '500 44px "JetBrains Mono Variable", ui-monospace, monospace';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 64, 34);
  const tex = new T.CanvasTexture(cv);
  tex.anisotropy = 4;
  const m = mat.clone();
  m.map = tex;
  m.transparent = true;
  m.depthWrite = false;
  m.polygonOffset = false;
  return { mesh: new T.Mesh(new T.PlaneGeometry(h * 2, h), m), m };
}

type Params = { reading: number; eye: React.RefObject<EyeRefs>; tilt: React.RefObject<number> };
type Pick = {
  T: typeof THREEns;
  camera: THREEns.Camera;
  inst: THREEns.Object3D;
  parts: THREEns.Object3D[];
};

export default function SextantModel({ n = 2 }: { n?: number | string }) {
  const [reading, setReading] = useState(20);
  const eyeRef = useRef<EyeRefs>({ hor: null, sun: null, up: null, down: null });
  const tiltRef = useRef(0); // the tilt of the last drawn 3D frame
  // a React render writes the still (untilted) eyepiece; put it back at the drawn frame's tilt
  useLayoutEffect(() => applyEye(eyeRef.current, sextantView(reading, tiltRef.current)), [reading]);
  const pickRef = useRef<Pick | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<number | null>(null);
  /** Pointer → ray into the scene, using the canvas's own camera. */
  const ray = (e: React.PointerEvent) => {
    const pk = pickRef.current;
    const cv = wrapRef.current?.querySelector('canvas');
    if (!pk || !cv) return null;
    const r = cv.getBoundingClientRect();
    const ndc = new pk.T.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1,
    );
    const rc = new pk.T.Raycaster();
    rc.setFromCamera(ndc, pk.camera);
    return { rc, pk };
  };
  const hitsArm = (e: React.PointerEvent) => {
    const q = ray(e);
    if (!q) return false;
    q.rc.params.Line = { threshold: 0 };
    return q.rc
      .intersectObjects(q.pk.parts, true)
      .some((h) => (h.object as THREEns.Mesh).isMesh && !h.object.userData.inkHull);
  };
  const grabRef = useRef<{ ang: number; reading: number } | null>(null);
  const readingRef = useRef(reading);
  readingRef.current = reading;
  /** Pointer angle about the pivot, in the instrument's own plane (degrees). */
  const pointerAngle = (e: React.PointerEvent, minR = 0.25) => {
    const q = ray(e);
    if (!q) return null;
    const { T } = q.pk;
    const m = q.pk.inst.matrixWorld;
    const n = new T.Vector3(0, 0, 1).transformDirection(m);
    const o = new T.Vector3(0, 0, 0.05).applyMatrix4(m);
    const plane = new T.Plane().setFromNormalAndCoplanarPoint(n, o);
    const hit = q.rc.ray.intersectPlane(plane, new T.Vector3());
    if (!hit) return null;
    const loc = q.pk.inst.worldToLocal(hit);
    if (Math.hypot(loc.x, loc.y) < minR) return null;
    return (Math.atan2(loc.y, loc.x) * 180) / Math.PI;
  };
  const dragTo = (e: React.PointerEvent) => {
    const g = grabRef.current;
    const ang = pointerAngle(e);
    if (!g || ang === null) return;
    let d = ang - g.ang;
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    const r = g.reading - 2 * d;
    setReading(Math.max(0, Math.min(120, Math.round(r * 10) / 10)));
  };
  const params = useMemo<Params>(() => ({ reading, eye: eyeRef, tilt: tiltRef }), [reading]);
  const nudge = (d: number) =>
    setReading((r) => Math.max(0, Math.min(120, Math.round((r + d) * 10) / 10)));
  // the still view (no tilt) for the first paint and the text; frames re-apply with their tilt
  const view = sextantView(reading, 0);
  const theta = reading / 2;

  return (
    <Plate
      n={n}
      wide
      title="The sextant: measuring the sun's height"
      kicker="drag the index arm"
      caption={
        <>
          The sun and the sea horizon stay put; the instrument moves. The red line is a real ray
          from the noon sun over Istanbul on 25 October 2026, bounced by the index mirror on the arm
          and then by the silvered half of the horizon mirror. The blue dashed line is the horizon,
          seen straight through the clear half. As the deck tips the instrument, both lines swing
          across the telescope together, so in the eyepiece (top right, drawn at true size, 5°
          across) the sun and the horizon drift up and down as a pair and the gap between them never
          changes. Only the arm changes the gap.
        </>
      }
    >
      <div
        ref={wrapRef}
        style={{ position: 'relative' }}
        onPointerDownCapture={(e) => {
          if (e.button !== 0 || !hitsArm(e)) return;
          const ang = pointerAngle(e, 0.5); // the outer half of the arm: a long lever, fine control
          if (ang === null) return;
          e.stopPropagation();
          e.preventDefault();
          grabRef.current = { ang, reading: readingRef.current };
          dragRef.current = e.pointerId;
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMoveCapture={(e) => {
          if (dragRef.current === e.pointerId) {
            e.stopPropagation();
            dragTo(e);
            return;
          }
          if (e.pointerType === 'mouse') {
            const cv = wrapRef.current?.querySelector('canvas');
            if (cv) cv.style.cursor = hitsArm(e) && pointerAngle(e, 0.5) !== null ? 'pointer' : '';
          }
        }}
        onPointerUpCapture={(e) => {
          if (dragRef.current === e.pointerId) {
            dragRef.current = null;
            e.stopPropagation();
          }
        }}
        onPointerCancelCapture={() => (dragRef.current = null)}
      >
        <BlueprintCanvas<Params>
          aspect={16 / 10}
          fit={1.02}
          animate
          yaw={-0.42}
          pitch={0.2}
          params={params}
          label="Rotatable line drawing of a marine sextant with the ray path from the sun through both mirrors"
          poster={<Poster />}
          build={(T, ctx) => {
            const world = new T.Group();
            world.position.set(0.3, 0.2, 0);
            ctx.root.add(world);
            const inst = new T.Group(); // everything that rolls with the ship
            world.add(inst);
            const deg = Math.PI / 180;
            const ext = (shape: THREEns.Shape, depth: number, z = 0) => {
              const g = new T.ExtrudeGeometry(shape, {
                depth,
                bevelEnabled: false,
                curveSegments: 48,
              });
              const m = new T.Mesh(g);
              m.position.z = z;
              return m;
            };

            // ── frame: limb, two radial bars, a brace, the pivot boss ──
            const a0 = armAngle(ARC_TO) * deg;
            const a1 = armAngle(ARC_FROM) * deg;
            const limb = new T.Shape();
            limb.absarc(0, 0, 1.03, a0, a1, false);
            limb.absarc(0, 0, 0.86, a1, a0, true);
            const limbM = ext(limb, 0.03);
            ctx.inkify(limbM, { threshold: 30, outline: 0 });
            inst.add(limbM);
            const barShape = (angDeg: number, r0: number, r1: number, w: number) => {
              const a = angDeg * deg;
              const u = { x: Math.cos(a), y: Math.sin(a) };
              const n = { x: -u.y * w, y: u.x * w };
              const s = new T.Shape();
              s.moveTo(u.x * r0 + n.x, u.y * r0 + n.y);
              s.lineTo(u.x * r1 + n.x, u.y * r1 + n.y);
              s.lineTo(u.x * r1 - n.x, u.y * r1 - n.y);
              s.lineTo(u.x * r0 - n.x, u.y * r0 - n.y);
              s.closePath();
              return s;
            };
            for (const a of [-121, -59]) {
              const b = ext(barShape(a, 0.06, 0.9, 0.04), 0.03);
              ctx.inkify(b, { outline: 0 });
              inst.add(b);
            }
            const brace = new T.Shape();
            const pL = pol(0.6, -121);
            const pR = pol(0.6, -59);
            brace.moveTo(pL.x, pL.y + 0.03);
            brace.lineTo(pR.x, pR.y + 0.03);
            brace.lineTo(pR.x, pR.y - 0.03);
            brace.lineTo(pL.x, pL.y - 0.03);
            brace.closePath();
            const braceM = ext(brace, 0.03);
            ctx.inkify(braceM, { outline: 0 });
            inst.add(braceM);
            const boss = new T.Mesh(new T.CylinderGeometry(0.075, 0.075, 0.05, 40));
            boss.rotation.x = Math.PI / 2;
            boss.position.z = 0.025;
            ctx.inkify(boss, { threshold: 50 });
            inst.add(boss);

            // handle on the back
            const handle = new T.Mesh(new T.CylinderGeometry(0.045, 0.045, 0.42, 24));
            handle.position.set(0.04, -0.52, -0.17);
            ctx.inkify(handle, { threshold: 50, hidden: true });
            inst.add(handle);
            for (const y of [-0.33, -0.71]) {
              const post = new T.Mesh(new T.BoxGeometry(0.035, 0.035, 0.15));
              post.position.set(0.04, y, -0.085);
              ctx.inkify(post, { outline: 0 });
              inst.add(post);
            }

            // ── graduations and numerals on the limb face ──
            const tick: number[] = [];
            for (let v = ARC_FROM; v <= ARC_TO + 1e-6; v += 1) {
              const L = v % 10 === 0 ? 0.07 : v % 5 === 0 ? 0.05 : 0.03;
              const p = pol(1.02, armAngle(v));
              const q = pol(1.02 - L, armAngle(v));
              tick.push(p.x, p.y, 0.031, q.x, q.y, 0.031);
            }
            const tg = new T.BufferGeometry();
            tg.setAttribute('position', new T.Float32BufferAttribute(tick, 3));
            inst.add(new T.LineSegments(tg, ctx.lineMaterial('ink')));
            const inkFill = ctx.fillMaterial('ink');
            const texMats: THREEns.MeshBasicMaterial[] = [];
            for (let v = 0; v <= 120; v += 10) {
              const { mesh, m } = numeral(T, String(v), inkFill, 0.045);
              texMats.push(m);
              const a = armAngle(v);
              const p = pol(0.905, a);
              mesh.position.set(p.x, p.y, 0.032);
              mesh.rotation.z = (a + 90) * deg;
              inst.add(mesh);
            }

            // ── index arm + index mirror (they turn together about the pivot) ──
            const arm = new T.Group();
            arm.position.z = 0.05;
            inst.add(arm);
            const armBar = ext(barShape(0, -0.1, 1.0, 0.035), 0.022);
            ctx.inkify(armBar, { outline: 0 });
            arm.add(armBar);
            const clamp = new T.Mesh(new T.BoxGeometry(0.14, 0.12, 0.04));
            clamp.position.set(1.0, 0, 0.02);
            ctx.inkify(clamp, { outline: 0 });
            arm.add(clamp);
            const screw = new T.Mesh(new T.CylinderGeometry(0.03, 0.03, 0.12, 20));
            screw.rotation.z = Math.PI / 2;
            screw.position.set(1.12, 0.0, 0.02);
            ctx.inkify(screw, { threshold: 50 });
            arm.add(screw);
            const knob = new T.Mesh(new T.CylinderGeometry(0.045, 0.045, 0.04, 24));
            knob.rotation.z = Math.PI / 2;
            knob.position.set(1.2, 0, 0.02);
            ctx.inkify(knob, { threshold: 50 });
            arm.add(knob);
            const idxLine = new T.BufferGeometry().setFromPoints([
              new T.Vector3(0.93, 0, -0.018),
              new T.Vector3(1.05, 0, -0.018),
            ]);
            const idx = new T.Line(idxLine, ctx.lineMaterial('red', { onTop: true }));
            idx.renderOrder = 5;
            arm.add(idx);
            // the index mirror: normal at NH_DEG − θ, so relative to the arm it is fixed
            const mirrorI = new T.Group();
            mirrorI.rotation.z = (NH_DEG + 60) * deg;
            arm.add(mirrorI);
            const glassI = new T.Mesh(new T.BoxGeometry(0.014, 0.22, 0.17));
            glassI.position.set(0, 0, 0.09);
            ctx.inkify(glassI, { fill: 'deep', edge: 'brass', outline: 0, threshold: 10 });
            mirrorI.add(glassI);
            const backI = new T.Mesh(new T.BoxGeometry(0.03, 0.24, 0.19));
            backI.position.set(0.024, 0, 0.09); // silvered face towards the sun (−normal side), backing behind it
            ctx.inkify(backI, { outline: 0 });
            mirrorI.add(backI);

            // ── horizon mirror: silvered half low, clear half high ──
            const mirrorH = new T.Group();
            mirrorH.position.set(H.x, H.y, 0.03);
            mirrorH.rotation.z = NH_DEG * deg;
            inst.add(mirrorH);
            const silver = new T.Mesh(new T.BoxGeometry(0.012, 0.14, 0.075));
            silver.position.z = 0.0375 + 0.01;
            ctx.inkify(silver, { fill: 'deep', outline: 0, threshold: 10 });
            mirrorH.add(silver);
            const clear = new T.Mesh(new T.BoxGeometry(0.012, 0.14, 0.075));
            clear.position.z = 0.1125 + 0.01;
            ctx.inkify(clear, { fill: 'none', edge: 'soft', outline: 0, threshold: 10 });
            mirrorH.add(clear);
            const hFrame = new T.Mesh(new T.BoxGeometry(0.03, 0.17, 0.02));
            hFrame.position.set(-0.025, 0, 0.0);
            ctx.inkify(hFrame, { outline: 0 });
            mirrorH.add(hFrame);

            // ── shades, folded flat beside the light paths ──
            const shade = (x: number, y: number, n: number, rotDeg: number) => {
              for (let i = 0; i < n; i++) {
                const p = new T.Mesh(new T.BoxGeometry(0.1, 0.1, 0.008));
                p.position.set(x, y, 0.035 + i * 0.012);
                p.rotation.z = (rotDeg + i * 9) * deg;
                ctx.inkify(p, { fill: i === 0 ? 'faint' : 'paper', outline: 0 });
                inst.add(p);
              }
            };
            shade(0.02, -0.24, 3, -28);
            shade(-0.42, -0.5, 2, 10);

            // ── telescope on a bracket from the right bar ──
            const tel = new T.Group();
            tel.position.set(0, H.y, 0.12);
            inst.add(tel);
            const tube = new T.Mesh(
              new T.CylinderGeometry(0.042, 0.042, TEL_TO - TEL_FROM - 0.12, 32),
            );
            tube.rotation.z = Math.PI / 2;
            tube.position.x = (TEL_FROM + TEL_TO) / 2;
            ctx.inkify(tube, { threshold: 50 });
            tel.add(tube);
            const obj = new T.Mesh(new T.CylinderGeometry(0.058, 0.058, 0.1, 32));
            obj.rotation.z = Math.PI / 2;
            obj.position.x = TEL_FROM + 0.05;
            ctx.inkify(obj, { threshold: 50 });
            tel.add(obj);
            const eyeP = new T.Mesh(new T.CylinderGeometry(0.034, 0.03, 0.08, 24));
            eyeP.rotation.z = Math.PI / 2;
            eyeP.position.x = TEL_TO - 0.02;
            ctx.inkify(eyeP, { threshold: 50 });
            tel.add(eyeP);
            const bracket = new T.Mesh(new T.BoxGeometry(0.05, 0.05, 0.1));
            bracket.position.set(0.27, H.y, 0.065);
            ctx.inkify(bracket, { outline: 0 });
            inst.add(bracket);

            // ── light: the sun, its path, the horizon ray ──
            const sunG = new T.Group();
            world.add(sunG);
            const disc = new T.Mesh(new T.CircleGeometry(0.06, 40));
            ctx.inkify(disc, { edge: 'red', outline: 1.4 });
            sunG.add(disc);
            const rays: number[] = [];
            for (let i = 0; i < 12; i++) {
              const a = (i / 12) * Math.PI * 2;
              rays.push(
                Math.cos(a) * 0.085,
                Math.sin(a) * 0.085,
                0,
                Math.cos(a) * 0.12,
                Math.sin(a) * 0.12,
                0,
              );
            }
            const rg = new T.BufferGeometry();
            rg.setAttribute('position', new T.Float32BufferAttribute(rays, 3));
            sunG.add(new T.LineSegments(rg, ctx.lineMaterial('red')));

            const pathGeo = new T.BufferGeometry();
            pathGeo.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(18), 3));
            const pathLine = new T.Line(pathGeo, ctx.lineMaterial('red', { onTop: true }));
            pathLine.renderOrder = 6;
            world.add(pathLine);
            const horGeo = new T.BufferGeometry();
            horGeo.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(12), 3));
            const horLine = new T.Line(
              horGeo,
              ctx.lineMaterial('sky', { dashed: true, dash: 0.035, gap: 0.025 }),
            );
            world.add(horLine);

            pickRef.current = { T, camera: ctx.camera, inst, parts: [armBar, clamp, screw, knob] };
            /** Write a path into a fixed-size buffer (repeat the last point), last segment on the axis. */
            const setPath = (
              geo: THREEns.BufferGeometry,
              pts: V2[],
              z: number,
              axisEnd: boolean,
            ) => {
              const pos = geo.attributes.position as THREEns.BufferAttribute;
              for (let i = 0; i < pos.count; i++) {
                const k = Math.min(i, pts.length - 1);
                const q = pts[k];
                pos.setXYZ(i, q.x, q.y, axisEnd && k === pts.length - 1 ? Z_AXIS : z);
              }
              pos.needsUpdate = true;
              geo.computeBoundingSphere();
            };
            return (_dt, t, p) => {
              texMats.forEach((m) => m.color.copy(inkFill.color));
              const tilt = ctx.reduced ? 0 : tiltAt(t);
              p.tilt.current = tilt;
              const v = sextantView(p.reading, tilt);
              inst.rotation.z = tilt * deg;
              arm.rotation.z = armAngle(p.reading) * deg;
              sunG.position.set(v.sun.x, v.sun.y, Z_SUN);
              // the last point is at the eye: there the two light paths meet on the axis
              setPath(pathGeo, v.red, Z_SUN, v.red.length === 5);
              setPath(horGeo, v.blue, Z_HOR, v.blue.length === 4);
              horLine.computeLineDistances();
              applyEye(p.eye.current, v);
              return false;
            };
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: 4,
            right: 4,
            width: 'clamp(76px, 21%, 132px)',
            pointerEvents: 'none',
          }}
        >
          <Eyepiece view={view} refs={eyeRef} />
          <div
            style={{
              textAlign: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.6rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: C.soft,
              marginTop: 2,
            }}
          >
            eyepiece
          </div>
        </div>
      </div>
      <Controls>
        <Slider
          label="index arm"
          value={reading}
          min={0}
          max={120}
          step={0.1}
          onChange={setReading}
          format={(v) => `${v.toFixed(1)}°`}
          width="12rem"
        />
        <span className="in-seg" role="group" aria-label="Tangent screw, fine adjustment">
          <button
            type="button"
            onClick={() => nudge(-0.1)}
            aria-label="Turn the arm back 0.1 degree"
          >
            −0.1
          </button>
          <button type="button" onClick={() => nudge(0.1)} aria-label="Turn the arm on 0.1 degree">
            +0.1
          </button>
        </span>
      </Controls>
      <Working label="Worked equation: arc reading and the sun's altitude">
        <Eq>
          reading <Sym>R</Sym> = 2 × mirror angle <Sym role="brass">θ</Sym>
        </Eq>
        <Eq>
          <V>{reading.toFixed(2)}°</V> = 2 × <V role="brass">{theta.toFixed(2)}°</V>
        </Eq>
        <Eq>
          sun over horizon <Sym role="soft">g</Sym> = <Sym role="red">h</Sym> − <Sym>R</Sym>
        </Eq>
        <Eq>
          <V role="soft">{sgn(view.gap, 2)}</V> = <V role="red">{SUN_ALT.toFixed(2)}°</V> −{' '}
          <V>{reading.toFixed(2)}°</V>
        </Eq>
        <Eq>
          {view.status === 'on' ? (
            <>
              they touch: <Sym role="red">h</Sym> = <Sym>R</Sym> ={' '}
              <V role="red">{fmtDeg(reading)}</V>
            </>
          ) : (
            <>
              sun {fmtDeg(Math.abs(view.gap))} {view.gap > 0 ? 'above' : 'below'} the horizon: move
              the arm {view.gap > 0 ? 'on' : 'back'}
            </>
          )}
        </Eq>
      </Working>
    </Plate>
  );
}
