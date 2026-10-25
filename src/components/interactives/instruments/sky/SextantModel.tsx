/**
 * Plate 2 (wide): a marine sextant as a blueprint. The index arm carries the index mirror; the
 * horizon mirror is half silvered; the telescope looks at both. Move the arm (slider or the
 * fine tangent screw) and watch the real ray from the sun: it reaches the eyepiece only when
 * the arc reads the sun's altitude, which is twice the angle the mirror turned. The frame
 * rolls gently, like a ship: both images in the eyepiece move together.
 */
import { useMemo, useRef, useState } from 'react';
import type * as THREEns from 'three';
import { ArcScale, BlueprintCanvas, C, Controls, InkText, Plate, Readout, Slider, polar } from '../shared';
import { ARC_FROM, ARC_TO, EYE, H, NH, NH_DEG, TEL_FROM, TEL_TO, armAngle, optics, pol, reflect, rot, type V2 } from './sextantGeom';

/** The noon sun over Istanbul on 25 October 2026 (USNO approximate solar coordinates). */
export const SUN_ALT = 36.82;
const ROLL_AMP = 1.6; // degrees
const ROLL_PERIOD = 5.5; // seconds
const Z_SUN = 0.085; // height of the reflected path above the frame (silvered half)
const Z_HOR = 0.155; // the direct horizon ray (clear half)

const fmtDeg = (v: number) => {
  const d = Math.floor(v + 1e-9);
  const m = Math.round((v - d) * 60);
  return m === 60 ? `${d + 1}°00′` : `${d}°${String(m).padStart(2, '0')}′`;
};

/** Trace the actual sunlight through both mirrors, in the frame plane, world coordinates. */
function trace(reading: number, roll: number) {
  const s: V2 = { x: -Math.cos((SUN_ALT * Math.PI) / 180), y: Math.sin((SUN_ALT * Math.PI) / 180) };
  const d1 = { x: -s.x, y: -s.y };
  const { nI } = optics(reading);
  const nIw = rot(nI, roll);
  const nHw = rot(NH, roll);
  const Hw = rot(H, roll);
  const start = { x: s.x * 1.25, y: s.y * 1.25 };
  const d2 = reflect(d1, nIw);
  // O + a d2 = Hw + b t, t = mirror direction
  const t = { x: -nHw.y, y: nHw.x };
  const det = d2.x * -t.y - d2.y * -t.x;
  const pts: V2[] = [start, { x: 0, y: 0 }];
  let hit = false;
  if (Math.abs(det) > 1e-9) {
    const a = (Hw.x * -t.y - Hw.y * -t.x) / det;
    const b = (d2.x * Hw.y - d2.y * Hw.x) / det;
    if (a > 0 && Math.abs(b) <= 0.07) {
      const P = { x: a * d2.x, y: a * d2.y };
      const d3 = reflect(d2, nHw);
      pts.push(P, { x: P.x + d3.x * 0.86, y: P.y + d3.y * 0.86 });
      hit = true;
    }
  }
  if (!hit) pts.push({ x: d2.x * 0.95, y: d2.y * 0.95 });
  return { pts, sun: start, Hw };
}

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
  const { pts } = trace(reading, 0);
  const path = pts.map((p, i) => `${i ? 'L' : 'M'} ${P(p).x} ${P(p).y}`).join(' ');
  const bar = (deg: number) => {
    const p = polar(ox, oy, S * 0.88, deg);
    return `M ${ox} ${oy} L ${p.x} ${p.y}`;
  };
  const hE = P(H);
  const tA = P({ x: TEL_FROM, y: H.y });
  const tB = P({ x: TEL_TO, y: H.y });
  const arm = polar(ox, oy, S * 1.06, a);
  const nAng = NH_DEG; // horizon mirror normal angle
  const mHalf = 0.07 * S;
  const mDir = ((nAng + 90) * Math.PI) / 180;
  const iDir = ((NH_DEG - reading / 2 + 90) * Math.PI) / 180;
  const iHalf = 0.11 * S;
  return (
    <svg viewBox={`0 0 ${W} ${Hh}`}>
      <ArcScale cx={ox} cy={oy} r={S * 0.98} from={-120} to={-60} major={5} minor={1} labelEvery={10} value={(x) => -2 * (x + 60)} inward tick={9} size={10} />
      <path d={`${bar(-120)} ${bar(-60)}`} style={{ ...ink, strokeWidth: 6, stroke: C.soft, opacity: 0.35 }} />
      <path d={`${bar(-120)} ${bar(-60)}`} style={ink} />
      <path d={`M ${ox} ${oy} L ${arm.x} ${arm.y}`} style={{ ...ink, strokeWidth: 2.2 }} />
      <circle cx={ox} cy={oy} r={7} style={{ ...ink, fill: C.paper }} />
      <line x1={ox - Math.cos(iDir) * iHalf} y1={oy + Math.sin(iDir) * iHalf} x2={ox + Math.cos(iDir) * iHalf} y2={oy - Math.sin(iDir) * iHalf} style={{ ...ink, strokeWidth: 3.2 }} />
      <line x1={hE.x - Math.cos(mDir) * mHalf} y1={hE.y + Math.sin(mDir) * mHalf} x2={hE.x + Math.cos(mDir) * mHalf} y2={hE.y - Math.sin(mDir) * mHalf} style={{ ...ink, strokeWidth: 3.2 }} />
      <rect x={tA.x} y={tA.y - 9} width={tB.x - tA.x} height={18} style={{ ...ink, fill: C.paper }} />
      <line x1={8} y1={hE.y} x2={tA.x} y2={hE.y} style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '5 4' }} />
      <path d={path} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.4 }} />
      <circle cx={P(pts[0]).x} cy={P(pts[0]).y} r={11} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.2 }} />
      <InkText x={16} y={hE.y - 8} size={13} role="soft">horizon</InkText>
      <InkText x={P(pts[0]).x + 16} y={P(pts[0]).y + 4} size={13} role="soft">sun</InkText>
    </svg>
  );
}

/* ───────────── the eyepiece: horizon (clear half) and sun (silvered half) ───────────── */

function Eyepiece({ reading, gRef }: { reading: number; gRef: React.Ref<SVGGElement> }) {
  const R = 50;
  const k = 15; // px per degree
  const off = (SUN_ALT - reading) * k;
  const shown = Math.max(-R - 20, Math.min(R + 20, -off));
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
        <g ref={gRef}>
          {/* horizon: seen directly through the clear half, and faintly in the mirror half */}
          <rect x={-R} y={0} width={R} height={R * 2} style={{ fill: C.sky, opacity: 0.22 }} />
          <line x1={-R} y1={0} x2={0} y2={0} style={{ stroke: C.ink, strokeWidth: 1.2 }} />
          <circle cx={R * 0.45} cy={shown} r={4.5} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.6 }} />
        </g>
      </g>
      {Math.abs(off) > R - 6 && (
        <path
          d={off > 0 ? `M ${R * 0.45 - 6} ${-R + 16} L ${R * 0.45} ${-R + 8} L ${R * 0.45 + 6} ${-R + 16}` : `M ${R * 0.45 - 6} ${R - 16} L ${R * 0.45} ${R - 8} L ${R * 0.45 + 6} ${R - 16}`}
          style={{ fill: 'none', stroke: C.red, strokeWidth: 1.6 }}
        />
      )}
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

type Params = { reading: number; eye: React.RefObject<SVGGElement | null> };
type Pick = {
  T: typeof THREEns;
  camera: THREEns.Camera;
  inst: THREEns.Object3D;
  parts: THREEns.Object3D[];
};

export default function SextantModel() {
  const [reading, setReading] = useState(20);
  const eyeRef = useRef<SVGGElement | null>(null);
  const pickRef = useRef<Pick | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<number | null>(null);
  /** Pointer → ray into the scene, using the canvas's own camera. */
  const ray = (e: React.PointerEvent) => {
    const pk = pickRef.current;
    const cv = wrapRef.current?.querySelector('canvas');
    if (!pk || !cv) return null;
    const r = cv.getBoundingClientRect();
    const ndc = new pk.T.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    const rc = new pk.T.Raycaster();
    rc.setFromCamera(ndc, pk.camera);
    return { rc, pk };
  };
  const hitsArm = (e: React.PointerEvent) => {
    const q = ray(e);
    return !!q && q.rc.intersectObjects(q.pk.parts, true).length > 0;
  };
  const dragTo = (e: React.PointerEvent) => {
    const q = ray(e);
    if (!q) return;
    const { T } = q.pk;
    const m = q.pk.inst.matrixWorld;
    const n = new T.Vector3(0, 0, 1).transformDirection(m);
    const o = new T.Vector3(0, 0, 0.05).applyMatrix4(m);
    const plane = new T.Plane().setFromNormalAndCoplanarPoint(n, o);
    const hit = q.rc.ray.intersectPlane(plane, new T.Vector3());
    if (!hit) return;
    const loc = q.pk.inst.worldToLocal(hit);
    const ang = (Math.atan2(loc.y, loc.x) * 180) / Math.PI;
    const r = -2 * (ang + 60);
    setReading(Math.max(0, Math.min(120, Math.round(r * 10) / 10)));
  };
  const params = useMemo<Params>(() => ({ reading, eye: eyeRef }), [reading]);
  const nudge = (d: number) => setReading((r) => Math.max(0, Math.min(120, Math.round((r + d) * 10) / 10)));
  const err = reading - SUN_ALT;
  const onIt = Math.abs(err) < 0.12;

  return (
    <Plate
      n={2}
      wide
      title="The sextant, set to the sun"
      kicker="drag the index arm"
      caption={
        <>
          The red line is a real ray from the noon sun over Istanbul on 25 October 2026. It bounces off
          the index mirror on the arm, then off the silvered half of the horizon mirror, and into
          the telescope only when the arc reads the sun&apos;s altitude. The arm turns the mirror by
          half that number. In the eyepiece (top right) the horizon is seen straight through the
          clear half; the frame rolls like a ship, and both images move together.
        </>
      }
    >
      <div
        ref={wrapRef}
        style={{ position: 'relative' }}
        onPointerDownCapture={(e) => {
          if (e.button !== 0 || !hitsArm(e)) return;
          e.stopPropagation();
          e.preventDefault();
          dragRef.current = e.pointerId;
          e.currentTarget.setPointerCapture(e.pointerId);
          dragTo(e);
        }}
        onPointerMoveCapture={(e) => {
          if (dragRef.current === e.pointerId) {
            e.stopPropagation();
            dragTo(e);
            return;
          }
          if (e.pointerType === 'mouse') {
            const cv = wrapRef.current?.querySelector('canvas');
            if (cv) cv.style.cursor = hitsArm(e) ? 'pointer' : '';
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
            world.position.set(0.3, 0.12, 0);
            ctx.root.add(world);
            const inst = new T.Group(); // everything that rolls with the ship
            world.add(inst);
            const deg = Math.PI / 180;
            const ext = (shape: THREEns.Shape, depth: number, z = 0) => {
              const g = new T.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments: 48 });
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
            const idxLine = new T.BufferGeometry().setFromPoints([new T.Vector3(0.93, 0, -0.018), new T.Vector3(1.05, 0, -0.018)]);
            const idx = new T.Line(idxLine, ctx.lineMaterial('red', { onTop: true }));
            idx.renderOrder = 5;
            arm.add(idx);
            // the index mirror: normal at NH_DEG − θ, so relative to the arm it is fixed
            const mirrorI = new T.Group();
            mirrorI.rotation.z = (NH_DEG + 60) * deg;
            arm.add(mirrorI);
            const glassI = new T.Mesh(new T.BoxGeometry(0.014, 0.22, 0.17));
            glassI.position.set(0, 0, 0.09);
            ctx.inkify(glassI, { fill: 'deep', outline: 0, threshold: 10 });
            mirrorI.add(glassI);
            const backI = new T.Mesh(new T.BoxGeometry(0.03, 0.24, 0.19));
            backI.position.set(-0.024, 0, 0.09);
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
            const tube = new T.Mesh(new T.CylinderGeometry(0.042, 0.042, TEL_TO - TEL_FROM - 0.12, 32));
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
              rays.push(Math.cos(a) * 0.085, Math.sin(a) * 0.085, 0, Math.cos(a) * 0.12, Math.sin(a) * 0.12, 0);
            }
            const rg = new T.BufferGeometry();
            rg.setAttribute('position', new T.Float32BufferAttribute(rays, 3));
            sunG.add(new T.LineSegments(rg, ctx.lineMaterial('red')));

            const pathGeo = new T.BufferGeometry();
            pathGeo.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(15), 3));
            const pathLine = new T.Line(pathGeo, ctx.lineMaterial('red', { onTop: true }));
            pathLine.renderOrder = 6;
            world.add(pathLine);
            const horGeo = new T.BufferGeometry();
            horGeo.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(6), 3));
            const horLine = new T.Line(horGeo, ctx.lineMaterial('sky', { dashed: true, dash: 0.035, gap: 0.025 }));
            world.add(horLine);

            pickRef.current = { T, camera: ctx.camera, inst, parts: [armBar, clamp, screw, knob, glassI, backI] };
            return (_dt, t, p) => {
              texMats.forEach((m) => m.color.copy(inkFill.color));
              const roll = ctx.reduced ? 0 : ROLL_AMP * Math.sin((2 * Math.PI * t) / ROLL_PERIOD);
              inst.rotation.z = roll * deg;
              arm.rotation.z = armAngle(p.reading) * deg;
              const tr = trace(p.reading, roll);
              sunG.position.set(tr.sun.x, tr.sun.y, Z_SUN);
              const pos = pathGeo.attributes.position as THREEns.BufferAttribute;
              for (let i = 0; i < 5; i++) {
                const q = tr.pts[Math.min(i, tr.pts.length - 1)];
                pos.setXYZ(i, q.x, q.y, Z_SUN);
              }
              pos.needsUpdate = true;
              pathGeo.computeBoundingSphere();
              const eyeW = rot(EYE, roll);
              const hp = horGeo.attributes.position as THREEns.BufferAttribute;
              hp.setXYZ(0, -1.45, tr.Hw.y, Z_HOR);
              hp.setXYZ(1, eyeW.x, tr.Hw.y, Z_HOR);
              hp.needsUpdate = true;
              horLine.computeLineDistances();
              const g = p.eye.current;
              if (g) g.setAttribute('transform', `translate(0 ${(-roll * 15).toFixed(2)})`);
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
          <Eyepiece reading={reading} gRef={eyeRef} />
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
          <button type="button" onClick={() => nudge(-0.1)} aria-label="Turn the arm back 0.1 degree">
            −0.1
          </button>
          <button type="button" onClick={() => nudge(0.1)} aria-label="Turn the arm on 0.1 degree">
            +0.1
          </button>
        </span>
        <Readout label="mirror turned">{(reading / 2).toFixed(2)}°</Readout>
        <Readout label="sun">
          <span style={{ color: onIt ? C.red : C.ink }}>
            {onIt ? `on the horizon: ${fmtDeg(SUN_ALT)}` : err > 0 ? 'below the horizon' : 'above the horizon'}
          </span>
        </Readout>
      </Controls>
    </Plate>
  );
}
