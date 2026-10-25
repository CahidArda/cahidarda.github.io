/**
 * The 3D blueprint astrolabe (BlueprintCanvas), shared by the "assemble it" and "tell the time"
 * plates. Every engraved line comes from ./plateGeom.ts (rasterised onto canvas textures by
 * ./Prims.tsx), so the 3D faces, the SVG poster and the 2D close-ups are the same drawing.
 *
 * World units = plate units (equator radius 1). The instrument hangs in its own vertical plane
 * (x right, y up, the FRONT faces +z). Parts, front to back: rule (z +0.115), rete (+0.085),
 * plate (+0.06), mater (front face +0.045, back face −0.045), alidade (−0.075).
 * Turning the rete by sidereal time `lst` rotates it CLOCKWISE (rotation.z = −lst): the day
 * passing, as on a real northern astrolabe.
 */
import { useMemo } from 'react';
import type * as THREEns from 'three';
import { BlueprintCanvas, onThemeChange, readInk, type BlueprintCtx, type THREE } from '../shared';
import { drawPrims, PrimsSvg } from './Prims';
import {
  RC,
  RM,
  VANE_D,
  almucantarRuns,
  backPrims,
  backPsi,
  eclipticPoint,
  limbPrims,
  platePrims,
  retePrims,
  sight,
} from './plateGeom';
import { PHI } from './astro';

export type ModelParams = {
  /** 0 = front faces you, 1 = back faces you (eased). */
  face: 0 | 1;
  /** Assembly: how many parts are in (1 mater .. 5 rule). */
  parts: number;
  /** Rete turn (sidereal time, degrees). */
  lst: number;
  /** Alidade setting (degrees above horizontal, on the back). */
  ali: number;
  /** Rule angle on the front, as an hour angle (degrees, + west); null = hidden. */
  ruleH: number | null;
  /** Draw the sun's rays (sun altitude, degrees); null = none. */
  rays: number | null;
  /** The sun's ecliptic longitude: a brass mark on the rete. null = none. */
  sunLam: number | null;
  /** Highlight one almucantar (altitude, degrees) on the plate, in sky blue. null = none. */
  almH: number | null;
  /** Bumped to start the hang-and-settle swing. */
  swing: number;
  phi?: number;
};

const Z = { rule: 0.115, rete: 0.085, plate: 0.06, front: 0.045, back: -0.045, ali: -0.075 };
const TEX = 1024;

export function AstrolabePoster({ p }: { p: ModelParams }) {
  const W = 400;
  const H = 400;
  const k = (W / 2 - 10) / (RM * 1.12);
  const f = { cx: W / 2, cy: H / 2 + 16, k };
  const phi = p.phi ?? PHI;
  if (p.face === 1) {
    const s = sight(p.ali, p.rays ?? p.ali);
    const tip = { x: f.cx + s.tip.u * k, y: f.cy - s.tip.v * k };
    const tail = { x: f.cx + s.tail.u * k, y: f.cy - s.tail.v * k };
    return (
      <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
        <circle cx={f.cx} cy={f.cy} r={RM * k} style={{ fill: 'var(--in-paper)', stroke: 'var(--in-ink)', strokeWidth: 1.3 }} />
        <PrimsSvg prims={backPrims()} f={f} scaleW={0.8} />
        <line x1={tip.x} y1={tip.y} x2={tail.x} y2={tail.y} style={{ stroke: 'var(--in-ink)', strokeWidth: 5, strokeLinecap: 'round', opacity: 0.85 }} />
      </svg>
    );
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <circle cx={f.cx} cy={f.cy} r={RM * k} style={{ fill: 'var(--in-paper)', stroke: 'var(--in-ink)', strokeWidth: 1.3 }} />
      <path
        d={`M ${f.cx - 0.32 * k} ${f.cy - RM * k + 4} L ${f.cx - 0.2 * k} ${f.cy - (RM + 0.22) * k} L ${f.cx + 0.2 * k} ${f.cy - (RM + 0.22) * k} L ${f.cx + 0.32 * k} ${f.cy - RM * k + 4} Z`}
        style={{ fill: 'var(--in-paper)', stroke: 'var(--in-ink)', strokeWidth: 1.1 }}
      />
      <circle cx={f.cx} cy={f.cy - (RM + 0.36) * k} r={0.13 * k} style={{ fill: 'none', stroke: 'var(--in-ink)', strokeWidth: 1.4 }} />
      <PrimsSvg prims={limbPrims()} f={f} scaleW={0.8} />
      {p.parts >= 2 && <PrimsSvg prims={platePrims(phi)} f={f} scaleW={0.8} />}
      {p.parts >= 3 && <PrimsSvg prims={retePrims(p.lst)} f={f} scaleW={0.9} />}
      {p.parts >= 3 && p.sunLam !== null && (() => {
        const s = eclipticPoint(p.sunLam!, p.lst);
        return <circle cx={f.cx + s.u * k} cy={f.cy - s.v * k} r={5} style={{ fill: 'var(--in-brass)' }} />;
      })()}
    </svg>
  );
}

/** Build the instrument in a BlueprintCanvas scene. */
function buildModel(T: THREE, ctx: BlueprintCtx, first: ModelParams) {
  const phi = first.phi ?? PHI;
  const inst = new T.Group(); // swings on the ring; turns to show front or back
  const hang = new T.Group(); // pivot at the ring; the instrument's centre sits at y = −0.3
  const ringY = RM + 0.53;
  hang.position.y = ringY - 0.3;
  inst.position.y = -ringY;
  hang.add(inst);
  ctx.root.add(hang);

  // ── engraved faces: canvas textures (theme-aware) ──
  const faces: { canvas: HTMLCanvasElement; tex: THREEns.CanvasTexture; draw: (ink: ReturnType<typeof readInk>) => void }[] = [];
  const face = (radius: number, z: number, flip: boolean, draw: (c: CanvasRenderingContext2D, ink: ReturnType<typeof readInk>) => void, transparent = false) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = TEX;
    const tex = new T.CanvasTexture(canvas);
    tex.colorSpace = T.SRGBColorSpace;
    tex.anisotropy = 4;
    const mat = new T.MeshBasicMaterial({ map: tex, transparent, depthWrite: !transparent, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
    const mesh = new T.Mesh(new T.CircleGeometry(radius, 128), mat);
    mesh.position.z = z;
    if (flip) mesh.rotation.y = Math.PI;
    const f = { canvas, tex, draw: (ink: ReturnType<typeof readInk>) => {
      const c = canvas.getContext('2d')!;
      draw(c, ink);
      tex.needsUpdate = true;
    } };
    faces.push(f);
    return mesh;
  };
  const paint = () => {
    const ink = readInk(document.documentElement);
    faces.forEach((f) => f.draw(ink));
    ctx.invalidate();
  };

  // mater: body + rim (inkified), front limb, back scales
  const body = new T.Mesh(new T.CylinderGeometry(RM, RM, 0.09, 128, 1));
  body.rotation.x = Math.PI / 2;
  ctx.inkify(body, { threshold: 30, outline: 1.4 });
  inst.add(body);
  const frontFace = face(RM, Z.front + 0.0005, false, (c, ink) => drawPrims(c, limbPrims(), { size: TEX, half: RM, bg: 'paper', ink }));
  inst.add(frontFace);
  const backFace = face(RM, Z.back - 0.0005, true, (c, ink) => drawPrims(c, backPrims(), { size: TEX, half: RM, bg: 'paper', ink }));
  inst.add(backFace);

  // throne (kursī) and ring
  const sh = new T.Shape();
  sh.moveTo(-0.34, 0);
  sh.lineTo(-0.22, 0.24);
  sh.quadraticCurveTo(0, 0.34, 0.22, 0.24);
  sh.lineTo(0.34, 0);
  sh.lineTo(-0.34, 0);
  const throne = new T.Mesh(new T.ExtrudeGeometry(sh, { depth: 0.09, bevelEnabled: false }));
  throne.position.set(0, RM - 0.06, -0.045);
  ctx.inkify(throne, { threshold: 20 });
  inst.add(throne);
  const shackle = new T.Mesh(new T.TorusGeometry(0.07, 0.018, 10, 40));
  shackle.position.set(0, RM + 0.33, 0);
  shackle.rotation.y = Math.PI / 2;
  ctx.inkify(shackle, { threshold: 40 });
  inst.add(shackle);
  const ring = new T.Mesh(new T.TorusGeometry(0.15, 0.022, 10, 56));
  ring.position.set(0, RM + 0.53, 0);
  ctx.inkify(ring, { threshold: 40 });
  inst.add(ring);

  // plate (tympan) for φ
  const plate = new T.Group();
  const plateFace = face(RC, 0.0, false, (c, ink) => drawPrims(c, platePrims(phi), { size: TEX, half: RC, bg: 'paper', ink }));
  const plateDisc = new T.Mesh(new T.CylinderGeometry(RC, RC, 0.012, 128));
  plateDisc.rotation.x = Math.PI / 2;
  plateDisc.position.z = -0.0065;
  ctx.inkify(plateDisc, { threshold: 30, outline: 0.9 });
  plate.add(plateDisc, plateFace);
  inst.add(plate);
  // a highlighted almucantar (the measured altitude), redrawn when it changes
  const almLine = new T.Line(new T.BufferGeometry(), ctx.lineMaterial('sky'));
  almLine.position.z = 0.002;
  almLine.renderOrder = 3;
  plate.add(almLine);
  let almFor: number | null = null;

  // rete: transparent texture (the pierced "spider"), drawn at lst = 0 and turned
  const rete = new T.Group();
  const reteFace = face(RC, 0.0, false, (c, ink) => drawPrims(c, retePrims(0, { signs: true }), { size: TEX, half: RC, bg: null, ink, lineScale: TEX / 520 }), true);
  rete.add(reteFace);
  // the sun's place: a brass bead on the ecliptic (moves with the rete)
  const sunMark = new T.Mesh(new T.CircleGeometry(0.035, 24), ctx.fillMaterial('brass'));
  sunMark.position.z = 0.004;
  rete.add(sunMark);
  inst.add(rete);

  // pin and horse
  const pin = new T.Mesh(new T.CylinderGeometry(0.035, 0.035, 0.3, 20));
  pin.rotation.x = Math.PI / 2;
  ctx.inkify(pin, { threshold: 40 });
  const horse = new T.Mesh(new T.BoxGeometry(0.05, 0.16, 0.03));
  horse.position.z = 0.15;
  ctx.inkify(horse, { threshold: 40 });
  const pinG = new T.Group();
  pinG.add(pin, horse);
  inst.add(pinG);

  // rule (front): a thin bar through the pin, its edge on the hour
  const rule = new T.Group();
  const ruleBar = new T.Mesh(new T.BoxGeometry(0.07, RC * 2.12, 0.012));
  ruleBar.position.x = 0.035; // the fiducial edge runs through the centre
  ctx.inkify(ruleBar, { threshold: 30, outline: 0.8 });
  const ruleEdge = new T.Line(
    new T.BufferGeometry().setFromPoints([new T.Vector3(0, -RC * 1.06, 0.007), new T.Vector3(0, RC * 1.06, 0.007)]),
    ctx.lineMaterial('red'),
  );
  rule.add(ruleBar, ruleEdge);
  rule.position.z = Z.rule;
  inst.add(rule);

  // alidade (back) with two vanes
  const ali = new T.Group();
  const aliBar = new T.Mesh(new T.BoxGeometry(RM * 1.9, 0.08, 0.014));
  ctx.inkify(aliBar, { threshold: 30, outline: 0.8 });
  ali.add(aliBar);
  for (const s of [1, -1]) {
    const vane = new T.Mesh(new T.BoxGeometry(0.02, 0.08, 0.14));
    vane.position.set(s * VANE_D, 0, -0.07);
    ctx.inkify(vane, { threshold: 30 });
    ali.add(vane);
  }
  ali.position.z = Z.ali;
  inst.add(ali);

  // sun rays (world-fixed direction; lines in the instrument plane, drawn on top)
  const rayMat = ctx.lineMaterial('brass', { onTop: true });
  const rayMatOn = ctx.lineMaterial('red', { onTop: true });
  const rays: THREEns.Line[] = [];
  for (let i = 0; i < 3; i++) {
    const l = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3()]), rayMat);
    l.renderOrder = 5;
    rays.push(l);
    inst.add(l);
  }

  paint();
  onThemeChange(() => requestAnimationFrame(paint)); // canvas faces re-read the ink on a theme flip

  // eased state
  let faceY = first.face * Math.PI;
  const placed: number[] = [1, 1, 1, 1, 1].map((_, i) => (i < first.parts ? 0 : 1)); // 0 = in place, 1 = out
  let swingT = 99;
  let lastSwing = first.swing;
  const out = (k: number) => 0.9 + k * 0.25;

  return (dt: number, _t: number, p: ModelParams) => {
    let more = false;
    const ease = (cur: number, target: number, rate = 6) => {
      const d = target - cur;
      if (Math.abs(d) < 1e-3) return target;
      more = true;
      return cur + d * Math.min(1, dt * rate || 1);
    };
    faceY = ctx.reduced ? p.face * Math.PI : ease(faceY, p.face * Math.PI, 5);
    inst.rotation.y = faceY;
    // assembly
    const parts = [plate, rete, pinG, rule];
    for (let i = 0; i < 4; i++) {
      const target = i + 2 <= p.parts ? 0 : 1;
      placed[i] = ctx.reduced ? target : ease(placed[i], target, 5);
      const g = parts[i];
      g.visible = placed[i] < 0.995;
      const zIn = i === 0 ? Z.plate : i === 1 ? Z.rete : i === 2 ? 0 : Z.rule;
      g.position.z = zIn + placed[i] * out(i);
    }
    rule.visible = rule.visible && p.ruleH !== null;
    // the measured almucantar
    if (p.almH !== almFor) {
      almFor = p.almH;
      if (p.almH !== null) {
        const pts = almucantarRuns(p.almH, phi).flat().map((q) => new T.Vector3(q.u, q.v, 0));
        almLine.geometry.dispose();
        almLine.geometry = new T.BufferGeometry().setFromPoints(pts);
        almLine.visible = true;
      } else almLine.visible = false;
    }
    // rete and its sun mark
    rete.rotation.z = (-p.lst * Math.PI) / 180;
    if (p.sunLam !== null) {
      const s = eclipticPoint(p.sunLam, 0);
      sunMark.position.set(s.u, s.v, 0.004);
      sunMark.visible = true;
    } else sunMark.visible = false;
    // rule: its edge along the hour angle H (screen angle 90° − H)
    if (p.ruleH !== null) rule.rotation.z = (-p.ruleH * Math.PI) / 180;
    // alidade on the back: as seen from behind, its tip at ψ = 180° − a, so in world x is mirrored
    const psi = backPsi(p.ali);
    ali.rotation.z = Math.PI - (psi * Math.PI) / 180;
    // rays
    if (p.rays !== null && p.face === 1) {
      const s = sight(p.ali, p.rays);
      const W = (u: number, v: number, z: number) => new T.Vector3(-u, v, z);
      const zr = Z.ali - 0.07;
      const L = 1.4;
      const offs = [-0.1, 0, 0.1];
      const nrm = { u: -s.dir.v, v: s.dir.u };
      rays.forEach((r, i) => {
        const o = offs[i];
        const a = { u: s.tip.u + nrm.u * o, v: s.tip.v + nrm.v * o };
        const start = { u: a.u - s.dir.u * L, v: a.v - s.dir.v * L };
        const end = i === 1 ? s.hit : { u: a.u, v: a.v };
        r.geometry.setFromPoints([W(start.u, start.v, zr), W(end.u, end.v, zr)]);
        r.material = i === 1 && s.on ? rayMatOn : rayMat;
        r.visible = true;
      });
    } else rays.forEach((r) => (r.visible = false));
    // hang and settle
    if (p.swing !== lastSwing) {
      lastSwing = p.swing;
      swingT = 0;
    }
    if (!ctx.reduced && swingT < 4) {
      swingT += dt;
      hang.rotation.z = 0.2 * Math.exp(-1.1 * swingT) * Math.cos(swingT * 3.2);
      more = true;
    } else hang.rotation.z = 0;
    return more;
  };
}

export function AstrolabeModel({ p, label, aspect = 1, fit = 2.35 }: { p: ModelParams; label: string; aspect?: number; fit?: number }) {
  const params = useMemo(() => p, [p.face, p.parts, p.lst, p.ali, p.ruleH, p.rays, p.sunLam, p.almH, p.swing, p.phi]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <BlueprintCanvas<ModelParams>
      aspect={aspect}
      fit={fit}
      yaw={-0.32}
      pitch={0.12}
      pitchLimit={0.6}
      params={params}
      poster={<AstrolabePoster p={p} />}
      label={label}
      build={(T, ctx) => buildModel(T, ctx, p)}
    />
  );
}
