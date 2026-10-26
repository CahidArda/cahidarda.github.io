/**
 * The pendulum clock's movement in 3D: a 30-tooth escape wheel with pointed, raked teeth, a
 * Graham DEADBEAT anchor whose pallets span 7½ teeth, and a pendulum hung from the anchor's
 * arbor. The wheel is driven by a weight and moves only when a pallet lets it: a tooth tip rests
 * on a locking face (the wheel stands dead still while the anchor slides on), slides over the
 * impulse face (pushing the pendulum), drops, and lands on the other pallet. Half a tooth per
 * swing. Every contact comes from ./deadbeat.ts (exact, swept per frame by deadbeat.sweep.ts).
 */
import { useMemo } from 'react';
import { BlueprintCanvas, C, type BlueprintCtx, type THREE } from '../shared';
import { AMP, ARMS, PALLETS, PIVOT, R_ROOT, initialState, palletsAt, stepMovement, wheelOutline, type Pt } from './deadbeat';
import { SECONDS_L, period, rodWorld } from './pendulumGeom';

const BOB_R = 0.3;
const FIT = 3.0;
const ASPECT = 0.8;
const rotPt = (p: Pt, c: Pt, a: number): Pt => {
  const k = Math.cos(a);
  const n = Math.sin(a);
  return [c[0] + (p[0] - c[0]) * k - (p[1] - c[1]) * n, c[1] + (p[0] - c[0]) * n + (p[1] - c[1]) * k];
};

/* the anchor body (arms) lives in ./deadbeat.ts, where the sweep checks it clears the teeth */
export { ARMS };


/* ───────────────────────── the SVG poster (front elevation) ───────────────────────── */

function Poster() {
  const W = 400;
  const H = 500;
  const k0 = Math.min(1, (2 * FIT * 0.93) / (PIVOT[1] + 0.45 - (PIVOT[1] - rodWorld(SECONDS_L) - BOB_R - 0.1)));
  const s = (400 / (2 * ASPECT * FIT)) * k0; // px per world unit, as the live camera
  const ox = W / 2;
  const oy = H / 2 + ((PIVOT[1] + 0.45 + PIVOT[1] - rodWorld(SECONDS_L) - BOB_R - 0.1) / 2) * s; // wheel centre, as the live model
  const P = (p: Pt) => `${(ox + p[0] * s).toFixed(1)},${(oy - p[1] * s).toFixed(1)}`;
  const poly = (pts: Pt[]) => pts.map(P).join(' ');
  const L = rodWorld(SECONDS_L);
  const bobY = PIVOT[1] - L;
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1 };
  const st0 = initialState();
  const pals = palletsAt(AMP * Math.sin(st0.phase));
  const swing = AMP * Math.sin(st0.phase);
  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      <polygon points={poly(wheelOutline(st0.w))} style={ink} />
      <circle cx={ox} cy={oy} r={R_ROOT * s * 0.85} style={{ ...ink, strokeWidth: 0.7 }} />
      <circle cx={ox} cy={oy} r={R_ROOT * s * 0.35} style={{ ...ink, strokeWidth: 0.7 }} />
      {[0, 1, 2, 3].map((k) => {
        const a = (k * Math.PI) / 2 + Math.PI / 4;
        return (
          <line
            key={k}
            x1={ox + Math.cos(a) * R_ROOT * s * 0.35}
            y1={oy - Math.sin(a) * R_ROOT * s * 0.35}
            x2={ox + Math.cos(a) * R_ROOT * s * 0.85}
            y2={oy - Math.sin(a) * R_ROOT * s * 0.85}
            style={{ ...ink, strokeWidth: 0.7 }}
          />
        );
      })}
      <circle cx={ox} cy={oy} r={5} style={ink} />
      {ARMS.map((a, i) => (
        <polygon key={i} points={poly(a.map((q) => rotPt(q, PIVOT, swing)))} style={{ ...ink, fill: C.paper }} />
      ))}
      {pals.map((p, i) => (
        <polygon key={i} points={poly(p)} style={{ ...ink, fill: C.paper }} />
      ))}
      <circle cx={ox} cy={oy - PIVOT[1] * s} r={9} style={{ ...ink, fill: C.paper }} />
      <g transform={`rotate(${(-swing * 180) / Math.PI} ${ox} ${oy - PIVOT[1] * s})`}>
        <line
          x1={ox}
          y1={oy - PIVOT[1] * s}
          x2={ox}
          y2={oy - (bobY + BOB_R) * s}
          style={{ ...ink, stroke: C.brass, strokeWidth: 1.4 }}
        />
        <circle cx={ox} cy={oy - bobY * s} r={BOB_R * s} style={{ ...ink, stroke: C.brass, fill: C.paper }} />
        <circle cx={ox} cy={oy - bobY * s} r={BOB_R * s * 0.55} style={{ ...ink, stroke: C.brass, strokeWidth: 0.6 }} />
      </g>
      {(() => {
        const rA = (L + BOB_R + 0.12) * s;
        const cy0 = oy - PIVOT[1] * s;
        const t = AMP * 1.6;
        return (
          <path
            d={`M ${ox - Math.sin(t) * rA} ${cy0 + Math.cos(t) * rA} A ${rA} ${rA} 0 0 0 ${ox + Math.sin(t) * rA} ${cy0 + Math.cos(t) * rA}`}
            style={{ fill: 'none', stroke: C.red, strokeWidth: 1 }}
          />
        );
      })()}
      {/* drive cord and weight, behind the wheel */}
      <line x1={ox + 0.56 * s} y1={oy} x2={ox + 0.56 * s} y2={oy + 1.5 * s} style={{ ...ink, strokeWidth: 0.8 }} />
      <rect
        x={ox + (0.56 - 0.085) * s}
        y={oy + 1.5 * s}
        width={0.17 * s}
        height={0.5 * s}
        style={{ ...ink, fill: C.paper }}
      />
    </svg>
  );
}

/* ───────────────────────── the canvas ───────────────────────── */

type Params = { L: number; tilt: number };

/** The rotatable 3D movement. `tilt` (radians): the case off level (the pendulum hangs plumb). */
export function Movement3D({ L, tilt = 0 }: { L: number; tilt?: number }) {
  const params = useMemo<Params>(() => ({ L, tilt }), [L, tilt]);
  return (
    <BlueprintCanvas<Params>
      aspect={ASPECT}
      fit={FIT}
      animate
      yaw={-0.42}
      pitch={0.12}
      pitchLimit={0.9}
      params={params}
      label="Rotatable line drawing of a pendulum clock's deadbeat escapement: a 30-tooth escape wheel, an anchor with two pallets, and a pendulum"
      poster={<Poster />}
      build={(T3, ctx) => buildMovement(T3, ctx)}
    />
  );
}

/* ───────────────────────── the 3D model ───────────────────────── */

function buildMovement(T: THREE, ctx: BlueprintCtx) {
  const model = new T.Group();
  ctx.root.add(model);

  const shapeFrom = (pts: Pt[]) => {
    const sh = new T.Shape();
    pts.forEach((p, i) => (i ? sh.lineTo(p[0], p[1]) : sh.moveTo(p[0], p[1])));
    sh.closePath();
    return sh;
  };
  const extrude = (sh: InstanceType<THREE['Shape']>, depth: number, z: number) => {
    const g = new T.ExtrudeGeometry(sh, { depth, bevelEnabled: false, curveSegments: 40 });
    g.translate(0, 0, z - depth / 2);
    return g;
  };

  /* escape wheel: toothed rim, four crossings (spokes), hub */
  const wheel = new T.Group();
  model.add(wheel);
  const rim = shapeFrom(wheelOutline(0));
  const r1 = R_ROOT * 0.84;
  const r0 = R_ROOT * 0.36;
  for (let k = 0; k < 4; k++) {
    const a0 = (k * Math.PI) / 2 + 0.16;
    const a1 = ((k + 1) * Math.PI) / 2 - 0.16;
    const hole = new T.Path();
    hole.absarc(0, 0, r1, a0, a1, false);
    hole.absarc(0, 0, r0, a1 - 0.12, a0 + 0.12, true);
    hole.closePath();
    rim.holes.push(hole);
  }
  const axle = new T.Path();
  axle.absarc(0, 0, 0.05, 0, Math.PI * 2, true);
  rim.holes.push(axle);
  const wheelMesh = new T.Mesh(extrude(rim, 0.05, 0));
  ctx.inkify(wheelMesh, { threshold: 30, outline: 0.8, hidden: true });
  wheel.add(wheelMesh);

  const cyl = (r: number, len: number, z: number, x = 0, y = 0, seg = 32) => {
    const m = new T.Mesh(new T.CylinderGeometry(r, r, len, seg));
    m.rotation.x = Math.PI / 2;
    m.position.set(x, y, z);
    return m;
  };
  const wheelArbor = cyl(0.045, 0.9, -0.3);
  ctx.inkify(wheelArbor, { threshold: 60, outline: 0.9 });
  wheel.add(wheelArbor);
  const collet = cyl(0.13, 0.08, -0.07);
  ctx.inkify(collet, { threshold: 60, outline: 0.9 });
  wheel.add(collet);

  /* the drive: a barrel on the wheel arbor, a cord, a weight that sinks as the wheel turns */
  const BARREL_R = 0.56;
  const BARREL_Z = -0.38;
  const barrel = cyl(BARREL_R, 0.16, BARREL_Z, 0, 0, 48);
  ctx.inkify(barrel, { threshold: 50, outline: 0.9 });
  wheel.add(barrel);
  const weight = new T.Group();
  model.add(weight);
  const wt = new T.Mesh(new T.CylinderGeometry(0.085, 0.085, 0.5, 32));
  ctx.inkify(wt, { threshold: 50, outline: 1 });
  weight.add(wt);
  const hook = new T.Mesh(new T.TorusGeometry(0.05, 0.012, 6, 20));
  hook.position.y = 0.3;
  ctx.inkify(hook, { threshold: 60, outline: 0.6 });
  weight.add(hook);
  const cordGeo = new T.BufferGeometry().setFromPoints([
    new T.Vector3(BARREL_R, 0, BARREL_Z),
    new T.Vector3(BARREL_R, -1, BARREL_Z),
  ]);
  const cord = new T.Line(cordGeo, ctx.lineMaterial('ink'));
  model.add(cord);
  const WEIGHT_TOP = -1.15;

  /* anchor: hub, two curved arms, two pallets (red working faces), on an arbor */
  const anchor = new T.Group();
  anchor.position.set(PIVOT[0], PIVOT[1], 0);
  model.add(anchor);
  const local = (pts: Pt[]) => pts.map((p) => [p[0] - PIVOT[0], p[1] - PIVOT[1]] as Pt);
  ARMS.forEach((a) => {
    const m = new T.Mesh(extrude(shapeFrom(local(a)), 0.05, 0.01));
    ctx.inkify(m, { threshold: 30, outline: 0 });
    anchor.add(m);
  });
  PALLETS.forEach((p) => {
    const m = new T.Mesh(extrude(shapeFrom(local(p)), 0.09, 0));
    ctx.inkify(m, { threshold: 20, outline: 0 });
    anchor.add(m);
  });
  const hub = cyl(0.12, 0.08, 0.01);
  ctx.inkify(hub, { threshold: 60, outline: 0.9 });
  anchor.add(hub);
  const anchorArbor = cyl(0.04, 0.75, -0.3);
  ctx.inkify(anchorArbor, { threshold: 60, outline: 0.9 });
  anchor.add(anchorArbor);

  /* pendulum: hangs from the anchor arbor, behind the movement */
  const pend = new T.Group();
  pend.position.set(PIVOT[0], PIVOT[1], -0.62);
  model.add(pend);
  const rodGeo = new T.BoxGeometry(0.05, 1, 0.025);
  const rod = new T.Mesh(rodGeo);
  ctx.inkify(rod, { threshold: 30, outline: 0, edge: 'brass' });
  pend.add(rod);
  const bob = cyl(BOB_R, 0.07, 0, 0, 0, 64);
  ctx.inkify(bob, { threshold: 40, outline: 1.1, edge: 'brass' });
  pend.add(bob);
  const bobRing = cyl(BOB_R * 0.55, 0.08, 0, 0, 0, 48);
  ctx.inkify(bobRing, { threshold: 40, outline: 0.6, edge: 'brass' });
  bob.add(bobRing);
  bobRing.position.set(0, 0, 0);
  bobRing.rotation.set(0, 0, 0);
  const nut = new T.Mesh(new T.CylinderGeometry(0.035, 0.035, 0.1, 16));
  ctx.inkify(nut, { threshold: 60, outline: 0.7, edge: 'brass' });
  pend.add(nut);
  // the bob's path, in red: one period is there and back along this arc
  const ARC_N = 24;
  const arcGeo = new T.BufferGeometry().setFromPoints(
    Array.from({ length: ARC_N + 1 }, () => new T.Vector3()),
  );
  const swingArc = new T.Line(arcGeo, ctx.lineMaterial('red'));
  swingArc.position.set(PIVOT[0], PIVOT[1], -0.62);
  model.add(swingArc);
  const crutchTop = cyl(0.09, 0.05, 0);
  ctx.inkify(crutchTop, { threshold: 60, outline: 0.8 });
  pend.add(crutchTop);

  /* a back cock (bracket) holding the arbors, drawn as a plate outline */
  const cockPts: Pt[] = [
    [-0.25, -0.2],
    [0.25, -0.2],
    [0.2, PIVOT[1] + 0.32],
    [-0.2, PIVOT[1] + 0.32],
  ];
  const cock = new T.Mesh(extrude(shapeFrom(cockPts), 0.04, -0.72));
  ctx.inkify(cock, { threshold: 30, outline: 0, hidden: true });
  model.add(cock);
  // suspension spring above the pivot
  const susp = new T.Mesh(new T.BoxGeometry(0.06, 0.24, 0.01));
  susp.position.set(PIVOT[0], PIVOT[1] + 0.16, -0.62);
  ctx.inkify(susp, { threshold: 30, outline: 0 });
  model.add(susp);

  let st = initialState();
  let lastL = -1;
  const setLength = (L: number) => {
    const Lw = rodWorld(L);
    rod.scale.y = Lw - BOB_R * 0.6;
    rod.position.y = -(Lw - BOB_R * 0.6) / 2;
    bob.position.y = -Lw;
    nut.position.y = -Lw - BOB_R - 0.03;
    const ap = arcGeo.attributes.position;
    const rA = Lw + BOB_R + 0.12;
    for (let i = 0; i <= ARC_N; i++) {
      const t = AMP * 1.6 * (2 * (i / ARC_N) - 1);
      ap.setXYZ(i, Math.sin(t) * rA, -Math.cos(t) * rA, 0);
    }
    ap.needsUpdate = true;
    arcGeo.computeBoundingSphere();
    const top = PIVOT[1] + 0.45;
    const bottom = PIVOT[1] - Lw - BOB_R - 0.1;
    const k = Math.min(1, (2 * FIT * 0.93) / (top - bottom));
    model.scale.setScalar(k);
    model.position.y = (-(top + bottom) / 2) * k;
  };

  return (dt: number, _t: number, p: Params) => {
    if (p.L !== lastL) {
      setLength(p.L);
      lastL = p.L;
    }
    // the case is off level by `tilt`; anchor and pendulum angles below are relative to it, and
    // a = θ − tilt, where θ = AMP·sin(phase) is the pendulum's swing about the true vertical
    model.rotation.z = p.tilt;
    if (!ctx.reduced) {
      // cap dt (tab switches); stepMovement substeps, so the escapement is right at any frame rate
      const n = stepMovement(st, Math.min(dt, 0.1), period(p.L), { tilt: p.tilt });
      st = { phase: n.phase, w: n.w };
    }
    const w = st.w;
    const a = AMP * Math.sin(st.phase) - p.tilt; // anchor angle relative to the case
    anchor.rotation.z = a;
    pend.rotation.z = a; // rigid with the anchor through the crutch: plumb ± AMP in the world
    wheel.rotation.z = -w;
    // cord unwinds as the wheel turns clockwise; loop the drop so the weight never leaves
    const drop = (((BARREL_R * w) % 1.0) + 1.0) % 1.0;
    const wy = WEIGHT_TOP - drop - 0.35;
    weight.position.set(BARREL_R, wy, BARREL_Z);
    const pos = cordGeo.attributes.position;
    pos.setY(1, wy + 0.35);
    pos.needsUpdate = true;
    cordGeo.computeBoundingSphere();
  };
}
