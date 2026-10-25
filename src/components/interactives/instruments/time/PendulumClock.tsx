/**
 * Plate 1 (wide): a pendulum clock movement as a blueprint. A 30-tooth escape wheel, a recoil
 * anchor whose pallets lock and release it, and a pendulum hung from the anchor's arbor. The
 * wheel is driven (by a weight we do not draw) and only moves when a pallet lets it: it locks,
 * recoils, drops, and gains half a tooth per swing, one tooth per full period. The slider
 * changes the pendulum length; the animation runs at the true period T = 2π√(L/g).
 */
import { useMemo, useState } from 'react';
import {
  BlueprintCanvas,
  C,
  Controls,
  InkText,
  Plate,
  Slider,
  useMeasuredWidth,
  type BlueprintCtx,
  type THREE,
} from '../shared';
import {
  AMP,
  N_TEETH,
  PALLETS,
  PIVOT,
  R_ROOT,
  palletsAt,
  solveWheel,
  wheelOutline,
  type Pt,
} from './escapement';

const G = 9.80665;
const SECONDS_L = G / (Math.PI * Math.PI); // 0.9936 m: period 2 s
const L_MIN = 0.1;
const L_MAX = 4;
const period = (L: number) => 2 * Math.PI * Math.sqrt(L / G);
/** Drawn rod length (world units, wheel tip radius = 1). Compressed so 0.1 m and 4 m both fit. */
const rodWorld = (L: number) => 3.0 + 2.6 * Math.sqrt((L - L_MIN) / (L_MAX - L_MIN));
const BOB_R = 0.3;
const FIT = 3.0;

/* ───────────────────────── geometry helpers (anchor body) ───────────────────────── */

const add = (a: Pt, b: Pt): Pt => [a[0] + b[0], a[1] + b[1]];
const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const mul = (a: Pt, k: number): Pt => [a[0] * k, a[1] * k];
const norm = (a: Pt): Pt => {
  const l = Math.hypot(a[0], a[1]) || 1;
  return [a[0] / l, a[1] / l];
};

/** A curved arm from the pivot hub out to a pallet's back, as a closed polygon. */
function armPoly(to: Pt, bulge: number, width = 0.075, n = 14): Pt[] {
  const p0 = PIVOT;
  const mid = mul(add(p0, to), 0.5);
  const dir = norm(sub(to, p0));
  const nrm: Pt = [-dir[1], dir[0]];
  const ctrl = add(mid, mul(nrm, bulge));
  const pts: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const a = mul(p0, (1 - t) * (1 - t));
    const b = mul(ctrl, 2 * t * (1 - t));
    const c = mul(to, t * t);
    pts.push(add(add(a, b), c));
  }
  const left: Pt[] = [];
  const right: Pt[] = [];
  pts.forEach((p, i) => {
    const q = pts[Math.min(i + 1, n)];
    const r = pts[Math.max(i - 1, 0)];
    const d = norm(sub(q, r));
    const k: Pt = [-d[1], d[0]];
    const w = width * (1 - 0.25 * (i / n));
    left.push(add(p, mul(k, w / 2)));
    right.push(add(p, mul(k, -w / 2)));
  });
  return [...left, ...right.reverse()];
}
const centroid = (p: Pt[]): Pt => mul(p.reduce(add, [0, 0] as Pt), 1 / p.length);
const ARM_ENDS = PALLETS.map((p) => {
  // aim at the back of the pallet, a little outside the wheel
  const c = centroid(p);
  return add(c, mul(norm(c), 0.04));
});
const ARMS = [armPoly(ARM_ENDS[0], 0.22), armPoly(ARM_ENDS[1], -0.22)];

/* ───────────────────────── the SVG poster (front elevation) ───────────────────────── */

function Poster() {
  const W = 400;
  const H = 625;
  const k0 = Math.min(1, (2 * FIT * 0.93) / (PIVOT[1] + 0.45 - (PIVOT[1] - rodWorld(SECONDS_L) - BOB_R - 0.1)));
  const s = (400 / (2 * 0.64 * FIT)) * k0; // px per world unit, as the live camera
  const ox = W / 2;
  const oy = H / 2 + ((PIVOT[1] + 0.45 + PIVOT[1] - rodWorld(SECONDS_L) - BOB_R - 0.1) / 2) * s; // wheel centre, as the live model
  const P = (p: Pt) => `${(ox + p[0] * s).toFixed(1)},${(oy - p[1] * s).toFixed(1)}`;
  const poly = (pts: Pt[]) => pts.map(P).join(' ');
  const L = rodWorld(SECONDS_L);
  const bobY = PIVOT[1] - L;
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1 };
  const pals = palletsAt(0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      <polygon points={poly(wheelOutline(0.08))} style={ink} />
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
        <polygon key={i} points={poly(a)} style={{ ...ink, fill: C.paper }} />
      ))}
      {pals.map((p, i) => (
        <polygon key={i} points={poly(p)} style={{ ...ink, stroke: C.red, fill: C.paper }} />
      ))}
      <circle cx={ox} cy={oy - PIVOT[1] * s} r={9} style={{ ...ink, fill: C.paper }} />
      <line
        x1={ox}
        y1={oy - PIVOT[1] * s}
        x2={ox}
        y2={oy - (bobY + BOB_R) * s}
        style={{ ...ink, strokeWidth: 1.4 }}
      />
      <circle cx={ox} cy={oy - bobY * s} r={BOB_R * s} style={{ ...ink, fill: C.paper }} />
      <circle cx={ox} cy={oy - bobY * s} r={BOB_R * s * 0.55} style={{ ...ink, strokeWidth: 0.6 }} />
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

/* ───────────────────────── the T(L) side drawing ───────────────────────── */

function PeriodCurve({ L }: { L: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(320);
  const H = Math.round(Math.min(230, Math.max(170, W * 0.62)));
  const m = { l: 34, r: 12, t: 14, b: 30 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const TMAX = 4.2;
  const x = (l: number) => m.l + (l / L_MAX) * iw;
  const y = (t: number) => m.t + ih - (t / TMAX) * ih;
  const d = useMemo(() => {
    let s = '';
    for (let i = 0; i <= 80; i++) {
      const l = (i / 80) * L_MAX;
      s += `${i ? 'L' : 'M'} ${x(l).toFixed(1)} ${y(period(l)).toFixed(1)} `;
    }
    return s;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W, H]);
  const T = period(L);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }}>
        {[0, 1, 2, 3, 4].map((t) => (
          <g key={t}>
            <line
              x1={m.l}
              x2={m.l + iw}
              y1={y(t)}
              y2={y(t)}
              style={{ stroke: C.faint, strokeWidth: 0.5, opacity: 0.6 }}
            />
            <InkText x={m.l - 6} y={y(t)} kind="num" anchor="end" baseline="central" size={10} role="soft" halo={false}>
              {t}
            </InkText>
          </g>
        ))}
        {[0, 1, 2, 3, 4].map((l) => (
          <InkText key={l} x={x(l)} y={m.t + ih + 14} kind="num" anchor="middle" size={10} role="soft" halo={false}>
            {l}
          </InkText>
        ))}
        <InkText x={m.l + iw} y={m.t + ih + 27} kind="note" anchor="end" size={11.5} role="soft" halo={false}>
          rod length, metres
        </InkText>
        <InkText x={m.l + 4} y={m.t + 2} kind="note" size={11.5} role="soft" baseline="hanging" halo={false}>
          period, seconds
        </InkText>
        <path
          d={`M ${m.l} ${m.t} V ${m.t + ih} H ${m.l + iw}`}
          style={{ stroke: C.ink, strokeWidth: 1, fill: 'none' }}
        />
        <path d={d} style={{ stroke: C.ink, strokeWidth: 1.3, fill: 'none' }} />
        {/* seconds pendulum */}
        <path
          d={`M ${x(SECONDS_L)} ${m.t + ih} V ${y(2)} H ${m.l}`}
          style={{ stroke: C.soft, strokeWidth: 0.7, strokeDasharray: '3 3', fill: 'none' }}
        />
        <InkText x={x(SECONDS_L) + 6} y={y(2) + 12} size={11.5} role="soft">
          the seconds pendulum
        </InkText>
        {/* current */}
        <path
          d={`M ${x(L)} ${m.t + ih} V ${y(T)} H ${m.l}`}
          style={{ stroke: C.red, strokeWidth: 1, fill: 'none' }}
        />
        <circle cx={x(L)} cy={y(T)} r={4} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.5 }} />
      </svg>
    </div>
  );
}

/* ───────────────────────── the plate ───────────────────────── */

type Params = { L: number };

export default function PendulumClock() {
  const [L, setL] = useState(SECONDS_L);
  const params = useMemo<Params>(() => ({ L }), [L]);
  const T = period(L);
  const wheelTurn = N_TEETH * T;

  return (
    <Plate
      n={1}
      wide
      id="plate-pendulum"
      title="Pendulum and anchor"
      caption={
        <>
          The falling weight always pushes the escape wheel clockwise. The red pallets let it
          move half a tooth per swing: a tooth lands on a pallet, is shoved back a little (the
          recoil), then slips off and the next tooth drops onto the other pallet. The rod is drawn
          shorter than scale; the swing runs at its true period.
        </>
      }
    >
      <div className="tm-split">
        <div className="tm-canvas">
          <BlueprintCanvas<Params>
            aspect={0.64}
            fit={FIT}
            animate
            yaw={-0.42}
            pitch={0.12}
            pitchLimit={0.9}
            params={params}
            label="Rotatable line drawing of a pendulum clock's anchor escapement: a 30-tooth escape wheel, an anchor with two pallets, and a pendulum"
            poster={<Poster />}
            build={(T3, ctx) => buildMovement(T3, ctx)}
          />
        </div>
        <div className="tm-side">
          <PeriodCurve L={L} />
          <dl className="tm-facts">
            <div>
              <dt>period</dt>
              <dd>{T.toFixed(2)} s</dd>
            </div>
            <div>
              <dt>ticks a minute</dt>
              <dd>{(120 / T).toFixed(1)}</dd>
            </div>
            <div>
              <dt>escape wheel turns once in</dt>
              <dd>{wheelTurn < 100 ? `${wheelTurn.toFixed(1)} s` : `${(wheelTurn / 60).toFixed(2)} min`}</dd>
            </div>
          </dl>
          <p className="tm-note">
            {Math.abs(L - SECONDS_L) < 0.004
              ? 'A 2 s period: one tick per second, and the 30-tooth wheel turns once a minute, so a seconds hand can sit on its arbor.'
              : 'Set the rod to 0.994 m and every tick is one second.'}
          </p>
        </div>
      </div>
      <Controls>
        <Slider
          label="rod length"
          value={L}
          min={L_MIN}
          max={L_MAX}
          step={0.001}
          onChange={(v) => setL(Math.abs(v - SECONDS_L) < 0.012 ? SECONDS_L : v)}
          format={(v) => `${v.toFixed(3)} m`}
          width="12rem"
        />
      </Controls>
      <style>{CSS}</style>
    </Plate>
  );
}

const CSS = `
.tm-split { display: grid; grid-template-columns: 1fr; gap: 1rem; align-items: center; }
@media (min-width: 40rem) { .tm-split { grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: 1.5rem; } }
.tm-canvas { max-width: 26rem; width: 100%; margin: 0 auto; }
.tm-facts { display: grid; gap: 0.35rem; margin: 0.9rem 0 0; }
.tm-facts > div { display: flex; justify-content: space-between; gap: 1rem; border-bottom: 1px dotted var(--in-ink-faint); padding-bottom: 0.2rem; }
.tm-facts dt { font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--in-ink-soft); }
.tm-facts dd { margin: 0; font-family: var(--font-mono); font-size: 0.88rem; font-variant-numeric: tabular-nums; color: var(--in-ink); }
.tm-note { margin: 0.7rem 0 0; font-family: var(--font-display); font-style: italic; font-size: 0.98rem; line-height: 1.45; color: var(--in-ink-soft); min-height: 4.3em; }
`;

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
    ctx.inkify(m, { threshold: 20, outline: 0, edge: 'red' });
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
  ctx.inkify(rod, { threshold: 30, outline: 0 });
  pend.add(rod);
  const bob = cyl(BOB_R, 0.07, 0, 0, 0, 64);
  ctx.inkify(bob, { threshold: 40, outline: 1.1 });
  pend.add(bob);
  const bobRing = cyl(BOB_R * 0.55, 0.08, 0, 0, 0, 48);
  ctx.inkify(bobRing, { threshold: 40, outline: 0.6 });
  bob.add(bobRing);
  bobRing.position.set(0, 0, 0);
  bobRing.rotation.set(0, 0, 0);
  const nut = new T.Mesh(new T.CylinderGeometry(0.035, 0.035, 0.1, 16));
  ctx.inkify(nut, { threshold: 60, outline: 0.7 });
  pend.add(nut);
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

  let w = 0;
  let phase = Math.PI / 2;
  let lastL = -1;
  const setLength = (L: number) => {
    const Lw = rodWorld(L);
    rod.scale.y = Lw - BOB_R * 0.6;
    rod.position.y = -(Lw - BOB_R * 0.6) / 2;
    bob.position.y = -Lw;
    nut.position.y = -Lw - BOB_R - 0.03;
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
    if (ctx.reduced) {
      phase = Math.PI * 0.3;
    } else {
      phase += (2 * Math.PI * dt) / period(p.L);
    }
    const a = AMP * Math.sin(phase);
    w = solveWheel(w, a, Math.max(0.004, dt * 2.2));
    anchor.rotation.z = a;
    pend.rotation.z = a;
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
