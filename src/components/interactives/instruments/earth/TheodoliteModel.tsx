/**
 * The 3D theodolite, as a component the triangulation step plates put in their view slot.
 * Parts, bottom to top: tripod and its plumb bob over the station mark; tribrach with three
 * levelling screws; the fixed horizontal circle (graduated every 2°); the alidade, which turns
 * about the vertical axis and carries the plate level, the index and the two standards; the
 * trunnion (horizontal) axis; the telescope with its vertical circle. The red line is the line
 * of sight. Horizontal angle = alidade rotation read against the fixed circle; vertical angle
 * = telescope tilt read against the index on the standard.
 * Horizontal readings run clockwise seen from above (like a real circle), so the alidade turns
 * by −h about the vertical axis. `targets` draws faint dashed lines from the instrument toward
 * each target (same circle readings as the eyepiece, ./theoGeom.ts), so the reader sees the
 * red line of sight swing onto them.
 */
import { useEffect, useMemo } from 'react';
import type * as THREEns from 'three';
import { BlueprintCanvas, C, InkText, useMeasuredWidth } from '../shared';

const RAD = Math.PI / 180;

export function dms(v: number, signed = false) {
  const s = v < 0 ? '−' : signed ? '+' : '';
  let a = Math.abs(v);
  let d = Math.floor(a);
  a = (a - d) * 60;
  let m = Math.floor(a);
  let sec = Math.round((a - m) * 60);
  if (sec === 60) {
    sec = 0;
    m += 1;
  }
  if (m === 60) {
    m = 0;
    d += 1;
  }
  return `${s}${d}°${String(m).padStart(2, '0')}′${String(sec).padStart(2, '0')}″`;
}

/* ── poster: a side elevation of the same instrument ── */
function Poster() {
  const W = 600;
  const H = 400;
  const cx = 290;
  const s = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const p = { fill: C.paper, stroke: C.ink, strokeWidth: 1.1 };
  const ty = 168; // trunnion
  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      {/* tripod */}
      <path d={`M ${cx - 20} 262 L ${cx - 130} 388 M ${cx + 20} 262 L ${cx + 130} 388 M ${cx} 262 L ${cx + 18} 392`} style={{ ...s, strokeWidth: 3 }} />
      <rect x={cx - 60} y={252} width={120} height={10} style={p} />
      <line x1={cx} y1={262} x2={cx} y2={330} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
      <path d={`M ${cx - 5} 330 L ${cx + 5} 330 L ${cx} 346 Z`} style={p} />
      {/* tribrach + screws */}
      <path d={`M ${cx - 70} 250 L ${cx + 70} 250 L ${cx + 56} 242 L ${cx - 56} 242 Z`} style={p} />
      {[-48, 0, 48].map((dx) => (
        <g key={dx}>
          <rect x={cx + dx - 4} y={226} width={8} height={18} style={p} />
          <rect x={cx + dx - 11} y={236} width={22} height={6} style={p} />
        </g>
      ))}
      <rect x={cx - 52} y={220} width={104} height={6} style={p} />
      {/* horizontal circle */}
      <rect x={cx - 80} y={212} width={160} height={8} style={p} />
      {Array.from({ length: 33 }, (_, i) => (
        <line key={i} x1={cx - 78 + i * 4.9} y1={212} x2={cx - 78 + i * 4.9} y2={i % 5 === 0 ? 217 : 215} style={{ stroke: C.soft, strokeWidth: 0.7 }} />
      ))}
      {/* alidade plate + standards */}
      <rect x={cx - 66} y={204} width={132} height={8} style={p} />
      <path d={`M ${cx - 58} 204 L ${cx - 30} ${ty - 14} L ${cx + 30} ${ty - 14} L ${cx + 58} 204`} style={p} />
      <rect x={cx - 26} y={190} width={52} height={9} rx={4} style={p} />
      {/* vertical circle */}
      <circle cx={cx} cy={ty} r={44} style={p} />
      <circle cx={cx} cy={ty} r={38} style={s} />
      {Array.from({ length: 36 }, (_, i) => {
        const a = i * 10 * RAD;
        return (
          <line key={i} x1={cx + 38 * Math.cos(a)} y1={ty + 38 * Math.sin(a)} x2={cx + 44 * Math.cos(a)} y2={ty + 44 * Math.sin(a)} style={{ stroke: C.soft, strokeWidth: 0.7 }} />
        );
      })}
      {/* telescope, tilted 10° up */}
      <g transform={`rotate(-10 ${cx} ${ty})`}>
        <rect x={cx - 120} y={ty - 13} width={230} height={26} style={p} />
        <rect x={cx + 110} y={ty - 17} width={34} height={34} style={p} />
        <rect x={cx - 150} y={ty - 9} width={30} height={18} style={p} />
        <line x1={cx + 144} y1={ty} x2={W + 40} y2={ty} style={{ stroke: C.red, strokeWidth: 1.4, strokeDasharray: '6 4' }} />
      </g>
      <circle cx={cx} cy={ty} r={6} style={p} />
      <path d={`M ${cx - 40} 392 H ${cx + 40} M ${cx} 384 V 398`} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
      <InkText x={cx + 50} y={392} size={12} role="soft">
        station mark
      </InkText>
    </svg>
  );
}

type Params = { h: number; v: number; targets: { h: number; v: number }[] };

/** Line of sight and up to three target directions, driven by circle readings (degrees). */
export function TheodoliteCanvas({ h, v, targets = [] }: { h: number; v: number; targets?: { h: number; v: number }[] }) {
  const params = useMemo<Params>(() => ({ h, v, targets }), [h, v, targets]);
  const [boxRef, boxW] = useMeasuredWidth<HTMLDivElement>(420);
  const narrow = boxW < 360;
  // StepPlate remounts its view on every step; release this canvas's WebGL context when it
  // goes, so stepping back and forth never piles up live contexts.
  useEffect(() => {
    const box = boxRef.current;
    return () => {
      const c = box?.querySelector('canvas');
      const gl = (c?.getContext('webgl2') ?? c?.getContext('webgl')) as WebGLRenderingContext | null | undefined;
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    };
  }, [boxRef]);
  return (
    <div ref={boxRef}>
      <BlueprintCanvas<Params>
        key={narrow ? 'n' : 'w'}
        aspect={narrow ? 1 : 4 / 3}
        fit={1.3}
        yaw={-0.75}
        pitch={0.3}
        params={params}
        poster={<Poster />}
        label="Rotatable line drawing of a theodolite on a tripod; the red line is its line of sight"
        build={(T, ctx) => {
          const ink = (o: THREEns.Object3D, opts?: Parameters<typeof ctx.inkify>[1]) => {
            ctx.inkify(o, opts);
            return o;
          };
          const cyl = (r: number, len: number, seg = 48, rTop = r) =>
            new T.Mesh(new T.CylinderGeometry(rTop, r, len, seg));
          const along = (m: THREEns.Mesh, axis: 'x' | 'z') => {
            if (axis === 'x') m.rotation.z = Math.PI / 2;
            else m.rotation.x = Math.PI / 2;
            return m;
          };
          const root = ctx.root;
          root.position.y = 0.36;

          /* tripod */
          const headY = -0.6;
          const head = cyl(0.3, 0.06, 6);
          head.position.y = headY;
          root.add(ink(head, { threshold: 30 }));
          const footY = -1.42;
          const up = new T.Vector3(0, 1, 0);
          for (let i = 0; i < 3; i++) {
            const a = (i * 120 + 30) * RAD;
            const top = new T.Vector3(Math.cos(a) * 0.22, headY - 0.03, Math.sin(a) * 0.22);
            const bot = new T.Vector3(Math.cos(a) * 0.95, footY, Math.sin(a) * 0.95);
            const dir = bot.clone().sub(top);
            const len = dir.length();
            const leg = new T.Mesh(new T.BoxGeometry(0.06, len, 0.035));
            leg.position.copy(top.clone().add(bot).multiplyScalar(0.5));
            leg.quaternion.setFromUnitVectors(up, dir.normalize());
            root.add(ink(leg, { outline: 0.8 }));
          }
          // plumb bob under the head, over the station mark
          const bobTop = -1.22;
          const thread = new T.BufferGeometry().setFromPoints([
            new T.Vector3(0, headY - 0.03, 0),
            new T.Vector3(0, bobTop, 0),
          ]);
          root.add(new T.Line(thread, ctx.lineMaterial('soft')));
          const bob = new T.Mesh(new T.ConeGeometry(0.035, 0.11, 16));
          bob.rotation.x = Math.PI;
          bob.position.y = bobTop - 0.055;
          root.add(ink(bob, { threshold: 40 }));
          const mark = new T.BufferGeometry().setFromPoints([
            new T.Vector3(-0.12, footY, 0),
            new T.Vector3(0.12, footY, 0),
            new T.Vector3(0, footY, -0.12),
            new T.Vector3(0, footY, 0.12),
          ]);
          const markL = new T.LineSegments(mark, ctx.lineMaterial('soft'));
          markL.position.y = 0;
          root.add(markL);
          const ring = new T.Mesh(new T.RingGeometry(0.05, 0.055, 32));
          ring.rotation.x = -Math.PI / 2;
          ring.position.y = footY;
          ctx.inkify(ring, { fill: 'soft', outline: 0, threshold: 80 });
          root.add(ring);

          /* tribrach with three levelling screws */
          const tri = cyl(0.27, 0.035, 3);
          tri.position.y = -0.55;
          tri.rotation.y = Math.PI / 6;
          root.add(ink(tri, { threshold: 30 }));
          for (let i = 0; i < 3; i++) {
            const a = (i * 120 + 90) * RAD;
            const x = Math.cos(a) * 0.19;
            const z = Math.sin(a) * 0.19;
            const shaft = cyl(0.022, 0.12, 12);
            shaft.position.set(x, -0.48, z);
            root.add(ink(shaft, { outline: 0.8, threshold: 60 }));
            const knob = cyl(0.06, 0.035, 14);
            knob.position.set(x, -0.5, z);
            root.add(ink(knob, { threshold: 20 }));
          }
          const basePlate = cyl(0.23, 0.025, 64);
          basePlate.position.y = -0.41;
          root.add(ink(basePlate, { threshold: 40 }));

          /* fixed horizontal circle, graduated every 2°, long ticks every 10° */
          const circleY = -0.37;
          const hc = cyl(0.36, 0.045, 96);
          hc.position.y = circleY;
          root.add(ink(hc, { threshold: 40 }));
          const tk: number[] = [];
          const topY = circleY + 0.0235;
          for (let d = 0; d < 360; d += 2) {
            const a = d * RAD;
            const r0 = d % 10 === 0 ? 0.31 : 0.335;
            tk.push(Math.cos(a) * r0, topY, -Math.sin(a) * r0, Math.cos(a) * 0.357, topY, -Math.sin(a) * 0.357);
          }
          const tg = new T.BufferGeometry();
          tg.setAttribute('position', new T.Float32BufferAttribute(tk, 3));
          root.add(new T.LineSegments(tg, ctx.lineMaterial('soft')));
          // zero mark on the circle: a small wedge
          const zero = new T.Mesh(new T.ConeGeometry(0.018, 0.04, 3));
          zero.rotation.z = Math.PI / 2;
          zero.position.set(0.385, circleY, 0);
          root.add(ink(zero, { outline: 0.6 }));

          /* the alidade: turns about the vertical axis */
          const alidade = new T.Group();
          root.add(alidade);
          const axis = cyl(0.07, 0.1, 32);
          axis.position.y = -0.31;
          alidade.add(ink(axis, { threshold: 40 }));
          const plate = cyl(0.3, 0.03, 80);
          plate.position.y = -0.26;
          alidade.add(ink(plate, { threshold: 40 }));
          // index arm reaching over the graduations
          const idx = new T.Mesh(new T.BoxGeometry(0.1, 0.012, 0.035));
          idx.position.set(0.33, -0.34, 0);
          alidade.add(ink(idx, { outline: 0.6 }));
          const idxLine = new T.BufferGeometry().setFromPoints([
            new T.Vector3(0.3, topY + 0.002, 0),
            new T.Vector3(0.375, topY + 0.002, 0),
          ]);
          alidade.add(new T.Line(idxLine, ctx.lineMaterial('ink')));
          // plate level (spirit level) across the alidade
          const vialBase = new T.Mesh(new T.BoxGeometry(0.2, 0.02, 0.05));
          vialBase.position.set(-0.08, -0.235, 0.2);
          vialBase.rotation.y = Math.PI / 2;
          alidade.add(ink(vialBase, { outline: 0.6 }));
          const vial = along(cyl(0.02, 0.17, 20), 'z');
          vial.position.set(-0.08, -0.205, 0.2);
          vial.rotation.set(Math.PI / 2, 0, 0);
          alidade.add(ink(vial, { threshold: 60, outline: 0.8 }));
          const bubble = new T.Mesh(new T.SphereGeometry(0.012, 12, 8));
          bubble.position.set(-0.08, -0.188, 0.2);
          alidade.add(ink(bubble, { threshold: 80, outline: 0.7 }));

          // standards: two A-shaped uprights carrying the trunnion axis
          const trunY = 0.12;
          const shape = new T.Shape();
          shape.moveTo(-0.17, 0);
          shape.lineTo(0.17, 0);
          shape.lineTo(0.06, trunY + 0.245 + 0.04);
          shape.lineTo(-0.06, trunY + 0.245 + 0.04);
          shape.closePath();
          const hole = new T.Path();
          hole.moveTo(-0.07, 0.04);
          hole.lineTo(0.07, 0.04);
          hole.lineTo(0.02, 0.2);
          hole.lineTo(-0.02, 0.2);
          hole.closePath();
          shape.holes.push(hole);
          const stdGeo = new T.ExtrudeGeometry(shape, { depth: 0.035, bevelEnabled: false });
          for (const z of [-0.2, 0.2]) {
            const st = new T.Mesh(stdGeo);
            st.position.set(0, -0.245, z - 0.0175);
            alidade.add(ink(st, { threshold: 30 }));
          }
          // index for the vertical circle, on the near standard
          const vIdx = new T.Mesh(new T.BoxGeometry(0.016, 0.06, 0.012));
          vIdx.position.set(0, trunY + 0.215, 0.235);
          alidade.add(ink(vIdx, { outline: 0.5 }));

          /* telescope on the trunnion axis */
          const scope = new T.Group();
          scope.position.y = trunY;
          alidade.add(scope);
          const trun = along(cyl(0.028, 0.52, 24), 'z');
          scope.add(ink(trun, { threshold: 50 }));
          const tube = along(cyl(0.055, 0.6, 40), 'x');
          tube.position.x = 0.02;
          scope.add(ink(tube, { threshold: 50, hidden: true }));
          const obj = along(cyl(0.07, 0.1, 40), 'x');
          obj.position.x = 0.37;
          scope.add(ink(obj, { threshold: 50 }));
          const hood = along(cyl(0.072, 0.03, 40, 0.072), 'x');
          hood.position.x = 0.435;
          scope.add(ink(hood, { threshold: 50 }));
          const eye = along(cyl(0.035, 0.12, 24), 'x');
          eye.position.x = -0.33;
          scope.add(ink(eye, { threshold: 50 }));
          const eyeRing = along(cyl(0.045, 0.04, 16), 'x');
          eyeRing.position.x = -0.36;
          scope.add(ink(eyeRing, { threshold: 20 }));
          const focus = along(cyl(0.064, 0.05, 14), 'x');
          focus.position.x = -0.12;
          scope.add(ink(focus, { threshold: 20 }));
          // telescope level on top
          const tvial = along(cyl(0.016, 0.16, 16), 'x');
          tvial.position.set(0.05, 0.085, 0);
          scope.add(ink(tvial, { threshold: 60, outline: 0.7 }));
          for (const x of [-0.02, 0.12]) {
            const post = new T.Mesh(new T.BoxGeometry(0.012, 0.03, 0.012));
            post.position.set(x, 0.065, 0);
            scope.add(ink(post, { outline: 0 }));
          }
          // vertical circle: turns with the telescope
          const vc = along(cyl(0.2, 0.018, 96), 'z');
          vc.position.z = 0.265;
          scope.add(ink(vc, { threshold: 40 }));
          const vt: number[] = [];
          const zf = 0.265 + 0.0095;
          for (let d = 0; d < 360; d += 2) {
            const a = d * RAD;
            const r0 = d % 10 === 0 ? 0.16 : 0.18;
            vt.push(Math.cos(a) * r0, Math.sin(a) * r0, zf, Math.cos(a) * 0.198, Math.sin(a) * 0.198, zf);
          }
          const vg = new T.BufferGeometry();
          vg.setAttribute('position', new T.Float32BufferAttribute(vt, 3));
          scope.add(new T.LineSegments(vg, ctx.lineMaterial('soft')));
          const hub = along(cyl(0.04, 0.03, 24), 'z');
          hub.position.z = 0.285;
          scope.add(ink(hub, { threshold: 40 }));
          // line of sight
          const los = new T.BufferGeometry().setFromPoints([
            new T.Vector3(0.45, 0, 0),
            new T.Vector3(3.2, 0, 0),
          ]);
          const losLine = new T.Line(los, ctx.lineMaterial('red', { dashed: true, dash: 0.05, gap: 0.03 }));
          losLine.computeLineDistances();
          scope.add(losLine);

          // target directions: dashed, faint, from the vertical axis at trunnion height
          const tLines = [0, 1, 2].map(() => {
            const g = new T.BufferGeometry().setFromPoints([new T.Vector3(0, 0, 0), new T.Vector3(3.2, 0, 0)]);
            const l = new T.Line(g, ctx.lineMaterial('soft', { dashed: true, dash: 0.06, gap: 0.05 }));
            l.computeLineDistances();
            l.position.y = trunY; // the trunnion axis (the alidade group sits at the origin)
            l.visible = false;
            root.add(l);
            return l;
          });
          return (_dt, _t, p) => {
            alidade.rotation.y = -p.h * RAD;
            scope.rotation.z = p.v * RAD;
            tLines.forEach((l, i) => {
              const t = p.targets[i];
              l.visible = !!t;
              if (t) l.rotation.set(0, -t.h * RAD, t.v * RAD, 'YXZ');
            });
          };
        }}
      />
    </div>
  );
}
