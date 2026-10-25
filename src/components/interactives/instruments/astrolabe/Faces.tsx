/**
 * Flat (2D) views of the astrolabe's faces, drawn from the same prims as the 3D model:
 * FrontView (limb + plate + rete + rule) and BackView (scales + alidade). Used for the close-up
 * steps where a reading has to be crisp. Geometry: ./plateGeom.ts, ./faceGeom.ts (swept).
 */
import { C, InkText, useMeasuredWidth, type Role } from '../shared';
import { PrimsSvg } from './Prims';
import { almucantarRuns, backPrims, limbPrims, platePrims, retePrims, type UV } from './plateGeom';
import { backScene, frontScene } from './faceGeom';

const P = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');

export function FrontView({
  phi,
  lst,
  sunLam,
  almH,
  ruleH,
  zoom = 1,
  focus,
  note,
  noteRole = 'soft',
  showRete = true,
}: {
  phi: number;
  lst: number;
  sunLam: number | null;
  almH: number | null;
  ruleH: number | null;
  zoom?: number;
  focus?: UV;
  note?: string;
  noteRole?: Role;
  showRete?: boolean;
}) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = frontScene(W, { phi, lst, sunLam, ruleH, zoom, focus });
  const clip = `fv-${Math.round(W)}-${Math.round(zoom * 10)}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={note ?? 'The front of the astrolabe'}>
        <defs>
          <clipPath id={clip}>
            <rect x={0} y={0} width={W} height={g.Hd} />
          </clipPath>
        </defs>
        <g clipPath={`url(#${clip})`}>
          <circle cx={g.f.cx} cy={g.f.cy} r={g.RMpx} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
          <circle cx={g.f.cx} cy={g.f.cy} r={g.RCpx} style={{ fill: C.paper }} />
          <PrimsSvg prims={limbPrims()} f={g.f} scaleW={0.9} />
          <PrimsSvg prims={platePrims(phi)} f={g.f} scaleW={0.9} />
          {almH !== null &&
            almucantarRuns(almH, phi).map((run, i) => (
              <path key={i} d={P(run.map((q) => ({ x: g.f.cx + q.u * g.f.k, y: g.f.cy - q.v * g.f.k })))} style={{ fill: 'none', stroke: C.sky, strokeWidth: 2 }} />
            ))}
          {showRete && <PrimsSvg prims={retePrims(lst)} f={g.f} scaleW={1} />}
          {g.rule && (
            <>
              <line x1={g.rule.a.x} y1={g.rule.a.y} x2={g.rule.b.x} y2={g.rule.b.y} style={{ stroke: C.red, strokeWidth: 1.6 }} />
              <circle cx={g.rule.limb.x} cy={g.rule.limb.y} r={3} style={{ fill: C.red }} />
            </>
          )}
          {g.sun && <circle cx={g.sun.x} cy={g.sun.y} r={Math.max(4, 0.035 * g.f.k)} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1 }} />}
          <circle cx={g.f.cx} cy={g.f.cy} r={3} style={{ fill: C.ink }} />
        </g>
        {note && (
          <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={noteRole}>
            {note}
          </InkText>
        )}
      </svg>
    </div>
  );
}

export function BackView({ ali, mark, zoom = 1, focus, note, noteRole = 'soft' }: { ali: number; mark?: 'alt' | 'date'; zoom?: number; focus?: UV; note?: string; noteRole?: Role }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = backScene(W, { ali, zoom, focus });
  const clip = `bv-${Math.round(W)}-${Math.round(zoom * 10)}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={note ?? 'The back of the astrolabe'}>
        <defs>
          <clipPath id={clip}>
            <rect x={0} y={0} width={W} height={g.Hd} />
          </clipPath>
        </defs>
        <g clipPath={`url(#${clip})`}>
          <circle cx={g.f.cx} cy={g.f.cy} r={g.RMpx} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
          <PrimsSvg prims={backPrims()} f={g.f} scaleW={0.9} />
          {/* the alidade: a bar through the pin with its fiducial edge (red) */}
          <path d={P(g.bar)} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1, opacity: 0.55 }} />
          <line x1={g.edge.a.x} y1={g.edge.a.y} x2={g.edge.b.x} y2={g.edge.b.y} style={{ stroke: mark ? C.red : C.ink, strokeWidth: 1.4 }} />
          {mark === 'alt' && <circle cx={g.altMark.x} cy={g.altMark.y} r={3.5} style={{ fill: C.sky }} />}
          {mark === 'date' && (
            <>
              <circle cx={g.zodMark.x} cy={g.zodMark.y} r={3.5} style={{ fill: C.brass }} />
              <circle cx={g.calMark.x} cy={g.calMark.y} r={3.5} style={{ fill: C.brass }} />
            </>
          )}
          <circle cx={g.f.cx} cy={g.f.cy} r={3} style={{ fill: C.ink }} />
        </g>
        {note && (
          <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={noteRole}>
            {note}
          </InkText>
        )}
      </svg>
    </div>
  );
}
