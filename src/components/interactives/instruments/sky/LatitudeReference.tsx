/**
 * The latitude post's stable reference view (every procedure plate): the Earth cut along your
 * meridian with the light's parallel rays, your horizon and zenith, h at you, φ and δ at the
 * centre. Geometry: ./latRefGeom.ts (`latRef`), swept by each plate's sweep script.
 * `hi` lists the quantities to highlight for the current step; the layout never changes.
 */
import { C, InkText, arcPath, useMeasuredWidth, type Role } from '../shared';
import { latRef, type RefSpec } from './latRefGeom';

export type RefKey = 'sun' | 'h' | 'horizon' | 'dec' | 'phi' | 'theta' | 'sight';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

export default function LatitudeReference({ spec, hi = [], label }: { spec: RefSpec; hi?: RefKey[]; label: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = latRef(spec, W);
  const on = new Set(hi);
  const o = (k: RefKey) => (on.has(k) || on.size === 0 ? 1 : 0.4);
  const wd = (k: RefKey, base: number) => (on.has(k) ? base * 1.9 : base);
  const T = (id: string, role: Role, k?: RefKey) => {
    const l = g.labels[id];
    if (!l) return null;
    return (
      <g style={{ opacity: k ? o(k) : 1 }}>
        <InkText x={l.x} y={l.y} anchor={l.anchor} baseline="central" size={l.size} role={role}>
          {l.text}
        </InkText>
      </g>
    );
  };
  const rayRole: Role = g.star ? 'sky' : 'brass';
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={label}>
        {/* the light: parallel rays */}
        <g style={{ opacity: o('sun') }}>
          {g.rays.map((r) => (
            <line key={r.o} x1={r.a.x} y1={r.a.y} x2={r.b.x} y2={r.b.y} style={{ ...solid(rayRole, 0.7), opacity: 0.6 }} />
          ))}
          <line x1={g.sunFrom.x} y1={g.sunFrom.y} x2={g.obs.x} y2={g.obs.y} style={solid(rayRole, wd('sun', 1.1))} />
          {g.sun2 && g.obs2 && <line x1={g.sun2.x} y1={g.sun2.y} x2={g.obs2.x} y2={g.obs2.y} style={solid(rayRole, wd('sun', 1.1))} />}
        </g>
        {T('sun', 'soft', 'sun')}
        {/* the Earth */}
        <circle cx={g.cx} cy={g.cy} r={g.Re} style={{ ...solid('ink', 1.3), fill: C.paper }} />
        <line x1={g.segs[0].a.x} y1={g.cy} x2={g.segs[0].b.x} y2={g.cy} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
        <line x1={g.cx} y1={g.segs[1].a.y} x2={g.cx} y2={g.segs[1].b.y} style={solid('faint', 0.7)} />
        {T('equator', 'soft')}
        {T('pole', 'soft')}
        <line x1={g.cx} y1={g.cy} x2={g.zen.x} y2={g.zen.y} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
        {g.two && g.zen2 && g.obs2 ? (
          <>
            <line x1={g.cx} y1={g.cy} x2={g.zen2.x} y2={g.zen2.y} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
            <path d={arcPath(g.cx, g.cy, g.thetaArcR, spec.second!.phi, g.phi)} style={{ ...solid('red', wd('theta', 1.3)), opacity: o('theta') }} />
            {T('theta', 'red', 'theta')}
            <path d={arcPath(g.obs.x, g.obs.y, g.zArcR, g.dir, g.phi)} style={{ ...solid('sky', wd('h', 1.3)), opacity: o('h') }} />
            {T('z', 'sky', 'h')}
            <circle cx={g.obs2.x} cy={g.obs2.y} r={3.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
            {T('second', 'ink')}
          </>
        ) : (
          <>
            {!g.star && (
              <>
                <line x1={g.cx} y1={g.cy} x2={g.sub.x} y2={g.sub.y} style={{ ...solid('brass', 0.9), strokeDasharray: '3 3', opacity: o('dec') }} />
                <path d={arcPath(g.cx, g.cy, g.decArcR, 0, g.dec)} style={{ ...solid('brass', wd('dec', 1.3)), opacity: o('dec') }} />
                {T('dec', 'brass', 'dec')}
              </>
            )}
            <path d={arcPath(g.cx, g.cy, g.phiArcR, 0, g.phi)} style={{ ...solid('red', wd('phi', 1.3)), opacity: o('phi') }} />
            {T('phi', 'red', 'phi')}
            <line x1={g.horFar.x} y1={g.horFar.y} x2={g.horSun.x} y2={g.horSun.y} style={{ ...solid('ink', wd('horizon', 1.3)), opacity: on.has('horizon') || on.size === 0 ? 1 : 0.7 }} />
            {T('horizon', 'soft', 'horizon')}
            {g.visible && (
              <>
                <path d={arcPath(g.obs.x, g.obs.y, g.hArcR, g.hFrom, g.hTo)} style={{ ...solid('sky', wd('h', 1.4)), opacity: o('h') }} />
                {T('h', 'sky', 'h')}
              </>
            )}
          </>
        )}
        {g.sightEnd && (
          <line x1={g.obs.x} y1={g.obs.y} x2={g.sightEnd.x} y2={g.sightEnd.y} style={{ ...solid('red', wd('sight', 1.1)), strokeDasharray: '4 3' }} />
        )}
        <circle cx={g.cx} cy={g.cy} r={2} style={{ fill: C.ink }} />
        <circle cx={g.obs.x} cy={g.obs.y} r={3.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
        {T('you', 'ink')}
        {T('zenith', 'soft')}
      </svg>
    </div>
  );
}
