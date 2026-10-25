/**
 * The stable reference view of the astrolabe post: the sky over you at 41° N, east on the left
 * (as on the astrolabe's face). Same camera and size in every plate; `hi` picks what is
 * highlighted for the current step. Geometry: ./skyGeom.ts (swept).
 */
import { C, InkText, useMeasuredWidth, type Role } from '../shared';
import { dome, type DomeOpts, type Run } from './skyGeom';

export type DomeHi = 'horizon' | 'alm' | 'day' | 'sun' | 'h' | 'hour' | 'pole' | 'star' | 'equator' | 'meridian';

function RunPath({ runs, role, w, op = 1, dashBack = true }: { runs: Run[]; role: Role; w: number; op?: number; dashBack?: boolean }) {
  return (
    <>
      {runs.map((r, i) => (
        <polyline
          key={i}
          points={r.pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
          style={{
            fill: 'none',
            stroke: C[role],
            strokeWidth: r.front ? w : w * 0.7,
            strokeDasharray: !r.front && dashBack ? '3 4' : undefined,
            opacity: r.front ? op : op * 0.55,
          }}
        />
      ))}
    </>
  );
}

export function Dome({ opts, hi = [], note, caption }: { opts: DomeOpts; hi?: DomeHi[]; note?: string; caption?: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = dome(W, opts);
  const has = (k: DomeHi) => hi.includes(k);
  const o = (k: DomeHi) => (hi.length === 0 || has(k) ? 1 : 0.45);
  const w = (k: DomeHi, b: number) => (has(k) ? b * 1.9 : b);
  const lab = (id: string, role: Role, k?: DomeHi) => {
    const l = g.labels.find((x) => x.id === id);
    if (!l) return null;
    return (
      <g style={{ opacity: k ? o(k) : 1 }}>
        <InkText x={l.x} y={l.y} anchor={l.anchor} size={l.size} role={role}>
          {l.text}
        </InkText>
      </g>
    );
  };
  return (
    <div ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${g.H}`}
        width="100%"
        style={{ display: 'block', background: C.paper }}
        role="img"
        aria-label={note ?? 'The sky over you at latitude 41 degrees north, east on the left'}
      >
        {/* the celestial sphere's outline and the ground */}
        <circle cx={g.cx} cy={g.cy} r={g.R} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.8 }} />
        <RunPath runs={g.horizon} role="ink" w={w('horizon', 1.2)} op={o('horizon')} />
        <RunPath runs={g.meridian} role="faint" w={w('meridian', 0.7)} op={o('meridian')} />
        <RunPath runs={g.equator} role="soft" w={w('equator', 0.8)} op={o('equator')} />
        {/* polar axis */}
        <line x1={g.ctr.x} y1={g.ctr.y} x2={g.pole.x} y2={g.pole.y} style={{ stroke: C.soft, strokeWidth: w('pole', 0.8), strokeDasharray: '4 3', opacity: o('pole') }} />
        {/* the sun's path today: whole circle faint, the part above the horizon in brass */}
        <RunPath runs={g.day} role="faint" w={0.6} op={o('day')} />
        <RunPath runs={g.dayUp} role="brass" w={w('day', 1)} op={o('day')} dashBack={false} />
        {g.star.length > 0 && <RunPath runs={g.star} role="sky" w={w('star', 0.8)} op={o('star')} />}
        {g.alm.length > 0 && <RunPath runs={g.alm} role="sky" w={w('alm', 0.9)} op={o('alm')} />}
        {g.hourArc.length > 0 && <RunPath runs={g.hourArc} role="red" w={w('hour', 1.1)} op={o('hour')} />}
        {/* altitude: the vertical from the horizon up to the sun */}
        {g.hArc.length > 0 && <RunPath runs={g.hArc} role="sky" w={w('h', 1.2)} op={o('h')} />}
        {g.hFoot && g.sun && (
          <line x1={g.ctr.x} y1={g.ctr.y} x2={g.hFoot.x} y2={g.hFoot.y} style={{ stroke: C.sky, strokeWidth: 0.7, strokeDasharray: '2 3', opacity: o('h') }} />
        )}
        {g.sun && g.sun.h > 0 && (
          <line x1={g.ctr.x} y1={g.ctr.y} x2={g.sun.x} y2={g.sun.y} style={{ stroke: C.brass, strokeWidth: 0.7, opacity: o('sun') }} />
        )}
        <circle cx={g.ctr.x} cy={g.ctr.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
        <circle cx={g.zen.x} cy={g.zen.y} r={1.8} style={{ fill: C.soft }} />
        <circle cx={g.pole.x} cy={g.pole.y} r={2.2} style={{ fill: C.soft, opacity: o('pole') }} />
        {g.starPt && (
          <circle cx={g.starPt.x} cy={g.starPt.y} r={has('star') ? 4 : 3} style={{ fill: C.sky, opacity: g.starPt.front ? o('star') : 0.4 }} />
        )}
        {g.sun && (
          <circle
            cx={g.sun.x}
            cy={g.sun.y}
            r={has('sun') ? 7 : 5.5}
            style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1.2, opacity: g.sun.h > 0 ? o('sun') : 0.35 }}
          />
        )}
        {lab('N', 'soft')}
        {lab('S', 'soft')}
        {lab('E', 'ink')}
        {lab('W', 'ink')}
        {lab('zenith', 'soft')}
        {lab('pole', 'soft', 'pole')}
        {lab('you', 'ink')}
        {lab('sun', 'ink', 'sun')}
        {lab('h', 'sky', 'h')}
        {caption && (
          <InkText x={g.W / 2} y={g.H - 8} anchor="middle" size={12} role="soft">
            {caption}
          </InkText>
        )}
      </svg>
    </div>
  );
}
