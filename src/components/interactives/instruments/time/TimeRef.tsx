/**
 * The stable reference view shared by every episode plate of the time post: the sun over
 * Istanbul at the TRUE time, a 24 h line with true time vs the device, and the error-per-day
 * gauge with every device so far. Same layout in every plate and every step; the current
 * step's row is highlighted. Geometry: ./timeRefGeom.ts.
 */
import { useId } from 'react';
import { C, InkText, useMeasuredWidth, type Role } from '../shared';
import { refScene, type RefInput } from './timeRefGeom';

const line = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

export default function TimeRef(inp: RefInput) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = refScene(inp, W);
  const clip = `tr-sky-${useId().replace(/:/g, '')}`;
  const hi = inp.hi ?? null;
  const op = (row: 'sky' | 'clock' | 'gauge') => (hi === null || hi === row ? 1 : 0.55);
  const L = Object.fromEntries(g.labels.map((l) => [l.id, l]));
  const T = (id: string, role: Role = 'ink') => {
    const l = L[id];
    if (!l) return null;
    return (
      <InkText x={l.x} y={l.y} anchor={l.anchor} kind={l.kind} size={l.size} role={role} baseline="alphabetic">
        {l.text}
      </InkText>
    );
  };
  const pathD = g.path.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  return (
    <div ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${g.H}`}
        width="100%"
        style={{ display: 'block' }}
        role="img"
        aria-label={`True local time ${L.true?.text ?? ''}; ${L.read?.text ?? ''}; ${L.gap0?.text ?? ''}`}
      >
        {/* row 1: the sun */}
        <g style={{ opacity: op('sky') }}>
          {T('sunHead', 'soft')}
          <clipPath id={clip}>
            <rect x={0} y={0} width={W} height={g.horizonY} />
          </clipPath>
          <path d={pathD} clipPath={`url(#${clip})`} style={{ ...line('brass', 0.9), strokeDasharray: '4 3' }} />
          <line x1={g.x0} x2={g.x1} y1={g.horizonY} y2={g.horizonY} style={line('ink', 1.1)} />
          <line x1={g.noonX} x2={g.noonX} y1={g.skyTop - 4} y2={g.horizonY} style={{ ...line('faint', 0.7), strokeDasharray: '2 3' }} />
          {[g.rise, g.set].map((x, i) => (
            <line key={i} x1={x} x2={x} y1={g.horizonY - 4} y2={g.horizonY + 4} style={line('soft', 0.9)} />
          ))}
          {g.sun.up ? (
            <circle cx={g.sun.x} cy={g.sun.y} r={6} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1 }} />
          ) : (
            <circle cx={g.sun.x} cy={g.horizonY} r={4} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1, strokeDasharray: '2 2' }} />
          )}
          {T('sun', 'soft')}
        </g>

        {/* row 2: true time vs the device */}
        <g style={{ opacity: op('clock') }}>
          {T('clockHead', 'soft')}
          <line x1={g.x0} x2={g.x1} y1={g.axisY} y2={g.axisY} style={line('ink', 1)} />
          {g.hourTicks.map((t) => (
            <g key={t.h}>
              <line x1={t.x} x2={t.x} y1={g.axisY - 3} y2={g.axisY + 3} style={line('soft', 0.8)} />
              <InkText x={t.x} y={g.axisY + 13} kind="num" anchor={t.h === 0 ? 'start' : t.h === 24 ? 'end' : 'middle'} size={9.5} role="soft" halo={false}>
                {`${t.h}h`}
              </InkText>
            </g>
          ))}
          <line x1={g.trueX} x2={g.trueX} y1={g.axisY - 9} y2={g.axisY + 4} style={line('ink', 2)} />
          {g.readX !== null && (
            <line x1={g.readX} x2={g.readX} y1={g.axisY - 9} y2={g.axisY + 4} style={line('red', 2)} />
          )}
          {T('true', 'ink')}
          {T('read', g.readX === null ? 'soft' : 'red')}
          {T('gap0', 'red')}
          {T('gap1', 'soft')}
        </g>

        {/* row 3: the error gauge */}
        <g style={{ opacity: op('gauge') }}>
          {T('gaugeHead', 'soft')}
          <line x1={g.x0} x2={g.x1} y1={g.gaugeY} y2={g.gaugeY} style={line('soft', 0.8)} />
          {g.gTicks.map((t) => (
            <g key={t.v}>
              <line x1={t.x} x2={t.x} y1={g.gaugeY - 3} y2={g.gaugeY + 3} style={line('soft', 0.8)} />
              <InkText x={t.x} y={g.gaugeY + 15} kind="num" anchor="middle" size={9.5} role="soft" halo={false}>
                {t.s}
              </InkText>
            </g>
          ))}
          {g.bands.map((b) => (
            <rect
              key={b.i}
              x={b.a}
              y={g.gaugeY - 4}
              width={b.b - b.a}
              height={8}
              style={{
                fill: b.cur ? (b.oom ? C.paper : C.red) : b.oom ? C.paper : C.faint,
                stroke: b.cur ? C.red : C.soft,
                strokeWidth: b.cur ? 1.2 : 0.6,
                opacity: b.cur ? 1 : 0.7,
              }}
            />
          ))}
          {g.rateX !== null && (
            <line x1={g.rateX} x2={g.rateX} y1={g.gaugeY - 8} y2={g.gaugeY + 8} style={line('red', 2.2)} />
          )}
          {T('dev', 'red')}
        </g>
      </svg>
    </div>
  );
}
