/**
 * What the theodolite's telescope shows: crosshairs and the targets at their offsets from the
 * line of sight (./theoGeom.ts eyepiece, the same function that decides "on target" for the
 * steps). Targets outside the field become arrows at its edge.
 */
import { C, InkText, useMeasuredWidth } from '../shared';
import { eyepiece, type Target } from './theoGeom';

export function Eyepiece({ h, v, targets, say }: { h: number; v: number; targets: Target[]; say: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = eyepiece(W, h, v, targets);
  const clip = `eye-${Math.round(W)}-${targets.map((t) => t.name).join('')}`;
  const anyOn = g.targets.some((t) => t.on);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Through the telescope: ${say}`}>
        <defs>
          <clipPath id={clip}>
            <circle cx={g.cx} cy={g.cy} r={g.R} />
          </clipPath>
        </defs>
        <circle cx={g.cx} cy={g.cy} r={g.R + 5} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
        <circle cx={g.cx} cy={g.cy} r={g.R} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 0.8 }} />
        <g clipPath={`url(#${clip})`}>
          {g.targets.map((t) =>
            t.inside ? (
              t.kind === 'pole' ? (
                <g key={t.name}>
                  <line x1={t.x} y1={t.y - 60} x2={t.x} y2={t.y + 60} style={{ stroke: t.on ? C.red : C.ink, strokeWidth: 4 }} />
                  <InkText x={t.x + 6} y={t.y - 30} size={12} kind="title" role={t.on ? 'red' : 'ink'}>{t.name}</InkText>
                </g>
              ) : (
                <g key={t.name}>
                  <path d={`M ${t.x} ${t.y} L ${t.x - 5} ${t.y + 40} L ${t.x + 5} ${t.y + 40} Z`} style={{ fill: C.paper, stroke: t.on ? C.red : C.ink, strokeWidth: 1.2 }} />
                  <rect x={t.x - 3} y={t.y + 40} width={6} height={60} style={{ fill: C.paper, stroke: t.on ? C.red : C.ink, strokeWidth: 1.2 }} />
                  <InkText x={t.x + 9} y={t.y + 4} size={11} role={t.on ? 'red' : 'ink'}>{t.name}</InkText>
                </g>
              )
            ) : null,
          )}
        </g>
        {g.targets.map((t) =>
          t.inside ? null : (
            <g key={`a-${t.name}`} transform={`translate(${t.edge.x} ${t.edge.y}) rotate(${(-t.edge.ang * 180) / Math.PI})`}>
              <path d="M 6 0 L -4 -5 L -4 5 Z" style={{ fill: C.soft }} />
            </g>
          ),
        )}
        <line x1={g.cx - g.R} y1={g.cy} x2={g.cx + g.R} y2={g.cy} style={{ stroke: C.ink, strokeWidth: 0.7 }} />
        <line x1={g.cx} y1={g.cy - g.R} x2={g.cx} y2={g.cy + g.R} style={{ stroke: C.ink, strokeWidth: 0.7 }} />
        <InkText x={W / 2} y={g.H - 6} anchor="middle" size={12} role={anyOn ? 'red' : 'soft'}>{say}</InkText>
      </svg>
    </div>
  );
}
