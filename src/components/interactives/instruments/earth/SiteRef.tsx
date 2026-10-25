/**
 * The stable reference drawing shared by every earth plate: the same composite landscape
 * (./site.ts) in section and plan, with the current episode's quantity highlighted and an
 * optional overlay drawn in the same coordinates. Layout never changes between steps.
 */
import type { ReactNode } from 'react';
import { C, DimLine, HatchDefs, InkText, useMeasuredWidth, type Role } from '../shared';
import { SITE, siteScene, type SiteScene } from './site';

export type SiteKey = 'minaret' | 'slope' | 'line' | 'road' | 'strait' | 'height' | 'baseline' | 'fall' | 'channel' | 'wall';

/** Width of one character of the 10px mono tag with its letter spacing (px). */
const TAG_EM = 6.9;

const solid = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

/** The minaret in section, foot at f, tip at t (drawn to the section's scale). */
function MinaretIcon({ f, t, hi }: { f: { x: number; y: number }; t: { x: number; y: number }; hi: boolean }) {
  const h = f.y - t.y;
  const w = Math.max(2.2, h * 0.05);
  const cap = h * 0.16;
  const s = { fill: C.paper, stroke: hi ? C.red : C.ink, strokeWidth: hi ? 1.2 : 0.9 };
  return (
    <g>
      <rect x={f.x - w * 0.6} y={t.y + cap} width={w * 1.2} height={h - cap} style={s} />
      <rect x={f.x - w} y={f.y - h * 0.47} width={w * 2} height={w * 0.7} style={s} />
      <rect x={f.x - w} y={f.y - h * 0.68} width={w * 2} height={w * 0.7} style={s} />
      <path d={`M ${f.x - w * 0.6} ${t.y + cap} L ${f.x} ${t.y} L ${f.x + w * 0.6} ${t.y + cap} Z`} style={s} />
      {/* the dome beside it, for scale and recognition */}
      <path d={`M ${f.x + w * 3} ${f.y} a ${h * 0.24} ${h * 0.24} 0 0 1 ${h * 0.48} 0`} style={{ ...s, stroke: C.ink, strokeWidth: 0.8 }} />
    </g>
  );
}

export function SiteRef({
  hi = [],
  best,
  overlay,
  label,
}: {
  /** Quantities to highlight. */
  hi?: SiteKey[];
  /** "best so far" line at the top. A string is squeezed to fit the width if it is too long. */
  best?: string;
  /** Extra marks in the scene's coordinates (sight lines, chain, beam, error regions). */
  overlay?: (g: SiteScene) => ReactNode;
  /** Accessible description. */
  label: string;
}) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = siteScene(W);
  const on = new Set(hi);
  const { names: n, labels: L } = g;
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  const id = `site-h-${W}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={label}>
        <HatchDefs id={id} gap={5} role="faint" />
        {best && (
          <text
            x={g.tag.x}
            y={g.tag.y}
            style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.08em', fill: C.soft }}
            {...(typeof best === 'string' && best.length * TAG_EM > W - 2 * g.tag.x
              ? { textLength: W - 2 * g.tag.x, lengthAdjust: 'spacingAndGlyphs' }
              : {})}
          >
            {best}
          </text>
        )}
        {/* ── section ── */}
        <InkText x={L.section.x} y={L.section.y} anchor="end" kind="label" size={L.section.size} role="soft">
          {L.section.text}
        </InkText>
        <rect x={g.nearShoreX} y={g.sea} width={g.farShoreX - g.nearShoreX} height={g.secBottom - g.sea} style={{ fill: C.sky, opacity: 0.16 }} />
        <line x1={g.nearShoreX} y1={g.sea} x2={g.farShoreX} y2={g.sea} style={solid('sky', on.has('strait') ? 1.4 : 0.9)} />
        <path d={`${path(g.nearGround)} L ${g.nearShoreX} ${g.secBottom} L ${g.pad.l} ${g.secBottom} Z`} style={{ fill: `url(#${id})`, stroke: 'none' }} />
        <path d={`${path(g.farGround)} L ${W - g.pad.r} ${g.secBottom} L ${g.farShoreX} ${g.secBottom} Z`} style={{ fill: `url(#${id})`, stroke: 'none' }} />
        <path d={path(g.nearGround)} style={solid('ink', 1.1)} />
        <path d={path(g.farGround)} style={solid('ink', 1.1)} />
        {on.has('channel') && <line x1={n.flat.x} y1={n.flat.y - 2} x2={n.crest.x} y2={n.crest.y - 2} style={solid('red', 2.6)} />}
        {on.has('slope') && <line x1={n.crest.x} y1={n.crest.y} x2={n.A.x} y2={n.A.y} style={solid('red', 2.2)} />}
        <MinaretIcon f={n.M} t={n.tip} hi={on.has('minaret')} />
        {on.has('minaret') && <line x1={n.M.x} y1={n.tip.y - 6} x2={n.M.x} y2={n.M.y + 8} style={{ ...solid('red', 0.8), strokeDasharray: '3 2' }} />}
        {on.has('height') && <DimLine x1={n.M.x - 7} y1={n.M.y} x2={n.M.x - 7} y2={n.tip.y} role="red" />}
        {on.has('fall') && <DimLine x1={n.A.x + 8} y1={n.M.y} x2={n.A.x + 8} y2={n.A.y} role="red" />}
        {on.has('fall') && <line x1={n.M.x} y1={n.M.y} x2={n.A.x + 12} y2={n.M.y} style={{ ...solid('red', 0.7), strokeDasharray: '2 3' }} />}
        {on.has('line') && <DimLine x1={n.M.x} y1={g.secBottom - 6} x2={n.A.x} y2={g.secBottom - 6} role="red" />}
        {on.has('strait') && <DimLine x1={g.nearShoreX} y1={g.sea + 10} x2={g.farShoreX} y2={g.sea + 10} role="red" />}
        <circle cx={n.A.x} cy={n.A.y} r={2.2} style={{ fill: C.ink }} />
        <circle cx={n.C.x} cy={n.C.y} r={2.2} style={{ fill: C.ink }} />
        <InkText x={L.M.x} y={L.M.y} size={L.M.size} kind="title">M</InkText>
        <InkText x={L.A.x} y={L.A.y} anchor="end" size={L.A.size} kind="title">A</InkText>
        <InkText x={L.C.x} y={L.C.y} size={L.C.size} kind="title">C</InkText>

        {/* ── plan ── */}
        <InkText x={L.plan.x} y={L.plan.y} anchor="end" kind="label" size={L.plan.size} role="soft">
          {L.plan.text}
        </InkText>
        <rect x={g.nearShoreX} y={g.planTop} width={g.farShoreX - g.nearShoreX} height={g.planH} style={{ fill: C.sky, opacity: 0.12 }} />
        <rect x={g.pad.l} y={g.planTop} width={g.nearShoreX - g.pad.l} height={g.planH} style={{ fill: `url(#${id})`, opacity: 0.55 }} />
        <rect x={g.farShoreX} y={g.planTop} width={W - g.pad.r - g.farShoreX} height={g.planH} style={{ fill: `url(#${id})`, opacity: 0.55 }} />
        <line x1={g.nearShoreX} y1={g.planTop} x2={g.nearShoreX} y2={g.planTop + g.planH} style={solid('ink', 1.1)} />
        <line x1={g.farShoreX} y1={g.planTop} x2={g.farShoreX} y2={g.planTop + g.planH} style={solid('ink', 1.1)} />
        <rect x={g.pad.l} y={g.planTop} width={W - g.pad.l - g.pad.r} height={g.planH} style={solid('faint', 0.6)} />
        <path d={path(g.road)} style={{ ...solid(on.has('road') ? 'red' : 'soft', on.has('road') ? 1.8 : 0.9), strokeDasharray: on.has('road') ? undefined : '3 2' }} />
        {on.has('line') && <line x1={n.pM.x} y1={n.pM.y} x2={n.pA.x} y2={n.pA.y} style={solid('red', 1.6)} />}
        {on.has('strait') && <line x1={n.pA.x} y1={n.pA.y} x2={n.pC.x} y2={n.pC.y} style={solid('red', 1.6)} />}
        {on.has('baseline') && <line x1={n.pA.x} y1={n.pA.y} x2={n.pB.x} y2={n.pB.y} style={solid('red', 2)} />}
        {overlay?.(g)}
        <rect x={n.pM.x - 3} y={n.pM.y - 3} width={6} height={6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
        {[n.pA, n.pB, n.pC].map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={2.4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
        ))}
        <InkText x={L.pM.x} y={L.pM.y} size={L.pM.size} kind="title">M</InkText>
        <InkText x={L.pA.x} y={L.pA.y} anchor="end" size={L.pA.size} kind="title">A</InkText>
        <InkText x={L.pB.x} y={L.pB.y} anchor="end" size={L.pB.size} kind="title">B</InkText>
        <InkText x={L.pC.x} y={L.pC.y} size={L.pC.size} kind="title">C</InkText>
        <path d={`M ${L.north.x} ${L.north.y + 16} V ${L.north.y + 4} M ${L.north.x - 3} ${L.north.y + 8} L ${L.north.x} ${L.north.y + 4} L ${L.north.x + 3} ${L.north.y + 8}`} style={solid('soft', 0.9)} />
        <InkText x={L.north.x} y={L.north.y} anchor="middle" kind="label" size={L.north.size} role="soft">N</InkText>
        <InkText x={g.pad.l + 2} y={g.H - 8} size={10} role="soft">
          {`M minaret (${SITE.MINARET_H} m) · A, B near quay · C far quay`}
        </InkText>
      </svg>
    </div>
  );
}
