/**
 * Earth: the triangulation drawings with their error regions, used as the last step of the
 * width and height procedure plates. Measure one length on the ground and two angles, get a
 * length you cannot walk. All coordinates come from ./geometry.ts (minaretLayout, straitLayout), which
 * the sweep script checks.
 *  - Minaret (elevation view): stations A (far) and B (near) in line with the foot, baseline b,
 *    elevation angles α at A and β at B: h = b·sin α·sin β / sin(β − α). True h = 76 m.
 *  - Bosphorus (plan view): baseline AB on the Asian shore, C on the European shore sighted
 *    square to the baseline at A (α = 90°) and at β from B: AC = b·sin β / sin(α + β).
 * Each angle may be off by ±ε. The red region is where the two sight lines can then cross (the
 * overlap of the two error wedges); the bracket beside it gives its lowest and highest point,
 * which is the range of the answer. It is not symmetric about the true value.
 * Colours: α and everything at A in sky, β and everything at B in brass, the baseline in ink,
 * the answer (height, width, error region) in red. The worked equation uses the same roles.
 */
import { C, DimLine, Eq, HatchDefs, InkText, V, arcPath, useMeasuredWidth } from '../shared';
import { deg, minaretLayout, straitLayout, type Pt } from './geometry';

export const fmtEps = (m: number) => (m === 0 ? '0′' : m < 1 ? `${Math.round(m * 60)}″` : `${m.toFixed(m % 1 ? 1 : 0)}′`);

/** A Süleymaniye-style pencil minaret, foot at (x, g), drawn tip exactly at g − Hpx. */
function Minaret({ x, g, Hpx }: { x: number; g: number; Hpx: number }) {
  const w = Math.max(4, Hpx * 0.07);
  const capH = Hpx * 0.16;
  const shaftTop = g - Hpx + capH;
  const base = Hpx * 0.18;
  const s = { fill: C.paper, stroke: C.ink, strokeWidth: 1 };
  const balcony = (y: number) => (
    <g key={y}>
      <path
        d={`M ${x - w * 1.05} ${y} L ${x + w * 1.05} ${y} L ${x + w * 0.7} ${y + w * 0.9} L ${x - w * 0.7} ${y + w * 0.9} Z`}
        style={s}
      />
      <rect x={x - w * 1.05} y={y - w * 0.55} width={w * 2.1} height={w * 0.55} style={s} />
    </g>
  );
  return (
    <g>
      <rect x={x - w * 0.95} y={g - base} width={w * 1.9} height={base} style={s} />
      <rect x={x - w * 0.62} y={shaftTop} width={w * 1.24} height={g - base - shaftTop} style={s} />
      {[0.45, 0.66].map((f) => balcony(g - Hpx * f))}
      <path d={`M ${x - w * 0.7} ${shaftTop} L ${x + w * 0.7} ${shaftTop} L ${x} ${g - Hpx} Z`} style={s} />
    </g>
  );
}

function Station({ x, y, role }: { x: number; y: number; role: 'sky' | 'brass' }) {
  return (
    <g>
      <path
        d={`M ${x} ${y} L ${x - 6} ${y + 9} M ${x} ${y} L ${x + 6} ${y + 9} M ${x} ${y} L ${x} ${y + 9}`}
        style={{ stroke: C[role], strokeWidth: 1.1 }}
      />
      <circle cx={x} cy={y} r={2.8} style={{ fill: C.paper, stroke: C[role], strokeWidth: 1.5 }} />
    </g>
  );
}

/** The region's lowest and highest corners (screen y largest and smallest). */
const extremes = (q: Pt[]) => [q.reduce((a, b) => (b.y > a.y ? b : a)), q.reduce((a, b) => (b.y < a.y ? b : a))];
const poly = (q: Pt[]) => `M ${q.map((p) => `${p.x} ${p.y}`).join(' L ')} Z`;

/** The range bracket: a vertical line from lo to hi with end ticks, an open arrow if clipped. */
function RangeBracket({ x, y1, y2, clippedTop }: { x: number; y1: number; y2: number; clippedTop: boolean }) {
  const st = { stroke: C.red, strokeWidth: 1.3, fill: 'none' };
  return (
    <g>
      <line x1={x} y1={y1} x2={x} y2={y2} style={st} />
      <line x1={x - 5} y1={y1} x2={x + 5} y2={y1} style={st} />
      {clippedTop ? (
        <path d={`M ${x - 4} ${y2 + 7} L ${x} ${y2} L ${x + 4} ${y2 + 7}`} style={st} />
      ) : (
        <line x1={x - 5} y1={y2} x2={x + 5} y2={y2} style={st} />
      )}
    </g>
  );
}

export function MinaretView({ dB, b, eps }: { dB: number; b: number; eps: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const L = minaretLayout(W, dB, b, eps);
  const { g, k, fx, ax, bx, al, be, tip, quad, range, rangeLabel } = L;
  return (
    <div ref={ref}>
    <svg viewBox={`0 0 ${W} ${L.Hs}`} width="100%" style={{ display: 'block', height: 'auto' }}>
      <HatchDefs id="tri-hatch" gap={6} />
      <clipPath id="tri-clip">
        <rect x={0} y={0} width={W} height={g} />
      </clipPath>
      <rect x={0} y={g} width={W} height={L.Hs - g} style={{ fill: 'url(#tri-hatch)' }} />
      <line x1={0} y1={g} x2={W} y2={g} style={{ stroke: C.ink, strokeWidth: 1.1 }} />
      {/* where the sight lines can cross, given ±ε on each angle */}
      {eps > 0 && quad && (
        <g clipPath="url(#tri-clip)">
          <path d={poly(quad)} style={{ fill: C.red, opacity: 0.18, stroke: C.red, strokeWidth: 0.8 }} />
          {/* level leaders from the region's lowest and highest corners to the bracket */}
          {extremes(quad).filter((p) => p.y > 1).map((p, i) => (
            <line
              key={i}
              x1={p.x}
              y1={p.y}
              x2={range.x}
              y2={p.y}
              style={{ stroke: C.red, strokeWidth: 0.6, strokeDasharray: '2 3', opacity: 0.8 }}
            />
          ))}
        </g>
      )}
      {/* sight lines */}
      <line x1={ax} y1={g} x2={tip.x} y2={tip.y} style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '5 3' }} />
      <line x1={bx} y1={g} x2={tip.x} y2={tip.y} style={{ stroke: C.brass, strokeWidth: 1, strokeDasharray: '5 3' }} />
      <path d={arcPath(ax, g, L.rA, 0, deg(al))} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.2 }} />
      <path d={arcPath(bx, g, L.rB, 0, deg(be))} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.2 }} />
      {L.alLabel.leader && (
        <line x1={L.alLabel.x} y1={L.alLabel.y + 7} x2={L.alArcMid.x} y2={L.alArcMid.y} style={{ stroke: C.sky, strokeWidth: 0.6 }} />
      )}
      {L.beLabel.leader && (
        <line x1={L.beLabel.x} y1={L.beLabel.y + 7} x2={L.beArcMid.x} y2={L.beArcMid.y} style={{ stroke: C.brass, strokeWidth: 0.6 }} />
      )}
      <InkText x={L.alLabel.x} y={L.alLabel.y} anchor="middle" baseline="central" size={14} role="sky">
        α
      </InkText>
      <InkText x={L.beLabel.x} y={L.beLabel.y} anchor="middle" baseline="central" size={14} role="brass">
        β
      </InkText>
      <Minaret x={fx} g={g} Hpx={L.h * k} />
      <DimLine
        x1={fx}
        y1={g}
        x2={fx}
        y2={tip.y}
        offset={-L.dimOff}
        label={L.hLabelText}
        role="red"
        labelRole="red"
        size={11}
      />
      {eps > 0 && (
        <g>
          <RangeBracket x={range.x} y1={range.y1} y2={range.y2} clippedTop={range.clippedTop} />
          <InkText
            x={rangeLabel.x}
            y={rangeLabel.y}
            anchor="middle"
            baseline="central"
            kind="num"
            size={rangeLabel.size}
            role="red"
            rotate={90}
          >
            {rangeLabel.text}
          </InkText>
        </g>
      )}
      <DimLine x1={ax} y1={L.baseDim.y} x2={bx} y2={L.baseDim.y} ends="tick" role="ink" />
      <InkText x={L.baseLabel.x} y={L.baseLabel.y} anchor="middle" kind="num" size={11}>
        {L.baseLabel.text}
      </InkText>
      <Station x={ax} y={g} role="sky" />
      <Station x={bx} y={g} role="brass" />
      <InkText x={L.nameA.x} y={L.nameA.y} anchor={L.nameA.anchor} kind="title" size={13} role="sky">
        A
      </InkText>
      <InkText x={L.nameB.x} y={L.nameB.y} anchor={L.nameB.anchor} kind="title" size={13} role="brass">
        B
      </InkText>
    </svg>
    </div>
  );
}

export function StraitView({ b, eps }: { b: number; eps: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const L = straitLayout(W, b, eps);
  const { nearY, farY, ax, bx, be, quad, range, rangeLabel } = L;
  const Hs = L.Hs;
  const shore = (y: number, seed: number) => {
    const pts: string[] = [];
    for (let i = 0; i <= 24; i++) {
      const x = (i / 24) * W;
      const yy = y + Math.sin(i * 0.9 + seed) * 3 + Math.sin(i * 2.3 + seed * 2) * 1.5;
      pts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${yy.toFixed(1)}`);
    }
    return pts.join(' ');
  };
  const ripples: string[] = [];
  for (let r = 0; r < 6; r++) {
    const y = farY + ((r + 0.5) / 6) * (nearY - farY);
    for (let i = 0; i < 5; i++) {
      const x = ((i + (r % 2) * 0.5 + 0.3) / 5) * W;
      ripples.push(`M ${x} ${y} q 6 -3 12 0 q 6 3 12 0`);
    }
  }
  return (
    <div ref={ref}>
    <svg viewBox={`0 0 ${W} ${Hs}`} width="100%" style={{ display: 'block', height: 'auto' }}>
      <HatchDefs id="st-hatch" gap={6} />
      <clipPath id="st-clip">
        <rect x={0} y={0} width={W} height={nearY} />
      </clipPath>
      <rect x={0} y={farY} width={W} height={nearY - farY} style={{ fill: C.sky, opacity: 0.06 }} />
      <path d={ripples.join(' ')} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.8 }} />
      <path d={`${shore(farY, 1)} L ${W} 0 L 0 0 Z`} style={{ fill: 'url(#st-hatch)', stroke: 'none' }} />
      <path d={shore(farY, 1)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      <path d={`${shore(nearY, 4)} L ${W} ${Hs} L 0 ${Hs} Z`} style={{ fill: 'url(#st-hatch)', stroke: 'none' }} />
      <path d={shore(nearY, 4)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      <InkText x={W - 10} y={farY - 12} anchor="end" size={12} role="soft">
        Europe
      </InkText>
      <InkText x={W - 10} y={nearY + 24} anchor="end" size={12} role="soft">
        Asia
      </InkText>
      {eps > 0 && quad && (
        <g clipPath="url(#st-clip)">
          <path d={poly(quad)} style={{ fill: C.red, opacity: 0.2, stroke: C.red, strokeWidth: 0.8 }} />
          {extremes(quad).filter((p) => p.y > 1).map((p, i) => (
            <line
              key={i}
              x1={range.x}
              y1={p.y}
              x2={p.x}
              y2={p.y}
              style={{ stroke: C.red, strokeWidth: 0.6, strokeDasharray: '2 3', opacity: 0.8 }}
            />
          ))}
        </g>
      )}
      {/* sight lines: A square across, B at β */}
      <line x1={ax} y1={nearY} x2={L.C.x} y2={L.C.y} style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '5 3' }} />
      <line x1={bx} y1={nearY} x2={L.C.x} y2={L.C.y} style={{ stroke: C.brass, strokeWidth: 1, strokeDasharray: '5 3' }} />
      <line x1={ax} y1={nearY} x2={bx} y2={nearY} style={{ stroke: C.ink, strokeWidth: 1.6 }} />
      <path
        d={`M ${ax + 10} ${nearY} L ${ax + 10} ${nearY - 10} L ${ax} ${nearY - 10}`}
        style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.2 }}
      />
      <path d={arcPath(bx, nearY, L.rB, 180 - deg(be), 180)} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.2 }} />
      <InkText x={L.beLabel.x} y={L.beLabel.y} anchor="middle" baseline="central" size={14} role="brass">
        β
      </InkText>
      <circle cx={L.C.x} cy={L.C.y} r={3} style={{ fill: C.red }} />
      <InkText x={ax + 8} y={L.C.y - 8} kind="title" size={13} role="red">
        C
      </InkText>
      <DimLine
        x1={ax}
        y1={nearY}
        x2={ax}
        y2={L.C.y}
        offset={L.dimOff}
        label={L.wText}
        role="red"
        labelRole="red"
        size={11}
      />
      {eps > 0 && (
        <g>
          <RangeBracket x={range.x} y1={range.y1} y2={range.y2} clippedTop={range.clippedTop} />
          <InkText
            x={rangeLabel.x}
            y={rangeLabel.y}
            anchor="middle"
            baseline="central"
            kind="num"
            size={rangeLabel.size}
            role="red"
            rotate={-90}
          >
            {rangeLabel.text}
          </InkText>
        </g>
      )}
      <DimLine x1={ax} y1={nearY + 28} x2={bx} y2={nearY + 28} ends="tick" role="ink" />
      <InkText x={L.baseLabel.x} y={L.baseLabel.y} anchor="middle" kind="num" size={11}>
        {L.baseLabel.text}
      </InkText>
      <circle cx={ax} cy={nearY} r={2.8} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.5 }} />
      <circle cx={bx} cy={nearY} r={2.8} style={{ fill: C.paper, stroke: C.brass, strokeWidth: 1.5 }} />
      <InkText x={ax - 8} y={nearY + 4} anchor="end" kind="title" size={13} role="sky">
        A
      </InkText>
      <InkText x={bx + 8} y={nearY + 4} kind="title" size={13} role="brass">
        B
      </InkText>
    </svg>
    </div>
  );
}

/** "63.5 to 94.1 m, i.e. 12.5 m low to 18.1 m high", or the unbounded case, said honestly. */
export function RangeLine({ lo, hi, truth, eps, dp }: { lo: number; hi: number; truth: number; eps: number; dp: number }) {
  return (
    <Eq>
      with each angle off by up to <V>{fmtEps(eps)}</V>:{' '}
      {Number.isFinite(hi) ? (
        <>
          <V role="red">{`${lo.toFixed(dp)} m`}</V> to <V role="red">{`${hi.toFixed(dp)} m`}</V> (
          <V>{`${(truth - lo).toFixed(dp)} m`}</V> low to <V>{`${(hi - truth).toFixed(dp)} m`}</V> high)
        </>
      ) : (
        <>
          anything above <V role="red">{`${lo.toFixed(0)} m`}</V>: the lines may never cross
        </>
      )}
    </Eq>
  );
}

