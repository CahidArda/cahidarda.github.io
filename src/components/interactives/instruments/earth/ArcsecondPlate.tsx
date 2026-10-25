/**
 * Earth, Plate 4: how an angle error grows with distance. Log-log chart: the sideways miss
 * of a sight line that is off by ε is d·tan ε, a straight line of slope 1. The Earth's
 * curvature drop below a level line, d²/2R, is a line of slope 2, and it overtakes the 1″
 * line at about 62 m. Drag along the chart (or use the slider) to move the distance marker.
 */
import { useRef, useState } from 'react';
import {
  C,
  Controls,
  Eq,
  InkText,
  Plate,
  Segmented,
  Slider,
  Sym,
  V,
  Working,
  useHoverPreview,
  useMeasuredWidth,
} from '../shared';
import { ARCSEC as ARC, CHART, EARTH_R as R, chartLayout, fmtLen } from './geometry';

export { fmtLen };

const Y_LABELS = ['1 µm', '100 µm', '1 cm', '1 m', '100 m', '10 km'];
type Eps = 0.2 | 1 | 200;
const EPS: { value: Eps; label: string; who: string }[] = [
  { value: 0.2, label: 'Wild T3, 0.2″', who: 'A 20th-century precision theodolite' },
  { value: 1, label: 'Ramsden, 1″', who: "Ramsden's great theodolite (1787), or a modern total station" },
  { value: 200, label: 'Pyramid, 200″', who: 'The Great Pyramid’s sides miss true north by 3 to 4 arcminutes' },
];
const MARKS = [
  { d: 76, name: 'minaret, 76 m' },
  { d: 700, name: 'Bosphorus, 700 m' },
  { d: 50_000, name: 'Nîmes aqueduct, 50 km' },
  { d: 160_000, name: 'Everest sights, 160 km' },
];
const fmtEpsV = (e: number) => `${e}″`;


export default function ArcsecondPlate() {
  const [sel, setSel] = useState<Eps>(1);
  const [eps, bindEps] = useHoverPreview(sel);
  const [logD, setLogD] = useState(Math.log10(160_000));
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const L = chartLayout(W, logD, eps);
  const { narrow, H, m, iw, ih, x, y, d, miss, ex, ey, cy } = L;
  const curve = L.drop;
  const { X0, X1, Y0, Y1 } = CHART;
  const line = (e: number) =>
    `M ${x(X0)} ${y(Math.log10(Math.tan(e * ARC)) + X0)} L ${x(X1)} ${y(Math.log10(Math.tan(e * ARC)) + X1)}`;
  const curvPath = () => {
    const pts: string[] = [];
    for (let ld = X0; ld <= X1 + 1e-9; ld += 0.25) {
      const le = 2 * ld - Math.log10(2 * R);
      pts.push(`${pts.length ? 'L' : 'M'} ${x(ld)} ${y(le)}`);
    }
    return pts.join(' ');
  };
  const onPointer = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === 'touch' && e.type === 'pointermove') return; // let touch scroll
    if (e.type === 'pointermove' && e.buttons === 0) return;
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return;
    const px = ((e.clientX - r.left) / r.width) * W;
    const ld = X0 + ((px - m.l) / iw) * (X1 - X0);
    setLogD(Math.max(0, Math.min(X1, ld)));
  };
  const who = EPS.find((o) => o.value === eps)?.who ?? '';
  return (
    <Plate
      n={4}
      title="How a small angle error grows with distance"
      kicker="log scales · drag along the chart"
      caption={
        <>
          The solid lines: how far sideways a sight line lands if its angle is off by a fixed
          amount. The farther the target, the bigger the miss. The dashed curve: how far the
          Earth falls away below a level line. For the {fmtEpsV(eps)} instrument the curve
          overtakes the miss beyond {fmtLen(L.crossover)}, which is why long surveys correct for
          the Earth’s shape.
        </>
      }
    >
      <Controls>
        <span className="in-field">angle error of the instrument</span>
        <Segmented label="Angle error" options={EPS.map(({ value, label }) => ({ value, label }))} value={sel} onChange={setSel} preview={bindEps} />
        <span className="in-field" style={{ textTransform: 'none', letterSpacing: 0, fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '0.9rem' }}>
          {who}
        </span>
      </Controls>
      <div ref={ref} style={{ marginTop: '0.6rem' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          style={{ display: 'block', height: 'auto', touchAction: 'pan-y', cursor: 'ew-resize' }}
          onPointerDown={onPointer}
          onPointerMove={onPointer}
        >
          {/* grid */}
          {Array.from({ length: X1 - X0 + 1 }, (_, i) => X0 + i).map((ld) => (
            <g key={`x${ld}`}>
              <line x1={x(ld)} y1={m.t} x2={x(ld)} y2={m.t + ih} style={{ stroke: C.faint, strokeWidth: 0.5 }} />
              {(!narrow || ld % 2 === 0) && (
                <InkText
                  x={x(ld)}
                  y={m.t + ih + 16}
                  kind="num"
                  anchor={ld === X1 ? 'end' : ld === X0 ? 'start' : 'middle'}
                  size={10}
                  role="soft"
                  halo={false}
                >
                  {['1 m', '10 m', '100 m', '1 km', '10 km', '100 km', '1000 km'][ld]}
                </InkText>
              )}
            </g>
          ))}
          {Array.from({ length: (Y1 - Y0) / 2 + 1 }, (_, i) => Y0 + i * 2).map((le) => (
            <g key={`y${le}`}>
              <line x1={m.l} y1={y(le)} x2={m.l + iw} y2={y(le)} style={{ stroke: C.faint, strokeWidth: 0.5 }} />
              <InkText x={m.l - 6} y={y(le)} kind="num" anchor="end" baseline="central" size={10} role="soft" halo={false}>
                {Y_LABELS[(le - Y0) / 2]}
              </InkText>
            </g>
          ))}
          <rect x={m.l} y={m.t} width={iw} height={ih} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }} />
          <InkText x={m.l + iw} y={H - 6} anchor="end" size={11} role="soft">
            distance to target
          </InkText>
          <InkText x={m.l + 4} y={m.t + 12} size={11} role="soft">
            sideways miss
          </InkText>
          {/* landmark distances */}
          {MARKS.map((mk) => (
            <g key={mk.name}>
              <line
                x1={x(Math.log10(mk.d))}
                y1={m.t + ih}
                x2={x(Math.log10(mk.d))}
                y2={m.t + ih - 6}
                style={{ stroke: C.ink, strokeWidth: 0.9 }}
              />
            </g>
          ))}
          <clipPath id="arc-clip">
            <rect x={m.l} y={m.t} width={iw} height={ih} />
          </clipPath>
          <g clipPath="url(#arc-clip)">
            <path d={curvPath()} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.3, strokeDasharray: '6 4' }} />
            {EPS.map((o) => (
              <path
                key={o.value}
                d={line(o.value)}
                style={{
                  fill: 'none',
                  stroke: o.value === eps ? C.red : C.soft,
                  strokeWidth: o.value === eps ? 1.8 : 0.9,
                }}
              />
            ))}
          </g>
          {L.curveLabel && (
            <InkText x={L.curveLabel.x} y={L.curveLabel.y} anchor="end" size={11} role="sky">
              Earth’s curve, d²/2R
            </InkText>
          )}
          {/* marker */}
          <line x1={ex} y1={m.t} x2={ex} y2={m.t + ih} style={{ stroke: C.ink, strokeWidth: 0.8, strokeDasharray: '2 3' }} />
          {L.dropOn && <circle cx={ex} cy={cy} r={3.2} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.5 }} />}
          {L.missOn && <circle cx={ex} cy={ey} r={4} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.8 }} />}
          {L.missOn && L.missLabel.leader && (
            <line x1={ex} y1={ey} x2={L.missLabel.x - 4} y2={L.missLabel.y - 4} style={{ stroke: C.red, strokeWidth: 0.6, strokeDasharray: '2 2' }} />
          )}
          {L.missOn && <InkText
            x={L.missLabel.x}
            y={L.missLabel.y}
            anchor={L.missLabel.anchor}
            kind="num"
            size={13}
            role="red"
          >
            {fmtLen(miss)}
          </InkText>}
        </svg>
      </div>
      <Controls>
        <Slider label="distance to target, d" value={logD} min={0} max={6} step={0.01} onChange={setLogD} format={(v) => fmtLen(10 ** v)} />
        <Segmented
          label="Example distances"
          options={MARKS.map((mk) => ({ value: mk.d, label: mk.name }))}
          value={Math.round(d)}
          onChange={(v) => setLogD(Math.log10(v))}
        />
      </Controls>
      <Working label="Sideways miss and the Earth's drop at this distance">
        <Eq>
          <Sym role="red">miss</Sym> = <Sym>d</Sym> · tan <Sym role="red">ε</Sym>
          {'   '}and{'   '}
          <Sym role="sky">drop</Sym> = <Sym>d</Sym>² / 2<Sym>R</Sym>
        </Eq>
        <Eq>
          <V role="red">{fmtLen(miss)}</V> = <V>{fmtLen(d)}</V> · tan <V role="red">{fmtEpsV(eps)}</V>
          {!L.missOn && <> ({L.missWhere} the chart)</>}
        </Eq>
        <Eq>
          <V role="sky">{fmtLen(curve)}</V> = (<V>{fmtLen(d)}</V>)² / (2 · <V>6,371 km</V>)
          {!L.dropOn && <> ({L.dropWhere} the chart)</>}
        </Eq>
      </Working>
    </Plate>
  );
}
