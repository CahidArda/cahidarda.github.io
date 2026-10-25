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
  InkText,
  Plate,
  Readout,
  Segmented,
  Slider,
  useHoverPreview,
  useMeasuredWidth,
} from '../shared';

const ARC = Math.PI / 180 / 3600;
const Y_LABELS = ['1 µm', '100 µm', '1 cm', '1 m', '100 m', '10 km'];
const R = 6_371_000;
type Eps = 0.2 | 1 | 200;
const EPS: { value: Eps; label: string; who: string }[] = [
  { value: 0.2, label: '0.2″', who: 'Wild T3 theodolite' },
  { value: 1, label: '1″', who: "Ramsden's great theodolite, a modern total station" },
  { value: 200, label: '200″', who: 'Great Pyramid alignment, about 3 to 4 arcmin' },
];
const MARKS = [
  { d: 76, name: 'minaret' },
  { d: 700, name: 'Bosphorus' },
  { d: 50_000, name: 'Nîmes' },
  { d: 160_000, name: 'Everest' },
];

export function fmtLen(m: number) {
  if (!Number.isFinite(m)) return '∞';
  if (m >= 1000) return `${(m / 1000).toFixed(m >= 10_000 ? 0 : 1)} km`;
  if (m >= 1) return `${m.toFixed(m >= 100 ? 0 : 2)} m`;
  if (m >= 0.01) return `${(m * 100).toFixed(1)} cm`;
  if (m >= 0.001) return `${(m * 1000).toFixed(1)} mm`;
  return `${(m * 1e6).toFixed(0)} µm`;
}

export default function ArcsecondPlate() {
  const [sel, setSel] = useState<Eps>(1);
  const [eps, bindEps] = useHoverPreview(sel);
  const [logD, setLogD] = useState(Math.log10(160_000));
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const narrow = W < 520;
  const H = narrow ? 300 : 330;
  const m = { l: narrow ? 50 : 60, r: narrow ? 12 : 20, t: 18, b: 40 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const X0 = 0; // 1 m
  const X1 = 6; // 1000 km
  const Y0 = -6; // 1 µm
  const Y1 = 4; // 10 km
  const x = (ld: number) => m.l + ((ld - X0) / (X1 - X0)) * iw;
  const y = (le: number) => m.t + (1 - (le - Y0) / (Y1 - Y0)) * ih;
  const d = 10 ** logD;
  const miss = d * Math.tan(eps * ARC);
  const curve = (d * d) / (2 * R);
  const line = (e: number) =>
    `M ${x(X0)} ${y(Math.log10(Math.tan(e * ARC)) + X0)} L ${x(X1)} ${y(Math.log10(Math.tan(e * ARC)) + X1)}`;
  const curvPath = () => {
    const pts: string[] = [];
    for (let ld = X0; ld <= X1 + 1e-9; ld += 0.25) {
      const le = 2 * ld - Math.log10(2 * R);
      pts.push(`${pts.length ? 'L' : 'M'} ${x(ld)} ${y(Math.max(Y0, le))}`);
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
  const ex = x(logD);
  const ey = y(Math.log10(miss));
  const cy = y(Math.max(Y0, Math.log10(curve)));
  return (
    <Plate
      n={4}
      title="One arcsecond, carried a long way"
      kicker="log scales"
      caption={
        <>
          A sight line that is off by a fixed angle misses by more the farther it goes: the solid
          lines. The dashed curve is how far the Earth falls away below a level line. For a
          1″ instrument the curve wins after about 62 m, which is why long surveys correct for
          it.
        </>
      }
    >
      <Controls>
        <Segmented label="Angle error" options={EPS.map(({ value, label }) => ({ value, label }))} value={sel} onChange={setSel} preview={bindEps} />
        <span className="in-field" style={{ textTransform: 'none', letterSpacing: 0, fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '0.9rem' }}>
          {who}
        </span>
      </Controls>
      <div ref={ref} style={{ marginTop: '0.6rem' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width={W}
          height={H}
          style={{ display: 'block', touchAction: 'pan-y', cursor: 'ew-resize' }}
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
          {!narrow && (
            <InkText x={x(4.9)} y={y(2 * 4.9 - Math.log10(2 * R)) - 10} anchor="end" size={11} role="sky">
              Earth’s curve, d²/2R
            </InkText>
          )}
          {/* marker */}
          <line x1={ex} y1={m.t} x2={ex} y2={m.t + ih} style={{ stroke: C.ink, strokeWidth: 0.8, strokeDasharray: '2 3' }} />
          <circle cx={ex} cy={cy} r={3} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.4 }} />
          <circle cx={ex} cy={ey} r={4} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.8 }} />
          <InkText
            x={ex + (logD > 4.6 ? -9 : 9)}
            y={ey + (ey < m.t + 30 ? 16 : -10)}
            anchor={logD > 4.6 ? 'end' : 'start'}
            kind="num"
            size={13}
            role="red"
          >
            {fmtLen(miss)}
          </InkText>
        </svg>
      </div>
      <Controls>
        <Slider label="distance" value={logD} min={0} max={6} step={0.01} onChange={setLogD} format={(v) => fmtLen(10 ** v)} />
        <Segmented
          label="Landmarks"
          options={MARKS.map((mk) => ({ value: mk.d, label: mk.name }))}
          value={Math.round(d)}
          onChange={(v) => setLogD(Math.log10(v))}
        />
        <Readout label="miss">{fmtLen(miss)}</Readout>
        <Readout label="curve">{fmtLen(curve)}</Readout>
      </Controls>
    </Plate>
  );
}
