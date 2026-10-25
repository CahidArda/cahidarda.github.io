/**
 * Earth, Plate 3: triangulation. Measure one length on the ground and two angles, get a
 * length you cannot walk.
 *  - Minaret: two stations in line with the foot, a baseline b between them, elevation
 *    angles α (far) and β (near): h = b·tanα·tanβ / (tanβ − tanα). True h = 76 m.
 *  - Bosphorus: a baseline AB along the shore; C on the far shore sighted square to A
 *    (α = 90°) and at β from B: w = b·sinα·sinβ / sin(α+β). True w = 700 m.
 * The error slider perturbs each angle by ±ε (worst case of the four sign combinations)
 * and paints the resulting uncertainty in red on the thing being measured.
 */
import { useState } from 'react';
import {
  C,
  Controls,
  DimLine,
  HatchDefs,
  InkText,
  Plate,
  Readout,
  Segmented,
  Slider,
  arcPath,
  useHoverPreview,
  useMeasuredWidth,
} from '../shared';

type Mode = 'minaret' | 'strait';
const RAD = Math.PI / 180;
const H_TRUE = 76;
const W_TRUE = 700;

const deg = (r: number) => r / RAD;
const fmtDeg = (r: number) => `${deg(r).toFixed(2)}°`;
const fmtEps = (m: number) => (m === 0 ? '0′' : m < 1 ? `${Math.round(m * 60)}″` : `${m.toFixed(0)}′`);

function heightFrom(b: number, a: number, be: number) {
  const ta = Math.tan(a);
  const tb = Math.tan(be);
  return (b * ta * tb) / (tb - ta);
}
function widthFrom(b: number, a: number, be: number) {
  return (b * Math.sin(a) * Math.sin(be)) / Math.sin(a + be);
}
function spread(f: (a: number, b: number) => number, a: number, b: number, e: number) {
  const vals = [
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
  ].map(([s1, s2]) => f(a + s1 * e, b + s2 * e));
  const ok = vals.filter((v) => Number.isFinite(v) && v > 0);
  if (ok.length < 4) return { lo: Math.min(...ok, 0), hi: Infinity };
  return { lo: Math.min(...ok), hi: Math.max(...ok) };
}

/** A Süleymaniye-style pencil minaret, drawn from its foot at (x, g), height Hpx. */
function Minaret({ x, g, Hpx }: { x: number; g: number; Hpx: number }) {
  const w = Math.max(5, Hpx * 0.07);
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
      <path
        d={`M ${x - w * 0.7} ${shaftTop} L ${x + w * 0.7} ${shaftTop} L ${x} ${g - Hpx} Z`}
        style={s}
      />
      <line x1={x} y1={g - Hpx} x2={x} y2={g - Hpx - 3} style={{ stroke: C.ink, strokeWidth: 1 }} />
    </g>
  );
}

function Station({ x, y, name }: { x: number; y: number; name: string }) {
  return (
    <g>
      <path
        d={`M ${x} ${y} L ${x - 6} ${y + 9} M ${x} ${y} L ${x + 6} ${y + 9} M ${x} ${y} L ${x} ${y + 9}`}
        style={{ stroke: C.ink, strokeWidth: 1 }}
      />
      <circle cx={x} cy={y} r={2.6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
      <InkText x={x} y={y + 22} anchor="middle" kind="title" size={13}>
        {name}
      </InkText>
    </g>
  );
}

function MinaretView({ W, dB, b, eps }: { W: number; dB: number; b: number; eps: number }) {
  const narrow = W < 520;
  const right = W - (narrow ? 40 : 70);
  const k = (right - 24) / 300; // px per metre, same both ways
  const Hs = Math.round(H_TRUE * k + (narrow ? 130 : 150));
  const g = Hs - 62;
  const dA = dB + b;
  const fx = right;
  const ax = fx - dA * k;
  const bx = fx - dB * k;
  const top = { x: fx, y: g - H_TRUE * k };
  const al = Math.atan(H_TRUE / dA);
  const be = Math.atan(H_TRUE / dB);
  const h = heightFrom(b, al, be);
  const sp = spread((x, y) => heightFrom(b, x, y), al, be, eps);
  const lo = Math.max(0, sp.lo);
  const hi = Math.min(sp.hi, 400);
  const clampY = (m: number) => Math.max(8, g - m * k);
  return (
    <svg viewBox={`0 0 ${W} ${Hs}`} width={W} height={Hs} style={{ display: 'block' }}>
      <HatchDefs id="tri-hatch" gap={6} />
      <rect x={0} y={g} width={W} height={Hs - g} style={{ fill: 'url(#tri-hatch)' }} />
      <line x1={0} y1={g} x2={W} y2={g} style={{ stroke: C.ink, strokeWidth: 1.1 }} />
      {/* sight lines */}
      {[ax, bx].map((x) => (
        <line
          key={x}
          x1={x}
          y1={g}
          x2={top.x}
          y2={top.y}
          style={{ stroke: C.soft, strokeWidth: 0.9, strokeDasharray: '5 3' }}
        />
      ))}
      <path
        d={arcPath(ax, g, 34, 0, deg(al))}
        style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }}
      />
      <path
        d={arcPath(bx, g, 26, 0, deg(be))}
        style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }}
      />
      <InkText x={ax + 40} y={g - 6} size={13}>
        α
      </InkText>
      <InkText x={bx + 31} y={g - 6} size={13}>
        β
      </InkText>
      {/* uncertainty band on the minaret */}
      {eps > 0 && (
        <g>
          <rect
            x={fx - 14}
            y={clampY(hi)}
            width={28}
            height={Math.max(1.5, clampY(lo) - clampY(hi))}
            style={{ fill: C.red, opacity: 0.16 }}
          />
          <line
            x1={fx - 16}
            y1={clampY(hi)}
            x2={fx + 16}
            y2={clampY(hi)}
            style={{ stroke: C.red, strokeWidth: 1.4 }}
          />
          <line
            x1={fx - 16}
            y1={clampY(lo)}
            x2={fx + 16}
            y2={clampY(lo)}
            style={{ stroke: C.red, strokeWidth: 1.4 }}
          />
        </g>
      )}
      <Minaret x={fx} g={g} Hpx={H_TRUE * k} />
      <DimLine x1={fx} y1={g} x2={fx} y2={top.y} offset={-(narrow ? 20 : 26)} label={`${h.toFixed(1)} m`} size={11} />
      <DimLine x1={ax} y1={g + 40} x2={bx} y2={g + 40} ends="tick" />
      <InkText x={(ax + bx) / 2} y={g + 56} anchor="middle" kind="num" size={11}>{`b = ${b} m`}</InkText>
      <Station x={ax} y={g} name="A" />
      <Station x={bx} y={g} name="B" />
      {eps > 0 && (
        <InkText x={fx - 20} y={Math.max(18, clampY(hi) - 8)} anchor="end" size={12} role="red">
          {Number.isFinite(sp.hi) ? `${lo.toFixed(0)} to ${hi.toFixed(0)} m` : 'no answer at all'}
        </InkText>
      )}
    </svg>
  );
}

function StraitView({ W, b, eps }: { W: number; b: number; eps: number }) {
  const narrow = W < 520;
  const Hs = narrow ? 310 : 330;
  const k = (Hs - 120) / W_TRUE;
  const nearY = Hs - 62;
  const farY = nearY - W_TRUE * k;
  const ax = narrow ? 70 : W * 0.28;
  const bx = ax + b * k;
  const al = Math.PI / 2;
  const be = Math.atan(W_TRUE / b);
  const w = widthFrom(b, al, be);
  const sp = spread((x, y) => widthFrom(b, x, y), al, be, eps);
  const hi = Math.min(sp.hi, 2500);
  const lo = sp.lo;
  const yOf = (m: number) => Math.max(10, nearY - m * k);
  const shore = (y: number, seed: number) => {
    const pts: string[] = [];
    for (let i = 0; i <= 24; i++) {
      const x = (i / 24) * W;
      const yy = y + Math.sin(i * 0.9 + seed) * 3 + Math.sin(i * 2.3 + seed * 2) * 1.5;
      pts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${yy.toFixed(1)}`);
    }
    return pts.join(' ');
  };
  // water ripples
  const ripples: string[] = [];
  for (let r = 0; r < 6; r++) {
    const y = farY + ((r + 0.5) / 6) * (nearY - farY);
    for (let i = 0; i < 5; i++) {
      const x = ((i + (r % 2) * 0.5 + 0.3) / 5) * W;
      ripples.push(`M ${x} ${y} q 6 -3 12 0 q 6 3 12 0`);
    }
  }
  return (
    <svg viewBox={`0 0 ${W} ${Hs}`} width={W} height={Hs} style={{ display: 'block' }}>
      <HatchDefs id="st-hatch" gap={6} />
      <rect x={0} y={farY} width={W} height={nearY - farY} style={{ fill: C.sky, opacity: 0.07 }} />
      <path d={ripples.join(' ')} style={{ fill: 'none', stroke: C.sky, strokeWidth: 0.8, opacity: 0.6 }} />
      <path d={`${shore(farY, 1)} L ${W} 0 L 0 0 Z`} style={{ fill: 'url(#st-hatch)', stroke: 'none' }} />
      <path d={shore(farY, 1)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      <path d={`${shore(nearY, 4)} L ${W} ${Hs} L 0 ${Hs} Z`} style={{ fill: 'url(#st-hatch)', stroke: 'none' }} />
      <path d={shore(nearY, 4)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
      <InkText x={W - 10} y={farY - 10} anchor="end" size={12} role="soft">
        Europe
      </InkText>
      <InkText x={W - 10} y={nearY + 22} anchor="end" size={12} role="soft">
        Asia
      </InkText>
      {/* uncertainty along A's square line */}
      {eps > 0 && (
        <g>
          <rect
            x={ax - 9}
            y={yOf(hi)}
            width={18}
            height={Math.max(1.5, yOf(lo) - yOf(hi))}
            style={{ fill: C.red, opacity: 0.16 }}
          />
          <line x1={ax - 11} y1={yOf(hi)} x2={ax + 11} y2={yOf(hi)} style={{ stroke: C.red, strokeWidth: 1.4 }} />
          <line x1={ax - 11} y1={yOf(lo)} x2={ax + 11} y2={yOf(lo)} style={{ stroke: C.red, strokeWidth: 1.4 }} />
        </g>
      )}
      {/* triangle */}
      <path
        d={`M ${ax} ${nearY} L ${ax} ${farY} M ${bx} ${nearY} L ${ax} ${farY}`}
        style={{ stroke: C.soft, strokeWidth: 0.9, strokeDasharray: '5 3' }}
      />
      <line x1={ax} y1={nearY} x2={bx} y2={nearY} style={{ stroke: C.ink, strokeWidth: 1.6 }} />
      <path
        d={`M ${ax + 10} ${nearY} L ${ax + 10} ${nearY - 10} L ${ax} ${nearY - 10}`}
        style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }}
      />
      <path d={arcPath(bx, nearY, 26, 180 - deg(be), 180)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.9 }} />
      <InkText x={bx - 34} y={nearY - 8} anchor="end" size={13}>
        β
      </InkText>
      <circle cx={ax} cy={farY} r={3} style={{ fill: C.ink }} />
      <InkText x={ax + 8} y={farY - 8} kind="title" size={13}>
        C
      </InkText>
      <DimLine x1={ax} y1={nearY} x2={ax} y2={farY} offset={narrow ? 26 : 34} label={`${w.toFixed(0)} m`} size={11} />
      <DimLine x1={ax} y1={nearY + 26} x2={bx} y2={nearY + 26} ends="tick" />
      <InkText x={(ax + bx) / 2} y={nearY + 42} anchor="middle" kind="num" size={11}>{`b = ${b} m`}</InkText>
      <circle cx={ax} cy={nearY} r={2.6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
      <circle cx={bx} cy={nearY} r={2.6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
      <InkText x={ax - 8} y={nearY + 4} anchor="end" kind="title" size={13}>
        A
      </InkText>
      <InkText x={bx + 8} y={nearY + 4} kind="title" size={13}>
        B
      </InkText>
      {eps > 0 && (
        <InkText x={ax + 16} y={Math.max(20, yOf(hi) + (narrow ? 14 : 4))} size={12} role="red">
          {Number.isFinite(sp.hi) ? `${lo.toFixed(0)} to ${hi.toFixed(0)} m` : 'anywhere past the hills'}
        </InkText>
      )}
    </svg>
  );
}

export default function TriangulationPlate() {
  const [sel, setSel] = useState<Mode>('minaret');
  const [mode, bindMode] = useHoverPreview(sel);
  const [dB, setDB] = useState(100);
  const [bm, setBm] = useState(50);
  const [bs, setBs] = useState(200);
  const [eps, setEps] = useState(30); // arcminutes
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const e = (eps / 60) * RAD;
  const bMin = Math.min(bm, 300 - dB);
  const al = Math.atan(H_TRUE / (dB + bMin));
  const be = Math.atan(H_TRUE / dB);
  const sp =
    mode === 'minaret'
      ? spread((x, y) => heightFrom(bMin, x, y), al, be, e)
      : spread((x, y) => widthFrom(bs, x, y), Math.PI / 2, Math.atan(W_TRUE / bs), e);
  const truth = mode === 'minaret' ? H_TRUE : W_TRUE;
  const errTxt = !Number.isFinite(sp.hi)
    ? 'unbounded'
    : `±${(Math.max(truth - sp.lo, sp.hi - truth)).toFixed(mode === 'minaret' ? 1 : 0)} m`;
  return (
    <Plate
      n={2}
      title={mode === 'minaret' ? 'How tall is the minaret?' : 'How wide is the Bosphorus?'}
      kicker="one length, two angles"
      caption={
        mode === 'minaret' ? (
          <>
            Walk a baseline toward the Süleymaniye minaret and read the elevation angle at both
            ends. Two angles and one length fix the triangle. Push the stations close together,
            or far away, and watch the same angle error blow up.
          </>
        ) : (
          <>
            Pace a baseline along the Asian shore. At A, sight a point C on the European shore at
            a right angle; at B, read the angle to C. Plan view, to scale. A short baseline makes
            the triangle a sliver, and slivers are where angle errors hide.
          </>
        )
      }
    >
      <Controls>
        <Segmented
          label="Problem"
          options={[
            { value: 'minaret', label: 'Minaret' },
            { value: 'strait', label: 'Bosphorus' },
          ]}
          value={sel}
          onChange={setSel}
          preview={bindMode}
        />
      </Controls>
      <div ref={ref} style={{ marginTop: '0.6rem' }}>
        {mode === 'minaret' ? (
          <MinaretView W={W} dB={dB} b={bMin} eps={e} />
        ) : (
          <StraitView W={W} b={bs} eps={e} />
        )}
      </div>
      <Controls>
        {mode === 'minaret' ? (
          <>
            <Slider label="B to foot" value={dB} min={20} max={250} step={5} onChange={setDB} format={(v) => `${v} m`} />
            <Slider
              label="baseline b"
              value={bMin}
              min={5}
              max={Math.max(5, 300 - dB)}
              step={5}
              onChange={setBm}
              format={(v) => `${v} m`}
            />
            <Readout label="α">{fmtDeg(al)}</Readout>
            <Readout label="β">{fmtDeg(be)}</Readout>
          </>
        ) : (
          <>
            <Slider label="baseline b" value={bs} min={20} max={600} step={10} onChange={setBs} format={(v) => `${v} m`} />
            <Readout label="α">90°</Readout>
            <Readout label="β">{fmtDeg(Math.atan(W_TRUE / bs))}</Readout>
          </>
        )}
        <Slider label="angle error ±" value={eps} min={0} max={60} step={0.5} onChange={setEps} format={fmtEps} />
        <Readout label="result">{errTxt}</Readout>
      </Controls>
    </Plate>
  );
}
