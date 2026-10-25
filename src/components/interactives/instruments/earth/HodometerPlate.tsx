/**
 * Episode 4 (small plate): Vitruvius's hodometer (De architectura 10.9). A cart wheel 4 ft
 * across; a one-tooth pin on its hub turns a 400-tooth drum one tooth per revolution; after a
 * full turn of the drum a pebble drops into a bronze bowl: one Roman mile. Vitruvius counts a
 * revolution as 12½ ft, but π × 4 ft is 12.57 ft, so the drum's mile is 5,027 ft, not 5,000.
 * Pure numbers in hodometer(); the drawing maps them. Distances in Roman feet.
 */
import { useState } from 'react';
import { C, Eq, InkText, Plate, Slider, Sym, V, Working, polar, useMeasuredWidth } from '../shared';

export const WHEEL_FT = 4;
export const TEETH = 400;
export const VITRUVIUS_REV = 12.5;
export const REV = Math.PI * WHEEL_FT;
export const MILE_FT = 5000;
export function hodometer(ft: number) {
  const revs = ft / REV;
  const teeth = Math.floor(revs);
  const pebbles = Math.floor(teeth / TEETH);
  const counted = pebbles * MILE_FT; // what the bowl says
  const wheelDeg = (revs % 1) * 360;
  const drumDeg = ((teeth % TEETH) / TEETH) * 360;
  return { revs, teeth, pebbles, counted, wheelDeg, drumDeg };
}

export default function HodometerPlate() {
  const [ft, setFt] = useState(15_100);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(560);
  const h = hodometer(ft);
  const H = 220;
  const wc = { x: Math.min(110, W * 0.26), y: 120, r: Math.min(80, W * 0.2) };
  const dc = { x: wc.x + wc.r + Math.min(90, W * 0.22), y: 70, r: 44 };
  const bowl = { x: dc.x + Math.min(110, W * 0.26), y: 160 };
  const spoke = (a: number) => polar(wc.x, wc.y, wc.r - 4, 90 - h.wheelDeg + a);
  const mark = polar(dc.x, dc.y, dc.r - 6, 90 - h.drumDeg);
  return (
    <Plate n={5} title="Vitruvius’s hodometer: a mile per pebble">
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`After ${ft} feet the bowl holds ${h.pebbles} pebbles`}>
          <line x1={0} y1={wc.y + wc.r} x2={W} y2={wc.y + wc.r} style={{ stroke: C.ink, strokeWidth: 1.1 }} />
          <circle cx={wc.x} cy={wc.y} r={wc.r} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 2 }} />
          {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
            const p = spoke(a);
            return <line key={a} x1={wc.x} y1={wc.y} x2={p.x} y2={p.y} style={{ stroke: a === 0 ? C.red : C.soft, strokeWidth: a === 0 ? 1.6 : 0.8 }} />;
          })}
          <InkText x={wc.x} y={wc.y + wc.r + 16} anchor="middle" size={11} role="soft">wheel, 4 ft across</InkText>
          <circle cx={dc.x} cy={dc.y} r={dc.r} style={{ fill: C.paper, stroke: C.brass, strokeWidth: 1.6 }} />
          {Array.from({ length: 40 }, (_, i) => {
            const a = polar(dc.x, dc.y, dc.r, i * 9);
            const b = polar(dc.x, dc.y, dc.r - 4, i * 9);
            return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={{ stroke: C.brass, strokeWidth: 0.6 }} />;
          })}
          <line x1={dc.x} y1={dc.y} x2={mark.x} y2={mark.y} style={{ stroke: C.red, strokeWidth: 1.4 }} />
          <InkText x={dc.x} y={dc.y + dc.r + 14} anchor="middle" size={11} role="brass">{`drum: ${h.teeth % TEETH} of 400 teeth`}</InkText>
          <path d={`M ${bowl.x - 34} ${bowl.y - 18} Q ${bowl.x} ${bowl.y + 26} ${bowl.x + 34} ${bowl.y - 18}`} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
          {Array.from({ length: Math.min(h.pebbles, 12) }, (_, i) => (
            <circle key={i} cx={bowl.x - 18 + (i % 5) * 9} cy={bowl.y - 2 - Math.floor(i / 5) * 7} r={3.5} style={{ fill: C.soft }} />
          ))}
          <InkText x={bowl.x} y={bowl.y + 26} anchor="middle" size={11} role="soft">{`bowl: ${h.pebbles} pebble${h.pebbles === 1 ? '' : 's'}`}</InkText>
        </svg>
      </div>
      <Slider label="distance driven" value={ft} min={0} max={30_000} step={50} onChange={setFt} format={(v) => `${v.toLocaleString('en')} ft`} />
      <Working label="Hodometer count">
        <Eq>
          <Sym role="red">miles</Sym> = pebbles; one pebble per <Sym role="brass">400</Sym> turns × <Sym>π · 4 ft</Sym>
        </Eq>
        <Eq>
          <V role="red">{`${h.pebbles}`}</V> pebbles = <V role="red">{`${h.counted.toLocaleString('en')} ft`}</V> counted; driven <V>{`${ft.toLocaleString('en')} ft`}</V>
        </Eq>
        <Eq>
          a drum’s mile: 400 × <V>{`${REV.toFixed(2)} ft`}</V> = <V>{`${(TEETH * REV).toFixed(0)} ft`}</V>, not 5,000 (Vitruvius took 12½ ft a turn)
        </Eq>
      </Working>
    </Plate>
  );
}
