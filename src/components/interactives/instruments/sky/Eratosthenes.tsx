/**
 * Plate 1: Eratosthenes. Parallel sunlight, a well at Syene with no shadow, a bowl sundial at
 * Alexandria with a shadow. The shadow angle equals the angle at Earth's centre, so the
 * circumference is (360 / angle) × the distance. Move the measured angle and pick a stadion:
 * the angle was good, the unit is where the uncertainty lives.
 */
import { useState } from 'react';
import { C, Controls, InkText, Plate, Readout, Segmented, Slider, arcPath, polar, useMeasuredWidth } from '../shared';

const DIST = 5000; // stades, Syene to Alexandria
const STADIA = [
  { value: '157.5', label: '157.5 m' },
  { value: '176.4', label: '176.4 m' },
  { value: '184.8', label: '184.8 m' },
] as const;
type St = (typeof STADIA)[number]['value'];
const EARTH_KM = 40008; // meridional circumference
const EXAG = 2.5; // the drawing shows angles 2.5× larger so they can be seen

const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

export default function Eratosthenes() {
  const [theta, setTheta] = useState(7.2);
  const [st, setSt] = useState<St>('157.5');
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const narrow = W < 520;
  const H = narrow ? 300 : 340;
  const cx = W * (narrow ? 0.55 : 0.56);
  const R = H * 0.7;
  const cy = H * 0.24 + R; // the centre sits near the bottom edge
  const top = 90; // Syene straight up
  const aAlex = top + theta * EXAG; // Alexandria to the left (north)
  const syene = polar(cx, cy, R, top);
  const alex = polar(cx, cy, R, aAlex);
  const g = narrow ? 40 : 50; // gnomon length
  const tipS = polar(cx, cy, R + g, top);
  const tipA = polar(cx, cy, R + g, aAlex);
  // sunlight falls straight down; the tip's shadow lands on the ground plane at Alexandria
  const away = { x: Math.cos(((aAlex + 90) * Math.PI) / 180), y: -Math.sin(((aAlex + 90) * Math.PI) / 180) };
  const s = (tipA.x - alex.x) / away.x;
  const sh = { x: alex.x + away.x * s, y: alex.y + away.y * s };
  const C_st = (360 / theta) * DIST;
  const km = (C_st * Number(st)) / 1000;
  const errPct = ((km - EARTH_KM) / EARTH_KM) * 100;
  const rays = narrow ? [0.08, 0.86] : [0.12, 0.24, 0.76, 0.88];
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };

  return (
    <Plate
      n={1}
      title="A shadow and a walk"
      kicker="angles drawn 2.5× larger"
      caption={
        <>
          The sun is so far away that its rays arrive parallel. At Syene a gnomon casts no shadow at
          noon on midsummer day; at Alexandria the shadow makes an angle, and that is the same angle as
          the slice of Earth between the two cities. Eratosthenes measured 1/50 of a circle (7.2°).
          Nobody knows exactly how long his stadion was, so the answer swings with it.
        </>
      }
    >
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img"
          aria-label={`Parallel sunlight over Syene and Alexandria; shadow angle ${theta.toFixed(1)} degrees`}>
          {/* parallel sunlight */}
          {rays.map((f) => (
            <g key={f}>
              <line x1={W * f} y1={4} x2={W * f} y2={H * 0.22} style={{ stroke: C.red, strokeWidth: 0.9, opacity: 0.7 }} />
              <path d={`M ${W * f - 3.5} ${H * 0.22 - 7} L ${W * f} ${H * 0.22} L ${W * f + 3.5} ${H * 0.22 - 7}`} style={{ fill: 'none', stroke: C.red, strokeWidth: 0.9, opacity: 0.7 }} />
            </g>
          ))}
          <InkText x={W * rays[0] + 8} y={16} size={12} role="soft">sunlight, parallel</InkText>
          {/* Earth's surface */}
          <path d={arcPath(cx, cy, R, 38, 142)} style={{ ...ink, strokeWidth: 1.4 }} />
          <path d={arcPath(cx, cy, R - 5, 38, 142)} style={{ ...ink, stroke: C.faint, strokeWidth: 0.6 }} />
          {/* radii towards the centre */}
          {[top, aAlex].map((a) => {
            const p = polar(cx, cy, R + g, a);
            const q = polar(cx, cy, R * 0.28, a);
            return <line key={a} x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />;
          })}
          {/* angle at the centre */}
          <path d={arcPath(cx, cy, R * 0.3, top, aAlex)} style={{ ...ink, stroke: C.red, strokeWidth: 1.2 }} />
          <circle cx={cx} cy={cy} r={2.2} style={{ fill: C.ink }} />
          {(() => {
            const p = polar(cx, cy, R * 0.3 + 10, aAlex + 4);
            return <InkText x={p.x} y={p.y} anchor="end" size={13} role="red">{theta.toFixed(1)}°</InkText>;
          })()}
          <InkText x={cx + 8} y={cy - 6} size={11} role="soft">Earth&apos;s centre</InkText>
          {/* the walk: 5,000 stades along the ground */}
          <path d={arcPath(cx, cy, R - 16, top, aAlex)} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.8 }} />
          {[top, aAlex].map((a) => {
            const p = polar(cx, cy, R - 11, a);
            const q = polar(cx, cy, R - 21, a);
            return <line key={a} x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={{ stroke: C.soft, strokeWidth: 0.8 }} />;
          })}
          {(() => {
            const p = polar(cx, cy, R - 32, (top + aAlex) / 2);
            return <InkText x={p.x} y={p.y} anchor="middle" kind="num" size={11} role="soft">5,000 stades</InkText>;
          })()}
          {/* sunbeam into Syene's well */}
          <line x1={syene.x} y1={H * 0.24} x2={syene.x} y2={syene.y} style={{ stroke: C.red, strokeWidth: 1.2 }} />
          <rect x={syene.x - 6} y={syene.y} width={12} height={16} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 0.9 }} />
          <line x1={syene.x} y1={syene.y} x2={tipS.x} y2={tipS.y} style={{ ...ink, strokeWidth: 2 }} />
          {/* Alexandria's gnomon, the sunbeam past its tip, and the shadow */}
          <line x1={alex.x} y1={alex.y} x2={tipA.x} y2={tipA.y} style={{ ...ink, strokeWidth: 2 }} />
          <line x1={tipA.x} y1={H * 0.24} x2={sh.x} y2={sh.y} style={{ stroke: C.red, strokeWidth: 1.2 }} />
          <line x1={alex.x} y1={alex.y} x2={sh.x} y2={sh.y} style={{ stroke: C.ink, strokeWidth: 3.2, opacity: 0.55 }} />
          <path d={arcPath(tipA.x, tipA.y, 22, 270, aAlex + 180)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.1 }} />
          {(() => {
            const p = polar(tipA.x, tipA.y, 34, 270 + (aAlex - 90) / 2);
            return <InkText x={p.x + 4} y={p.y + 4} size={13} role="red">{theta.toFixed(1)}°</InkText>;
          })()}
          {/* labels */}
          <InkText x={syene.x + 10} y={syene.y + 22} size={13}>Syene</InkText>
          <InkText x={syene.x + 10} y={syene.y + 36} size={11} role="soft">no shadow</InkText>
          <InkText x={alex.x - 10} y={alex.y + 22} anchor="end" size={13}>Alexandria</InkText>
          <InkText x={alex.x - 10} y={alex.y + 36} anchor="end" size={11} role="soft">a short shadow</InkText>
        </svg>
      </div>
      <Controls>
        <Slider label="shadow angle" value={theta} min={5} max={10} step={0.1} onChange={setTheta} format={(v) => `${v.toFixed(1)}°`} />
        <Segmented label="Length of one stadion" options={STADIA.map((s) => ({ ...s }))} value={st} onChange={setSt} />
      </Controls>
      <Controls>
        <Readout label="circumference">{fmt(C_st)} stades</Readout>
        <Readout label="in km">
          {fmt(km)} <span style={{ color: Math.abs(errPct) < 3 ? C.ink : C.red }}>({errPct >= 0 ? '+' : ''}{errPct.toFixed(1)}%)</span>
        </Readout>
      </Controls>
    </Plate>
  );
}
