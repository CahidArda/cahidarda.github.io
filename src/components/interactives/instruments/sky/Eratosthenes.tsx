/**
 * Plate 1: Eratosthenes. Geometry lives in
 * ./eratosthenesGeom.ts (pure, so it can be swept numerically). Parallel sunlight, a well at Syene with no shadow, a bowl sundial at
 * Alexandria with a shadow. The shadow angle equals the angle at Earth's centre, so the
 * circumference is (360 / angle) × the distance. Move the measured angle and pick a stadion:
 * the angle was good, the unit is where the uncertainty lives.
 */
import { useState } from 'react';
import { C, Controls, Eq, HatchDefs, InkText, Plate, Segmented, Slider, Sym, V, Working, arcPath, useMeasuredWidth, type Role } from '../shared';
import { EARTH_KM, THETA_MAX, THETA_MIN, circumference, eratosthenes, polar } from './eratosthenesGeom';

// Wikipedia, Earth's circumference: a short stadion of 155 to 160 m (midpoint shown), the
// Olympic stade (176.4 m) and the Italian stade (184.8 m).
const STADIA = [
  { value: '157.5', label: 'short · 157.5 m', name: 'short' },
  { value: '176.4', label: 'Olympic · 176.4 m', name: 'Olympic' },
  { value: '184.8', label: 'Italian · 184.8 m', name: 'Italian' },
] as const;
type St = (typeof STADIA)[number]['value'];
const fmt = (n: number) => Math.round(n).toLocaleString('en-US');

export default function Eratosthenes() {
  const [theta, setTheta] = useState(7.2);
  const [st, setSt] = useState<St>('157.5');
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const g = eratosthenes(theta, W);
  const { H, cx, cy, R, top, aAlex, syene, alex, tipS, tipA, shadowEnd: sh, labels: L } = g;
  const res = circumference(theta, Number(st));
  const stName = STADIA.find((x) => x.value === st)!.name;
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const T = (l: { x: number; y: number; size: number; text: string; anchor: 'start' | 'middle' | 'end'; mono?: boolean }, role: Role = 'ink') => (
    <InkText x={l.x} y={l.y} anchor={l.anchor} size={l.size} kind={l.mono ? 'num' : 'note'} role={role}>{l.text}</InkText>
  );

  return (
    <Plate
      n={1}
      title="Eratosthenes measures the Earth"
      kicker="angles drawn 4× larger"
      caption={
        <>
          The sun is so far away that its rays arrive parallel. At Syene a gnomon casts no shadow at
          noon on midsummer day; at Alexandria the shadow makes an angle θ, the same angle as the slice
          of Earth between the two cities. Eratosthenes measured 1/50 of a circle (7.2°). There are three
          stadion lengths because nobody knows which one he used: historians argue for a short one of
          155 to 160 m, the Olympic stade or the Italian stade, and the answer in kilometres swings with
          the choice.
        </>
      }
    >
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img"
          aria-label={`Parallel sunlight over Syene and Alexandria; shadow angle ${theta.toFixed(1)} degrees`}>
          {/* parallel sunlight */}
          {g.rays.map((x) => (
            <g key={x}>
              <line x1={x} y1={4} x2={x} y2={92} style={{ stroke: C.soft, strokeWidth: 0.9 }} />
              <path d={`M ${x - 3.5} 85 L ${x} 92 L ${x + 3.5} 85`} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.9 }} />
            </g>
          ))}
          {T(L.sun, 'soft')}
          {/* Earth's surface */}
          <path d={arcPath(cx, cy, R, g.surfFrom, g.surfTo)} style={{ ...ink, strokeWidth: 1.4 }} />
          {/* radii towards the centre */}
          {[top, aAlex].map((a) => {
            const p = polar(cx, cy, R + g.g, a);
            return <line key={a} x1={p.x} y1={p.y} x2={cx} y2={cy} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />;
          })}
          {/* the slice of Earth between the cities */}
          <HatchDefs id="er-hatch" gap={6} role="faint" />
          <path d={`M ${cx} ${cy} L ${syene.x} ${syene.y} ${arcPath(cx, cy, R, top, aAlex).replace(/^M [^A]+/, '')} Z`} style={{ fill: 'url(#er-hatch)', stroke: 'none', opacity: 0.8 }} />
          {/* θ at the centre */}
          <path d={arcPath(cx, cy, g.rc, top, aAlex)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.6 }} />
          <circle cx={cx} cy={cy} r={2.2} style={{ fill: C.ink }} />
          {T(L.centre, 'sky')}
          {T(L.earth, 'soft')}
          {/* d: the 5,000 stadia along the ground */}
          <path d={arcPath(cx, cy, g.walkR, top, aAlex)} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.8 }} />
          {[top, aAlex].map((a) => {
            const p = polar(cx, cy, g.walkR + 5, a);
            const q = polar(cx, cy, g.walkR - 5, a);
            return <line key={a} x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={{ stroke: C.brass, strokeWidth: 1.2 }} />;
          })}
          {T(L.walk, 'brass')}
          {/* Syene: the sun straight down the well */}
          <rect x={syene.x - 6} y={syene.y} width={12} height={16} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 0.9 }} />
          <line x1={syene.x} y1={syene.y} x2={tipS.x} y2={tipS.y} style={{ ...ink, strokeWidth: 2 }} />
          {/* Alexandria: gnomon, the sunbeam past its tip, the shadow, and θ at the tip */}
          <line x1={alex.x} y1={alex.y} x2={tipA.x} y2={tipA.y} style={{ ...ink, strokeWidth: 2 }} />
          <line x1={tipA.x} y1={96} x2={sh.x} y2={sh.y} style={{ stroke: C.soft, strokeWidth: 1, strokeDasharray: '3 3' }} />
          <line x1={alex.x} y1={alex.y} x2={sh.x} y2={sh.y} style={{ stroke: C.ink, strokeWidth: 3.2, opacity: 0.55 }} />
          <path d={arcPath(tipA.x, tipA.y, g.tipArcR, 270, aAlex + 180)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.6 }} />
          {T(L.tip, 'sky')}
          {T(L.syene)}
          {T(L.syeneSub, 'soft')}
          {T(L.alex)}
          {T(L.alexSub, 'soft')}
        </svg>
      </div>
      <Controls>
        <Slider label="shadow angle θ" value={theta} min={THETA_MIN} max={THETA_MAX} step={0.1} onChange={setTheta} format={(v) => `${v.toFixed(1)}°`} />
        <span className="in-field">
          length of one stadion
          <Segmented label="Length of one stadion" options={STADIA.map(({ value, label }) => ({ value, label }))} value={st} onChange={setSt} />
        </span>
      </Controls>
      <Working label="Circumference from the shadow angle">
        <Eq>
          <Sym role="red">C</Sym> = (360° / <Sym role="sky">θ</Sym>) × <Sym role="brass">d</Sym>
        </Eq>
        <Eq>
          <V role="red">{fmt(res.stadia)} stadia</V> = (360° / <V role="sky">{theta.toFixed(1)}°</V>) × <V role="brass">5,000 stadia</V>
        </Eq>
        <Eq>
          × <V>{st} m</V> ({stName}) = <V role="red">{fmt(res.km)} km</V>{' '}
          <span style={{ fontSize: '0.9em' }}>
            ({res.errPct >= 0 ? '+' : '−'}{Math.abs(res.errPct).toFixed(1)}% vs today&apos;s {fmt(EARTH_KM)} km)
          </span>
        </Eq>
      </Working>
    </Plate>
  );
}
