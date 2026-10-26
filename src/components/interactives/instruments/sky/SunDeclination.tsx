/**
 * Plate: why the sun's height at noon changes through the year. The celestial sphere as an
 * armillary: the equator, and the ecliptic tilted ε = 23.44° to it. A date slider moves the sun
 * along the ecliptic; its angle above or below the equator is the declination δ the noon sight
 * needs. Same date convention and formula (USNO) as the noon-sight plate.
 * Geometry lives in ./armillaryGeom.ts (swept by the verification script).
 */
import { useState } from 'react';
import { C, Controls, Eq, InkText, Plate, Slider, Sym, V, Working, useMeasuredWidth, type Role } from '../shared';
import { armillaryScene } from './armillaryGeom';
import { DOY_PRESET, dateLabel, dms, sdm } from './noonGeom';

const RAD = Math.PI / 180;

export default function SunDeclination() {
  const [doy, setDoy] = useState(DOY_PRESET);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const s = armillaryScene(doy, W);
  const { R, HH, cx, cy, sun, foot } = s;
  const T = (id: string, role: Role = 'soft') => {
    const l = s.labels.find((x) => x.id === id);
    return l ? <InkText key={id} x={l.x} y={l.y} anchor={l.anchor} size={l.size} role={role}>{l.text}</InkText> : null;
  };
  const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
  const dashed = (role: Role, w = 1) => ({ ...solid(role, w), strokeDasharray: '4 4' });
  const sinPart = Math.sin(s.eps * RAD) * Math.sin(s.lambda * RAD);

  return (
    <Plate
      n={9}
      title="Why the sun's height changes through the year"
      kicker="drag the date"
      caption={
        <>
          The sky drawn as an armillary sphere, with the Earth at its centre. The sun moves once a year
          around the ecliptic, which is tilted 23.44° to the celestial equator. Its angle above or below
          the equator is the declination δ: 0 at the equinoxes, ±23.44° at the solstices. The almanac
          the navigator carries is, among other things, a table of δ for every day.
        </>
      }
    >
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${HH}`} width="100%" style={{ display: 'block' }} role="img"
          aria-label={`Celestial sphere on ${dateLabel(doy)}: the sun on the ecliptic at declination ${s.dec.toFixed(1)} degrees`}>
          {/* outline of the sphere and the polar axis */}
          <circle cx={cx} cy={cy} r={R} style={solid('ink', 1.2)} />
          <line x1={s.south.x} y1={s.south.y} x2={s.pole.x} y2={s.pole.y} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
          <circle cx={s.pole.x} cy={s.pole.y} r={2.5} style={{ fill: C.ink }} />
          {T('pole')}
          {/* the back halves first, dashed */}
          <path d={s.equator.back} style={dashed('ink')} />
          <path d={s.ecliptic.back} style={dashed('soft')} />
          <path d={s.equator.front} style={solid('ink', 1.3)} />
          <path d={s.ecliptic.front} style={solid('soft', 1.3)} />
          {/* legend */}
          <line x1={s.legend.x[0]} y1={s.legend.eq} x2={s.legend.x[1]} y2={s.legend.eq} style={solid('ink', 1.3)} />
          {T('legend-eq')}
          <line x1={s.legend.x[0]} y1={s.legend.ecl} x2={s.legend.x[1]} y2={s.legend.ecl} style={solid('soft', 1.3)} />
          {T('legend-ecl')}
          {/* ε at the December solstice */}
          <path d={s.epsArc.front} style={solid('ink', 0.9)} />
          {T('eps', 'ink')}
          {/* the equinoxes: where the two circles cross */}
          {[s.march, s.sept].map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={2.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          ))}
          {T('march')}
          {T('sept')}
          {/* Earth at the centre */}
          <circle cx={cx} cy={cy} r={5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          {/* δ: along the sun's hour circle, from the equator to the sun */}
          <path d={s.decArc.front} style={solid('brass', 2.2)} />
          <path d={s.decArc.back} style={{ ...solid('brass', 1.6), strokeDasharray: '3 3' }} />
          <circle cx={foot.x} cy={foot.y} r={2} style={{ fill: C.brass }} />
          {s.showDec && T('dec', 'brass')}
          {/* the sun */}
          <circle cx={sun.x} cy={sun.y} r={8} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.4, opacity: sun.front ? 1 : 0.6, strokeDasharray: sun.front ? undefined : '2 2' }} />
          <circle cx={sun.x} cy={sun.y} r={2.4} style={{ fill: C.red }} />
          {T('sun', 'red')}
        </svg>
      </div>
      <Controls>
        <Slider label="date" value={doy} min={1} max={365} step={1} onChange={setDoy} format={dateLabel} width="12rem" />
      </Controls>
      <Working label="Declination from the date">
        <Eq>
          sin <Sym role="brass">δ</Sym> = sin <Sym>ε</Sym> · sin <Sym>λ</Sym>
        </Eq>
        <Eq>
          sin(<V role="brass">{sdm(s.dec)}</V>) = sin <V>23.44°</V> · sin <V>{s.lambda.toFixed(1)}°</V> = <V>{sinPart.toFixed(3)}</V>
        </Eq>
        <Eq>
          <span style={{ fontSize: '0.9em' }}>
            λ is how far the sun has gone round the ecliptic since the March equinox; on {dateLabel(doy)} the
            sun is {dms(Math.abs(s.dec), '')} {s.dec >= 0 ? 'north' : 'south'} of the equator
          </span>
        </Eq>
      </Working>
    </Plate>
  );
}
