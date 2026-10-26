/**
 * Plate 5 (episode 5): one plate per latitude, then one plate for all (the saphea).
 * 1 Travel · 2 Measure the sun there · 3 Use the Istanbul plate anyway · 4 Swap in the right
 * plate · 5 The universal plate. Numbers: sapheaGeom.ts + astro.ts (swept).
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Segmented, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { DOY_DEFAULT, PHI, altitude, deg1, dmin, eclRA, hm, sunAt, noteSize, wrap360, type MomentId } from './astro';
import { Dome, type DomeHi } from './Dome';
import { FrontView } from './Faces';
import { DateMoment, isMorning, momentT } from './common';
import { PLACES, horizonDiameter, placeLat, plateReading, sapheaGrid, saphea, sunsetOn, type PlaceId } from './sapheaGeom';

function Saphea({ lat, dec, name }: { lat: number; dec: number; name: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const H = Math.round(Math.max(280, Math.min(400, W * 0.9)));
  const R = Math.min(W, H - 30) / 2 - 12;
  const cx = W / 2;
  const cy = (H - 30) / 2 + 4;
  const X = (p: { x: number; y: number }) => ({ x: cx + p.x * R, y: cy - p.y * R });
  const grid = sapheaGrid();
  const hz = horizonDiameter(lat);
  const ss = sunsetOn(lat, dec);
  const day = grid.days;
  const path = (run: { x: number; y: number }[]) => run.map((p, i) => `${i ? 'L' : 'M'} ${X(p).x.toFixed(1)} ${X(p).y.toFixed(1)}`).join(' ');
  const sunRun = (() => {
    const pts: { x: number; y: number }[] = [];
    for (let Hh = 0; Hh <= 180; Hh += 1) pts.push(saphea(dec, Hh));
    return pts.filter((p) => Math.hypot(p.x, p.y) <= 1);
  })();
  const s = X(ss.pt);
  const a = X(hz.a);
  const b = X(hz.b);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={`The universal plate set for ${name}: sunset at ${hm(ss.t)}`}>
        <circle cx={cx} cy={cy} r={R} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.2 }} />
        {day.map((d) => d.runs.map((r, i) => <path key={`${d.d}-${i}`} d={path(r)} style={{ fill: 'none', stroke: d.d === 0 ? C.soft : C.faint, strokeWidth: d.d === 0 ? 0.9 : 0.6 }} />))}
        {grid.hours.map((h) => h.runs.map((r, i) => <path key={`h${h.H}-${i}`} d={path(r)} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.6 }} />))}
        <path d={path(sunRun)} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.6 }} />
        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={{ stroke: C.ink, strokeWidth: 2 }} />
        <circle cx={s.x} cy={s.y} r={5} style={{ fill: C.red }} />
        <InkText x={cx} y={cy - R - 4} anchor="middle" size={11} role="soft">north pole</InkText>
        <InkText x={b.x + (b.x > cx ? -4 : 4)} y={b.y + (b.y > cy ? 14 : -6)} anchor={b.x > cx ? 'end' : 'start'} size={11} role="ink">{`horizon at ${lat.toFixed(1)}°`}</InkText>
        <InkText x={W / 2} y={H - 10} anchor="middle" size={noteSize(`${name}: the sun sets at ${hm(ss.t)} sun time`, W)} role="red">{`${name}: the sun sets at ${hm(ss.t)} sun time`}</InkText>
      </svg>
    </div>
  );
}

const HI: DomeHi[][] = [['horizon', 'pole'], ['h', 'sun'], ['h', 'hour'], ['hour', 'sun'], ['horizon', 'day']];

export default function PlatesPlate() {
  const [doy, setDoy] = useState(DOY_DEFAULT);
  const [moment, setMoment] = useState<MomentId>('m1');
  const [place, setPlace] = useState<PlaceId>('cai');
  const [step, setStep] = useState(0);
  const lat = placeLat(place);
  const name = PLACES.find((p) => p.id === place)!.name;
  const t = momentT(moment);
  const side = isMorning(moment) ? 'am' : 'pm';
  const sun = sunAt(doy, t, lat);
  const h = Math.round(sun.h * 10) / 10;
  const wrong = plateReading(PHI, h, sun.dec, side);
  const right = plateReading(lat, h, sun.dec, side)!;
  const err = wrong ? wrong.t - right.t : null;
  const plateLat = step >= 3 ? lat : PHI;
  const used = step >= 3 ? right : wrong;
  const lst = used ? wrap360(used.H + eclRA(sun.lam)) : wrap360(eclRA(sun.lam) + 180);
  const ss = sunsetOn(lat, sun.dec);
  const steps: StepDef[] = [
    {
      id: 'travel',
      title: 'Travel south',
      text: <>Take the Istanbul astrolabe to {name}, {(PHI - lat).toFixed(1)}° of latitude further south. Pick the city above the plate.</>,
      controls: (
        <Segmented
          label="where you are"
          options={PLACES.filter((p) => p.id !== 'ist').map((p) => ({ value: p.id, label: `${p.name} ${p.lat.toFixed(1)}° N` }))}
          value={place}
          onChange={setPlace}
        />
      ),
    },
    {
      id: 'measure',
      title: 'Measure the sun',
      text: <>Sight the sun as before. In {name} at that moment it stands <span style={{ color: C.sky }}>{deg1(h)}</span> high.</>,
    },
    {
      id: 'wrong',
      title: 'Use the 41° plate',
      text: wrong ? (
        <>
          Do the procedure on the Istanbul plate: its height circles belong to a different sky. It gives <span style={{ color: C.red }}>{hm(wrong.t)}</span>, but
          it is really {hm(right.t)}: {dmin(err!)} out.
        </>
      ) : (
        <>
          On the Istanbul plate there is no such height that day: the sun never gets to {deg1(h)} at 41° N on this date. The plate cannot answer at all.
        </>
      ),
    },
    {
      id: 'right',
      title: 'Swap in the right plate',
      text: <>Take out the 41° plate and drop in the one cut for {lat.toFixed(1)}°. Same rete, same sight, and the answer is right: <span style={{ color: C.red }}>{hm(right.t)}</span>. So makers stacked several plates in one mater.</>,
    },
    {
      id: 'saphea',
      title: 'One plate for all',
      text: (
        <>
          Al-Zarqālī's universal plate projects the sky onto the meridian plane instead. Every horizon becomes a straight diameter, so one grid
          serves every latitude: turn the horizon to {lat.toFixed(1)}° and read sunset where the sun's day line meets it.
        </>
      ),
    },
  ];
  return (
    <StepPlate
      n={5}
      title="One plate per latitude, then one plate for all"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) =>
        i === 4 ? (
          <Saphea lat={lat} dec={sun.dec} name={name} />
        ) : (
          <FrontView
            phi={i >= 3 ? lat : PHI}
            lst={i >= 2 ? lst : wrap360(eclRA(sun.lam) + 180)}
            sunLam={sun.lam}
            almH={i >= 1 ? h : null}
            ruleH={i >= 2 && used ? used.H : null}
            note={
              i === 0
                ? `the plate in the mater: ${PHI}° N`
                : i === 1
                  ? `the ${deg1(h)} circle on the ${PHI}° plate`
                  : i === 2
                    ? wrong
                      ? `the ${PHI}° plate says ${hm(wrong.t)}`
                      : `no ${deg1(h)} circle reaches the sun's path on this plate`
                    : `the ${lat.toFixed(1)}° plate says ${hm(right.t)}`
            }
            noteRole={i === 2 ? 'red' : i === 3 ? 'red' : 'soft'}
          />
        )
      }
      viewLabel={step === 4 ? 'The universal plate (saphea)' : `The astrolabe, ${plateLat === PHI ? 'Istanbul plate' : `${lat.toFixed(1)}° plate`}`}
      reference={(i) => <Dome opts={{ phi: lat, dec: sun.dec, sunH: sun.H, alm: i >= 1 && i < 4 ? h : null, showHourArc: i === 2 || i === 3 }} hi={HI[i]} />}
      referenceLabel={`The real sky over ${name} (east on the left)`}
      settings={<DateMoment doy={doy} setDoy={setDoy} moment={moment} setMoment={setMoment} />}
      equation={
        <Working label="Same sight, two plates">
          <Eq>
            cos <Sym role="red">H</Sym> = (sin <Sym role="sky">h</Sym> − sin <Sym role="soft">φ</Sym> sin δ) / (cos <Sym role="soft">φ</Sym> cos δ)
          </Eq>
          <Eq>
            φ = 41° (wrong plate): <Known when={step >= 2} role="red">{wrong ? hm(wrong.t) : 'no answer'}</Known>
          </Eq>
          <Eq>
            φ = {lat.toFixed(1)}° (right plate): <Known when={step >= 3} role="red">{hm(right.t)}</Known>
          </Eq>
          <Eq>
            error: <Known when={step >= 2} role="red">{err === null ? 'the plate has no such height' : dmin(err)}</Known>
          </Eq>
          {step >= 4 && (
            <Eq>
              sunset in {name}: cos H₀ = −tan φ tan δ, <span style={{ color: C.red }}>{hm(ss.t)}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The sun's place is the same everywhere on a given date; only the height circles change with latitude. Al-Zarqālī's universal projection (11th
          century) is on the meridian plane, where the horizons of all latitudes are diameters (Museum of the History of Science, Oxford). In {name} the true
          altitude is {deg1(altitude(sun.dec, sun.H, lat))}; the plates use it rounded to 0.1°.
        </>
      }
    />
  );
}
