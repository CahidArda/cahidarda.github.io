/**
 * Plate 10, procedure: latitude to a tenth of an arcsecond with a zenith camera.
 *   1 Level it · 2 Photograph the zenith · 3 Match the stars · 4 Compare with GPS.
 * Reference: the post's Earth section with a star straight overhead (h = 90°).
 * Geometry: ./zenithGeom.ts (swept by zenithGeom.sweep.ts).
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type Role, type StepDef } from '../shared';
import LatitudeReference, { type RefKey } from './LatitudeReference';
import {
  DEFLECT,
  G_MAX,
  G_MIN,
  G_STEP,
  MATCH_TOL,
  PHI_ASTRO,
  PHI_GPS,
  TILT_MAX,
  TILT_MIN,
  TILT_STEP,
  TILT_TOL,
  bubbleScene,
  dms1,
  fieldScene,
  guessLat,
  truthG,
} from './zenithGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const sec = (v: number) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(1)}″`;

function Bubble({ tilt }: { tilt: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = bubbleScene(tilt, W);
  const on = Math.abs(tilt) <= TILT_TOL;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Tilt meter: ${sec(tilt)} north-south`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">tilt meter, north-south (arcseconds)</InkText>
        <rect x={g.cx - g.len / 2} y={g.cy - 16} width={g.len} height={32} rx={16} style={{ ...solid('ink', 1.3), fill: C.deep }} />
        {[-20, -10, 0, 10, 20].map((v) => (
          <g key={v}>
            <line x1={g.cx + v * g.k} y1={g.cy + 18} x2={g.cx + v * g.k} y2={g.cy + 26} style={solid('ink', 0.8)} />
            <InkText x={g.cx + v * g.k} y={g.cy + 40} kind="num" anchor="middle" size={10} role="soft">{`${v}`}</InkText>
          </g>
        ))}
        <ellipse cx={g.bubble.x} cy={g.cy} rx={14} ry={9} style={{ fill: on ? C.sky : C.paper, stroke: C.ink, strokeWidth: 1 }} />
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={on ? 'sky' : 'soft'}>{on ? 'level: the camera points along the plumb line' : `tilted ${Math.abs(tilt).toFixed(1)}″ ${tilt > 0 ? 'north' : 'south'}`}</InkText>
      </svg>
    </div>
  );
}

function Field({ tilt, guess }: { tilt: number; guess: number | null }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = fieldScene(tilt, guess, W);
  const clip = `zf-${Math.round(W)}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label="Stars round the zenith, with the catalogue's predicted positions as circles">
        <defs>
          <clipPath id={clip}>
            <rect x={g.cx - g.size / 2} y={g.cy - g.size / 2} width={g.size} height={g.size} />
          </clipPath>
        </defs>
        <rect x={g.cx - g.size / 2} y={g.cy - g.size / 2} width={g.size} height={g.size} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
        <g clipPath={`url(#${clip})`}>
          {g.stars.map((s, i) => (
            <circle key={i} cx={s.x} cy={s.y} r={s.r} style={{ fill: C.ink }} />
          ))}
          {g.cat.map((s, i) => (
            <circle key={i} cx={s.x} cy={s.y} r={7} style={solid('sky', 1.1)} />
          ))}
        </g>
        <line x1={g.cx - 8} y1={g.cy} x2={g.cx + 8} y2={g.cy} style={solid('red', 1.2)} />
        <line x1={g.cx} y1={g.cy - 8} x2={g.cx} y2={g.cy + 8} style={solid('red', 1.2)} />
        <InkText x={g.cx + g.size / 2 - 4} y={g.cy - g.size / 2 + 14} anchor="end" size={11} role="soft">N ↑ · 20′ across</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">
          {g.shift === null ? 'stars overhead; the cross is the image centre' : Math.abs(g.shift) <= MATCH_TOL ? 'every star in its circle' : `circles ${Math.abs(g.shift).toFixed(1)}″ ${g.shift > 0 ? 'north of' : 'south of'} the stars`}
        </InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['sun'], ['sun'], ['phi', 'dec'], ['phi']];

export default function ZenithCameraProcedure() {
  const [tilt, setTilt] = useState(12);
  const [g, setG] = useState(30);
  const [step, setStep] = useState(0);
  const level = Math.abs(tilt) <= TILT_TOL;
  const shift = (PHI_ASTRO - guessLat(g)) * 3600;
  const match = level && Math.abs(shift) <= MATCH_TOL;
  const phi = guessLat(g);
  const diff = (phi - PHI_GPS) * 3600;

  const steps: StepDef[] = [
    {
      id: 'level',
      title: 'Level it',
      text: <>The camera points straight up. Tilt meters on it find the plumb line, the local “down”, to a fraction of an arcsecond.</>,
      controls: <Slider label="tilt the camera" value={tilt} min={TILT_MIN} max={TILT_MAX} step={TILT_STEP} onChange={setTilt} format={sec} />,
      gated: true,
      done: level,
      hint: 'Centre the bubble',
    },
    {
      id: 'shoot',
      title: 'Photograph the zenith',
      text: <>A short exposure of the stars round the point overhead, timed by GPS. The centre of the image is your zenith.</>,
    },
    {
      id: 'match',
      title: 'Match the stars',
      text: (
        <>
          A star catalogue says where each star should be. Shift the predicted positions until every star sits in its
          circle: the latitude that does it is the declination of your zenith, which is your latitude.
        </>
      ),
      controls: <Slider label="latitude to try" value={g} min={G_MIN} max={G_MAX} step={G_STEP} onChange={setG} format={(v) => dms1(guessLat(v))} width="11rem" />,
      gated: true,
      done: match,
      hint: 'Put every star in its circle',
    },
    {
      id: 'gps',
      title: 'Compare with GPS',
      text: (
        <>
          GPS gives a latitude too, measured against a smooth model of the Earth. The two differ by a few arcseconds,
          because mountains and dense rock pull the plumb line aside. That difference is what these cameras are built to
          measure.
        </>
      ),
    },
  ];

  return (
    <StepPlate
      n={10}
      wide
      title="Latitude from the stars overhead, with a zenith camera"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (i === 0 ? <Bubble tilt={tilt} /> : <Field tilt={tilt} guess={i >= 2 ? g : null} />)}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: PHI_GPS, src: { kind: 'sun', dec: PHI_GPS }, srcName: 'a star overhead', visible: false }}
          hi={HI[i]}
          label="The Earth cut along Istanbul's meridian, with a star straight overhead: its declination equals your latitude."
        />
      )}
      referenceLabel="What you are measuring (your meridian)"
      onSkip={() => {
        setTilt(0);
        setG(Math.round(truthG / G_STEP) * G_STEP);
      }}
      equation={
        <Working label="Latitude from a zenith camera">
          <Eq>
            <Sym role="red">φ</Sym> = <Sym role="brass">δ</Sym> of the zenith
          </Eq>
          <Eq>
            <Known when={step >= 2 && match} role="red">{dms1(phi)}</Known>
          </Eq>
          <Eq>
            <Sym role="red">φ</Sym> − <Sym>φ</Sym>
            <sub>GPS</sub> = deflection of the vertical
          </Eq>
          <Eq>
            <Known when={step >= 3 && match} role="red">{sec(diff)}</Known> north
          </Eq>
        </Working>
      }
      caption={
        <>
          Istanbul’s GPS latitude is the series’ reference point (41.0136° N). The plumb line here is deflected {DEFLECT}″
          north for illustration only; real deflections are several arcseconds and the cameras measure them to about
          0.1″ to 0.3″. The star field is made up; the field is 20′ across.
        </>
      }
    />
  );
}
