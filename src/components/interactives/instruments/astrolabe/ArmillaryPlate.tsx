/**
 * Plate 1 (episode 1): telling the time with an armillary sphere, step by step.
 * View: the brass sphere (armGeom.ts). Reference: the sky dome (skyGeom.ts), same in every plate.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { DOY_DEFAULT, PHI, deg1, noteSize, eclDec, hm, sunLambda, sunAt, type MomentId } from './astro';
import { ARM_H_MAX, ARM_H_MIN, ARM_TOL, armReading, armScene, AXIS_MAX, AXIS_MIN } from './armGeom';
import { Dome, type DomeHi } from './Dome';
import { DateMoment, lamText, momentT } from './common';
import type { Run } from './skyGeom';

const line = (runs: Run[], stroke: string, w: number, back = true) =>
  runs.map((r, i) => (
    <polyline
      key={i}
      points={r.pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
      style={{ fill: 'none', stroke, strokeWidth: r.front ? w : w * 0.6, strokeDasharray: !r.front && back ? '3 4' : undefined, opacity: r.front ? 1 : 0.5 }}
    />
  ));

function Sphere({ axis, lam, H, step, h }: { axis: number; lam: number; H: number; step: number; h: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = armScene(W, { axis, lam, H });
  const showSun = step >= 1;
  const onH = Math.abs(g.alt - h) <= ARM_TOL;
  const note =
    step === 0
      ? `polar axis ${axis.toFixed(0)}° above the north horizon`
      : step === 1
        ? `the sun's bead at ${lamText(lam)} on the ecliptic ring`
        : step === 2
          ? `the bead stands ${deg1(g.alt)} above the horizon ring`
          : step === 3
            ? onH
              ? `${deg1(g.alt)}: on the measured height`
              : `bead at ${deg1(g.alt)}, the sun is at ${deg1(h)}`
            : step === 4
              ? `the equator ring reads ${hm(armReading(H))} under the bead`
              : 'rings in front of rings, from every side';
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={`An armillary sphere: ${note}`}>
        <line x1={g.stand.top.x} y1={g.stand.top.y} x2={g.stand.foot.x} y2={g.stand.foot.y} style={{ stroke: C.ink, strokeWidth: 3 }} />
        <line x1={g.stand.foot.x - 30} y1={g.stand.foot.y} x2={g.stand.foot.x + 30} y2={g.stand.foot.y} style={{ stroke: C.ink, strokeWidth: 3 }} />
        {line(g.meridian, C.ink, 1.6)}
        {line(g.horizon, C.ink, step === 0 || step === 3 ? 2.4 : 1.6)}
        {line(g.equator, step === 4 ? C.red : C.soft, step === 4 ? 2 : 1.3)}
        {step >= 4 &&
          g.ticks.map((t, i) => (
            <g key={i} style={{ opacity: t.front ? 1 : 0.35 }}>
              <line x1={t.a.x} y1={t.a.y} x2={t.b.x} y2={t.b.y} style={{ stroke: C.red, strokeWidth: 1 }} />
              {t.front && (
                <InkText x={t.lab.x} y={t.lab.y} kind="num" anchor="middle" baseline="central" size={10} role="red">
                  {t.text}
                </InkText>
              )}
            </g>
          ))}
        {line(g.ecliptic, C.brass, step === 1 ? 2.2 : 1.3)}
        <line x1={g.poleS.x} y1={g.poleS.y} x2={g.poleN.x} y2={g.poleN.y} style={{ stroke: step === 0 ? C.red : C.ink, strokeWidth: step === 0 ? 2.2 : 1.4 }} />
        {showSun && step >= 2 && line(g.hourCircle, C.faint, 0.8, false)}
        {showSun && (
          <circle cx={g.sun.x} cy={g.sun.y} r={6} style={{ fill: C.brass, stroke: step === 3 && onH ? C.sky : C.paper, strokeWidth: step === 3 && onH ? 2.5 : 1.2, opacity: g.sun.front ? 1 : 0.5 }} />
        )}
        <circle cx={g.cx} cy={g.cy} r={2.5} style={{ fill: C.ink }} />
        <InkText x={W / 2} y={g.H - 12} anchor="middle" size={noteSize(note, W)} role={step === 3 && onH ? 'sky' : 'soft'}>
          {note}
        </InkText>
      </svg>
    </div>
  );
}

const HI: DomeHi[][] = [['pole'], ['day', 'sun'], ['h', 'sun'], ['h', 'sun'], ['hour', 'sun'], []];

export default function ArmillaryPlate() {
  const [doy, setDoy] = useState(DOY_DEFAULT);
  const [moment, setMoment] = useState<MomentId>('m1');
  const [axis, setAxis] = useState(20);
  const [H, setH] = useState(0);
  const [step, setStep] = useState(0);
  const t = momentT(moment);
  const sun = sunAt(doy, t);
  const lam = sunLambda(doy);
  const dec = eclDec(lam);
  const g = armScene(400, { axis, lam, H });
  const axisOk = Math.abs(axis - PHI) < 0.5;
  const sideOk = Math.sign(H) === Math.sign(sun.H) || sun.H === 0;
  const onH = axisOk && Math.abs(g.alt - sun.h) <= ARM_TOL && sideOk;
  const read = armReading(H);
  const kn = (i: number) => step >= i;

  const steps: StepDef[] = [
    {
      id: 'axis',
      title: 'Tilt the axis',
      text: <>Set the polar axis to your latitude: in Istanbul it stands 41° above the north horizon, pointing at the celestial pole.</>,
      controls: <Slider label="axis above the north horizon" value={axis} min={AXIS_MIN} max={AXIS_MAX} step={1} onChange={setAxis} format={(v) => `${v}°`} />,
      gated: true,
      done: axisOk,
      hint: 'Tilt it to 41°',
    },
    {
      id: 'sun',
      title: "Find today's sun",
      text: (
        <>
          The sun moves round the ecliptic ring once a year. On {sun && lamText(lam)} it sits at δ = {deg1(dec)} from the equator; slide its bead there
          (the date is set above the plate).
        </>
      ),
    },
    {
      id: 'measure',
      title: 'Measure the sun',
      text: (
        <>
          Outside, with any sighting instrument, the sun stands <span style={{ color: C.sky }}>{deg1(sun.h)}</span> above the horizon, in the{' '}
          {sun.H < 0 ? 'east (morning)' : 'west (afternoon)'}. That one number is the whole observation.
        </>
      ),
    },
    {
      id: 'turn',
      title: 'Turn the sphere',
      text: <>Turn the inner sphere about its axis, as the sky turns, until the bead stands at that height above the horizon ring, on the {sun.H < 0 ? 'east' : 'west'} side.</>,
      controls: <Slider label="turn the sphere (hour angle)" value={H} min={ARM_H_MIN} max={ARM_H_MAX} step={0.5} onChange={setH} format={(v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}°`} />,
      gated: true,
      done: onH,
      hint: `Bead at ${deg1(sun.h)}, ${sun.H < 0 ? 'east' : 'west'} side`,
    },
    {
      id: 'read',
      title: 'Read the hour',
      text: <>The equator ring is divided into 24 hours. Read it where the bead's hour circle crosses it: <span style={{ color: C.red }}>{hm(read)}</span> sun time.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          It works, and it is the sky in three dimensions: big, fiddly to read through rings that hide each other, and every circle has to be a
          separate ring of brass. Can it be flat?
        </>
      ),
    },
  ];

  return (
    <StepPlate
      n={1}
      title="Telling the time with an armillary sphere"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Sphere axis={axis} lam={lam} H={i >= 3 ? H : i === 2 ? sun.H : 0} step={i} h={sun.h} />}
      viewLabel="The armillary sphere"
      reference={(i) => <Dome opts={{ dec, sunH: sun.H, showHourArc: i >= 4 }} hi={HI[i]} />}
      referenceLabel="The real sky over Istanbul (east on the left)"
      settings={<DateMoment doy={doy} setDoy={setDoy} moment={moment} setMoment={setMoment} />}
      onSkip={() => {
        setAxis(PHI);
        setH(sun.H);
      }}
      equation={
        <Working label="What the sphere is computing">
          <Eq>
            <Sym role="sky">h</Sym> from <Sym role="soft">φ</Sym>, <Sym role="brass">δ</Sym> and <Sym role="red">H</Sym>, turned around:
          </Eq>
          <Eq>
            <Sym role="red">H</Sym> = <Known when={kn(3) && onH} role="red">{`${H > 0 ? '+' : H < 0 ? '−' : ''}${Math.abs(H).toFixed(1)}°`}</Known> (φ{' '}
            <Known when={axisOk} role="soft">41°</Known>, δ <Known when={kn(1)} role="brass">{deg1(dec)}</Known>, h <Known when={kn(2)} role="sky">{deg1(sun.h)}</Known>)
          </Eq>
          <Eq>
            <Sym role="red">t</Sym> = 12:00 + <Sym role="red">H</Sym> / 15° = <Known when={kn(4) && onH} role="red">{hm(read)}</Known>
          </Eq>
        </Working>
      }
      caption={
        <>
          The first view is the instrument; the second, “the real sky”, is what it models, and it stays the same in every plate of this post.
          Turning the sphere until the sun matches its measured height solves the triangle of pole, zenith and sun without any trigonometry.
          Sun positions: USNO approximate formula.
        </>
      }
    />
  );
}
