/**
 * Episode 7, step by step: measuring the strait with light (EDM / total station).
 *   1 Aim at the reflector on C · 2 Coarse frequency: match the phase, a rough distance ·
 *   3 Fine frequency: match the phase, the fraction of a 15 m "tape" · 4 Put them together ·
 *   5 Where it fails: the air sets the speed of light, and it needs a line of sight.
 * Geometry and numbers: ./edmGeom.ts.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import {
  D_TRUE,
  F_COARSE,
  F_FINE,
  N_TRUE,
  PHASE_STEP,
  PHI_COARSE,
  PHI_FINE,
  PPM_PER_C,
  beamScene,
  coarseD,
  distance,
  halfLambda,
  phaseOk,
  phaseScene,
} from './edmGeom';

const solid = (role: 'ink' | 'soft' | 'faint' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function PhaseView({ truth, read, label }: { truth: number; read: number; label: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = phaseScene(W, truth, read);
  const ok = phaseOk(read, truth);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`${label}: phase ${read.toFixed(1)} degrees`}>
        <InkText x={14} y={18} kind="label" size={11} role="soft">{label}</InkText>
        <path d={g.out} style={solid(ok ? 'red' : 'sky', 1.4)} />
        <path d={g.ret} style={solid('brass', 1.4)} />
        <InkText x={W - 14} y={g.wy[0] - 20} anchor="end" size={11} role={ok ? 'red' : 'sky'}>{`reference, shifted ${read.toFixed(1)}°`}</InkText>
        <InkText x={W - 14} y={g.wy[1] + 30} anchor="end" size={11} role="brass">light back from C</InkText>
        <line x1={g.peakRet} y1={g.wy[0] - 18} x2={g.peakRet} y2={g.wy[1] + 18} style={{ ...solid('faint', 0.8), strokeDasharray: '2 3' }} />
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={12} role={ok ? 'red' : 'soft'}>
          {ok ? 'the peaks line up: read the phase' : 'shift the reference until its peaks sit over the returning ones'}
        </InkText>
      </svg>
    </div>
  );
}

function BeamView({ n }: { n: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = beamScene(W, n);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label="The beam across the strait, marked in half wavelengths">
        <InkText x={14} y={18} kind="label" size={11} role="soft">plan, A to C, to scale</InkText>
        <rect x={g.A} y={g.y - 22} width={Math.max(0, g.Cx - g.A)} height={44} style={{ fill: C.sky, opacity: 0.12 }} />
        <line x1={g.A} y1={g.y} x2={g.Cx} y2={g.y} style={solid('red', 1.2)} />
        {g.marks.map((x, i) => (
          <line key={i} x1={x} y1={g.y - (i % 10 === 0 ? 9 : 4)} x2={x} y2={g.y + (i % 10 === 0 ? 9 : 4)} style={solid('ink', i % 10 === 0 ? 1 : 0.6)} />
        ))}
        <InkText x={g.A} y={g.y + 26} anchor="middle" kind="title" size={12}>A</InkText>
        <InkText x={g.Cx} y={g.y + 26} anchor="middle" kind="title" size={12}>C</InkText>
        <path d={`M ${g.Cx} ${g.y - 7} l 7 7 l -7 7 Z`} style={{ fill: C.brass }} />
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={12} role="soft">{`${Math.min(n, N_TRUE)} whole half-wavelengths of ${g.hl.toFixed(2)} m, then a fraction`}</InkText>
      </svg>
    </div>
  );
}

export default function EdmSteps() {
  const [step, setStep] = useState(0);
  const [coarse, setCoarse] = useState(40);
  const [fine, setFine] = useState(60);
  const [air, setAir] = useState(15);
  const cOk = phaseOk(coarse, PHI_COARSE);
  const fOk = phaseOk(fine, PHI_FINE);
  const dc = coarseD(coarse);
  const N = Math.floor(dc / halfLambda(F_FINE) - fine / 360 + 0.5);
  const D = distance(N, fine);
  const airErr = D_TRUE * PPM_PER_C * 1e-6 * (air - 15);
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'aim',
      title: 'Aim at the reflector',
      text: (
        <>
          Baselines took months. Light crosses the strait in two microseconds. Set a glass prism on C and aim the instrument at
          A at it. The light goes out, bounces, and comes back.
        </>
      ),
    },
    {
      id: 'coarse',
      title: 'Coarse: match the phase',
      text: <>Timing two microseconds directly is too hard, so the light is made to flicker. At 100 kHz one flicker spans 1.5 km out and back. Match the phase.</>,
      controls: <Slider label="phase shift, 100 kHz" value={coarse} min={0} max={359.5} step={PHASE_STEP} onChange={setCoarse} format={(v) => `${v.toFixed(1)}°`} />,
      gated: true,
      done: cOk,
      hint: 'Line up the peaks',
    },
    {
      id: 'fine',
      title: 'Fine: match the phase',
      text: <>At 10 MHz one flicker is a 15 m “tape”. Its phase gives the fraction of the last tape length to centimetres; the coarse reading says how many whole ones.</>,
      controls: <Slider label="phase shift, 10 MHz" value={fine} min={0} max={359.5} step={PHASE_STEP} onChange={setFine} format={(v) => `${v.toFixed(1)}°`} />,
      gated: true,
      done: fOk,
      hint: 'Line up the peaks',
    },
    {
      id: 'sum',
      title: 'Put them together',
      text: <>Whole tapes from the coarse reading, the fraction from the fine one. Bergstrand’s Geodimeter did this in the 1940s; today a total station does it in a second.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: <>Light is slower in air, and how much slower depends on temperature and pressure: about 1 mm per km for each degree you get wrong. And you still need to see the other end.</>,
      controls: <Slider label="air temperature assumed" value={air} min={-5} max={35} step={1} onChange={setAir} format={(v) => `${v < 0 ? '−' : ''}${Math.abs(v)} °C`} />,
    },
  ];
  return (
    <StepPlate
      n={11}
      wide
      title="The width of the strait, by light"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) =>
        i === 1 ? (
          <PhaseView truth={PHI_COARSE} read={coarse} label="100 kHz: one cycle = 1,499 m" />
        ) : i === 2 ? (
          <PhaseView truth={PHI_FINE} read={fine} label="10 MHz: one cycle = 14.99 m" />
        ) : (
          <BeamView n={i >= 3 ? N_TRUE : 0} />
        )
      }
      viewLabel="The instrument at A"
      reference={(i) => (
        <SiteRef
          hi={['strait']}
          best={i >= 3 ? `known: the strait is ${D.toFixed(2)} m wide` : 'known: the strait 700 m (from angles)'}
          label="The site, with the light beam across the strait"
          overlay={(g) => <line x1={g.names.pA.x} y1={g.names.pA.y - 3} x2={g.names.pC.x} y2={g.names.pC.y - 3} style={{ stroke: C.red, strokeWidth: 1, strokeDasharray: i === 0 ? '4 3' : undefined }} />}
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setCoarse(Math.round(PHI_COARSE / PHASE_STEP) * PHASE_STEP);
        setFine(Math.round(PHI_FINE / PHASE_STEP) * PHASE_STEP);
      }}
      equation={
        <Working label="Distance from the phase of modulated light">
          <Eq>
            <Sym role="red">D</Sym> = (<Sym role="brass">N</Sym> + <Sym role="sky">Δφ</Sym> / 360°) · λ/2
          </Eq>
          <Eq>
            coarse: <Known when={kn(1) && cOk}>{`${dc.toFixed(0)} m`}</Known> = <Known when={kn(1) && cOk}>{`${coarse.toFixed(1)}°`}</Known> / 360° · 1,499 m, so{' '}
            <Sym role="brass">N</Sym> = <Known when={kn(2) && cOk && fOk} role="brass">{`${N}`}</Known>
          </Eq>
          <Eq>
            <Known when={kn(3) && cOk && fOk} role="red">{`${D.toFixed(2)} m`}</Known> = (<Known when={kn(2) && cOk && fOk} role="brass">{`${N}`}</Known> +{' '}
            <Known when={kn(2) && fOk} role="sky">{`${fine.toFixed(1)}°`}</Known> / 360°) · <Known when>{`${halfLambda(F_FINE).toFixed(3)} m`}</Known>
          </Eq>
          {kn(4) && (
            <Eq>
              air assumed {air - 15 < 0 ? `${15 - air} °C too cold` : air - 15 > 0 ? `${air - 15} °C too warm` : 'right'}: off by{' '}
              <Known when role="red">{`${(Math.abs(airErr) * 1000).toFixed(1)} mm`}</Known> over 700 m
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The phase method as in Bergstrand’s Geodimeter and modern total stations. The fine {F_FINE / 1e6} MHz is the Geodimeter’s
          typical frequency; the coarse {F_COARSE / 1000} kHz is illustrative. Phase dial read to {PHASE_STEP}°, which is 2 cm of distance. Speed of light in
          standard air; the 1 ppm per °C rule of thumb sets the last line.
        </>
      }
    />
  );
}
