/**
 * Episode 5b, step by step: the height of the minaret from two stations on the precinct.
 *   1 At the far station, tip the telescope up to the finial and read the vertical circle ·
 *   2 Walk 50 m toward the minaret and read again · 3 Work out h · 4 Where it fails: near the
 *   top the two sight lines are almost parallel, so an angle error moves the answer a long way.
 * View: the 3D theodolite tilting (TheodoliteModel) and its eyepiece (Eyepiece), from the same
 * vertical-circle reading; the last step is the exact error drawing (./triViews.tsx MinaretView).
 */
import { useMemo, useState } from 'react';
import { C, Eq, Known, Slider, StepPlate, Sym, Working, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { Eyepiece } from './Eyepiece';
import { TheodoliteCanvas } from './TheodoliteModel';
import { MinaretView, RangeLine, fmtEps } from './triViews';
import { RAD, minaretLayout } from './geometry';
import { EL_FAR, EL_NEAR, MIN_STEP, ON_TOL, TOWER, V_MAX, dm, heightFrom, snap, toDeg } from './theoGeom';

const FACE = 270; // the telescope faces the minaret (west); the circle reading is not used here

function Instrument({ v, target, say }: { v: number; target: number; say: string }) {
  const t3 = useMemo(() => [{ h: FACE, v: target }], [target]);
  return (
    <div style={{ display: 'grid', gap: '0.4rem' }}>
      <TheodoliteCanvas h={FACE} v={v} targets={t3} />
      <Eyepiece h={FACE} v={v} targets={[{ name: 'finial', h: FACE, v: target, kind: 'tip' }]} say={say} />
    </div>
  );
}

export default function HeightSteps() {
  const [step, setStep] = useState(0);
  const [far, setFar] = useState({ d: 20, m: 0 });
  const [near, setNear] = useState({ d: 40, m: 0 });
  const [eps, setEps] = useState(30);
  const a = toDeg(far.d, far.m);
  const b = toDeg(near.d, near.m);
  const onFar = Math.abs(a - EL_FAR) <= ON_TOL + 1e-12;
  const onNear = Math.abs(b - EL_NEAR) <= ON_TOL + 1e-12;
  const base = TOWER.far - TOWER.near;
  const h = heightFrom(base, a, b);
  const ctl = (st: { d: number; m: number }, set: (x: { d: number; m: number }) => void) => (
    <>
      <Slider label="tip up: whole degrees" value={st.d} min={0} max={V_MAX - 1} step={1} onChange={(x) => set({ ...st, d: x })} format={(x) => `${x}°`} />
      <Slider label="tangent screw: minutes" value={st.m} min={0} max={59.9} step={MIN_STEP} onChange={(x) => set({ ...st, m: Math.round(x * 10) / 10 })} format={(x) => `${x.toFixed(1)}′`} />
    </>
  );
  const say = (v: number, ok: boolean) => (ok ? `on the finial: the vertical circle reads ${dm(v)}` : `reading ${dm(v)}: bring the crosshairs onto the finial`);
  const L = minaretLayout(400, TOWER.near, base, (eps / 60) * RAD);
  const steps: StepDef[] = [
    {
      id: 'far',
      title: 'Far station: tip up',
      text: (
        <>
          A chain cannot climb a minaret either. Set up on the paving {TOWER.far} m from its foot, aim at it, and tip the
          telescope up until the crosshairs sit on the tip of the finial. Read the vertical circle: α.
        </>
      ),
      controls: ctl(far, setFar),
      gated: true,
      done: onFar,
      hint: 'Crosshairs on the finial',
    },
    {
      id: 'near',
      title: 'Near station: tip up',
      text: <>Chain {base} m straight toward the minaret, set up again and read the steeper angle: β.</>,
      controls: ctl(near, setNear),
      gated: true,
      done: onNear,
      hint: 'Crosshairs on the finial',
    },
    {
      id: 'solve',
      title: 'Work out the height',
      text: <>Two angles and the distance between the stations fix the triangle, and its top is the finial. You never needed to reach the foot.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: <>Near the top the two sight lines are almost parallel, so a small angle error slides their crossing a long way, and further up than down.</>,
      controls: <Slider label="error in each angle" value={eps} min={0} max={60} step={0.5} onChange={setEps} format={fmtEps} />,
    },
  ];
  return (
    <StepPlate
      n={9}
      wide
      title="The height of the minaret, from two angles"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) =>
        i === 0 ? (
          <Instrument v={a} target={EL_FAR} say={say(a, onFar)} />
        ) : i === 1 ? (
          <Instrument v={b} target={EL_NEAR} say={say(b, onNear)} />
        ) : (
          <MinaretView dB={TOWER.near} b={base} eps={i === 3 ? (eps / 60) * RAD : 0} />
        )
      }
      viewLabel={step <= 1 ? 'The theodolite and its telescope' : 'Side view, to scale'}
      reference={(i) => (
        <SiteRef
          hi={i >= 2 ? ['height'] : ['minaret']}
          best={i >= 2 ? `known: the minaret is ${h.toFixed(1)} m` : 'known: the strait is 700 m wide'}
          label="The site, with the minaret's height highlighted"
          overlay={(g) => {
            const pFar = g.S(TOWER.far, 40);
            const pNear = g.S(TOWER.near, 40);
            return (
              <g>
                {[pFar, pNear].map((p, j) => (
                  <line key={j} x1={p.x} y1={p.y} x2={g.names.tip.x} y2={g.names.tip.y} style={{ stroke: j ? C.brass : C.sky, strokeWidth: 0.9, strokeDasharray: '3 2', opacity: i >= j ? 1 : 0.25 }} />
                ))}
              </g>
            );
          }}
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setFar(snap(EL_FAR));
        setNear(snap(EL_NEAR));
      }}
      equation={
        <Working label="Height from two elevation angles">
          <Eq>
            <Sym role="red">h</Sym> = <Sym>b</Sym> · sin <Sym role="sky">α</Sym> · sin <Sym role="brass">β</Sym> / sin(<Sym role="brass">β</Sym> − <Sym role="sky">α</Sym>)
          </Eq>
          <Eq>
            <Known when={step >= 2} role="red">{`${h.toFixed(1)} m`}</Known> = <Known when>{`${base} m`}</Known> · sin <Known when={onFar} role="sky">{dm(a)}</Known> · sin{' '}
            <Known when={step >= 1 && onNear} role="brass">{dm(b)}</Known> / sin <Known when={step >= 2}>{dm(b - a)}</Known>
          </Eq>
          {step >= 3 && eps > 0 && <RangeLine lo={L.reg.lo} hi={L.reg.hi} truth={L.h} eps={eps} dp={1} />}
        </Working>
      }
      caption={
        <>
          The angles are measured from the ground mark; a real survey adds the instrument’s height. Readings to 0.1′, so the
          answer can differ from 76 m by a few centimetres. The section in the reference has heights ×3, which steepens the
          sight lines there; the side view is to scale.
        </>
      }
    />
  );
}
