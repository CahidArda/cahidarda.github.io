/**
 * Episode 5a, step by step: the width of the strait by triangulation, with the theodolite.
 *   1 Set up over A, sight B, read the circle · 2 Turn to C on the far quay, read · 3 Move to B,
 *   sight A, read · 4 Turn to C, read · 5 Work out w · 6 Where it fails: angle errors.
 * View: the 3D theodolite (TheodoliteModel) beside its eyepiece (Eyepiece); both are driven by
 * the same circle readings and the same ./theoGeom.ts targets. The reader reads the circle;
 * the angles are differences of readings, as on a real instrument (each station has its own
 * arbitrary zero). Step 6 reuses the exact error-region drawing (./triViews.tsx StraitView).
 */
import { useMemo, useState } from 'react';
import { C, Eq, Known, Slider, StepPlate, Sym, Working, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { Eyepiece } from './Eyepiece';
import { TheodoliteCanvas } from './TheodoliteModel';
import { RangeLine, StraitView, fmtEps } from './triViews';
import { RAD, straitLayout } from './geometry';
import { ALPHA, BASE, BETA, MIN_STEP, ON_TOL, PA, PB, PC, diff, dm, readingTo, snap, toDeg, widthFrom } from './theoGeom';
import { ROAD_LEN } from './site';

type Sight = { d: number; m: number };
const START: Record<string, Sight> = { AB: { d: 300, m: 0 }, AC: { d: 60, m: 0 }, BA: { d: 200, m: 0 }, BC: { d: 160, m: 0 } };
const TARGET = {
  AB: readingTo('A', PB),
  AC: readingTo('A', PC),
  BA: readingTo('B', PA),
  BC: readingTo('B', PC),
};
type Key = keyof typeof TARGET;

function Instrument({ station, h, targets, say }: { station: 'A' | 'B'; h: number; targets: { name: string; h: number }[]; say: string }) {
  const t3 = useMemo(() => targets.map((t) => ({ h: t.h, v: 0 })), [targets]);
  return (
    <div style={{ display: 'grid', gap: '0.4rem' }}>
      <TheodoliteCanvas h={h} v={0} targets={t3} />
      <Eyepiece h={h} v={0} targets={targets.map((t) => ({ name: t.name, h: t.h, v: 0, kind: 'pole' as const }))} say={say} />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: C.soft }}>
        {`theodolite at ${station}`}
      </span>
    </div>
  );
}

export default function StraitSteps() {
  const [step, setStep] = useState(0);
  const [s, setS] = useState<Record<Key, Sight>>(START);
  const [eps, setEps] = useState(30); // arcminutes
  const val = (k: Key) => toDeg(s[k].d, s[k].m);
  const on = (k: Key) => Math.abs(diff(val(k), TARGET[k])) <= ON_TOL + 1e-12;
  const set = (k: Key, part: 'd' | 'm', x: number) => setS((p) => ({ ...p, [k]: { ...p[k], [part]: x } }));
  const alpha = (((val('AC') - val('AB')) % 360) + 360) % 360;
  const beta = (((val('BA') - val('BC')) % 360) + 360) % 360;
  const w = widthFrom(BASE, alpha, beta);
  const kn = (k: Key, i: number) => step >= i && on(k);
  const ctl = (k: Key) => (
    <>
      <Slider label="turn: whole degrees" value={s[k].d} min={0} max={359} step={1} onChange={(x) => set(k, 'd', x)} format={(x) => `${x}°`} />
      <Slider label="tangent screw: minutes" value={s[k].m} min={0} max={59.9} step={MIN_STEP} onChange={(x) => set(k, 'm', Math.round(x * 10) / 10)} format={(x) => `${x.toFixed(1)}′`} />
    </>
  );
  const sayFor = (k: Key, name: string) => (on(k) ? `on ${name}: the circle reads ${dm(val(k))}` : `circle ${dm(val(k))}: put the vertical hair on ${name}`);
  const steps: StepDef[] = [
    {
      id: 'AB',
      title: 'At A, sight B',
      text: (
        <>
          Nobody can chain across 700 m of water. Measure angles instead. Set the theodolite over A, plumb bob on the mark,
          level it, and turn it until the vertical hair cuts the pole at B, 200 m up the quay (that baseline was chained).
        </>
      ),
      controls: ctl('AB'),
      gated: true,
      done: on('AB'),
      hint: 'Put the hair on B',
    },
    {
      id: 'AC',
      title: 'At A, sight C',
      text: <>Turn to the pole at C on the far quay and read the circle again. The angle at A is the difference of the two readings.</>,
      controls: ctl('AC'),
      gated: true,
      done: on('AC'),
      hint: 'Put the hair on C',
    },
    {
      id: 'BA',
      title: 'At B, sight A',
      text: <>Carry the instrument to B and set it up again. Its circle has a new, arbitrary zero. Sight back to A.</>,
      controls: ctl('BA'),
      gated: true,
      done: on('BA'),
      hint: 'Put the hair on A',
    },
    {
      id: 'BC',
      title: 'At B, sight C',
      text: <>Turn to C. The difference of the two readings at B is the angle at B.</>,
      controls: ctl('BC'),
      gated: true,
      done: on('BC'),
      hint: 'Put the hair on C',
    },
    {
      id: 'solve',
      title: 'Work out the width',
      text: <>One side and two angles fix a triangle. The law of sines gives the distance from the baseline to C: the width of the strait.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Every reading has an error. Give each angle a few arcminutes and the answer becomes a region (red). A short baseline
          makes a thin triangle, and a thin triangle magnifies the error.
        </>
      ),
      controls: <Slider label="error in each angle" value={eps} min={0} max={60} step={0.5} onChange={setEps} format={fmtEps} />,
    },
  ];
  const view = (i: number) => {
    if (i === 0) return <Instrument station="A" h={val('AB')} targets={[{ name: 'B', h: TARGET.AB }, { name: 'C', h: TARGET.AC }]} say={sayFor('AB', 'B')} />;
    if (i === 1) return <Instrument station="A" h={val('AC')} targets={[{ name: 'B', h: TARGET.AB }, { name: 'C', h: TARGET.AC }]} say={sayFor('AC', 'C')} />;
    if (i === 2) return <Instrument station="B" h={val('BA')} targets={[{ name: 'A', h: TARGET.BA }, { name: 'C', h: TARGET.BC }]} say={sayFor('BA', 'A')} />;
    if (i === 3) return <Instrument station="B" h={val('BC')} targets={[{ name: 'A', h: TARGET.BA }, { name: 'C', h: TARGET.BC }]} say={sayFor('BC', 'C')} />;
    return <StraitView b={BASE} eps={i === 5 ? (eps / 60) * RAD : 0} />;
  };
  const S = straitLayout(400, BASE, (eps / 60) * RAD);
  return (
    <StepPlate
      n={8}
      wide
      title="The width of the strait, from two angles"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel={step <= 3 ? 'The theodolite and its telescope' : 'Plan, to scale'}
      reference={(i) => (
        <SiteRef
          hi={i >= 4 ? ['baseline', 'strait'] : ['baseline']}
          best={i >= 4 ? `known: the strait is ${w.toFixed(0)} m wide` : `known: the road is ${ROAD_LEN.toFixed(0)} m`}
          label="The site: the baseline A to B on the near quay and the target C on the far quay"
          overlay={(g) => (
            <g>
              {i >= 0 && <line x1={g.names.pA.x} y1={g.names.pA.y} x2={(i === 0 ? g.names.pB : g.names.pC).x} y2={(i === 0 ? g.names.pB : g.names.pC).y} style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '4 3' }} />}
              {i >= 1 && <line x1={g.names.pA.x} y1={g.names.pA.y} x2={g.names.pB.x} y2={g.names.pB.y} style={{ stroke: C.sky, strokeWidth: 1, strokeDasharray: '4 3' }} />}
              {i >= 2 && <line x1={g.names.pB.x} y1={g.names.pB.y} x2={(i === 2 ? g.names.pA : g.names.pC).x} y2={(i === 2 ? g.names.pA : g.names.pC).y} style={{ stroke: C.brass, strokeWidth: 1, strokeDasharray: '4 3' }} />}
              {i >= 3 && <line x1={g.names.pB.x} y1={g.names.pB.y} x2={g.names.pC.x} y2={g.names.pC.y} style={{ stroke: C.brass, strokeWidth: 1, strokeDasharray: '4 3' }} />}
            </g>
          )}
        />
      )}
      referenceLabel="The site"
      onSkip={() =>
        setS({ AB: snap(TARGET.AB), AC: snap(TARGET.AC), BA: snap(TARGET.BA), BC: snap(TARGET.BC) })
      }
      equation={
        <Working label="Width of the strait by the law of sines">
          <Eq>
            <Sym role="sky">α</Sym> = <Known when={kn('AC', 1)} role="sky">{dm(val('AC'))}</Known> − <Known when={kn('AB', 0)} role="sky">{dm(val('AB'))}</Known> ={' '}
            <Known when={kn('AC', 1) && on('AB')} role="sky">{dm(alpha)}</Known>
          </Eq>
          <Eq>
            <Sym role="brass">β</Sym> = <Known when={kn('BA', 2)} role="brass">{dm(val('BA'))}</Known> − <Known when={kn('BC', 3)} role="brass">{dm(val('BC'))}</Known> ={' '}
            <Known when={kn('BC', 3) && on('BA')} role="brass">{dm(beta)}</Known>
          </Eq>
          <Eq>
            <Sym role="red">w</Sym> = <Sym>b</Sym> · sin <Sym role="sky">α</Sym> · sin <Sym role="brass">β</Sym> / sin(<Sym role="sky">α</Sym> + <Sym role="brass">β</Sym>)
          </Eq>
          <Eq>
            <Known when={step >= 4} role="red">{`${w.toFixed(1)} m`}</Known> = <Known when>{`${BASE} m`}</Known> · sin <Known when={step >= 4} role="sky">{dm(alpha)}</Known> · sin{' '}
            <Known when={step >= 4} role="brass">{dm(beta)}</Known> / sin <Known when={step >= 4}>{dm(alpha + beta)}</Known>
          </Eq>
          {step >= 5 && eps > 0 && <RangeLine lo={S.reg.lo} hi={S.reg.hi} truth={S.w} eps={eps} dp={0} />}
        </Working>
      }
      caption={
        <>
          The circle is read clockwise from an arbitrary zero, so only differences of readings mean anything. True angles here:
          α = {dm(ALPHA)}, β = {dm(BETA)}, on a 200 m baseline: 700 m across, about the Bosphorus at its narrows. Drag the 3D
          instrument to turn it; the dashed lines point to the poles.
        </>
      }
    />
  );
}
