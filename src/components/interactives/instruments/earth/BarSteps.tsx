/**
 * Episode 6, step by step: the baseline and temperature.
 *   1 Lay plain steel bars on a warm day · 2 See the error over the baseline (and everything
 *   scaled from it) · 3 Colby's pair: brass and iron, joined by tongues · 4 Find the point that
 *   does not move · 5 Where it fails: superb, slow.
 * Geometry: ./barGeom.ts. One temperature (a plate setting) feeds every view and the equation.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { ALPHA, BARS, BASELINE, L0, T, T0, X_STAR, baseScene, compLen, compScene, plainErr, plainRead, tongueMove } from './barGeom';
import { SITE } from './site';

const solid = (role: 'ink' | 'soft' | 'faint' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const mm = (m: number) => `${m >= 0 ? '' : '−'}${Math.abs(m * 1000).toFixed(1)} mm`;
const um = (m: number) => `${m >= 0 ? '' : '−'}${Math.abs(m * 1e6).toFixed(1)} µm`;

function BaseView({ t, comp }: { t: number; comp: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = baseScene(W, t, comp);
  const bars = Math.min(BARS, 66);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Baseline read ${comp ? 'with compensated bars' : 'with plain steel bars'}`}>
        <InkText x={14} y={18} kind="label" size={11} role="soft">{`baseline A–B, ${BASELINE} m · error ×${g.EX2}`}</InkText>
        {Array.from({ length: bars }, (_, i) => {
          const x0 = g.pad + (i * (g.readEnd - g.pad)) / bars;
          const x1 = g.pad + ((i + 1) * (g.readEnd - g.pad)) / bars;
          return <rect key={i} x={x0} y={g.y - 4} width={Math.max(0.5, x1 - x0 - 0.4)} height={8} style={{ fill: i % 2 ? C.deep : C.paper, stroke: comp ? C.brass : C.soft, strokeWidth: 0.4 }} />;
        })}
        <line x1={g.trueEnd} y1={g.y - 18} x2={g.trueEnd} y2={g.y + 18} style={solid('ink', 1.4)} />
        <InkText x={g.trueEnd + 4} y={g.y - 20} size={11}>B</InkText>
        <InkText x={g.pad} y={g.y - 12} size={11}>A</InkText>
        <line x1={g.readEnd} y1={g.y + 8} x2={g.readEnd} y2={g.y + 26} style={solid('red', 1.6)} />
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={comp || Math.abs(g.err) < 1e-4 ? 'soft' : 'red'}>
          {comp ? 'compensated bars: the count is right at any temperature' : `${BARS} bars at ${t} °C read ${plainRead(t).toFixed(3)} m`}
        </InkText>
      </svg>
    </div>
  );
}

function CompView({ t, dotX, showDot }: { t: number; dotX: number; showDot: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = compScene(W, t, dotX);
  const still = Math.abs(dotX - X_STAR) < 0.05;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Compensation bar at ${t} degrees`}>
        <InkText x={14} y={18} kind="label" size={11} role="soft">{`Colby’s bar from above, expansion ×${g.EX}`}</InkText>
        <line x1={g.ends[0].b.x} y1={g.yBrass} x2={g.ends[1].b.x} y2={g.yBrass} style={solid('brass', 7)} />
        <line x1={g.ends[0].i.x} y1={g.yIron} x2={g.ends[1].i.x} y2={g.yIron} style={solid('soft', 7)} />
        <InkText x={g.cx} y={g.yBrass - 10} anchor="middle" size={11} role="brass">brass</InkText>
        <InkText x={g.cx} y={g.yIron + 18} anchor="middle" size={11} role="soft">iron</InkText>
        <line x1={g.cx} y1={g.yBrass - 4} x2={g.cx} y2={g.yIron + 4} style={solid('ink', 2)} />
        {g.ends.map((e) => (
          <g key={e.side}>
            <line x1={e.b.x} y1={e.b.y} x2={e.tip.x} y2={e.tip.y} style={solid('ink', 2.2)} />
            <circle cx={e.b.x} cy={e.b.y} r={2.2} style={{ fill: C.paper, stroke: C.ink }} />
            <circle cx={e.i.x} cy={e.i.y} r={2.2} style={{ fill: C.paper, stroke: C.ink }} />
            {showDot && <circle cx={e.probe.x} cy={e.probe.y} r={3.4} style={{ fill: still ? C.red : C.sky }} />}
          </g>
        ))}
        {showDot &&
          g.rest.map((x, i) => (
            <line key={i} x1={x} y1={g.ends[0].probe.y - 10} x2={x} y2={g.ends[0].probe.y + 10} style={{ ...solid('faint', 0.9), strokeDasharray: '2 2' }} />
          ))}
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={showDot && still ? 'red' : 'soft'}>
          {showDot ? (still ? 'this point stays put: the dot goes here' : `this point moves ${um(g.probeMove)} at each end`) : `${t} °C: brass grows more than iron, so the tongues tilt`}
        </InkText>
      </svg>
    </div>
  );
}

export default function BarSteps() {
  const [step, setStep] = useState(0);
  const [t, setT] = useState(T.START);
  const [dotX, setDotX] = useState(0.5);
  const still = Math.abs(dotX - X_STAR) < 0.05;
  const err = plainErr(t);
  const wErr = SITE.WIDTH * ALPHA.steel * (t - T0);
  const steps: StepDef[] = [
    {
      id: 'lay',
      title: 'Lay steel bars',
      text: <>Every length in the triangle came from one measured side, the 200 m baseline. Measure it with 10-foot steel bars laid end to end.</>,
    },
    {
      id: 'warm',
      title: 'Feel the heat',
      text: (
        <>
          Steel grows about 11.5 millionths of its length per degree. On a warm day each bar is long, fewer fit, and the
          baseline reads short. The whole map shrinks with it.
        </>
      ),
    },
    {
      id: 'pair',
      title: 'Pair brass and iron',
      text: (
        <>
          Colby and Drummond’s answer (1820s): a brass bar and an iron bar side by side, fixed at the middle, with steel tongues
          across the ends. Brass grows more, so the tongues tilt.
        </>
      ),
    },
    {
      id: 'dot',
      title: 'Find the still point',
      text: <>Somewhere along each tongue, beyond the iron bar, a point does not move at all. Slide along the tongue to find it.</>,
      controls: <Slider label="along the tongue, gap widths past the iron" value={dotX} min={0} max={2.6} step={0.05} onChange={setDotX} format={(v) => v.toFixed(2)} />,
      gated: true,
      done: still,
      hint: 'Find the point that stays put',
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          It worked: the Lough Foyle base (1827-28) was later remeasured about an inch longer over nearly 8 miles. But it took
          about 60 days and 70 men. Every new baseline cost months.
        </>
      ),
    },
  ];
  return (
    <StepPlate
      n={11}
      wide
      title="Baseline bars that ignore the weather"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (i <= 1 ? <BaseView t={i === 0 ? T0 : t} comp={false} /> : i === 4 ? <BaseView t={t} comp /> : <CompView t={t} dotX={dotX} showDot={i === 3} />)}
      viewLabel="On the quay"
      reference={(i) => (
        <SiteRef
          hi={['baseline', 'strait']}
          best={i >= 1 && i < 4 ? `plain bars, ${t} °C: strait reads ${(SITE.WIDTH - wErr).toFixed(3)} m` : 'known: the strait 700 m, the minaret 76 m'}
          label="The site, with the baseline and the width of the strait highlighted"
        />
      )}
      referenceLabel="The site"
      settings={<Slider label="air and bar temperature" value={t} min={T.MIN} max={T.MAX} step={T.STEP} onChange={setT} format={(v) => `${v < 0 ? '−' : ''}${Math.abs(v)} °C`} />}
      onSkip={() => setDotX(X_STAR)}
      equation={
        <Working label="Thermal error and the compensation point">
          <Eq>
            <Sym role="red">ΔL</Sym> = <Sym>L</Sym> · <Sym role="brass">α</Sym> · <Sym role="sky">ΔT</Sym>
          </Eq>
          <Eq>
            <Known when={step >= 1} role="red">{mm(err)}</Known> = <Known when>{`${BASELINE} m`}</Known> · <Known when role="brass">11.5×10⁻⁶</Known> · <Known when={step >= 1} role="sky">{`${t - T0 < 0 ? '−' : ''}${Math.abs(t - T0)} °C`}</Known>
          </Eq>
          <Eq>
            <Sym role="red">x</Sym> / <Sym>s</Sym> = <Sym role="soft">α</Sym>
            <sub>iron</sub> / (<Sym role="brass">α</Sym>
            <sub>brass</sub> − <Sym role="soft">α</Sym>
            <sub>iron</sub>) = <Known when={step >= 3 && still} role="red">{`${X_STAR.toFixed(1)}`}</Known>
          </Eq>
          <Eq>
            bar between the dots: <Known when={step >= 3 && still}>{`${compLen(t).toFixed(6)} m`}</Known> at any temperature (plain: {(L0 * (1 + ALPHA.steel * (t - T0))).toFixed(6)} m)
          </Eq>
        </Working>
      }
      caption={
        <>
          The compensation bar principle after Herschel’s 1867 description: two metals, tongues across the ends, two fine dots
          that keep the same distance. Expansion coefficients are textbook values (brass to iron about 3 to 2); the bar
          proportions are illustrative. The still point x = 2s is derived from that ratio. Bars are 10 ft, true at {T0} °C.
          {` `}A tongue moves {um(tongueMove(0, t))} at the iron bar.
        </>
      }
    />
  );
}
