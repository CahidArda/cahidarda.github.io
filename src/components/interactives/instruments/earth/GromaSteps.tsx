/**
 * Episode 3a, step by step: a right angle with the groma, at A on the near quay.
 *   1 Turn the cross until one pair of strings covers B · 2 Look along the other pair: an
 *   assistant on the far quay moves a pole until the strings hide it · 3 The right angle is
 *   built into the cross: AC is square to AB · 4 Where it fails: wind swings the strings.
 * Plan view to scale (./romanGeom.ts gromaScene); the cross is shown enlarged in an inset.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { GROMA, farHit, gromaScene, onB, poleHidden, windAngleMin, windMiss } from './romanGeom';

function GromaView({ psi, poleY, sway, showPole }: { psi: number; poleY: number; sway: number; showPole: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = gromaScene(W, psi, poleY, sway);
  const hidden = poleHidden(psi, poleY);
  const lineB = onB(psi);
  const solid = (role: 'ink' | 'soft' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Groma at A, cross turned ${psi.toFixed(1)} degrees`}>
        <rect x={g.A.x} y={14} width={g.farX - g.A.x} height={g.H - 40} style={{ fill: C.sky, opacity: 0.1 }} />
        <line x1={g.A.x} y1={14} x2={g.A.x} y2={g.H - 26} style={solid('ink', 1.1)} />
        <line x1={g.farX} y1={14} x2={g.farX} y2={g.H - 26} style={solid('ink', 1.1)} />
        <InkText x={g.A.x + 6} y={26} size={11} role="soft">near quay</InkText>
        <InkText x={g.farX - 6} y={26} anchor="end" size={11} role="soft">far quay</InkText>
        {/* pair 1: toward B */}
        <line x1={g.A.x} y1={g.A.y} x2={g.sight1.x} y2={g.sight1.y} style={{ ...solid(lineB ? 'sky' : 'soft', 1), strokeDasharray: '5 3' }} />
        {/* pair 2: across the water */}
        {g.ray0 && <line x1={g.A.x} y1={g.A.y} x2={g.ray0.x} y2={g.ray0.y} style={{ ...solid('brass', 1), strokeDasharray: '5 3' }} />}
        {sway !== 0 && g.rayW && <line x1={g.A.x} y1={g.A.y} x2={g.rayW.x} y2={g.rayW.y} style={solid('red', 1.2)} />}
        <circle cx={g.B.x} cy={g.B.y} r={4} style={{ fill: C.paper, stroke: lineB ? C.sky : C.ink, strokeWidth: 1.4 }} />
        <InkText x={g.B.x - 8} y={g.B.y + 4} anchor="end" kind="title" size={12}>B</InkText>
        <circle cx={g.A.x} cy={g.A.y} r={4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.4 }} />
        <InkText x={g.A.x - 8} y={g.A.y + 4} anchor="end" kind="title" size={12}>A</InkText>
        {showPole && (
          <g>
            <circle cx={g.pole.x} cy={g.pole.y} r={4} style={{ fill: hidden ? C.red : C.paper, stroke: C.red, strokeWidth: 1.4 }} />
            <InkText x={g.pole.x - 8} y={g.pole.y - 8} anchor="end" kind="title" size={12} role="red">
              {hidden ? 'C' : 'pole'}
            </InkText>
          </g>
        )}
        {/* inset: the cross from above */}
        <circle cx={g.inset.cx} cy={g.inset.cy} r={g.inset.r} style={{ fill: C.paper, stroke: C.soft, strokeWidth: 0.8 }} />
        <line x1={g.strings[0].x} y1={g.strings[0].y} x2={g.strings[2].x} y2={g.strings[2].y} style={solid('sky', 2.2)} />
        <line x1={g.strings[1].x} y1={g.strings[1].y} x2={g.strings[3].x} y2={g.strings[3].y} style={solid('brass', 2.2)} />
        {g.strings.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={3} style={{ fill: C.ink }} />
        ))}
        <InkText x={g.inset.cx} y={g.inset.cy + g.inset.r + 13} anchor="middle" size={10} role="soft">the cross from above</InkText>
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={12} role={hidden && lineB ? 'red' : 'soft'}>
          {!lineB
            ? `cross at ${psi.toFixed(1)}°: turn it until the first pair covers B`
            : !showPole
              ? 'the first pair covers B'
              : sway
                ? `wind: the line misses by ${windMiss(sway).toFixed(1)} m on the far quay`
                : hidden
                  ? 'the pole is hidden behind both strings: mark C'
                  : 'move the pole until both strings hide it'}
        </InkText>
      </svg>
    </div>
  );
}

export default function GromaSteps() {
  const [step, setStep] = useState(0);
  const [psi, setPsi] = useState<number>(GROMA.PSI_START);
  const [poleY, setPoleY] = useState(120);
  const [sway, setSway] = useState(5);
  const lineB = onB(psi);
  const y0 = farHit(psi);
  const hidden = poleHidden(psi, poleY);
  const steps: StepDef[] = [
    {
      id: 'turn',
      title: 'Sight B',
      text: (
        <>
          Plant the groma over A. Its cross has a plumb line hanging from each arm. Turn the cross until one pair of strings,
          one behind the other, covers the pole at B up the quay.
        </>
      ),
      controls: <Slider label="turn the cross" value={psi} min={GROMA.PSI_MIN} max={GROMA.PSI_MAX} step={GROMA.PSI_STEP} onChange={setPsi} format={(v) => `${v.toFixed(1)}°`} />,
      gated: true,
      done: lineB,
      hint: 'Cover B with the first pair',
    },
    {
      id: 'pole',
      title: 'Hide the far pole',
      text: <>Now look along the other pair, across the water. A helper on the far quay moves a pole until both strings hide it.</>,
      controls: <Slider label="helper’s pole, m north of A" value={poleY} min={GROMA.POLE_MIN} max={GROMA.POLE_MAX} step={GROMA.POLE_STEP} onChange={setPoleY} format={(v) => `${v} m`} />,
      gated: true,
      done: hidden,
      hint: 'Move the pole behind the strings',
    },
    {
      id: 'square',
      title: 'Read the right angle',
      text: <>The arms are square to each other, so the line A to C is square to A to B. No number was read: the angle is built into the cross.</>,
    },
    {
      id: 'wind',
      title: 'Where it fails',
      text: (
        <>
          Wind swings the strings. A few millimetres of sway on strings under a metre apart turns the line by many
          arcminutes, metres on the far quay. And the groma gives directions only: no height, no distance.
        </>
      ),
      controls: <Slider label="wind sway of a string" value={sway} min={0} max={GROMA.SWAY_MAX} step={1} onChange={setSway} format={(v) => `${v} mm`} />,
    },
  ];
  return (
    <StepPlate
      n={3}
      wide
      title="A right angle with the groma"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (
        <GromaView psi={psi} poleY={poleY} sway={i === 3 ? sway : 0} showPole={i >= 1} />
      )}
      viewLabel="Plan of the quay, to scale"
      reference={(i) => (
        <SiteRef
          hi={i >= 2 ? ['baseline', 'strait'] : ['baseline']}
          best={i >= 2 ? 'known: a right angle at A' : 'known: the slope, a unit of length'}
          label="The site, with the baseline A to B and the line across the strait highlighted"
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setPsi(90);
        setPoleY(0);
      }}
      equation={
        <Working label="The groma's right angle and the wind's error">
          <Eq>
            angle <Sym role="sky">B</Sym>–A–<Sym role="brass">C</Sym> = 90° (the cross)
          </Eq>
          <Eq>
            C lies <Known when={step >= 1 && (hidden || step >= 2)} role="brass">{`${(y0 ?? 0).toFixed(1)} m`}</Known> north of A, straight across
          </Eq>
          <Eq>
            miss = <Sym>w</Sym> · <Sym role="red">s</Sym> / <Sym>S</Sym> = <Known when>700 m</Known> · <Known when={step >= 3} role="red">{`${sway} mm`}</Known> /{' '}
            <Known when>{`${GROMA.S * 1000} mm`}</Known> = <Known when={step >= 3} role="red">{`${windMiss(sway).toFixed(1)} m`}</Known> (
            <Known when={step >= 3}>{`${windAngleMin(sway).toFixed(0)}′`}</Known>)
          </Eq>
        </Working>
      }
      caption={
        <>
          The groma is a Roman surveyor’s cross on a staff with four plumb lines. The string spacing S is illustrative
          (reconstructions differ). The wind sway is the sideways offset of one string; the error grows with the distance
          sighted, so across 700 m of water it is metres.
        </>
      }
    />
  );
}
