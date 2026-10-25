/**
 * Episode 3b, step by step: holding a gradient with the chorobates, on the water channel at
 * the top of the hill (100 m falling 2.5 cm, the Nîmes aqueduct's mean gradient, site.ts).
 *   1 Set the bench on the channel bed · 2 Shim the downhill leg until the water touches both
 *   rims · 3 Sight along the top to the staff at the end of the run · 4 Work out the fall ·
 *   5 Where it fails: "looks level" has a tolerance, and over 94 m it is as big as the fall.
 * Geometry: ./romanGeom.ts (benchScene, sightScene). One tilt function feeds the bench, the
 * water, the sight line, the staff reading and the equation.
 */
import { useState } from 'react';
import { C, Eq, HatchDefs, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import {
  CHORO,
  DROP_UNC,
  SIGHT_D,
  TILT_TOL,
  TRUE_DROP,
  benchScene,
  instHeight,
  looksLevel,
  measuredDrop,
  sightScene,
  staffReading,
  tiltOf,
} from './romanGeom';

const solid = (role: 'ink' | 'soft' | 'faint' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const cm = (m: number, dp = 1) => {
  const v = m * 100;
  return `${Math.abs(v) < 0.5 * 10 ** -dp ? (0).toFixed(dp) : v.toFixed(dp)} cm`;
};

function BenchView({ shim }: { shim: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = benchScene(W, shim);
  const lvl = looksLevel(shim);
  const id = `ch-h-${W}`;
  // water fills the groove to its lower rim; its surface is level
  const wy = Math.max(g.gL.y, g.gR.y);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={g.status}>
        <HatchDefs id={id} gap={6} />
        <InkText x={14} y={18} kind="label" size={11} role="soft">{`the chorobates, tilts ×${g.EX}`}</InkText>
        <path
          d={`M 0 ${g.bedY(0)} L ${W} ${g.bedY(W)} L ${W} ${g.H} L 0 ${g.H} Z`}
          style={{ fill: `url(#${id})`, stroke: C.ink, strokeWidth: 1 }}
        />
        {/* legs */}
        <path d={`M ${g.pivot.x} ${g.pivot.y} L ${g.pivot.x + Math.sin(g.t) * 70} ${g.pivot.y - Math.cos(g.t) * 70}`} style={solid('ink', 3)} />
        <path d={`M ${g.footR.x} ${g.footR.y} L ${g.footR.x + Math.sin(g.t) * 70} ${g.footR.y - Math.cos(g.t) * 70}`} style={solid('ink', 3)} />
        {g.shimPx > 0.3 && (
          <rect x={g.footR.x - 7} y={g.footR.y} width={14} height={Math.max(0.6, g.bedY(g.footR.x) - g.footR.y)} style={{ fill: C.brass, opacity: 0.8 }} />
        )}
        {/* the bench top */}
        <path d={`M ${g.topL.x} ${g.topL.y} L ${g.topR.x} ${g.topR.y}`} style={solid('ink', 7)} />
        <path d={`M ${g.topL.x} ${g.topL.y} L ${g.topR.x} ${g.topR.y}`} style={{ fill: 'none', stroke: C.paper, strokeWidth: 5 }} />
        {/* the groove and its water */}
        <path d={`M ${g.gL.x} ${g.gL.y} L ${g.gR.x} ${g.gR.y}`} style={solid('ink', 1)} />
        <line x1={g.gL.x} y1={wy} x2={g.gR.x} y2={wy} style={solid('sky', 2.2)} />
        <circle cx={g.gL.x} cy={g.gL.y} r={2.4} style={{ fill: lvl || g.gL.y >= g.gR.y ? C.sky : C.paper, stroke: C.sky, strokeWidth: 1 }} />
        <circle cx={g.gR.x} cy={g.gR.y} r={2.4} style={{ fill: lvl || g.gR.y >= g.gL.y ? C.sky : C.paper, stroke: C.sky, strokeWidth: 1 }} />
        {/* plumb lines and their marks */}
        {g.bobs.map((b, i) => (
          <g key={i}>
            <line x1={b.hang.x} y1={b.hang.y} x2={b.bob.x} y2={b.bob.y} style={solid('ink', 0.8)} />
            <circle cx={b.bob.x} cy={b.bob.y + 3} r={3} style={{ fill: C.brass }} />
            <line x1={b.mark.x - 6 * Math.cos(g.t)} y1={b.mark.y - 6 * Math.sin(g.t)} x2={b.mark.x + 6 * Math.cos(g.t)} y2={b.mark.y + 6 * Math.sin(g.t)} style={solid('red', 1)} />
            <line x1={b.mark.x + 10 * Math.sin(g.t)} y1={b.mark.y - 10 * Math.cos(g.t)} x2={b.mark.x - 10 * Math.sin(g.t)} y2={b.mark.y + 10 * Math.cos(g.t)} style={solid('red', 1)} />
          </g>
        ))}
        <InkText x={(g.gL.x + g.gR.x) / 2} y={Math.min(g.gL.y, g.gR.y) - 14} anchor="middle" size={11} role="sky">water groove</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={lvl ? 'red' : 'soft'}>{g.status}</InkText>
      </svg>
    </div>
  );
}

function SightView({ shim, band }: { shim: number; band: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = sightScene(W, shim, band);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={g.status}>
        <InkText x={14} y={14} kind="label" size={11} role="soft">{`${CHORO.RUN} m of channel · falls and errors ×${g.VE}`}</InkText>
        <line x1={g.bed0.x} y1={g.bed0.y} x2={g.bedEnd.x} y2={g.bedEnd.y} style={solid('ink', 1.3)} />
        <line x1={g.legX} y1={g.levelY} x2={g.staffX + 10} y2={g.levelY} style={{ ...solid('faint', 0.8), strokeDasharray: '4 3' }} />
        <InkText x={g.legX + 4} y={g.levelY + 13} size={10} role="soft">true level</InkText>
        {g.band && (
          <rect x={g.staffX - 9} y={g.band.y0} width={18} height={g.band.y1 - g.band.y0} style={{ fill: C.red, opacity: 0.18 }} />
        )}
        {/* the bench, small */}
        <rect x={g.X(0)} y={g.levelY - 3} width={g.legX - g.X(0) + 6} height={5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
        <line x1={g.legX} y1={g.levelY} x2={g.legBed.x} y2={g.legBed.y} style={solid('ink', 1.6)} />
        <line x1={g.X(0) + 3} y1={g.levelY} x2={g.X(0) + 3} y2={g.bed0.y} style={solid('ink', 1.6)} />
        {/* the sight along the top */}
        <line x1={g.legX} y1={g.levelY} x2={g.hit.x} y2={g.hit.y} style={{ ...solid('sky', 1.2), strokeDasharray: '6 3' }} />
        {/* the staff */}
        <line x1={g.staffX} y1={12} x2={g.staffX} y2={g.bedEnd.y} style={solid('ink', 3)} />
        {g.ticks.map((t) => (
          <g key={t.cm}>
            <line x1={g.staffX} y1={t.y} x2={g.staffX + (t.label ? 8 : 4)} y2={t.y} style={solid('ink', 0.8)} />
            {t.label && (
              <InkText x={g.staffX + 11} y={t.y} baseline="central" kind="num" size={9} role="soft">
                {t.cm}
              </InkText>
            )}
          </g>
        ))}
        <circle cx={g.hit.x} cy={g.hit.y} r={3.4} style={{ fill: C.sky }} />
        <InkText x={g.staffX - 8} y={g.hit.y - 8} anchor="end" kind="num" size={11} role="sky">
          {`${(g.read * 100).toFixed(1)} cm`}
        </InkText>
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={11} role="soft">{g.status}</InkText>
      </svg>
    </div>
  );
}

export default function ChorobatesSteps() {
  const [step, setStep] = useState(0);
  const [shim, setShim] = useState(CHORO.SHIM_START);
  const lvl = looksLevel(shim);
  const drop = measuredDrop(shim);
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'set',
      title: 'Set it on the bed',
      text: (
        <>
          At the top of the hill a water channel runs 100 m from the precinct to the head of the road. It must fall, but
          only a little. Stand the chorobates, a 20-foot bench with plumb lines and a water groove, on its bed.
        </>
      ),
    },
    {
      id: 'shim',
      title: 'Level the bench',
      text: <>Pack the downhill leg until the plumb lines hang on their marks and the water touches both ends of the groove.</>,
      controls: <Slider label="packing under the downhill leg" value={shim} min={CHORO.SHIM_MIN} max={CHORO.SHIM_MAX} step={CHORO.SHIM_STEP} onChange={setShim} format={(v) => `${v.toFixed(1)} mm`} />,
      gated: true,
      done: lvl,
      hint: 'Until the water touches both rims',
    },
    {
      id: 'sight',
      title: 'Sight the staff',
      text: <>Sight along the top edge to a staff held at the end of the channel and read where the line meets it.</>,
    },
    {
      id: 'fall',
      title: 'Work out the fall',
      text: <>The fall is the staff reading minus the height of the bench top above the bed, measured at the downhill leg.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          “Touches both rims” is as good as the eye: about a millimetre over the 5-foot groove. Over a {SIGHT_D.toFixed(0)} m sight that
          hidden tilt is several centimetres, bigger than the fall you are trying to hold. Try the packing across its whole
          level range.
        </>
      ),
      controls: <Slider label="packing under the downhill leg" value={shim} min={CHORO.SHIM_MIN} max={CHORO.SHIM_MAX} step={CHORO.SHIM_STEP} onChange={setShim} format={(v) => `${v.toFixed(1)} mm`} />,
    },
  ];
  return (
    <StepPlate
      n={4}
      wide
      title="Holding a gradient with the chorobates"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (i <= 1 ? <BenchView shim={shim} /> : <SightView shim={shim} band={i === 4} />)}
      viewLabel={step <= 1 ? 'The bench' : 'Along the channel'}
      reference={(i) => (
        <SiteRef
          hi={['channel']}
          best={i >= 3 ? `known: the channel falls ${cm(drop)} in ${SIGHT_D.toFixed(0)} m` : 'known: a right angle, the slope of the road'}
          label="The site, with the water channel on the hilltop highlighted"
        />
      )}
      referenceLabel="The site"
      onSkip={() => setShim(Math.round((CHORO.FALL / CHORO.RUN) * CHORO.LEG_SPAN * 1000 * 2) / 2)}
      equation={
        <Working label="Fall of the channel from the chorobates">
          <Eq>
            <Sym role="red">fall</Sym> = <Sym role="sky">staff</Sym> − <Sym role="brass">bench height</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3) && lvl} role="red">{cm(drop, 2)}</Known> = <Known when={kn(2) && lvl} role="sky">{cm(staffReading(shim), 2)}</Known> −{' '}
            <Known when={kn(1) && lvl} role="brass">{cm(instHeight(shim), 2)}</Known>
          </Eq>
          <Eq>
            true fall over {SIGHT_D.toFixed(0)} m: <Known when={kn(3)}>{cm(TRUE_DROP, 2)}</Known>; hidden tilt up to{' '}
            <Known when={kn(4)}>{`${(TILT_TOL * 1000).toFixed(2)}‰`}</Known> = <Known when={kn(4)} role="red">{`±${cm(DROP_UNC)}`}</Known>
          </Eq>
          {kn(4) && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>{`this setting: bench tilted ${(tiltOf(shim) * 1000).toFixed(2)}‰, ${lvl ? 'and it still looks level' : 'and the water shows it'}`}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          Vitruvius (8.5) describes the chorobates: a straightedge about twenty feet long with plumb lines, and for windy days
          a groove five feet long filled with water. The one-millimetre judgement of the water’s edge is my assumption. The
          channel falls 2.5 cm in 100 m, the mean gradient of the Nîmes aqueduct; Roman surveyors held it by levelling over
          and over and averaging.
        </>
      }
    />
  );
}
