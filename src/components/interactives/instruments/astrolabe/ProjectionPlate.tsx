/**
 * Plate 2 (episode 2): flattening the sky by stereographic projection, step by step.
 * View: the meridian cut (side) + the plate face on (plan), one construction (projGeom.ts).
 * Reference: the same sky dome as every plate.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { PHI, deg0, num3, tand } from './astro';
import { almucantar, RC, RCAN } from './plateGeom';
import { ALT_MAX, ALT_MIN, DEC_MAX, DEC_MIN, projScene } from './projGeom';
import { Dome, type DomeHi } from './Dome';

type Mode = 'star' | 'alm';

function Construction({ mode, dec, h, step }: { mode: Mode; dec: number; h: number; step: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = projScene(W, { mode, dec, h, phi: PHI });
  const showRay = step >= 1;
  const showCross = step >= 2;
  const showCircle = step >= 2;
  const note =
    mode === 'star'
      ? step === 0
        ? `a star ${deg0(dec)} from the equator, on the meridian`
        : step === 1
          ? 'a straight line from the south pole through the star'
          : `it lands ${num3(g.pts[0].x)} from the pole; the day circle has that radius`
      : `the altitude-${h}° circle lands as a circle, r ${num3(g.circle!.rr)}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={note}>
        {/* ── side view: the sphere cut along the meridian ── */}
        <InkText x={g.sideBox.x0 + 8} y={g.sideBox.y0 + 14} kind="label" size={11} role="soft">the sky, cut north to south</InkText>
        <circle cx={g.sc.x} cy={g.sc.y} r={g.ks} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
        <line x1={g.plateLine[0].x} y1={g.plateLine[0].y} x2={g.plateLine[1].x} y2={g.plateLine[1].y} style={{ stroke: C.ink, strokeWidth: 2.4, opacity: 0.8 }} />
        <line x1={g.north.x} y1={g.north.y} x2={g.south.x} y2={g.south.y} style={{ stroke: C.faint, strokeWidth: 0.7, strokeDasharray: '3 3' }} />
        {mode === 'alm' && <line x1={g.horizonLine[0].x} y1={g.horizonLine[0].y} x2={g.horizonLine[1].x} y2={g.horizonLine[1].y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 3' }} />}
        {g.chord && <line x1={g.chord[0].x} y1={g.chord[0].y} x2={g.chord[1].x} y2={g.chord[1].y} style={{ stroke: C.sky, strokeWidth: 1.6 }} />}
        {g.pts.map((p, i) => (
          <g key={i}>
            {showRay && <line x1={g.south.x} y1={g.south.y} x2={p.rayEnd.x} y2={p.rayEnd.y} style={{ stroke: C.brass, strokeWidth: 1 }} />}
            <circle cx={p.on.x} cy={p.on.y} r={4.5} style={{ fill: mode === 'star' ? C.sky : C.sky, stroke: C.paper, strokeWidth: 1 }} />
            {showCross && <circle cx={p.cross.x} cy={p.cross.y} r={3.5} style={{ fill: C.red }} />}
          </g>
        ))}
        <circle cx={g.south.x} cy={g.south.y} r={3.5} style={{ fill: C.ink }} />
        <InkText x={g.south.x + 8} y={g.south.y + 12} size={11} role="ink">south pole (the eye)</InkText>
        <InkText x={g.north.x + 7} y={g.north.y - 4} size={11} role="soft">north pole</InkText>
        {mode === 'alm' && <InkText x={g.zenith.x + 6} y={g.zenith.y - 5} size={11} role="soft">zenith</InkText>}
        <InkText x={g.plateLine[0].x} y={g.plateLine[0].y - 6} anchor="start" size={11} role="soft">the plate</InkText>

        {/* ── plan view: the plate, face on ── */}
        <InkText x={g.planBox.x0 + 8} y={g.planBox.y0 + 14} kind="label" size={11} role="soft">the plate, face on</InkText>
        <circle cx={g.pc.x} cy={g.pc.y} r={g.tropics.cap} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
        <circle cx={g.pc.x} cy={g.pc.y} r={g.tropics.eq} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.8 }} />
        <circle cx={g.pc.x} cy={g.pc.y} r={g.tropics.cancer} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.8 }} />
        <line x1={g.pc.x} y1={g.pc.y - g.tropics.cap} x2={g.pc.x} y2={g.pc.y + g.tropics.cap} style={{ stroke: C.faint, strokeWidth: 0.7 }} />
        {step >= 4 && mode === 'alm' && (
          <>
            {(() => {
              const hz = almucantar(0, PHI);
              return <circle cx={g.pc.x} cy={g.pc.y - hz.cv * g.kp} r={hz.r * g.kp} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.6 }} clipPath="url(#pclip)" />;
            })()}
          </>
        )}
        <defs>
          <clipPath id="pclip">
            <circle cx={g.pc.x} cy={g.pc.y} r={g.tropics.cap} />
          </clipPath>
        </defs>
        {showCircle && g.circle && (
          <circle cx={g.circle.c.x} cy={g.circle.c.y} r={g.circle.r} clipPath="url(#pclip)" style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.8 }} />
        )}
        {showCross && g.pts.filter((p) => Math.abs(p.x) <= RC).map((p, i) => <circle key={i} cx={p.planPt.x} cy={p.planPt.y} r={3.5} style={{ fill: C.red }} />)}
        <circle cx={g.pc.x} cy={g.pc.y} r={2.5} style={{ fill: C.ink }} />
        <InkText x={g.pc.x + 6} y={g.pc.y + 12} size={10} role="soft">pole</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={showCircle ? 'sky' : 'soft'}>{note}</InkText>
      </svg>
    </div>
  );
}

const HI: DomeHi[][] = [['pole'], ['pole'], ['pole'], ['alm'], ['alm', 'horizon'], ['horizon', 'alm']];

export default function ProjectionPlate() {
  const [dec, setDec] = useState(30);
  const [h, setH] = useState(30);
  const [step, setStep] = useState(0);
  const mode: Mode = step >= 3 ? 'alm' : 'star';
  const r = tand((90 - dec) / 2);
  const a = almucantar(step >= 4 ? 0 : h, PHI);
  const hh = step >= 4 ? 0 : h;
  const steps: StepDef[] = [
    {
      id: 'pick',
      title: 'Pick a star',
      text: <>Take a star on the meridian, at declination δ from the celestial equator. Every star turns in a circle round the pole.</>,
      controls: <Slider label="the star's declination δ" value={dec} min={DEC_MIN} max={DEC_MAX} step={1} onChange={setDec} format={(v) => `${v}°`} />,
    },
    {
      id: 'ray',
      title: 'Look from the south pole',
      text: <>Put your eye at the south celestial pole and draw a straight line through the star until it meets the plane of the equator. That plane is the plate.</>,
      controls: <Slider label="the star's declination δ" value={dec} min={DEC_MIN} max={DEC_MAX} step={1} onChange={setDec} format={(v) => `${v}°`} />,
    },
    {
      id: 'mark',
      title: 'Mark where it lands',
      text: <>It lands at r = tan((90° − δ)/2) from the pole. As the star turns, the point turns round the pin: a circle. Northern stars land near the middle, the tropic of Capricorn is the edge.</>,
      controls: <Slider label="the star's declination δ" value={dec} min={DEC_MIN} max={DEC_MAX} step={1} onChange={setDec} format={(v) => `${v}°`} />,
    },
    {
      id: 'alm',
      title: 'Project a height circle',
      text: <>Now a circle of equal altitude over Istanbul (an almucantar). It is tilted, but its image is still a circle: project its two meridian points and the circle runs through both.</>,
      controls: <Slider label="altitude of the circle h" value={h} min={ALT_MIN + 10} max={ALT_MAX} step={10} onChange={setH} format={(v) => `${v}°`} />,
    },
    {
      id: 'horizon',
      title: 'Project the horizon',
      text: <>The horizon is the almucantar of altitude 0. Its circle is off-centre and bigger than the equator, and it runs off the plate at the top: what lies outside never rises.</>,
    },
    {
      id: 'limit',
      title: 'Where it fails',
      text: (
        <>
          The star circles are centred on the pin for every place on Earth. The horizon and height circles are not: they depend on your latitude.
          One drawing cannot hold both, unless the two can move against each other.
        </>
      ),
    },
  ];
  return (
    <StepPlate
      n={2}
      title="Flattening the sky onto a plate"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Construction mode={i >= 3 ? 'alm' : 'star'} dec={dec} h={i >= 4 ? 0 : h} step={i} />}
      viewLabel="The construction"
      reference={(i) => <Dome opts={{ dec, sunH: null, alm: i >= 3 ? (i >= 4 ? 0.0001 : h) : null, starDec: i < 3 ? dec : null, starH: 0 }} hi={i < 3 ? ['star', 'pole'] : HI[i]} />}
      referenceLabel="The real sky over Istanbul (east on the left)"
      equation={
        <Working label="Stereographic projection">
          <Eq>
            <Sym role="red">r</Sym> = tan((90° − <Sym role="sky">δ</Sym>) / 2)
          </Eq>
          <Eq>
            <Known when={step >= 2 && mode === 'star'} role="red" ph={step >= 3 ? '·' : '?'}>{num3(r)}</Known> = tan((90° − <Known when={mode === 'star'} role="sky" ph="·">{`${dec}°`}</Known>) / 2)
          </Eq>
          <Eq>
            height circle h: centre <Sym role="sky">c</Sym> = cos φ / (sin φ + sin h), radius cos h / (sin φ + sin h)
          </Eq>
          <Eq>
            h <Known when={mode === 'alm'} role="sky" ph="·">{`${hh}°`}</Known>, φ 41°: c = <Known when={mode === 'alm'} role="sky" ph="·">{num3(a.cv)}</Known>, radius{' '}
            <Known when={mode === 'alm'} role="sky" ph="·">{num3(a.r)}</Known>
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em', color: C.soft }}>
              {`Unit: the equator's radius. Cancer ${num3(RCAN)}, Capricorn (the edge) ${num3(RC)}.`}
            </span>
          </Eq>
        </Working>
      }
      caption={
        <>
          The side view cuts the sky along the meridian; the face-on view is the plate. A point crossing the plate at distance x in the side view lands
          at x above or below the pin in the face-on view. The projection keeps circles as circles and preserves angles, which is why every line on an
          astrolabe plate can be drawn with compasses (Pingree, Encyclopaedia Iranica; formulas checked numerically).
        </>
      }
    />
  );
}
