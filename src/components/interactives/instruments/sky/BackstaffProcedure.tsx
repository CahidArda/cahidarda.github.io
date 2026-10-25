/**
 * Plate 4, procedure: the noon sun's height with a backstaff (Davis quadrant).
 *   1 Preset the shadow vane · 2 Turn your back to the sun · 3 Line up shadow and horizon ·
 *   4 Add the two arcs · 5 Where it fails (no shadow, no stars).
 * The shadow is lined up by the middle of its blurred edge (the sun's centre), so no
 * semi-diameter correction is needed. Reference: the post's Earth section, sun mode, Istanbul,
 * 25 October 2026 (same sight as the cross-staff). Geometry: ./staffGeom.ts.
 */
import { useState } from 'react';
import {
  C,
  Eq,
  InkText,
  Known,
  Slider,
  StepPlate,
  Sym,
  Working,
  arcPath,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import LatitudeReference, { type RefKey } from './LatitudeReference';
import {
  A30_MAX,
  A30_MIN,
  A30_STEP,
  A60_MAX,
  A60_MIN,
  A60_STEP,
  BS_TOL,
  DEC,
  H_SUN,
  IST,
  bsPresetOk,
  bsSay,
  bsSide,
  bsVane,
  latFrom,
} from './staffGeom';
import { dm } from './noonProcGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function Side({ a60, a30, back }: { a60: number; a30: number; back: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = bsSide(a60, a30, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Side view of the backstaff: shadow vane at ${a60} degrees on the small arc, sight vane at ${a30.toFixed(1)} on the large arc`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">side view (sun behind you)</InkText>
        {/* the world: the horizon ahead (left), the sun behind (up and right) */}
        <line x1={g.horizonEnd.x} y1={g.O.y} x2={g.eye.x + 6} y2={g.O.y} style={{ ...solid('ink', 0.9), strokeDasharray: '4 3' }} />
        <InkText x={g.horizonEnd.x + 2} y={g.O.y - 8} size={11} role="soft">horizon</InkText>
        {back && (
          <>
            <line x1={g.sunIn.x} y1={g.sunIn.y} x2={g.shadowVane.x} y2={g.shadowVane.y} style={solid('brass', 1.1)} />
            <line x1={g.shadowVane.x} y1={g.shadowVane.y} x2={g.shadowHit.x} y2={g.shadowHit.y} style={{ ...solid('brass', 1.1), strokeDasharray: '2 3' }} />
            <InkText x={g.sunIn.x - 4} y={g.sunIn.y - 6} anchor="end" size={11} role="brass">sunlight</InkText>
          </>
        )}
        {/* arcs and staff */}
        <path d={arcPath(g.O.x, g.O.y, g.r30, 0, 30)} style={solid('ink', 1.6)} />
        <path d={arcPath(g.O.x, g.O.y, g.r60, g.a30, g.a30 + 60)} style={solid('ink', 1.6)} />
        <line x1={g.O.x} y1={g.O.y} x2={g.staffEnd.x} y2={g.staffEnd.y} style={solid('ink', 2.2)} />
        <line x1={g.O.x} y1={g.O.y} x2={g.sightVane.x} y2={g.sightVane.y} style={{ ...solid('sky', 1), strokeDasharray: '4 3' }} />
        {/* vanes */}
        <rect x={g.O.x - 2} y={g.O.y - 14} width={4} height={28} style={{ fill: C.ink }} />
        <rect x={g.sightVane.x - 3} y={g.sightVane.y - 9} width={6} height={18} style={{ fill: C.sky }} />
        <circle cx={g.shadowVane.x} cy={g.shadowVane.y} r={5} style={{ fill: C.brass }} />
        <InkText x={g.shadowVane.x + 8} y={g.shadowVane.y - 6} size={11} role="brass">{`shadow vane ${a60}°`}</InkText>
        <InkText x={g.sightVane.x - 4} y={g.sightVane.y + 24} anchor="end" size={11} role="sky">{`sight vane ${a30.toFixed(1)}°`}</InkText>
        <InkText x={g.O.x - 6} y={g.O.y + 26} anchor="middle" size={11} role="ink">horizon vane</InkText>
        <circle cx={g.eye.x} cy={g.eye.y} r={4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
        <InkText x={g.eye.x - 2} y={g.eye.y - 10} size={11} role="ink">eye</InkText>
      </svg>
    </div>
  );
}

function Vane({ a60, a30 }: { a60: number; a30: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = bsVane(a60, a30, W);
  const { vane } = g;
  const clip = `bsv-${Math.round(W)}`;
  const n = 14;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Through the sight vane: ${bsSay(g)}`}>
        <defs>
          <clipPath id={clip}>
            <rect x={vane.x0} y={vane.y0} width={vane.x1 - vane.x0} height={vane.y1 - vane.y0} />
          </clipPath>
        </defs>
        <rect x={vane.x0} y={vane.y0} width={vane.x1 - vane.x0} height={vane.y1 - vane.y0} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
        <g clipPath={`url(#${clip})`}>
          {/* the vane's shadow: dark from the top down to the blurred edge */}
          <rect x={vane.x0} y={vane.y0 - 400} width={vane.x1 - vane.x0} height={g.edgeY - g.blur - vane.y0 + 400} style={{ fill: C.ink, opacity: 0.55 }} />
          {Array.from({ length: n }, (_, i) => {
            const y = g.edgeY - g.blur + ((2 * g.blur) / n) * i;
            return <rect key={i} x={vane.x0} y={y} width={vane.x1 - vane.x0} height={(2 * g.blur) / n + 0.3} style={{ fill: C.ink, opacity: 0.55 * (1 - (i + 0.5) / n) }} />;
          })}
        </g>
        {/* the slit with the horizon in it */}
        <rect x={vane.x0} y={g.slitY - 2} width={vane.x1 - vane.x0} height={4} style={{ fill: C.sky, opacity: 0.5 }} />
        <line x1={vane.x0} y1={g.slitY} x2={vane.x1} y2={g.slitY} style={solid('sky', 1.4)} />
        <InkText x={vane.x1 + 6} y={g.slitY} baseline="central" size={11} role="sky">slit</InkText>
        {Math.abs(g.offMm) <= 6.5 && (
          <line x1={vane.x0 - 8} y1={g.edgeY} x2={vane.x0} y2={g.edgeY} style={solid(g.on ? 'sky' : 'red', 1.6)} />
        )}
        <InkText x={W / 2} y={18} anchor="middle" kind="label" size={12} role="soft">the horizon vane, close up</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={g.on ? 'sky' : 'soft'}>{bsSay(g)}</InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['h'], ['sun'], ['h', 'horizon'], ['h'], ['sun']];

export default function BackstaffProcedure() {
  const [a60, setA60] = useState(20);
  const [a30, setA30] = useState(5);
  const [step, setStep] = useState(0);
  const preset = bsPresetOk(a60);
  const h = a60 + a30;
  const on = Math.abs(h - H_SUN) <= BS_TOL;
  const phi = latFrom(h);
  const kn = (i: number) => step >= i && on;

  const steps: StepDef[] = [
    {
      id: 'preset',
      title: 'Preset the shadow vane',
      text: (
        <>
          John Davis’s backstaff has two arcs: a small one of 60° and a large one of 30°. Set the shadow vane on the
          small arc to a round number a little below the sun’s expected height; the large arc will read the rest finely.
        </>
      ),
      controls: <Slider label="shadow vane on the 60° arc" value={a60} min={A60_MIN} max={A60_MAX} step={A60_STEP} onChange={setA60} format={(v) => `${v}°`} />,
      gated: true,
      done: preset,
      hint: `The rest must fit on the 30° arc`,
    },
    {
      id: 'back',
      title: 'Turn your back',
      text: <>Face north, away from the noon sun. The shadow vane’s shadow now falls on the horizon vane in front of you.</>,
    },
    {
      id: 'align',
      title: 'Line them up',
      text: (
        <>
          Look through the sight vane and the slit at the horizon, and slide the sight vane until the shadow’s edge sits
          on the slit. The edge is soft because the sun is a disc: put its middle on the slit.
        </>
      ),
      controls: <Slider label="sight vane on the 30° arc" value={a30} min={A30_MIN} max={A30_MAX} step={A30_STEP} onChange={setA30} format={(v) => dm(v)} />,
      gated: true,
      done: on,
      hint: 'Put the shadow edge on the slit',
    },
    {
      id: 'add',
      title: 'Add the arcs',
      text: <>The sun’s height is the sum of the two arcs. Then the noon rule gives the latitude.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          No sun, no shadow: haze or a thin cloud blurs it away. At night it is useless, since a star casts no shadow.
          And it is still held by hand on a moving deck.
        </>
      ),
    },
  ];

  const view = (i: number) => (i === 2 ? <Vane a60={a60} a30={a30} /> : <Side a60={a60} a30={a30} back={i >= 1} />);

  return (
    <StepPlate
      n={4}
      wide
      title="The noon sun's height with a backstaff"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: IST.lat, src: { kind: 'sun', dec: DEC }, sight: i >= 2 && i <= 3 ? h : null }}
          hi={HI[i]}
          label={`The Earth cut along Istanbul's meridian on 25 October 2026: the sun's noon height ${H_SUN.toFixed(1)} degrees.`}
        />
      )}
      referenceLabel="What you are measuring (your meridian)"
      onSkip={() => {
        setA60(30);
        setA30(Math.round((H_SUN - 30) / A30_STEP) * A30_STEP);
      }}
      equation={
        <Working label="Latitude from a backstaff sight">
          <Eq>
            <Sym role="sky">h</Sym> = <Sym role="brass">a</Sym>
            <sub>60</sub> + <Sym role="sky">a</Sym>
            <sub>30</sub>
          </Eq>
          <Eq>
            <Known when={kn(3)} role="sky">{dm(h)}</Known> = <Known when={step >= 1 && preset} role="brass">{`${a60}°`}</Known> +{' '}
            <Known when={kn(2)} role="sky">{dm(a30)}</Known>
          </Eq>
          <Eq>
            <Sym role="red">φ</Sym> = 90° − <Sym role="sky">h</Sym> + <Sym role="brass">δ</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3)} role="red">{dm(phi)}</Known> = 90° − <Known when={kn(3)} role="sky">{dm(h)}</Known> + (<Known when={kn(3)} role="brass">{dm(DEC)}</Known>)
          </Eq>
        </Working>
      }
      caption={
        <>
          Arcs of 30 cm and 75 cm radius; the large arc reads to 2′ (real ones had transversals, down to half an
          arcminute on a Greenwich example). The shadow counts as lined up within 3′. Same sun, date and simplifications
          as Plate 3.
        </>
      }
    />
  );
}
