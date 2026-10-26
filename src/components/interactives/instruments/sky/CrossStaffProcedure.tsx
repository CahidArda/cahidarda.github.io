/**
 * Plate 3, procedure: the noon sun's height with a cross-staff.
 *   1 Pick a transom · 2 Frame the sun and the horizon · 3 Read the staff · 4 Work out φ ·
 *   5 Where it fails (you stare at the sun, two ways at once).
 * Reference: the post's Earth section, sun mode, Istanbul on 25 October 2026.
 * Geometry: ./staffGeom.ts (swept by staffGeom.sweep.ts).
 */
import { useState } from 'react';
import {
  C,
  Eq,
  InkText,
  Known,
  Segmented,
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
  CS_TOL,
  DEC,
  H_SUN,
  IST,
  TRANSOMS,
  X_MAX,
  xMinFor,
  X_STEP,
  csPerMm,
  csSay,
  csSide,
  csSky,
  csTheta,
  csX,
  latFrom,
  transomL,
  type TransomId,
} from './staffGeom';
import { dm } from './noonProcGeom';
import { nmi } from './latRefGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function SkyView({ L, x }: { L: number; x: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = csSky(L, x, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Looking south along the staff: ${csSay(g)}`}>
        {g.ticks.map((t) => (
          <g key={t.a}>
            <line x1={14} y1={t.y} x2={W - 14} y2={t.y} style={{ ...solid('faint', 0.6), strokeDasharray: '2 5' }} />
            <InkText x={16} y={t.y - 4} kind="num" size={10} role="soft">{`${t.a}°`}</InkText>
          </g>
        ))}
        {/* glare: a halo round the sun */}
        <circle cx={g.sun.x} cy={g.sun.y} r={g.sun.r * 4} style={{ fill: C.brass, opacity: 0.18 }} />
        <circle cx={g.sun.x} cy={g.sun.y} r={g.sun.r} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1 }} />
        {/* the transom, seen end on: a bar from the horizon up to its upper end */}
        <line x1={g.cx} y1={g.horizonY} x2={g.upper.x} y2={g.upper.y} style={solid('ink', 3)} />
        <line x1={g.cx - 12} y1={g.upper.y} x2={g.cx + 12} y2={g.upper.y} style={solid(g.on ? 'sky' : 'red', 2)} />
        <line x1={g.cx - 12} y1={g.horizonY} x2={g.cx + 12} y2={g.horizonY} style={solid('sky', 2)} />
        {g.clipped && <InkText x={g.cx + 16} y={g.upper.y + 4} size={11} role="red">↑ off the top</InkText>}
        <rect x={0} y={g.horizonY} width={W} height={g.H - g.horizonY} style={{ fill: C.sky, opacity: 0.14 }} />
        <line x1={0} y1={g.horizonY} x2={W} y2={g.horizonY} style={solid('ink', 1.3)} />
        <InkText x={g.cx} y={g.horizonY + 16} kind="num" anchor="middle" size={11} role="ink">S</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={g.on ? 'sky' : 'soft'}>{csSay(g)}</InkText>
      </svg>
    </div>
  );
}

function SideView({ L, x, perMm }: { L: number; x: number; perMm?: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = csSide(L, x, W);
  const lo = Math.atan2(-(g.a.y - g.eye.y), g.a.x - g.eye.x) * (180 / Math.PI);
  const hi = Math.atan2(-(g.b.y - g.eye.y), g.b.x - g.eye.x) * (180 / Math.PI);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Side view: the transom ${x.toFixed(1)} cm along the staff frames ${g.theta.toFixed(2)} degrees`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">side view, to scale</InkText>
        <line x1={g.eye.x} y1={g.eye.y} x2={g.a.x} y2={g.a.y} style={{ ...solid('ink', 0.9), strokeDasharray: '4 3' }} />
        <line x1={g.eye.x} y1={g.eye.y} x2={g.b.x} y2={g.b.y} style={{ ...solid('brass', 0.9), strokeDasharray: '4 3' }} />
        <line x1={g.eye.x} y1={g.eye.y} x2={g.end.x} y2={g.end.y} style={solid('ink', 2.4)} />
        {g.ticks.map((t) => (
          <circle key={t.cm} cx={t.p.x} cy={t.p.y} r={1.6} style={{ fill: C.ink }} />
        ))}
        <line x1={g.a.x} y1={g.a.y} x2={g.b.x} y2={g.b.y} style={solid('ink', 3)} />
        <path d={arcPath(g.eye.x, g.eye.y, g.arcR, lo, hi)} style={solid('sky', 1.5)} />
        <circle cx={g.eye.x} cy={g.eye.y} r={4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
        {g.labels.map((l) => (
          <InkText key={l.id} x={l.x} y={l.y} anchor={l.anchor} baseline="central" kind={l.id.startsWith('t') ? 'num' : 'note'} size={l.size} role={l.id === 'h' || l.id === 'x' ? 'sky' : l.id === 'eye' ? 'ink' : 'soft'}>
            {l.text}
          </InkText>
        ))}
        {perMm !== undefined && (
          <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">{`set for ${H_SUN.toFixed(0)}°: 1 mm on the staff = ${(perMm * 60).toFixed(1)}′`}</InkText>
        )}
      </svg>
    </div>
  );
}

function GlareView({ L }: { L: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const H = Math.round(Math.max(290, Math.min(400, W * 0.86)));
  const cx = W / 2;
  const r = Math.min(W * 0.26, H * 0.26);
  const gy = 40 + r + 20;
  const perMm = csPerMm(L);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img" aria-label="Two places to look at once: the sun at the top of the transom and the horizon at the bottom">
        <InkText x={14} y={18} kind="label" size={12} role="soft">what your eye has to do</InkText>
        <circle cx={cx} cy={gy - r} r={14} style={{ fill: C.brass }} />
        <circle cx={cx} cy={gy - r} r={34} style={{ fill: C.brass, opacity: 0.2 }} />
        <InkText x={cx + 42} y={gy - r + 4} size={12} role="brass">look here (the sun)</InkText>
        <line x1={20} y1={gy + r} x2={W - 20} y2={gy + r} style={solid('ink', 1.3)} />
        <InkText x={cx + 14} y={gy + r - 8} size={12} role="ink">and here (the horizon)</InkText>
        <line x1={cx} y1={gy - r + 16} x2={cx} y2={gy + r} style={{ ...solid('red', 1.2), strokeDasharray: '4 4' }} />
        <InkText x={cx - 8} y={gy} anchor="end" size={12} role="red">{`${H_SUN.toFixed(0)}° apart`}</InkText>
        <InkText x={W / 2} y={H - 32} anchor="middle" size={12} role="soft">{`1 mm on the staff moves the reading ${(perMm * 60).toFixed(1)}′`}</InkText>
        <InkText x={W / 2} y={H - 10} anchor="middle" size={12} role="soft">a ship rolls the ends apart while you look</InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['sun'], ['h', 'horizon'], ['h'], ['phi', 'dec'], ['h']];

export default function CrossStaffProcedure() {
  const [tid, setTid] = useState<TransomId>('long');
  const [x, setX] = useState(70);
  const [step, setStep] = useState(0);
  const L = transomL(tid);
  const theta = csTheta(L, x);
  const on = Math.abs(theta - H_SUN) <= CS_TOL;
  const phi = latFrom(theta);
  const kn = (i: number) => step >= i && on;

  const steps: StepDef[] = [
    {
      id: 'transom',
      title: 'Pick a transom',
      text: (
        <>
          A square staff about a metre long, with crosspieces (transoms) of different lengths. You use one at a time.
          Today’s noon sun over Istanbul is about {H_SUN.toFixed(0)}° up: a longer transom has to sit further out to
          frame it, and the further out it sits, the finer the staff reads.
        </>
      ),
      controls: (
        <Segmented
          label="transom length"
          options={TRANSOMS.map((t) => ({ value: t.id, label: t.name }))}
          value={tid}
          onChange={(v) => {
            setTid(v);
            setX((xx) => Math.max(xx, xMinFor(transomL(v)))); // the new transom's nearest setting
          }}
        />
      ),
    },
    {
      id: 'frame',
      title: 'Frame the sun',
      text: (
        <>
          Rest the end of the staff on your cheekbone and aim it between the sun and the horizon. Slide the transom
          until its lower end sits on the horizon and its upper end on the sun.
        </>
      ),
      controls: <Slider label="transom along the staff" value={x} min={xMinFor(L)} max={X_MAX} step={X_STEP} onChange={setX} format={(v) => `${v.toFixed(1)} cm`} />,
      gated: true,
      done: on,
      hint: 'Put the upper end on the sun',
    },
    {
      id: 'read',
      title: 'Read the staff',
      text: <>The staff is marked in degrees; in centimetres the angle is twice a right triangle’s.</>,
    },
    {
      id: 'solve',
      title: 'Work out φ',
      text: <>The noon sun is due south, so latitude is 90° minus its height plus the sun’s declination from the tables.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          You look straight at the noon sun (a smoked glass helped), and at the horizon {H_SUN.toFixed(0)}° below it,
          at the same moment, on a deck that rolls.
        </>
      ),
    },
  ];

  const view = (i: number) => {
    if (i === 1) return <SkyView L={L} x={x} />;
    if (i === 4) return <GlareView L={L} />;
    if (i === 0) return <SideView L={L} x={csX(L, H_SUN)} perMm={csPerMm(L)} />;
    return <SideView L={L} x={x} />;
  };

  return (
    <StepPlate
      n={3}
      wide
      title="The noon sun's height with a cross-staff"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: IST.lat, src: { kind: 'sun', dec: DEC }, sight: i >= 1 && i <= 2 ? theta : null }}
          hi={HI[i]}
          label={`The Earth cut along Istanbul's meridian on 25 October 2026: the sun's declination ${DEC.toFixed(1)} degrees, its noon height ${H_SUN.toFixed(1)}, latitude ${IST.lat.toFixed(1)}.`}
        />
      )}
      referenceLabel="What you are measuring (your meridian)"
      onSkip={() => {
        setTid('long');
        setX(Math.round(csX(60, H_SUN) / X_STEP) * X_STEP);
      }}
      equation={
        <Working label="Latitude from a cross-staff sight">
          <Eq>
            <Sym role="sky">h</Sym> = 2 · atan(<Sym>L</Sym> / 2<Sym role="sky">x</Sym>)
          </Eq>
          <Eq>
            <Known when={kn(2)} role="sky">{dm(theta)}</Known> = 2 · atan(<Known when={step >= 1}>{`${L} cm`}</Known> / 2 · <Known when={kn(1)} role="sky">{`${x.toFixed(1)} cm`}</Known>)
          </Eq>
          <Eq>
            <Sym role="red">φ</Sym> = 90° − <Sym role="sky">h</Sym> + <Sym role="brass">δ</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3)} role="red">{dm(phi)}</Known> = 90° − <Known when={kn(2)} role="sky">{dm(theta)}</Known> + (<Known when={kn(3)} role="brass">{dm(DEC)}</Known>)
          </Eq>
          {kn(3) && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>{`Istanbul is at ${dm(IST.lat)}: off by ${nmi(phi - IST.lat)}.`}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The noon sun over Istanbul on 25 October 2026 (declination from the USNO formula). The ends count as on the
          sun and the horizon within 0.1°. Refraction, the sun’s radius and the height of your eye (a few arcminutes in
          all) are left out; the sextant plate adds them back.
        </>
      }
    />
  );
}
