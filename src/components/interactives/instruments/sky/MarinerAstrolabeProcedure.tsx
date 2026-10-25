/**
 * Plate 5, procedure: the noon sun's height with a mariner's astrolabe.
 *   1 Hang it still · 2 Let the sun through both pinholes · 3 Read the ring · 4 Now at sea:
 *   the ring swings, take several readings and average · 5 Where it fails.
 * World fixed: the sun and the true horizontal; the ring tips by ψ when it swings.
 * Reference: the post's Earth section, sun mode (same sight as Plates 3 and 4).
 * Geometry: ./marinerGeom.ts (swept by marinerGeom.sweep.ts).
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
  V,
  Working,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import LatitudeReference, { type RefKey } from './LatitudeReference';
import {
  ALI_TOL,
  A_MAX,
  A_MIN,
  A_STEP,
  DEC,
  H_SUN,
  IST,
  N_READ,
  SWING_MAX,
  SWING_STEP,
  latFrom,
  plateScene,
  readings,
  ringSay,
  ringScene,
  scatterScene,
  swingAt,
} from './marinerGeom';
import { dm } from './noonProcGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const am = (deg: number) => `${deg < 0 ? '−' : '+'}${Math.abs(deg * 60).toFixed(0)}′`;

function Ring({ a, psi }: { a: number; psi: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = ringScene(a, psi, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The astrolabe hanging${psi ? `, swung ${psi.toFixed(1)} degrees` : ''}: the alidade at ${a.toFixed(2)} degrees. ${ringSay(a, psi)}`}>
        {g.rays.map((r, i) => (
          <line key={i} x1={r.a.x} y1={r.a.y} x2={r.b.x} y2={r.b.y} style={{ ...solid('brass', 0.8), opacity: 0.6 }} />
        ))}
        <line x1={g.level.a.x} y1={g.level.a.y} x2={g.level.b.x} y2={g.level.b.y} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
        <InkText x={g.level.a.x} y={g.level.a.y - 6} size={10} role="soft">level</InkText>
        <line x1={g.shackle.x} y1={g.shackle.y - 14} x2={g.shackle.x} y2={g.shackle.y} style={solid('ink', 1.2)} />
        <circle cx={g.shackle.x} cy={g.shackle.y + 5} r={6} style={solid('ink', 1.6)} />
        {/* the ring, its four spokes and the weight at the bottom, all tipped by ψ */}
        <circle cx={g.cx} cy={g.cy} r={g.R} style={{ ...solid('ink', 7), opacity: 0.85 }} />
        {[0, 90].map((d) => {
          const s = ((d + psi) * Math.PI) / 180;
          return <line key={d} x1={g.cx - g.R * Math.cos(s)} y1={g.cy + g.R * Math.sin(s)} x2={g.cx + g.R * Math.cos(s)} y2={g.cy - g.R * Math.sin(s)} style={solid('ink', 2)} />;
        })}
        {g.ticks.map((t) => (
          <line key={t.d} x1={t.p.x} y1={t.p.y} x2={t.q.x} y2={t.q.y} style={solid('paper', t.d % 10 === 0 ? 1.3 : 0.8)} />
        ))}
        {[0, 30, 60, 90].map((d) => {
          const s = ((d + psi) * Math.PI) / 180;
          return (
            <InkText key={d} x={g.cx + (g.R + 14) * Math.cos(s)} y={g.cy - (g.R + 14) * Math.sin(s)} kind="num" anchor="middle" baseline="central" size={10} role="soft">{`${d}`}</InkText>
          );
        })}
        {/* the alidade */}
        <line x1={g.ali.a.x} y1={g.ali.a.y} x2={g.ali.b.x} y2={g.ali.b.y} style={solid(g.on ? 'sky' : 'red', 4)} />
        <circle cx={g.ali.b.x} cy={g.ali.b.y} r={4} style={{ fill: C.paper, stroke: C.ink }} />
        <circle cx={g.ali.a.x} cy={g.ali.a.y} r={4} style={{ fill: C.paper, stroke: C.ink }} />
        <circle cx={g.cx} cy={g.cy} r={4} style={{ fill: C.ink }} />
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={g.on ? 'sky' : 'soft'}>{ringSay(a, psi)}</InkText>
      </svg>
    </div>
  );
}

function PlateView({ a, psi }: { a: number; psi: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = plateScene(a, psi, W);
  const r = Math.min(W, g.H) * 0.34;
  const visible = Math.abs(g.off) <= 6;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The lower sight plate: ${ringSay(a, psi)}`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">the lower sight plate, close up</InkText>
        <rect x={g.cx - r} y={g.cy - r} width={2 * r} height={2 * r} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
        {visible && <circle cx={g.spot.x} cy={g.spot.y} r={Math.max(g.pinR, 4)} style={{ fill: C.brass, opacity: 0.9 }} />}
        <circle cx={g.cx} cy={g.cy} r={Math.max(g.pinR, 3)} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.2 }} />
        {[-5, 5].map((mm) => (
          <g key={mm}>
            <line x1={g.cx + r + 4} y1={g.cy - mm * g.k} x2={g.cx + r + 10} y2={g.cy - mm * g.k} style={solid('soft', 0.8)} />
            <InkText x={g.cx + r + 13} y={g.cy - mm * g.k} kind="num" baseline="central" size={10} role="soft">{`${mm > 0 ? '+' : '−'}5 mm`}</InkText>
          </g>
        ))}
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={Math.abs(a + psi - H_SUN) <= ALI_TOL ? 'sky' : 'soft'}>{ringSay(a, psi)}</InkText>
      </svg>
    </div>
  );
}

function Scatter({ amp, n }: { amp: number; n: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = scatterScene(amp, n, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`${n} readings at sea, swing ±${amp.toFixed(1)} degrees: average ${dm(g.mean)}`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">your readings, arcminutes from the true height</InkText>
        <line x1={g.x0} y1={g.y} x2={g.x1} y2={g.y} style={solid('ink', 1)} />
        {g.ticks.map((t) => (
          <g key={t.m}>
            <line x1={t.x} y1={g.y} x2={t.x} y2={g.y + 6} style={solid('ink', 0.8)} />
            <InkText x={t.x} y={g.y + 20} kind="num" anchor="middle" size={10} role="soft">{t.m === 0 ? '0' : `${t.m > 0 ? '+' : '−'}${Math.abs(t.m)}`}</InkText>
          </g>
        ))}
        <line x1={g.xTrue} y1={g.y - 64} x2={g.xTrue} y2={g.y} style={{ ...solid('ink', 1), strokeDasharray: '3 3' }} />
        <InkText x={g.xTrue + 4} y={g.y - 68} size={11} role="ink">true</InkText>
        {g.dots.map((d, i) => (
          <circle key={i} cx={d.x} cy={d.y} r={4} style={{ fill: C.sky, opacity: 0.85 }} />
        ))}
        {n > 0 && (
          <>
            <line x1={g.xMean} y1={g.y - 4} x2={g.xMean} y2={g.y + 34} style={solid('red', 1.8)} />
            <InkText x={g.xMean} y={g.y + 48} anchor="middle" size={12} role="red">{`average ${am(g.mean - H_SUN)}`}</InkText>
          </>
        )}
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">{`each dot: one reading, off by the swing (±${(amp * 60).toFixed(0)}′)`}</InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['sun'], ['h'], ['h'], ['h'], ['phi']];

export default function MarinerAstrolabeProcedure() {
  const [a, setA] = useState(20);
  const [amp, setAmp] = useState(1.5);
  const [n, setN] = useState(1);
  const [step, setStep] = useState(0);
  const still = Math.abs(a - H_SUN) <= ALI_TOL;
  const sea = readings(amp, n);
  const h = step >= 3 ? sea.mean : a;
  const phi = latFrom(h);
  const kn = (i: number) => step >= i && still;
  const psiNow = step >= 3 ? swingAt(amp, Math.max(0, n - 1)) : 0;

  const steps: StepDef[] = [
    {
      id: 'hang',
      title: 'Hang it still',
      text: (
        <>
          A heavy brass ring, cut away so the wind has less to push. Hang it from its shackle, on land or low in the
          ship near the mainmast, where the motion is smallest. Hanging still, gravity makes its 0° line level: no
          horizon needed.
        </>
      ),
    },
    {
      id: 'sight',
      title: 'Sunlight through both holes',
      text: (
        <>
          Never look at the sun. Turn the alidade until the sunlight through the upper pinhole falls on the pinhole in
          the lower plate.
        </>
      ),
      controls: <Slider label="turn the alidade" value={a} min={A_MIN} max={A_MAX} step={A_STEP} onChange={setA} format={(v) => dm(v)} />,
      gated: true,
      done: still,
      hint: 'Put the spot of light on the pinhole',
    },
    {
      id: 'read',
      title: 'Read the ring',
      text: <>The alidade points along the sunlight, so its reading on the ring is the sun’s height, h.</>,
    },
    {
      id: 'sea',
      title: 'Now at sea',
      text: (
        <>
          On a rolling deck the ring swings, and a ring tipped by a degree reads a degree wrong. You take reading after
          reading as it swings and average them.
        </>
      ),
      controls: (
        <>
          <Slider label="swing of the ring" value={amp} min={0} max={SWING_MAX} step={SWING_STEP} onChange={setAmp} format={(v) => `±${v.toFixed(1)}°`} />
          <Slider label="readings taken" value={n} min={1} max={N_READ} step={1} onChange={setN} format={(v) => `${v}`} />
        </>
      ),
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Here is the ring caught mid-swing at your last reading. A good observer on land got within about 10′; at sea,
          much worse. A 178 mm ring has little room for fine divisions, and gravity is a poor reference on a ship that
          never stops moving.
        </>
      ),
    },
  ];

  const view = (i: number) => {
    if (i === 0) return <Ring a={a} psi={0} />;
    if (i === 1) return <PlateView a={a} psi={0} />;
    if (i === 2) return <Ring a={a} psi={0} />;
    if (i === 3) return <Scatter amp={amp} n={n} />;
    return <Ring a={readingsLast(amp, n)} psi={psiNow} />;
  };

  return (
    <StepPlate
      n={5}
      wide
      title="The noon sun's height with a mariner's astrolabe"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: IST.lat, src: { kind: 'sun', dec: DEC }, sight: i >= 2 && (i > 2 || still) ? h : null }}
          hi={HI[i]}
          label={`The Earth cut along Istanbul's meridian on 25 October 2026: the sun's noon height ${H_SUN.toFixed(1)} degrees.`}
        />
      )}
      referenceLabel="What you are measuring (your meridian)"
      onSkip={() => setA(Math.round(H_SUN / A_STEP) * A_STEP)}
      equation={
        <Working label="Latitude from an astrolabe sight">
          <Eq>
            <Sym role="sky">h</Sym> = {step >= 3 ? <>average of the readings</> : <>the alidade’s reading</>}
          </Eq>
          <Eq>
            <Known when={kn(2)} role="sky">{dm(h)}</Known>
            {step >= 3 && kn(2) ? <> from <V>{`${n}`}</V> reading{n > 1 ? 's' : ''}, swing ±<V>{`${amp.toFixed(1)}°`}</V></> : null}
          </Eq>
          <Eq>
            <Sym role="red">φ</Sym> = 90° − <Sym role="sky">h</Sym> + <Sym role="brass">δ</Sym>
          </Eq>
          <Eq>
            <Known when={kn(2)} role="red">{dm(phi)}</Known> = 90° − <Known when={kn(2)} role="sky">{dm(h)}</Known> + (<Known when={kn(2)} role="brass">{dm(DEC)}</Known>)
          </Eq>
          {kn(2) && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>{`Off by ${(Math.abs(phi - IST.lat) * 60).toFixed(0)} nautical miles.`}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The ring is drawn the size of the Valentia astrolabe at Greenwich (178 mm). The spot counts as on the pinhole
          within 0.1°. At sea each reading is taken at a moment in the swing (fixed random moments, so the plate always
          shows the same ones); the reference shows the averaged height against the true one.
        </>
      }
    />
  );
}

/** The alidade setting a careful observer leaves at the latest reading (spot on the pinhole then). */
function readingsLast(amp: number, n: number) {
  return readings(amp, n).r[Math.max(0, n - 1)];
}
