/**
 * Plate 1, procedure: Eratosthenes measures the Earth with a stick.
 *   1 Wait for noon · 2 Mark the shadow's tip · 3 Work out the angle · 4 Scale it to the Earth ·
 *   5 Where it fails (the blurred tip).
 * Reference: the post's Earth section in two-city mode (Alexandria, Syene, parallel sunlight,
 * θ at the centre equal to θ at the stick). Geometry: ./gnomonGeom.ts (swept by gnomonGeom.sweep.ts).
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
  V,
  Working,
  arcPath,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import LatitudeReference, { type RefKey } from './LatitudeReference';
import {
  ALEX,
  DEC,
  DIST,
  STADIA,
  SYENE,
  S_MAX,
  S_MIN,
  S_STEP,
  THETA_HI,
  THETA_LO,
  TIP,
  T_MAX,
  T_MIN,
  T_START,
  circumference,
  groundSay,
  groundScene,
  hhmm,
  onEdge,
  rulerSay,
  rulerScene,
  snapTime,
  thetaOf,
  triangleScene,
  type StadionId,
} from './gnomonGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const deg2 = (v: number) => `${v.toFixed(2)}°`;
const km = (v: number) => `${Math.round(v).toLocaleString('en-US')} km`;
const stad = (v: number) => `${Math.round(v / 100) * 100}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',');

function GroundView({ t }: { t: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = groundScene(t, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The ground at Alexandria seen from above: ${groundSay(t)}`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">the ground, seen from above</InkText>
        <InkText x={W - 14} y={18} anchor="end" size={11} role="soft">north ↑</InkText>
        {/* noon line (due north of the stick) with 10 cm ticks */}
        <line x1={g.foot.x} y1={g.foot.y} x2={g.foot.x} y2={g.top} style={{ ...solid(g.noon ? 'sky' : 'faint', g.noon ? 1.2 : 0.8), strokeDasharray: g.noon ? undefined : '3 4' }} />
        {g.ticks.map((tk) => (
          <line key={tk.m} x1={g.foot.x - 3} y1={tk.y} x2={g.foot.x + 3} y2={tk.y} style={solid('faint', 0.8)} />
        ))}
        {/* the tip's track through the day */}
        <polyline points={g.track.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={{ ...solid('brass', 0.9), strokeDasharray: '4 3' }} />
        {/* the shadow: full shadow, then the blur */}
        <line x1={g.foot.x} y1={g.foot.y} x2={g.tipIn.x} y2={g.tipIn.y} style={{ ...solid('ink', 4), strokeLinecap: 'butt' }} />
        <line x1={g.tipIn.x} y1={g.tipIn.y} x2={g.tipOut.x} y2={g.tipOut.y} style={{ ...solid('soft', 4), opacity: 0.5, strokeLinecap: 'butt' }} />
        <circle cx={g.foot.x} cy={g.foot.y} r={5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.4 }} />
        <InkText x={g.foot.x + 10} y={g.foot.y + 16} size={11} role="ink">the stick</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={g.noon ? 'sky' : 'soft'}>{groundSay(t)}</InkText>
      </svg>
    </div>
  );
}

function RulerView({ s }: { s: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = rulerScene(s, W);
  const edge = onEdge(s);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`A ruler at the tip of the noon shadow: ${rulerSay(s)}`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">the tip, close up (cm from the stick)</InkText>
        {/* full shadow, blur (penumbra), sunlight */}
        <rect x={g.x0} y={g.shadowY0} width={g.xIn - g.x0} height={g.shadowY1 - g.shadowY0} style={{ fill: C.ink, opacity: 0.55 }} />
        {Array.from({ length: 16 }, (_, i) => {
          const x = g.xIn + ((g.xOut - g.xIn) * i) / 16;
          return <rect key={i} x={x} y={g.shadowY0} width={(g.xOut - g.xIn) / 16 + 0.3} height={g.shadowY1 - g.shadowY0} style={{ fill: C.ink, opacity: 0.55 * (1 - (i + 0.5) / 16) }} />;
        })}
        <InkText x={(g.x0 + g.xIn) / 2} y={(g.shadowY0 + g.shadowY1) / 2} anchor="middle" baseline="central" size={12} role="paper" halo={false}>full shadow</InkText>
        <InkText x={(g.xOut + g.x1) / 2} y={(g.shadowY0 + g.shadowY1) / 2} anchor="middle" baseline="central" size={12} role="soft">sunlight</InkText>
        <line x1={g.xIn} y1={g.shadowY1 + 4} x2={g.xOut} y2={g.shadowY1 + 4} style={solid('brass', 1.2)} />
        <InkText x={(g.xIn + g.xOut) / 2} y={g.shadowY0 - 8} anchor="middle" size={11} role="brass">blur</InkText>
        {/* ruler */}
        <line x1={g.x0} y1={g.rulerY} x2={g.x1} y2={g.rulerY} style={solid('ink', 1.2)} />
        {g.ticks.map((tk) => (
          <g key={tk.cm}>
            <line x1={tk.x} y1={g.rulerY} x2={tk.x} y2={g.rulerY + (tk.major ? 14 : tk.mid ? 9 : 5)} style={solid('ink', tk.major ? 1 : 0.6)} />
            {tk.major && <InkText x={tk.x} y={g.rulerY + 28} kind="num" anchor="middle" size={11} role="ink">{`${tk.cm}`}</InkText>}
          </g>
        ))}
        {/* the mark */}
        <line x1={g.xMark} y1={g.shadowY0 - 2} x2={g.xMark} y2={g.rulerY + 16} style={solid(edge ? 'sky' : 'red', 1.6)} />
        <path d={`M ${g.xMark} ${g.rulerY - 2} L ${g.xMark - 6} ${g.rulerY - 13} L ${g.xMark + 6} ${g.rulerY - 13} Z`} style={{ fill: edge ? C.sky : C.red }} />
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={edge ? 'sky' : 'soft'}>{rulerSay(s)}</InkText>
      </svg>
    </div>
  );
}

function TriangleView({ s, blur }: { s: number; blur?: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = triangleScene(s, W);
  const lo = triangleScene(Math.tan((THETA_LO * Math.PI) / 180) * 100, W);
  const hi = triangleScene(Math.tan((THETA_HI * Math.PI) / 180) * 100, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The stick and its shadow side on: angle ${g.theta.toFixed(2)} degrees`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">side on, to scale</InkText>
        <line x1={20} y1={g.base.y} x2={W - 20} y2={g.base.y} style={solid('ink', 1.2)} />
        {blur && (
          <>
            <path d={`M ${lo.top.x} ${lo.top.y} L ${lo.tip.x} ${lo.tip.y} L ${hi.tip.x} ${hi.tip.y} Z`} style={{ fill: C.brass, opacity: 0.3 }} />
            <line x1={lo.tip.x} y1={g.base.y + 8} x2={hi.tip.x} y2={g.base.y + 8} style={solid('brass', 1.4)} />
            <InkText x={hi.tip.x + 6} y={g.base.y + 12} size={11} role="brass">blur</InkText>
          </>
        )}
        <line x1={g.rayFrom.x} y1={g.rayFrom.y} x2={g.tip.x} y2={g.tip.y} style={solid('brass', 1.1)} />
        <line x1={g.base.x} y1={g.base.y} x2={g.top.x} y2={g.top.y} style={solid('ink', 3)} />
        <line x1={g.base.x} y1={g.base.y} x2={g.tip.x} y2={g.tip.y} style={{ ...solid('ink', 4), opacity: 0.6 }} />
        <path d={arcPath(g.top.x, g.top.y, g.arcR, -90, -90 + g.theta)} style={solid('red', 1.5)} />
        <InkText x={g.top.x + 8} y={g.top.y + g.arcR + 16} size={15} role="red">θ</InkText>
        <InkText x={g.base.x - 8} y={(g.base.y + g.top.y) / 2} anchor="end" size={13} role="ink">g = 100 cm</InkText>
        <InkText x={g.tip.x + 8} y={g.base.y - 8} size={13} role="sky">{`s = ${s.toFixed(2)} cm`}</InkText>
        <InkText x={g.sunLabel.x} y={g.sunLabel.y} size={11} role="soft">sunlight</InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['sun'], ['sun'], ['h'], ['theta'], ['theta', 'sun']];

export default function GnomonProcedure() {
  const [t, setT] = useState(T_START);
  const [s, setS] = useState(S_MIN + 0.5);
  const [st, setSt] = useState<StadionId>('olympic');
  const [step, setStep] = useState(0);
  const noon = t === 12;
  const edge = onEdge(s);
  const theta = thetaOf(s);
  const stadion = STADIA.find((x) => x.id === st)!;
  const c = circumference(theta, stadion.m);
  const cLo = circumference(THETA_HI, stadion.m);
  const cHi = circumference(THETA_LO, stadion.m);
  const kn = (i: number, cond: boolean) => step >= i && cond;
  const sKnown = kn(2, edge && noon);
  const cKnown = kn(3, edge && noon);

  const steps: StepDef[] = [
    {
      id: 'noon',
      title: 'Wait for noon',
      text: (
        <>
          Midsummer in Alexandria. Stand a stick upright (check it with a plumb line) and watch its shadow. The tip
          swings round and shortens until the sun crosses the meridian, then grows again. Noon is the shortest shadow,
          pointing due north.
        </>
      ),
      controls: <Slider label="local sun time" value={t} min={T_MIN} max={T_MAX} step={1 / 60} onChange={(v) => setT(snapTime(v))} format={hhmm} />,
      gated: true,
      done: noon,
      hint: 'Drag the time to noon (12:00)',
    },
    {
      id: 'mark',
      title: 'Mark the tip',
      text: (
        <>
          Lay a ruler along the noon line and mark where the shadow ends. Look closely: the sun is a disc, not a point,
          so the shadow fades out over a few millimetres instead of stopping.
        </>
      ),
      controls: <Slider label="mark on the ruler" value={s} min={S_MIN} max={S_MAX} step={S_STEP} onChange={setS} format={(v) => `${v.toFixed(2)} cm`} />,
      gated: true,
      done: edge,
      hint: 'Put the mark on the edge of the shadow',
    },
    {
      id: 'angle',
      title: 'Work out the angle',
      text: (
        <>
          The stick, its shadow and the sunlight make a right triangle. The angle at the top, between the stick and the
          sunlight, is θ: how far the noon sun stands from straight overhead.
        </>
      ),
    },
    {
      id: 'scale',
      title: 'Scale it to the Earth',
      text: (
        <>
          At Syene, {DIST.toLocaleString('en-US')} stadia to the south, the same noon sun shone straight down a well: no
          shadow. Sunlight arrives parallel, so θ is also the angle between the two places seen from the Earth’s centre.
          A full circle holds 360°/θ of those slices.
        </>
      ),
      controls: (
        <Segmented
          label="length of one stadion"
          options={STADIA.map((x) => ({ value: x.id, label: `${x.name} ${x.m} m` }))}
          value={st}
          onChange={setSt}
        />
      ),
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Anywhere in the blur is a fair mark, and that alone moves θ from {deg2(THETA_LO)} to {deg2(THETA_HI)}. The
          method also needs level ground and a stick that stands still. A ship has neither.
        </>
      ),
    },
  ];

  const view = (i: number) => {
    if (i === 0) return <GroundView t={t} />;
    if (i === 1) return <RulerView s={s} />;
    return <TriangleView s={s} blur={i === 4} />;
  };

  return (
    <StepPlate
      n={1}
      wide
      title="Eratosthenes measures the Earth with a stick"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: ALEX, src: { kind: 'sun', dec: DEC }, second: { phi: SYENE, name: 'Syene' }, you: 'Alexandria' }}
          hi={HI[i]}
          label={`The Earth cut along the Nile: Alexandria at ${ALEX} degrees north, Syene at ${SYENE}, the sun straight over Syene. The angle between them at the centre equals the shadow angle at Alexandria.`}
        />
      )}
      referenceLabel="What you are measuring (the Nile's meridian)"
      onSkip={() => {
        setT(12);
        setS(Math.round((TIP * 100) / S_STEP) * S_STEP);
      }}
      equation={
        <Working label="The Earth's circumference from a shadow">
          <Eq>
            tan <Sym role="red">θ</Sym> = <Sym role="sky">s</Sym> / <Sym>g</Sym>
          </Eq>
          <Eq>
            tan <Known when={sKnown} role="red">{deg2(theta)}</Known> = <Known when={sKnown} role="sky">{`${s.toFixed(2)} cm`}</Known> / <V>100 cm</V>
          </Eq>
          <Eq>
            <Sym role="brass">C</Sym> = (360° / <Sym role="red">θ</Sym>) · <Sym>d</Sym>
          </Eq>
          <Eq>
            <Known when={cKnown} role="brass">{`${stad(c.stadia)} stadia`}</Known> = (360° / <Known when={cKnown} role="red">{deg2(theta)}</Known>) · <V>5,000 stadia</V>
          </Eq>
          <Eq>
            <Known when={cKnown} role="brass">{km(c.km)}</Known> at <V>{`${stadion.m} m`}</V> a stadion{cKnown ? ` (${c.err >= 0 ? '+' : '−'}${Math.abs(c.err).toFixed(1)}%)` : ''}
          </Eq>
          {step >= 4 && cKnown && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>{`The blur alone: ${km(cLo.km)} to ${km(cHi.km)}.`}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          Drawn for midsummer noon with the modern latitudes of Alexandria ({ALEX}° N) and Aswan ({SYENE}° N), a 1 m
          stick and the sun’s disc 0.53° across. Cleomedes gives θ = 1/50 of a circle, 7.2°. Nobody knows which stadion
          Eratosthenes used, so the plate offers three; the percentage compares with the modern 40,008 km.
        </>
      }
    />
  );
}
