/**
 * Plate 6 (episode 6): half an astrolabe on a board. The 41° plate is cut to a quarter, a thread
 * and bead replace the rete. 1 Halve it · 2 Quarter it · 3 Set the bead · 4 Swing the thread ·
 * 5 Read the arc. Geometry: quadGeom.ts + plateGeom.ts (swept).
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { DOY_DEFAULT, PHI, altitude, deg1, hm, noteSize, sunAt, type MomentId } from './astro';
import { Dome, type DomeHi } from './Dome';
import { DateMoment, isMorning, momentT } from './common';
import { PrimsSvg } from './Prims';
import { RC, almucantarRuns, platePrims, radiusOf } from './plateGeom';
import { QUAD_MAX, THREAD_TOL, bead, quarterRuns, threadEnd } from './quadGeom';

function Board({ step, dec, absH, hMeas }: { step: number; dec: number; absH: number; hMeas: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const H = Math.round(Math.max(290, Math.min(400, W * 0.9)));
  const quarter = step >= 1;
  // whole plate (step 0) centred; quarter: apex bottom-left, so it fills the view
  const k = quarter ? Math.min(W - 64, H - 74) / RC : (Math.min(W, H - 30) / 2 - 10) / RC;
  const f = quarter ? { cx: 22, cy: H - 40, k } : { cx: W / 2, cy: (H - 30) / 2 + 2, k };
  const X = (p: { u: number; v: number }) => ({ x: f.cx + p.u * k, y: f.cy - p.v * k });
  const path = (run: { u: number; v: number }[]) => run.map((p, i) => `${i ? 'L' : 'M'} ${X(p).x.toFixed(1)} ${X(p).y.toFixed(1)}`).join(' ');
  const b = bead(dec, absH);
  const e = threadEnd(absH);
  const bp = X(b);
  const hb = altitude(dec, absH, PHI);
  const on = Math.abs(hb - hMeas) <= THREAD_TOL;
  const note =
    step === 0
      ? '41° plate: morning left, afternoon right, mirror images'
      : step === 1
        ? 'keep one quarter: the pole at the corner, noon along the upright edge'
        : step === 2
          ? `bead at the sun's distance from the pole: ${radiusOf(dec).toFixed(3)}`
          : step === 3
            ? on
              ? `the bead is on the ${deg1(hMeas)} circle`
              : `bead at ${deg1(hb)}, measured ${deg1(hMeas)}`
            : `the thread reads ${absH.toFixed(1)}° on the arc`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={note}>
        {!quarter && <PrimsSvg prims={platePrims(PHI, { labels: false })} f={f} scaleW={0.9} />}
        {!quarter && <line x1={f.cx} y1={f.cy - RC * k} x2={f.cx} y2={f.cy + RC * k} style={{ stroke: C.red, strokeWidth: 1.4, strokeDasharray: '6 4' }} />}
        {quarter && (
          <>
            <path d={`M ${f.cx} ${f.cy} L ${f.cx} ${f.cy - RC * k} A ${RC * k} ${RC * k} 0 0 1 ${f.cx + RC * k} ${f.cy} Z`} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.3 }} />
            {quarterRuns().map((q) => q.runs.map((r, i) => <path key={`${q.h}-${i}`} d={path(r)} style={{ fill: 'none', stroke: q.h === 0 ? C.ink : C.soft, strokeWidth: q.h === 0 ? 1.3 : 0.8 }} />))}
            {[0.656, 1].map((r) => (
              <path key={r} d={`M ${f.cx} ${f.cy - r * k} A ${r * k} ${r * k} 0 0 1 ${f.cx + r * k} ${f.cy}`} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.7 }} />
            ))}
            {Array.from({ length: 10 }, (_, i) => i * 10).map((a) => {
              const p = threadEnd(a);
              const q = { u: p.u * 1.04, v: p.v * 1.04 };
              const P1 = X(p);
              const P2 = X(q);
              const L = X({ u: p.u * 1.1, v: p.v * 1.1 });
              return (
                <g key={a}>
                  <line x1={P1.x} y1={P1.y} x2={P2.x} y2={P2.y} style={{ stroke: C.ink, strokeWidth: 0.8 }} />
                  {a % 30 === 0 && a > 0 && a < 90 && (
                    <InkText x={L.x} y={L.y} kind="num" anchor="middle" baseline="central" size={10} role="ink">{`${a}°`}</InkText>
                  )}
                </g>
              );
            })}
            {step >= 3 &&
              almucantarRuns(hMeas, PHI).map((run, i) => {
                const r = run.filter((p) => p.u >= -1e-9 && p.v >= -1e-9);
                return r.length > 1 ? <path key={i} d={path(r)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 2 }} /> : null;
              })}
            {step >= 2 && (
              <>
                <line x1={f.cx} y1={f.cy} x2={X(e).x} y2={X(e).y} style={{ stroke: step >= 4 ? C.red : C.ink, strokeWidth: 1.2 }} />
                <circle cx={bp.x} cy={bp.y} r={5} style={{ fill: C.brass, stroke: on && step >= 3 ? C.sky : C.paper, strokeWidth: on && step >= 3 ? 2.4 : 1 }} />
              </>
            )}
            <InkText x={f.cx + 6} y={f.cy - RC * k + 12} size={11} role="soft">noon</InkText>
            <InkText x={f.cx + RC * k} y={f.cy + 16} anchor="end" size={11} role="soft">90°: 6 hours from noon</InkText>
          </>
        )}
        <InkText x={W / 2} y={H - 10} anchor="middle" size={noteSize(note, W)} role={step === 4 ? 'red' : 'soft'}>{note}</InkText>
      </svg>
    </div>
  );
}

const HI: DomeHi[][] = [['meridian'], ['meridian'], ['day', 'sun'], ['h', 'sun'], ['hour', 'sun']];

export default function QuadrantPlate() {
  const [doy, setDoy] = useState(DOY_DEFAULT);
  const [moment, setMoment] = useState<MomentId>('m2');
  const [absH, setAbsH] = useState(80);
  const [step, setStep] = useState(0);
  const t = momentT(moment);
  const sun = sunAt(doy, t);
  const hMeas = Math.round(sun.h * 10) / 10;
  const hb = altitude(sun.dec, absH, PHI);
  const on = Math.abs(hb - hMeas) <= THREAD_TOL;
  const morning = isMorning(moment);
  const tRead = 12 + (morning ? -absH : absH) / 15;
  const steps: StepDef[] = [
    {
      id: 'half',
      title: 'Halve the plate',
      text: <>The morning half of the plate is the afternoon half in a mirror. Keep one half; you only need to know which side of noon you are on.</>,
    },
    {
      id: 'quarter',
      title: 'Quarter it',
      text: <>Cut again, along the east-west line. The quarter keeps the pole at its corner, noon along one edge and a 90° arc for six hours. This is what gets engraved on the board.</>,
    },
    {
      id: 'bead',
      title: 'Set the bead',
      text: <>The rete is gone. A thread hangs from the corner (the pole) with a bead on it; slide the bead to the sun's distance from the pole for today, read off a scale along the edge.</>,
    },
    {
      id: 'swing',
      title: 'Swing the thread',
      text: (
        <>
          With the sun measured at <span style={{ color: C.sky }}>{deg1(hMeas)}</span>, swing the thread until the bead sits on that height circle. The bead
          does exactly what the sun's point on the rete did.
        </>
      ),
      controls: <Slider label="swing the thread" value={absH} min={0} max={QUAD_MAX} step={0.25} onChange={(v) => setAbsH(Math.abs(altitude(sun.dec, v, PHI) - hMeas) <= THREAD_TOL ? Math.abs(sun.H) : v)} format={(v) => `${v.toFixed(1)}°`} />,
      gated: true,
      done: on,
      hint: `Bead on the ${deg1(hMeas)} circle`,
    },
    {
      id: 'read',
      title: 'Read the arc',
      text: (
        <>
          The thread's angle from the noon edge is the hour angle: {absH.toFixed(1)}° is {(absH / 15).toFixed(2)} hours{' '}
          {morning ? 'before' : 'after'} noon, <span style={{ color: C.red }}>{hm(tRead)}</span>.
        </>
      ),
    },
  ];
  return (
    <StepPlate
      n={6}
      title="From an astrolabe plate to a quadrant"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Board step={i} dec={sun.dec} absH={i >= 3 ? absH : i === 2 ? 0 : absH} hMeas={hMeas} />}
      viewLabel="The 41° plate, cut down"
      reference={(i) => <Dome opts={{ dec: sun.dec, sunH: sun.H, alm: i >= 3 ? hMeas : null, showHourArc: i >= 4 }} hi={HI[i]} />}
      referenceLabel="The real sky over Istanbul (east on the left)"
      settings={<DateMoment doy={doy} setDoy={setDoy} moment={moment} setMoment={setMoment} />}
      onSkip={() => setAbsH(Math.abs(sun.H))}
      equation={
        <Working label="The same equation, on a quarter">
          <Eq>
            <Sym role="brass">bead</Sym> = tan((90° − δ)/2) = <Known when={step >= 2} role="brass">{radiusOf(sun.dec).toFixed(3)}</Known>
          </Eq>
          <Eq>
            bead on <Sym role="sky">h</Sym> = <Known when={step >= 3} role="sky">{deg1(hMeas)}</Known> at <Sym role="red">|H|</Sym> ={' '}
            <Known when={step >= 4 && on} role="red">{`${absH.toFixed(1)}°`}</Known>
          </Eq>
          <Eq>
            <Sym role="red">t</Sym> = 12:00 {morning ? '−' : '+'} <Sym role="red">|H|</Sym> / 15° = <Known when={step >= 4 && on} role="red">{hm(tRead)}</Known>
          </Eq>
        </Working>
      }
      caption={
        <>
          The quarter covers hour angles up to 90°, six hours either side of noon; on summer days the earliest and latest hours fall off it. Museo Galileo
          describes the quadrant as an astrolabe "folded back twice on itself". Ottoman almucantar quadrants survive; the sine quadrant replaced the drawn
          circles with a grid that computes the same equation.
        </>
      }
    />
  );
}
