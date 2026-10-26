/**
 * Episode 8: the caesium clock, as a procedure. 1 Tune the microwaves · 2 Find the peak ·
 * 3 Lock the quartz to it · 4 Count a second · 5 How good it gets.
 * View: the resonance curve (illustrative width) with the microwave frequency marker and the
 * flip fraction. Numbers: ./atomicGeom.ts.
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
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import { fmtSec, sci } from './fmt';
import TimeRef from './TimeRef';
import {
  DET_MAX,
  DET_MIN,
  ERAS,
  F_CS,
  PEAK_TOL,
  QUARTZ_OFFSET,
  curveFrame,
  errFromDet,
  flipFraction,
  grouped,
  type Era,
} from './atomicGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function Curve({ det, locked, count }: { det: number; locked: boolean; count: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const f = curveFrame(W);
  const pts = Array.from({ length: 161 }, (_, i) => {
    const d = DET_MIN + ((DET_MAX - DET_MIN) * i) / 160;
    return `${f.x(d).toFixed(1)},${f.y(flipFraction(d)).toFixed(1)}`;
  }).join(' ');
  const p = flipFraction(det);
  const H = f.H + (count ? 36 : 0);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Caesium resonance: microwaves ${det >= 0 ? '+' : ''}${det.toFixed(0)} Hz from the line, ${(p * 100).toFixed(0)} percent of atoms flip`}>
        <InkText x={f.m.l} y={14} size={11.5} role="soft">atoms that flip, against the microwave frequency</InkText>
        <line x1={f.m.l} x2={f.m.l + f.iw} y1={f.y(0)} y2={f.y(0)} style={ink('ink', 0.9)} />
        <polyline points={pts} style={ink('sky', 1.5)} />
        <line x1={f.x(0)} x2={f.x(0)} y1={f.y(0) + 4} y2={f.y(1) - 6} style={{ ...ink('faint', 0.8), strokeDasharray: '2 3' }} />
        <InkText x={f.x(0)} y={f.y(0) + 16} kind="num" anchor="middle" size={10} role="soft">9 192 631 770 Hz</InkText>
        <InkText x={f.x(DET_MIN)} y={f.y(0) + 16} kind="num" size={10} role="soft">−1 kHz</InkText>
        <InkText x={f.x(DET_MAX)} y={f.y(0) + 16} kind="num" anchor="end" size={10} role="soft">+1 kHz</InkText>
        {/* the microwave frequency now */}
        <line x1={f.x(det)} x2={f.x(det)} y1={f.y(0)} y2={f.y(p)} style={ink('red', 1.6)} />
        <circle cx={f.x(det)} cy={f.y(p)} r={4.5} style={{ fill: C.red }} />
        <InkText x={f.x(det) + (det > 300 ? -8 : 8)} y={f.y(p) - 6} kind="num" anchor={det > 300 ? 'end' : 'start'} size={11} role="red">
          {`${Math.round(p * 100)}%`}
        </InkText>
        <InkText x={W / 2} y={f.H - 8} anchor="middle" size={11.5} role={locked ? 'red' : 'ink'}>
          {locked ? 'Locked: the feedback holds the microwaves on the peak.' : Math.abs(det) <= PEAK_TOL ? 'On the peak.' : `Off the line by ${Math.abs(det).toFixed(0)} Hz.`}
        </InkText>
        {count && (
          <InkText x={W / 2} y={f.H + 22} kind="num" anchor="middle" size={W < 400 ? 12 : 14} role="red">
            {`${grouped(F_CS)} cycles = 1 second`}
          </InkText>
        )}
      </svg>
    </div>
  );
}

export default function AtomicPlate() {
  const [step, setStep] = useState(0);
  const [det, setDet] = useState(QUARTZ_OFFSET - 300);
  const [tuned, setTuned] = useState(false);
  const [locked, setLocked] = useState(false);
  const [era, setEra] = useState<Era>('essen');
  const onPeak = Math.abs(det) <= PEAK_TOL;
  const detAt = (i: number) => (i >= 2 && locked ? 0 : det);
  const eraOf = ERAS.find((e) => e.id === era)!;
  const rateAt = (i: number) => (i >= 2 && locked ? (i === 4 ? eraOf.perDay : ERAS[0].perDay) : Math.abs(errFromDet(detAt(i))));

  const steps: StepDef[] = [
    {
      id: 'tune',
      title: 'Tune the microwaves',
      text: (
        <>
          A quartz oscillator, multiplied up to about 9.2 GHz, drives microwaves through a beam of caesium atoms. Change the
          frequency and watch how many atoms flip to the other energy state.
        </>
      ),
      controls: <Slider label="microwave frequency, from the caesium line" value={det} min={DET_MIN} max={DET_MAX} step={1} onChange={(v) => { setDet(v); setTuned(true); }} format={(v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)} Hz`} width="12rem" />,
      gated: true,
      done: tuned,
      hint: 'Move the frequency',
    },
    {
      id: 'peak',
      title: 'Find the peak',
      text: <>Only at one frequency do most of them flip. Nature sets it, the same for every caesium atom anywhere.</>,
      controls: <Slider label="microwave frequency, from the caesium line" value={det} min={DET_MIN} max={DET_MAX} step={1} onChange={setDet} format={(v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)} Hz`} width="12rem" />,
      gated: true,
      done: onPeak,
      hint: `Within ${PEAK_TOL} Hz of the peak`,
    },
    {
      id: 'lock',
      title: 'Lock the quartz',
      text: (
        <>
          A detector counts the flipped atoms and nudges the quartz whenever the count drops: now the quartz cannot drift.
          Louis Essen and Jack Parry ran the first one at NPL in 1955.
        </>
      ),
      controls: (
        <button type="button" className="in-btn" disabled={locked} onClick={() => setLocked(true)}>
          {locked ? 'Locked' : 'Close the loop'}
        </button>
      ),
      gated: true,
      done: locked,
      hint: 'Close the loop',
    },
    {
      id: 'count',
      title: 'Count a second',
      text: <>Since 1967 a second is, by definition, 9,192,631,770 cycles of this radiation. The clock just counts them.</>,
    },
    {
      id: 'better',
      title: 'How good it gets',
      text: (
        <>
          Fountains toss the atoms up and let them fall, for a longer, sharper look; optical clocks count light, a hundred
          thousand times faster than microwaves.
        </>
      ),
      controls: (
        <Segmented label="clock" options={ERAS.map((e) => ({ value: e.id, label: e.label }))} value={era} onChange={setEra} />
      ),
    },
  ];

  const r = rateAt(step);
  return (
    <StepPlate
      n={8}
      title="Counting caesium"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Curve det={detAt(i)} locked={i >= 2 && locked} count={i >= 3} />}
      viewLabel="The atoms' answer"
      reference={(i) => {
        const ri = rateAt(i);
        const yr = 365.25;
        return (
          <TimeRef
            ep={7}
            t={12}
            dec={0}
            dateText="a year after setting"
            read={12 + (ri * yr) / 3600}
            gapSec={ri * yr}
            gapNote={i >= 2 && locked ? (i === 4 ? eraOf.say : ERAS[0].say) : 'free-running, off the line'}
            rate={ri}
            hi={i >= 2 ? 'gauge' : 'clock'}
          />
        );
      }}
      equation={
        <Working label="From frequency to seconds">
          <Eq>
            1 s = <V>9 192 631 770</V> cycles of the caesium line
          </Eq>
          <Eq>
            <Sym role="red">error</Sym> = 86 400 s × Δ<Sym role="sky">f</Sym> ÷ <Sym role="sky">f</Sym> ={' '}
            <Known when={step >= 1} role="red">{`${fmtSec(r)} a day`}</Known>
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em' }}>
              {step >= 2 && locked
                ? `${step === 4 ? eraOf.label : ERAS[0].label}: ${step === 4 ? eraOf.say : ERAS[0].say}, Δf/f ≈ ${sci(r / 86400)}`
                : `off by ${Math.abs(detAt(step))} Hz: Δf/f = ${detAt(step) === 0 ? '0' : sci(Math.abs(detAt(step)) / F_CS)}`}
            </span>
          </Eq>
        </Working>
      }
      caption={
        <>
          The curve&apos;s width is illustrative; the definition is the BIPM&apos;s (1967), the figures for the three clocks
          are from the Science Museum Group (1955), NIST (NIST-F2) and Aeppli et al. 2024 (strontium). The reference shows
          the clock a year after it was set.
        </>
      }
    />
  );
}
