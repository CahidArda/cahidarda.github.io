/**
 * Episode 3: the hour from the sun's height with a sine quadrant, as a procedure (Arda's
 * 23:35 note). 1 Pick the day and the moment · 2 Sight the sun · 3 Read sin h · 4 The date
 * terms (the LEFT scale explained) · 5 Divide and read the hour.
 * The date/time set the world in step 1 only; the measured altitude is the reader's own value
 * and no other control ever changes it (a wrong or impossible one is flagged, not moved).
 * Geometry and numbers: ./quadrantGeom.ts.
 */
import { useState } from 'react';
import {
  ArcScale,
  C,
  Eq,
  InkText,
  Known,
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
import { PHI, dayLabel } from './astro';
import { fmtSec, hm } from './fmt';
import TimeRef from './TimeRef';
import { H_MAX, H_MIN, H_STEP, T_MAX, T_MIN, quadScene, quadrant, sightSay, type Quadrant } from './quadrantGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const p60 = (v: number) => (60 * v).toFixed(1);

function Board({ q, step }: { q: Quadrant; step: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const s = quadScene(q, step, W);
  const g = s.g;
  const { AX, AY, R, U, BAND } = g;
  const small = W < 400 ? 9 : 10;
  const grid: { d: string; major: boolean }[] = [];
  for (let k = 1; k < 60; k++) {
    const t = k * U;
    const len = Math.sqrt(R * R - t * t);
    grid.push({ d: `M ${AX + t} ${AY} V ${AY + len} M ${AX} ${AY + t} H ${AX + len}`, major: k % 5 === 0 });
  }
  const say =
    step === 0
      ? `Today the sun climbs to ${q.hNoon.toFixed(1)}° at noon (the red tick).`
      : step === 1
        ? sightSay(q)
        : !q.valid
          ? 'This altitude gives no hour today: go back and sight the sun.'
          : step === 2
            ? `The thread crosses the rim above ${p60(q.sinh)} on the top scale.`
            : step === 3
              ? `${q.dec >= 0 ? 'Subtract' : 'Add'} the date term: ${p60(q.sinh)} ${q.dec >= 0 ? '−' : '+'} ${p60(Math.abs(q.a))} = ${p60(q.diff)}.`
              : `The rim reads ${q.theta.toFixed(1)}°, so H = ${q.diff >= 0 ? `90° − ${q.theta.toFixed(1)}°` : `90° + ${q.theta.toFixed(1)}°`} = ${q.Hdeg.toFixed(1)}°.`;
  const showWork = q.valid && !q.tooHigh;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Sine quadrant: ${say}`}>
        {/* border bands: TOP scale = sines (0..60 to the right), LEFT scale = cosines (0..60 down) */}
        <path d={`M ${AX - BAND} ${AY - BAND} H ${AX + R + 4} V ${AY} M ${AX - BAND} ${AY - BAND} V ${AY + R + 4} H ${AX}`} style={ink('ink', 1)} />
        {Array.from({ length: 6 }, (_, i) => (i + 1) * 10).map((v) => (
          <g key={v}>
            <InkText x={AX + v * U} y={AY - BAND / 2} kind="num" anchor="middle" baseline="central" size={small} role={step === 2 ? 'sky' : 'soft'} halo={false}>
              {v}
            </InkText>
            <InkText x={AX - BAND / 2} y={AY + v * U} kind="num" anchor="middle" baseline="central" size={small} role={step === 3 ? 'brass' : 'soft'} halo={false}>
              {v}
            </InkText>
          </g>
        ))}
        {grid.map((l, i) => (
          <path key={i} d={l.d} style={ink(l.major ? 'soft' : 'faint', l.major ? 0.5 : 0.3)} />
        ))}
        <path d={`M ${AX + R} ${AY} H ${AX} V ${AY + R}`} style={ink('ink', 1.1)} />
        <path d={arcPath(AX, AY, R, -90, 0)} style={ink('ink', 1.1)} />
        <ArcScale cx={AX} cy={AY} r={R + 2} from={-90} to={0} major={10} minor={2} labelEvery={10} value={(v) => v + 90} tick={7} size={small} rim={false} />
        {/* sights on the top edge */}
        {[AX + 8, AX + R - 6].map((x) => (
          <rect key={x} x={x - 3.5} y={AY - BAND - 8} width={7} height={8} style={{ ...ink('ink', 0.9), fill: C.paper }} />
        ))}
        {/* the day's noon height on the rim: a limit the sun never passes */}
        {s.noonMark && (
          <line x1={s.noonMark.a.x} y1={s.noonMark.a.y} x2={s.noonMark.b.x} y2={s.noonMark.b.y} style={ink('red', 2)} />
        )}
        {/* step 2: where the sun really is (brass), for sighting */}
        {step === 1 && s.sunDir && (
          <>
            <line x1={s.sunDir.a.x} y1={s.sunDir.a.y} x2={s.sunDir.b.x} y2={s.sunDir.b.y} style={ink('brass', 2.2)} />
            <InkText x={s.sunDir.b.x + 3} y={s.sunDir.b.y + 10} size={small + 1} role="brass">sun</InkText>
          </>
        )}
        {/* step 4: the date terms on the latitude thread (brass), fixed at 41° */}
        {step >= 3 && showWork && (
          <g>
            <line x1={AX} y1={AY} x2={s.latEnd.x} y2={s.latEnd.y} style={{ ...ink('brass', 1.2), strokeDasharray: '6 3' }} />
            {Math.abs(q.a) > 1e-4 && (
              <>
                <path d={`M ${s.beadA.x} ${s.beadA.y} V ${AY}`} style={{ ...ink('brass', 0.9), strokeDasharray: '2 2' }} />
                <circle cx={s.beadA.x} cy={s.beadA.y} r={3.4} style={{ fill: C.paper, stroke: C.brass, strokeWidth: 1.4 }} />
              </>
            )}
            <path d={`M ${s.beadB.x} ${s.beadB.y} H ${AX}`} style={{ ...ink('brass', 0.9), strokeDasharray: '2 2' }} />
            <circle cx={s.beadB.x} cy={s.beadB.y} r={3.4} style={{ fill: C.brass }} />
            <circle cx={AX} cy={s.bY} r={2.6} style={{ fill: C.brass }} />
            {/* the shift along the row: from 60 sin h by the date term */}
            {q.diff >= 0 ? (
              <line x1={s.sinX} x2={s.diffX} y1={s.shiftY} y2={s.shiftY} style={ink('brass', 2.4)} />
            ) : (
              <>
                <line x1={s.sinX} x2={AX} y1={s.shiftY} y2={s.shiftY} style={ink('brass', 2.4)} />
                <line x1={AX} x2={s.diffX} y1={s.shiftY + 5} y2={s.shiftY + 5} style={{ ...ink('brass', 2), strokeDasharray: '3 2' }} />
              </>
            )}
            <circle cx={s.diffX} cy={q.diff >= 0 ? s.shiftY : s.shiftY + 5} r={3} style={{ fill: C.ink }} />
          </g>
        )}
        {/* step 3+: 60 sin h on the top scale */}
        {step >= 2 && showWork && (
          <>
            <path d={`M ${s.thread.b.x} ${s.thread.b.y} V ${AY}`} style={{ ...ink('sky', 1), strokeDasharray: '4 2.5' }} />
            <circle cx={s.sinX} cy={AY} r={3} style={{ fill: C.sky }} />
          </>
        )}
        {/* step 5: the dividing thread through the bead at the divisor */}
        {step >= 4 && showWork && s.divEnd && s.divBead && (
          <>
            <line x1={AX} y1={AY} x2={s.divEnd.x} y2={s.divEnd.y} style={ink('red', 1.4)} />
            <path d={`M ${s.divBead.x} ${s.divBead.y} V ${q.diff >= 0 ? s.shiftY : s.shiftY + 5}`} style={{ ...ink('red', 0.9), strokeDasharray: '2 2' }} />
            <circle cx={s.divBead.x} cy={s.divBead.y} r={3.6} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.5 }} />
            <circle cx={s.divEnd.x} cy={s.divEnd.y} r={3.4} style={{ fill: C.red }} />
          </>
        )}
        {/* the plumb thread at the measured altitude (the reader's value, never moved) */}
        {step >= 1 && (
          <>
            <line x1={AX} y1={AY} x2={s.thread.bob.x} y2={s.thread.bob.y} style={ink('sky', 1.7)} />
            <circle cx={s.thread.bob.x} cy={s.thread.bob.y} r={4} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.2 }} />
            <path d={arcPath(AX, AY, 46, -90, q.hMeas - 90)} style={ink('sky', 1)} />
          </>
        )}
        <circle cx={AX} cy={AY} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
        {s.labels.map((l) => (
          <InkText key={l.id} x={l.x} y={l.y} kind="num" anchor={l.anchor} baseline="central" size={l.size} role={l.role}>
            {l.text}
          </InkText>
        ))}
        <InkText x={W / 2} y={g.H - 12} anchor="middle" size={W < 400 ? 11 : 12} role={step === 1 && !q.onSun ? 'red' : 'ink'}>
          {say}
        </InkText>
      </svg>
    </div>
  );
}

export default function QuadrantPlate() {
  const [step, setStep] = useState(0);
  const [day, setDay] = useState(172);
  const [tTrue, setT] = useState(9);
  const [h, setH] = useState(20);
  const q = quadrant(day, tTrue, h);
  const sinDone = step >= 2 && q.valid;
  const dateDone = step >= 3 && q.valid;
  const hDone = step >= 4 && q.valid;

  const steps: StepDef[] = [
    {
      id: 'day',
      title: 'Pick the day and hour',
      text: (
        <>
          This sets the world: the date fixes the sun&apos;s path (and its noon height, the red tick on the rim), the
          true time puts the sun on it. The board, like the two in the case, is made for 41° N.
        </>
      ),
      controls: (
        <>
          <Slider label="date" value={day} min={1} max={365} step={1} onChange={setDay} format={dayLabel} width="8rem" />
          <Slider label="true time (morning)" value={tTrue} min={T_MIN} max={T_MAX} step={1 / 12} onChange={setT} format={hm} width="8rem" />
        </>
      ),
      gated: true,
      done: q.sunUp,
      hint: 'The sun is down: pick a later time',
    },
    {
      id: 'sight',
      title: 'Sight the sun',
      text: (
        <>
          Tilt the board until sunlight passes through both sights on the top edge. The thread hangs plumb and its angle
          from the left edge is the sun&apos;s altitude <Sym role="sky">h</Sym>.
        </>
      ),
      controls: (
        <Slider label="tilt the board: altitude h" value={h} min={H_MIN} max={H_MAX} step={H_STEP} onChange={setH} format={(v) => `${v.toFixed(1)}°`} width="11rem" />
      ),
      gated: true,
      done: q.onSun && !q.tooHigh,
      hint: q.tooHigh ? 'Higher than the sun ever gets today' : 'Line the sights up on the sun',
    },
    {
      id: 'sin',
      title: 'Read sin h',
      text: (
        <>
          Follow the grid line straight up from where the thread crosses the rim. The top edge is the sine scale: it
          reads <Sym role="sky">60 sin h</Sym>, in parts of 60.
        </>
      ),
    },
    {
      id: 'date',
      title: 'Apply the date',
      text: (
        <>
          Lay the thread on the board&apos;s fixed latitude line, 41° (brass). The left edge is the cosine scale: a point on a
          thread at angle θ sits 60·cos θ down it. A bead at 60·|sin δ| on this thread reads the date term on the top
          scale; one at 60·cos δ reads the divisor on the left scale. The sun is {q.dec >= 0 ? 'north' : 'south'} of the
          equator today, so the term is {q.dec >= 0 ? 'subtracted' : 'added'}.
        </>
      ),
    },
    {
      id: 'divide',
      title: 'Divide, read the hour',
      text: (
        <>
          Slide the bead to the divisor and turn the thread until the bead stands over the difference. The rim then gives
          90° − H, the sun&apos;s distance from noon. At 15° an hour that is the time.
        </>
      ),
    },
  ];

  return (
    <StepPlate
      n={3}
      label="Levha"
      title="Telling the time with a sine quadrant"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Board q={q} step={i} />}
      viewLabel="The board (drawn after the 1848 sheet)"
      reference={(i) => (
        <TimeRef
          ep={2}
          t={tTrue}
          dec={q.dec}
          dateText={dayLabel(day)}
          read={i >= 4 && q.valid ? q.read : null}
          gapSec={i >= 4 && q.errSec !== null ? Math.round(Math.abs(q.errSec)) : null}
          gapNote={i >= 4 && q.valid && !q.onSun ? 'from a sight that missed the sun' : undefined}
          noRead={i >= 4 ? 'no hour from this sight' : 'not worked out yet'}
          hi={i <= 1 ? 'sky' : 'clock'}
        />
      )}
      equation={
        <Working label="The muvakkit's working, in parts of 60">
          <Eq>
            cos <Sym role="red">H</Sym> = (sin <Sym role="sky">h</Sym> − <Sym role="brass">sin φ sin δ</Sym>) ÷ <Sym role="brass">cos φ cos δ</Sym>
          </Eq>
          <Eq>
            <Known when={hDone} role="red">{q.valid ? p60(q.cosH) : '?'}</Known> = (<Known when={sinDone} role="sky">{p60(q.sinh)}</Known>{' '}
            {q.a >= 0 ? '−' : '+'} <Known when={dateDone} role="brass">{p60(Math.abs(q.a))}</Known>) ÷{' '}
            <Known when={dateDone} role="brass">{p60(q.b)}</Known> × 60
          </Eq>
          <Eq>
            <Sym role="red">H</Sym> = <Known when={hDone} role="red">{q.valid ? `${q.Hdeg.toFixed(1)}°` : '?'}</Known>, time ={' '}
            <Known when={hDone} role="red">{q.read !== null ? hm(q.read) : '?'}</Known>
            {hDone && q.errSec !== null && (
              <>
                {' '}
                (true <V>{hm(tTrue)}</V>, off by <V role="red">{fmtSec(Math.abs(q.errSec))}</V>)
              </>
            )}
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em' }}>{`φ = ${PHI}°, δ = ${q.dec.toFixed(1)}° on ${dayLabel(day)}`}</span>
          </Eq>
        </Working>
      }
      caption={
        <>
          Morning only: the afternoon gives the same H after noon. Numbers are my arithmetic from the altitude formula,
          sin h = sin φ sin δ + cos φ cos δ cos H (local solar time, no refraction). A sight counts as on the sun within
          0.25°, about half the sun&apos;s width, which alone moves the hour by a minute or two.
        </>
      }
    />
  );
}
