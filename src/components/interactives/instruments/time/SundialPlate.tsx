/**
 * Episode 1: the sundial, as a procedure. 1 Pick the day · 2 Move the sun · 3 Read the hour ·
 * 4 How long is an hour? · 5 Tilt the gnomon (equal hours) · 6 Where it fails.
 * View: a horizontal dial at 41° N seen from above, hour lines computed from the sun's real
 * path (./sundialGeom.ts). Reference: the shared TimeRef (true time vs the dial).
 */
import { useEffect, useState } from 'react';
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
  useHoverPreview,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import { dayLabel } from './astro';
import { hm } from './fmt';
import TimeRef from './TimeRef';
import {
  STYLE_FOOT,
  T_MAX,
  T_MIN,
  dayCurve,
  dialFrame,
  northMark,
  hourLabels,
  hourLines,
  sundial,
  type Mode,
  type Sundial,
} from './sundialGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function Dial({ s, cloud }: { s: Sundial; cloud: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const nm = northMark(W);
  const f = dialFrame(W);
  const lines = hourLines(s.mode);
  const labs = hourLabels(s.mode, W);
  const curve = dayCurve(s.day, s.mode);
  const P = (p: { x: number; y: number }) => f.toS(p);
  const o = P({ x: 0, y: 0 });
  const showShadow = s.up && s.shadowEnd && !cloud;
  const tip = s.tipW && !s.tipOff ? P(s.tipW) : null;
  const end = s.shadowEnd ? P(s.shadowEnd) : null;
  const foot = P({ x: 0, y: STYLE_FOOT });
  const runD = (r: { x: number; y: number }[]) => r.map((p, i) => `${i ? 'L' : 'M'} ${P(p).x.toFixed(1)} ${P(p).y.toFixed(1)}`).join(' ');
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${f.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`A horizontal sundial at 41 degrees north, seen from above. ${s.say}`}>
        <InkText x={f.x0} y={14} size={11.5} role="soft">
          {s.mode === 'unequal' ? 'pin gnomon, 12 hours from sunrise to sunset' : 'style tilted 41° to the pole, equal hours'}
        </InkText>
        <rect x={f.x0} y={f.y0} width={f.x1 - f.x0} height={f.y1 - f.y0} style={{ ...ink('ink', 1.1), fill: C.paper }} />
        {lines.map((l) =>
          l.runs.map((r, i) => (
            <path key={`${l.key}-${i}`} d={runD(r)} style={ink(l.key === (s.mode === 'unequal' ? 6 : 12) ? 'ink' : 'soft', l.key === (s.mode === 'unequal' ? 6 : 12) ? 1 : 0.7)} />
          )),
        )}
        {labs.map((l) => (
          <InkText key={l.id} x={l.x} y={l.y} kind="num" anchor="middle" baseline="central" size={l.size} role="soft">
            {l.text}
          </InkText>
        ))}
        {/* today's path of the shadow tip */}
        {curve.map((r, i) => (
          <path key={i} d={runD(r)} style={{ ...ink('brass', 1), strokeDasharray: '4 3' }} />
        ))}
        {/* the gnomon */}
        {s.mode === 'equal' ? (
          <>
            <line x1={o.x} y1={o.y} x2={foot.x} y2={foot.y} style={ink('ink', 2.2)} />
            <InkText x={foot.x + 6} y={foot.y + 4} size={10.5} role="soft">style</InkText>
          </>
        ) : null}
        {/* the shadow */}
        {showShadow && end && (
          <line x1={o.x} y1={o.y} x2={end.x} y2={end.y} style={{ ...ink('red', 3), strokeLinecap: 'round', opacity: 0.85 }} />
        )}
        {showShadow && tip && <circle cx={tip.x} cy={tip.y} r={3.5} style={{ fill: C.red }} />}
        {showShadow && s.tipOff && end && (
          <path d={`M ${end.x} ${end.y} m -5 -5 l 10 10 m 0 -10 l -10 10`} style={ink('red', 1.3)} />
        )}
        <circle cx={o.x} cy={o.y} r={4} style={{ fill: C.ink }} />
        {/* north arrow */}
        <g>
          <line x1={nm.ax} y1={nm.yBot} x2={nm.ax} y2={nm.yTop + 2} style={ink('soft', 1)} />
          <path d={`M ${nm.ax - 4} ${nm.yTop + 8} L ${nm.ax} ${nm.yTop} L ${nm.ax + 4} ${nm.yTop + 8}`} style={ink('soft', 1)} />
          <InkText x={nm.label.x} y={nm.label.y} kind="label" anchor="end" size={11} role="soft">N</InkText>
        </g>
        {cloud && s.up && (
          <rect x={f.x0} y={f.y0} width={f.x1 - f.x0} height={f.y1 - f.y0} style={{ fill: C.deep, opacity: 0.55 }} />
        )}
        <InkText x={(f.x0 + f.x1) / 2} y={f.H - 12} anchor="middle" size={12} role={s.read === null || cloud ? 'soft' : 'red'}>
          {cloud && s.up ? 'Clouds: no shadow, no time.' : s.say}
        </InkText>
      </svg>
    </div>
  );
}

export default function SundialPlate() {
  const [step, setStep] = useState(0);
  const [day, setDay] = useState(172);
  const [t, setT] = useState(9.25);
  const [selMode, setMode] = useState<Mode>('unequal');
  const [mode, bindMode] = useHoverPreview<Mode>(selMode);
  const [cloud, setCloud] = useState(false);
  const [moved, setMoved] = useState(false);
  const [dayMoved, setDayMoved] = useState(false);
  const [tilted, setTilted] = useState(false);
  useEffect(() => {
    if (selMode === 'equal') setTilted(true);
  }, [selMode]);

  const s = sundial(mode, day, t);
  const sUn = sundial('unequal', day, t);
  /** Steps 1-4 use the upright pin; from step 5 the reader chooses. */
  const modeAt = (i: number): Mode => (i >= 4 ? mode : 'unequal');
  const readAt = (i: number) => (i === 5 && cloud ? null : sundial(modeAt(i), day, t).read);

  const dateSlider = (
    <Slider label="day of the year" value={day} min={1} max={365} step={1} onChange={(v) => { setDay(v); setDayMoved(true); }} format={dayLabel} width="9rem" />
  );
  const timeSlider = (
    <Slider label="true local time" value={t} min={T_MIN} max={T_MAX} step={1 / 12} onChange={(v) => { setT(v); setMoved(true); }} format={hm} width="10rem" />
  );
  const modeSeg = (
    <Segmented
      label="gnomon"
      options={[
        { value: 'unequal', label: 'upright pin' },
        { value: 'equal', label: 'style tilted to 41°' },
      ]}
      value={selMode}
      onChange={setMode}
      preview={bindMode}
    />
  );

  const steps: StepDef[] = [
    {
      id: 'day',
      title: 'Pick the day',
      text: (
        <>
          The dashed brass curve is where the tip of the pin&apos;s shadow will travel today. Move the day: in
          winter the sun is low and the shadow long, in summer short.
        </>
      ),
      controls: dateSlider,
      gated: true,
      done: dayMoved,
      hint: 'Move the day',
    },
    {
      id: 'sun',
      title: 'Let the sun move',
      text: <>The sun crosses the sky and the shadow swings from west, through north at noon, to east.</>,
      controls: timeSlider,
      gated: true,
      done: moved,
      hint: 'Move the time',
    },
    {
      id: 'read',
      title: 'Read the hour',
      text: (
        <>
          The lines number the hours of daylight, 1 to 11; the 6th ends at noon. The tip, not the whole
          shadow, gives the hour. {s.read !== null && mode === 'unequal' ? s.say : ''}
        </>
      ),
      controls: timeSlider,
    },
    {
      id: 'len',
      title: 'How long is an hour?',
      text: (
        <>
          Twelve hours always fill sunrise to sunset, so on {dayLabel(day)} each one lasts{' '}
          <b style={{ color: C.red }}>{Math.round(sUn.hourMin)} minutes</b>. Move the day and watch it stretch from
          about 45 (December) to 75 (June).
        </>
      ),
      controls: dateSlider,
    },
    {
      id: 'tilt',
      title: 'Tilt the gnomon',
      text: (
        <>
          Lean the shadow edge until it points at the celestial pole, 41° up here. Its shadow now turns 15° of the
          sun&apos;s hour angle every hour, whatever the date: equal hours, and one set of lines for the whole year.
        </>
      ),
      controls: (
        <>
          {modeSeg}
          {timeSlider}
        </>
      ),
      gated: true,
      done: tilted,
      hint: 'Switch to the tilted style',
    },
    {
      id: 'fail',
      title: 'Where it fails',
      text: (
        <>
          A cloud, or the sun below the horizon, and the dial says nothing. Half of every day, a sundial is a
          stone.
        </>
      ),
      controls: (
        <>
          {timeSlider}
          <Segmented
            label="sky"
            options={[
              { value: 'clear', label: 'clear sky' },
              { value: 'cloud', label: 'cloud' },
            ]}
            value={cloud ? 'cloud' : 'clear'}
            onChange={(v) => setCloud(v === 'cloud')}
          />
        </>
      ),
    },
  ];

  return (
    <StepPlate
      n={1}
      title="Telling the hour from a shadow"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Dial s={sundial(modeAt(i), day, t)} cloud={i === 5 && cloud} />}
      viewLabel="The dial, from above"
      reference={(i) => (
        <TimeRef
          ep={0}
          t={t}
          dec={s.dec}
          dateText={dayLabel(day)}
          read={readAt(i)}
          gapSec={readAt(i) === null ? null : Math.round(Math.abs(readAt(i)! - t) * 3600)}
          gapNote={modeAt(i) === 'equal' ? undefined : 'an hour of daylight is not a clock hour'}
          noRead={i === 5 && cloud ? 'cloud' : 'no shadow'}
          hi={i <= 1 ? 'sky' : 'clock'}
        />
      )}
      referenceLabel="What you are measuring"
      equation={
        <Working label="Length of one hour of daylight">
          <Eq>
            cos <Sym role="brass">H₀</Sym> = −tan <Sym>φ</Sym> · tan <Sym role="brass">δ</Sym>
          </Eq>
          <Eq>
            <Known when={step >= 3} role="red">{`${sUn.hourMin.toFixed(0)} min`}</Known> = 2 ·{' '}
            <Known when={step >= 3} role="brass">{`${sUn.H0.toFixed(1)}°`}</Known> ÷ 15°/h ÷ 12
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em' }}>
              {`φ = 41°, δ = ${s.dec.toFixed(1)}° on ${dayLabel(day)}; sunrise ${hm(s.rise)}, sunset ${hm(s.set)}`}
            </span>
          </Eq>
        </Working>
      }
      caption={
        <>
          Hour lines computed for 41° N from the sun&apos;s position through the year (local solar time, no
          refraction). On the upright-pin dial each line joins where the shadow tip falls at that hour of daylight on
          every day; on the tilted dial the line for hour angle H sits at tan θ = sin φ tan H from the noon line.
        </>
      }
    />
  );
}
