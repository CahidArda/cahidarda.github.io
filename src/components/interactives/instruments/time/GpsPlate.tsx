/**
 * Episode 9: GPS, as a procedure. 1 One satellite: a range · 2 Three satellites: a fix ·
 * 3 Your phone's clock is off · 4 Solve for the clock · 5 Relativity, uncorrected.
 * View: the ground around you, seen from above, a few km across; each satellite's range is a
 * line at right angles to its direction (it is 20,200 km away). Geometry: ./gpsGeom.ts.
 */
import { useState } from 'react';
import {
  C as Col,
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
import { fmtDist, fmtSec, signed } from './fmt';
import TimeRef from './TimeRef';
import { B_MAX, B_STEP, C, MEET_TOL, REL_PER_DAY, SATS, clipLine, fix, fmtM, halfWidth, scaleBar, type GpsState } from './gpsGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: Col[role], strokeWidth: w });
const ns = (s: number) => `${signed(s * 1e9, (x) => (x >= 1000 ? `${(x / 1000).toFixed(2)} µs` : `${Math.round(x)} ns`))}`;

function Ground({ s }: { s: GpsState }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const H = Math.min(W, 380);
  const side = H - 40;
  const cx = W / 2;
  const cy = 22 + side / 2;
  const hw = halfWidth(s);
  const k = side / 2 / hw;
  const P = (p: { x: number; y: number }) => ({ x: cx + p.x * k, y: cy - p.y * k });
  const f = fix(s);
  const met = s.n >= 3 && f.size <= MEET_TOL;
  const bar = scaleBar(hw);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Map around you, ${fmtM(2 * hw)} across, ${s.n} satellite ranges`}>
        <rect x={cx - side / 2} y={cy - side / 2} width={side} height={side} style={{ ...ink('faint', 0.8), fill: Col.paper }} />
        {/* satellites: arrows at the edge pointing to where each is in the sky */}
        {SATS.slice(0, s.n).map((sat, i) => {
          const e = { x: cx + sat.u.x * (side / 2 + 2), y: cy - sat.u.y * (side / 2 + 2) };
          const seg = clipLine(sat.u, f.d[i], hw);
          const lab = { x: cx + sat.u.x * (side / 2 - 18), y: cy - sat.u.y * (side / 2 - 18) };
          return (
            <g key={sat.id}>
              {seg && <line x1={P(seg[0]).x} y1={P(seg[0]).y} x2={P(seg[1]).x} y2={P(seg[1]).y} style={ink(i === 0 && s.relHours > 0 ? 'red' : 'sky', 1.5)} />}
              <path d={`M ${e.x} ${e.y} l ${-sat.u.x * 14 - sat.u.y * 5} ${sat.u.y * 14 - sat.u.x * 5} m ${sat.u.y * 10} ${sat.u.x * 10} L ${e.x} ${e.y}`} style={ink('sky', 1.2)} />
              <InkText x={lab.x} y={lab.y} kind="label" anchor="middle" baseline="central" size={12} role="sky">{sat.id}</InkText>
            </g>
          );
        })}
        {/* the fix: the triangle the lines make, or their meeting point */}
        {f.pts.length === 3 && !met && (
          <polygon points={f.pts.map((p) => `${P(p).x},${P(p).y}`).join(' ')} style={{ fill: Col.red, opacity: 0.18, stroke: Col.red, strokeWidth: 0.8 }} />
        )}
        {/* you (the truth), fixed */}
        <circle cx={cx} cy={cy} r={5} style={{ fill: Col.paper, stroke: Col.ink, strokeWidth: 1.6 }} />
        <InkText x={cx + 8} y={cy + 16} size={11} role="ink">you</InkText>
        {met && f.at && <circle cx={P(f.at).x} cy={P(f.at).y} r={3} style={{ fill: Col.red }} />}
        {/* scale bar */}
        <line x1={cx - side / 2 + 8} x2={cx - side / 2 + 8 + bar * k} y1={cy + side / 2 - 10} y2={cy + side / 2 - 10} style={ink('ink', 1.4)} />
        <InkText x={cx - side / 2 + 8} y={cy + side / 2 - 16} kind="num" size={10} role="soft">{fmtM(bar)}</InkText>
        <InkText x={W / 2} y={H - 6} anchor="middle" size={11.5} role={met ? 'ink' : 'red'}>
          {s.n === 1
            ? 'One range: you are somewhere on this line.'
            : met
              ? f.at && Math.hypot(f.at.x, f.at.y) > MEET_TOL
                ? `The lines meet, ${fmtDist(Math.hypot(f.at.x, f.at.y))} from where you are.`
                : 'The three lines meet where you are.'
              : `The lines miss each other by ${fmtDist(f.size)}.`}
        </InkText>
      </svg>
    </div>
  );
}

export default function GpsPlate() {
  const [step, setStep] = useState(0);
  const [b, setB] = useState(1e-6);
  const [bHat, setBHat] = useState(0);
  const [rel, setRel] = useState(0);
  const stateAt = (i: number): GpsState => ({
    n: i === 0 ? 1 : 3,
    b: i >= 2 ? b : 0,
    bHat: i >= 3 ? bHat : 0,
    relHours: i === 4 ? rel : 0,
  });
  const s = stateAt(step);
  const f = fix(s);
  const solved = Math.abs(b - bHat) * C <= MEET_TOL;
  const posErr = f.at ? Math.hypot(f.at.x, f.at.y) : null;

  const steps: StepDef[] = [
    {
      id: 'one',
      title: 'One satellite',
      text: (
        <>
          Each satellite carries atomic clocks and broadcasts the time. Your receiver notes when the signal arrives: the
          delay times the speed of light is your distance from it. That puts you on a sphere, here a line.
        </>
      ),
    },
    {
      id: 'three',
      title: 'Three satellites',
      text: <>Three ranges, three lines; with a perfect clock in your hand they cross at one point, which is you.</>,
    },
    {
      id: 'clock',
      title: 'Your clock is off',
      text: (
        <>
          But your phone&apos;s clock is a cheap quartz one. If it is late by b, every delay looks longer by b and every
          range grows by c·b: 30 cm for each nanosecond. The lines no longer meet.
        </>
      ),
      controls: <Slider label="your clock's error b" value={b} min={-B_MAX} max={B_MAX} step={B_STEP} onChange={setB} format={ns} width="11rem" />,
      gated: true,
      done: Math.abs(b) * C > MEET_TOL,
      hint: 'Put an error on your clock',
    },
    {
      id: 'solve',
      title: 'Solve for the clock',
      text: (
        <>
          The receiver treats its own clock error as one more unknown and finds the value that makes the lines meet. In
          three dimensions that takes a fourth satellite; on this map, three. Find it.
        </>
      ),
      controls: <Slider label="receiver's guess of its error" value={bHat} min={-B_MAX} max={B_MAX} step={B_STEP} onChange={setBHat} format={ns} width="11rem" />,
      gated: true,
      done: solved,
      hint: 'Make the lines meet',
    },
    {
      id: 'rel',
      title: 'Relativity, uncorrected',
      text: (
        <>
          Up there gravity is weaker and the satellites move fast: their clocks gain about 38 µs a day on ours. Satellite
          clocks are set slow before launch for this. Here is what satellite A would do without that.
        </>
      ),
      controls: <Slider label="hours since A's clock was set" value={rel} min={0} max={24} step={0.25} onChange={setRel} format={(x) => `${x.toFixed(2)} h`} width="10rem" />,
    },
  ];

  const residual = s.b - s.bHat;
  return (
    <StepPlate
      n={9}
      title="How clock error becomes position error"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Ground s={stateAt(i)} />}
      viewLabel="The ground around you, from above"
      reference={(i) => {
        const si = stateAt(i);
        const r = si.b - si.bHat;
        const relE = i === 4 ? REL_PER_DAY * (si.relHours / 24) : 0;
        const e = Math.abs(r) + relE;
        return (
          <TimeRef
            ep={8}
            t={12}
            dec={0}
            dateText="today"
            read={12 + (r + relE) / 3600}
            gapSec={e}
            metresPerSec={C}
            distWord="of range"
            gapNote={i === 4 ? "satellite A's clock" : i >= 2 ? "your receiver's clock" : undefined}
            rate={i === 4 ? REL_PER_DAY : 3e-9}
            hi="clock"
          />
        );
      }}
      equation={
        <Working label="Range from a time delay">
          <Eq>
            <Sym role="sky">d</Sym> = <Sym>c</Sym> · (Δ<Sym>t</Sym> − <Sym role="red">b</Sym>)
          </Eq>
          <Eq>
            <Sym role="red">b</Sym> left over: <Known when={step >= 2} role="red">{ns(residual)}</Known> × <V>c</V> ={' '}
            <Known when={step >= 2} role="red">{fmtDist(Math.abs(residual) * C)}</Known> on every range
          </Eq>
          {step === 4 && (
            <Eq>
              A gains <V role="red">{fmtSec(REL_PER_DAY * (rel / 24))}</V> = <V role="red">{fmtDist(C * REL_PER_DAY * (rel / 24))}</V>
              ; the fix is <V role="red">{posErr === null ? '?' : fmtDist(posErr)}</V> out
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The satellites are 20,200 km up (Ashby), so near you each range sphere is flat: a straight line at right angles to
          the satellite. c = 299,792,458 m/s; 38 µs a day: GPS World and Ashby (net of +45 µs from weaker gravity and −7 µs
          from speed). Real receivers solve in three dimensions with four or more satellites.
        </>
      }
    />
  );
}
