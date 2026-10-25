/**
 * Episode 2: the water clock (clepsydra), as a procedure. 1 Fill it at sunset · 2 Let it run ·
 * 3 Slope the walls · 4 Keep the head constant (Ctesibius) · 5 Where it fails.
 * View: the vessel with an evenly spaced hour scale (what the reader reads) and the true hour
 * marks (brass), beside a chart of level against time. Physics: ./waterGeom.ts.
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
  V,
  Working,
  useHoverPreview,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import { declination } from './astro';
import { fmtSec, hm } from './fmt';
import TimeRef from './TimeRef';
import { NIGHT, SUNSET, chartCurve, trueMarks, vesselFrame, waterClock, worstError, type Vessel, type WaterClock } from './waterGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const EQUINOX = 80;

function Vessel({ w, full }: { w: WaterClock; full: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const v = w.v;
  const f = vesselFrame(W, v);
  const h = full ? w.h : 0;
  const recv = v === 'head';
  // outline of the vessel (the bowl, or for 'head' the receiving jar)
  const side = (s: 1 | -1) =>
    Array.from({ length: 21 }, (_, i) => i / 20).map((u) => `${(f.cx + s * f.rAt(u)).toFixed(1)},${f.yAt(u).toFixed(1)}`);
  const outline = [...side(-1).reverse(), ...side(1)].join(' ');
  const water = (() => {
    if (h <= 0) return '';
    const n = 20;
    const L = Array.from({ length: n + 1 }, (_, i) => (i / n) * h).map((u) => `${(f.cx - f.rAt(u)).toFixed(1)},${f.yAt(u).toFixed(1)}`);
    const R = Array.from({ length: n + 1 }, (_, i) => (i / n) * h).map((u) => `${(f.cx + f.rAt(u)).toFixed(1)},${f.yAt(u).toFixed(1)}`);
    return [...L, ...R.reverse()].join(' ');
  })();
  const marks = trueMarks(v, w.clog);
  const curve = chartCurve(v, W, w.clog);
  const scaleX = f.cx + f.rAt(1) + 6;
  const evenHours = Array.from({ length: NIGHT + 1 }, (_, k) => k);
  const yEven = (k: number) => f.yAt(recv ? k / NIGHT : 1 - k / NIGHT);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${f.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Water clock, ${recv ? 'constant head' : v === 'slope' ? 'sloping walls' : 'straight walls'}: the scale reads ${hm(w.read)} at a true ${hm(w.now)}`}>
        <InkText x={14} y={18} size={11.5} role="soft">
          {recv ? 'receiving jar, fed at a constant head' : v === 'slope' ? 'bowl with sloping walls' : 'bowl with straight walls'}
        </InkText>
        {/* header tank for Ctesibius */}
        {recv && (
          <g>
            <rect x={f.cx - 22} y={24} width={44} height={14} style={{ ...ink('ink', 1), fill: C.paper }} />
            <rect x={f.cx - 21} y={26} width={42} height={11} style={{ fill: C.sky, opacity: 0.35 }} />
            <line x1={f.cx} x2={f.cx} y1={38} y2={f.yAt(h) - 1} style={{ ...ink('sky', 1.2), strokeDasharray: '2 2' }} />
            <InkText x={f.cx + 26} y={34} size={10} role="soft">full tank</InkText>
          </g>
        )}
        <polygon points={outline} style={{ fill: C.paper, stroke: 'none' }} />
        {water && <polygon points={water} style={{ fill: C.sky, opacity: 0.35 }} />}
        <polyline points={side(-1).join(' ')} style={ink('ink', 1.3)} />
        <polyline points={side(1).join(' ')} style={ink('ink', 1.3)} />
        <line x1={f.cx - f.rAt(0)} x2={f.cx + f.rAt(0)} y1={f.yBot} y2={f.yBot} style={ink('ink', 1.3)} />
        {/* the hole and its jet */}
        {!recv && (
          <>
            <circle cx={f.cx + f.rAt(0) - 4} cy={f.yBot - 4} r={1.8} style={{ fill: C.ink }} />
            {h > 0.002 && <path d={`M ${f.cx + f.rAt(0)} ${f.yBot - 4} q ${8 + 22 * Math.sqrt(h)} 0 ${10 + 24 * Math.sqrt(h)} 18`} style={ink('sky', 1.2)} />}
          </>
        )}
        {/* the evenly spaced scale (what you read) */}
        {evenHours.map((k) => (
          <g key={k}>
            <line x1={scaleX} x2={scaleX + (k % 3 ? 4 : 8)} y1={yEven(k)} y2={yEven(k)} style={ink('soft', 0.8)} />
            {k % 3 === 0 && (
              <InkText x={scaleX + 11} y={yEven(k)} kind="num" baseline="central" size={9.5} role="soft" halo={false}>{`${k}h`}</InkText>
            )}
          </g>
        ))}
        {/* where the level really is at each hour (brass) */}
        {marks.map((m, k) => (
          <line key={k} x1={f.cx - f.rAt(m) - 2} x2={f.cx - f.rAt(m) + 9} y1={f.yAt(m)} y2={f.yAt(m)} style={ink('brass', k % 3 ? 0.9 : 1.6)} />
        ))}
        {/* the water line, read across to the scale */}
        {full && (
          <line x1={f.cx - f.rAt(h)} x2={scaleX + 8} y1={f.yAt(h)} y2={f.yAt(h)} style={ink('red', 1.4)} />
        )}

        {/* chart: level against time */}
        <line x1={f.c.x0} x2={f.c.x0} y1={f.c.y0} y2={f.c.y1} style={ink('ink', 0.9)} />
        <line x1={f.c.x0} x2={f.c.x1} y1={f.c.y1} y2={f.c.y1} style={ink('ink', 0.9)} />
        {[0, 3, 6, 9, 12].map((k) => (
          <InkText key={k} x={f.xc(k)} y={f.c.y1 + 13} kind="num" anchor="middle" size={9.5} role="soft" halo={false}>{`${k}h`}</InkText>
        ))}
        <InkText x={f.c.x0} y={f.c.y0 - 8} size={10.5} role="soft">water level</InkText>
        <InkText x={f.c.x1} y={f.c.y1 + 27} anchor="end" size={10.5} role="soft">hours since sunset</InkText>
        <line x1={f.xc(0)} y1={f.yc(recv ? 0 : 1)} x2={f.xc(NIGHT)} y2={f.yc(recv ? 1 : 0)} style={{ ...ink('soft', 0.8), strokeDasharray: '4 3' }} />
        <polyline points={curve.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={ink('sky', 1.5)} />
        {full && (
          <>
            <line x1={f.xc(w.t)} x2={f.xc(w.t)} y1={f.c.y1} y2={f.yc(h)} style={{ ...ink('ink', 0.7), strokeDasharray: '2 2' }} />
            <circle cx={f.xc(w.t)} cy={f.yc(h)} r={3.5} style={{ fill: C.red }} />
          </>
        )}
        <InkText x={W / 2} y={f.H - 10} anchor="middle" size={11.5} role={full ? 'red' : 'soft'}>
          {!full
            ? 'Empty: fill it at sunset.'
            : w.empty
              ? 'The bowl is empty: the night is over, or the clock is.'
              : `Scale says ${hm(w.read)}; true time ${hm(w.now)}.`}
        </InkText>
      </svg>
    </div>
  );
}

export default function WaterClockPlate() {
  const [step, setStep] = useState(0);
  const [full, setFull] = useState(false);
  const [t, setT] = useState(3);
  const [ran, setRan] = useState(false);
  const [selV, setV] = useState<Vessel>('cyl');
  const [vv, bindV] = useHoverPreview<Vessel>(selV);
  const [clog, setClog] = useState(0);
  // a step stays done once its action was taken, even if a later step changes the same control
  const [triedSlope, setTriedSlope] = useState(false);
  const [triedHead, setTriedHead] = useState(false);
  useEffect(() => {
    if (selV === 'slope') setTriedSlope(true);
    if (selV === 'head') setTriedHead(true);
  }, [selV]);
  const vesselAt = (i: number): Vessel => (i <= 1 ? 'cyl' : i === 2 ? (vv === 'head' ? 'slope' : vv) : i === 3 ? (vv === 'cyl' ? 'head' : vv) : vv);
  const clogAt = (i: number) => (i === 4 ? clog : 0);
  const wAt = (i: number) => waterClock(vesselAt(i), t, clogAt(i));
  const w = wAt(step);
  const dec = declination(EQUINOX);
  const isFull = full || step > 0;

  const timeSlider = (
    <Slider label="hours since sunset" value={t} min={0} max={NIGHT} step={0.25} onChange={(v) => { setT(v); setRan(true); }} format={(v) => `${v.toFixed(2)} h`} width="10rem" />
  );
  const vesselSeg = (opts: Vessel[]) => (
    <Segmented
      label="vessel"
      options={opts.map((o) => ({ value: o, label: o === 'cyl' ? 'straight walls' : o === 'slope' ? 'sloping walls' : 'constant head' }))}
      value={selV}
      onChange={setV}
      preview={bindV}
    />
  );
  const steps: StepDef[] = [
    {
      id: 'fill',
      title: 'Fill it at sunset',
      text: <>A stone bowl with a small hole near the bottom. Fill it to the brim as the sun sets; the scale inside is marked in twelve equal steps, one per hour of the night.</>,
      controls: (
        <button type="button" className="in-btn" onClick={() => setFull(true)} disabled={full}>
          {full ? 'Full' : 'Fill the bowl'}
        </button>
      ),
      gated: true,
      done: full,
      hint: 'Fill the bowl',
    },
    {
      id: 'run',
      title: 'Let it run',
      text: (
        <>
          A full bowl pushes water out fast; a nearly empty one only dribbles. So the level falls quickly at first and
          slowly at the end, and an evenly marked scale runs ahead, then catches up. The brass marks show where the
          level really is at each hour.
        </>
      ),
      controls: timeSlider,
      gated: true,
      done: ran,
      hint: 'Move the time',
    },
    {
      id: 'slope',
      title: 'Slope the walls',
      text: (
        <>
          Make the bowl wider at the top. The Karnak clock (Amenhotep III, c. 1390 to 1350 BCE) is 46 cm across at the
          top and 26 cm at the bottom: the wide top drains slowly, so the level falls more evenly.
        </>
      ),
      controls: vesselSeg(['cyl', 'slope']),
      gated: true,
      done: triedSlope,
      hint: 'Choose the sloping walls',
    },
    {
      id: 'head',
      title: 'Keep the pressure constant',
      text: (
        <>
          Ctesibius of Alexandria (c. 270 BCE) turned it round: a header tank kept brim-full by an overflow feeds a
          second jar at a constant head, so the water arrives at a constant rate and a float in the jar rises evenly.
        </>
      ),
      controls: (
        <>
          {vesselSeg(['cyl', 'slope', 'head'])}
          {timeSlider}
        </>
      ),
      gated: true,
      done: triedHead,
      hint: 'Choose the constant head',
    },
    {
      id: 'fail',
      title: 'Where it fails',
      text: (
        <>
          Grit narrows the hole, cold water flows a little slower, and the whole night drifts with it. And someone still
          has to start it at the right moment, by the sun.
        </>
      ),
      controls: (
        <>
          <Slider label="hole narrowed by" value={clog} min={0} max={0.3} step={0.01} onChange={setClog} format={(v) => `${Math.round(v * 100)}%`} width="8rem" />
          {timeSlider}
        </>
      ),
    },
  ];

  const worst = worstError(w.v, clogAt(step));
  return (
    <StepPlate
      n={2}
      title="Keeping time with water"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Vessel w={wAt(i)} full={isFull || i > 0} />}
      viewLabel="The water clock"
      reference={(i) => {
        const wi = wAt(i);
        const on = isFull || i > 0;
        return (
          <TimeRef
            ep={1}
            t={(SUNSET + t) % 24}
            dec={dec}
            dateText="at the equinox"
            read={on && !wi.empty ? wi.read % 24 : null}
            gapSec={on && !wi.empty ? Math.round(Math.abs(wi.errSec)) : null}
            gapNote={on && !wi.empty ? (wi.errSec > 0 ? 'running ahead' : wi.errSec < 0 ? 'running behind' : undefined) : undefined}
            noRead={on ? 'empty bowl' : 'not filled yet'}
            hi={i === 0 ? 'sky' : 'clock'}
          />
        );
      }}
      equation={
        <Working label="How fast the level falls">
          <Eq>
            speed out = √(2 <Sym>g</Sym> <Sym role="sky">h</Sym>)
          </Eq>
          <Eq>
            {w.v === 'head' ? (
              <>
                level rises at <V role="sky">{`1/${NIGHT}`}</V> of the jar per hour, at every hour
              </>
            ) : (
              <>
                water depth <Known when={isFull} role="sky">{`${Math.round(w.h * 100)}%`}</Known>, so it flows{' '}
                <Known when={isFull} role="sky">{`${Math.round(Math.sqrt(w.h) * 100)}%`}</Known> as fast as when full
              </>
            )}
          </Eq>
          <Eq>
            worst error tonight: <Known when={step >= 1} role="red">{worst < 30 ? 'none' : fmtSec(worst)}</Known>
          </Eq>
        </Working>
      }
      caption={
        <>
          Each vessel is sized to empty (or fill) in exactly 12 hours, an equinox night at Istanbul. Flow from
          Torricelli&apos;s law; the sloping bowl uses the Karnak clock&apos;s outside diameters (46 and 26 cm, McMaster
          Ancient Egypt Archive). The scale is evenly spaced on purpose, to show why it cannot be.
        </>
      }
    />
  );
}
