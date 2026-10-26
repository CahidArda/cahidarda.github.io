/**
 * Episode 4: verge and foliot, as a procedure. 1 Hang the weight · 2 Watch it escape ·
 * 3 Regulate with the weights · 4 Change the drive (where it fails).
 * View: the foliot from above (bar, regulating weights, verge) and the crown wheel's teeth
 * unrolled under the two pallets (a schematic, not to scale), animated at the model's period.
 * Physics: ./vergeGeom.ts.
 */
import { useEffect, useRef, useState } from 'react';
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
  useInView,
  useMeasuredWidth,
  useReducedMotion,
  type Role,
  type StepDef,
} from '../shared';
import { fmtSec, hm, signed } from './fmt';
import TimeRef from './TimeRef';
import {
  CROWN_TEETH,
  DRIVE_MAX,
  DRIVE_MIN,
  KEEP_TOL,
  R_MAX,
  R_MIN,
  R_STEP,
  T_SET,
  crownTeeth,
  errPerDay,
  foliotAngle,
  foliotScene,
  period,
} from './vergeGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

/** Clock time for the animation, advancing only while visible and motion is allowed. */
function useClock(run: boolean) {
  const [t, setT] = useState(0.7);
  const raf = useRef(0);
  useEffect(() => {
    if (!run) return;
    let last = 0;
    const f = (now: number) => {
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      setT((v) => v + dt);
      raf.current = requestAnimationFrame(f);
    };
    raf.current = requestAnimationFrame(f);
    return () => cancelAnimationFrame(raf.current);
  }, [run]);
  return t;
}

function Movement({ r, drive, t, hung }: { r: number; drive: number; t: number; hung: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const T = period(r, drive);
  const th = hung ? foliotAngle(t, T) : 0;
  const g = foliotScene(th, r, W);
  const narrow = W < 420;
  // the unrolled crown wheel strip, under the foliot on phones, to its right on wide views
  const sx0 = narrow ? 16 : W * 0.58;
  const sx1 = W - 16;
  const sy = narrow ? g.H + 34 : 50;
  const pitch = 26;
  const shift = hung ? crownTeeth(t, T) * pitch : 0;
  const rowTop = sy;
  const rowBot = sy + 70;
  const mid = (sx0 + sx1) / 2;
  const teeth = (y: number, dir: 1 | -1) => {
    const out: string[] = [];
    for (let k = -8; k <= 8; k++) {
      const x = mid + k * pitch + dir * (shift % pitch);
      if (x < sx0 + 4 || x > sx1 - 14) continue;
      // a crown-wheel tooth in profile: a sloped back, a straight front, pointing at the verge
      out.push(dir === 1 ? `M ${x} ${y} L ${x + 12} ${y + (y < sy + 35 ? 14 : -14)} L ${x + 12} ${y}` : `M ${x + 12} ${y} L ${x} ${y + (y < sy + 35 ? 14 : -14)} L ${x} ${y}`);
    }
    return out.join(' ');
  };
  // pallets: flags on the verge, 90° apart, seen edge-on as they turn with the foliot
  const pal = (y: number, phase: number) => {
    const a = th + phase;
    const len = 22;
    return { x1: mid, y1: y, x2: mid + len * Math.sin(a), y2: y + (y < sy + 35 ? 1 : -1) * len * 0.35 * Math.cos(a) };
  };
  const pTop = pal(rowTop + 14, Math.PI / 4);
  const pBot = pal(rowBot - 14, -Math.PI / 4);
  const H = narrow ? sy + 110 : Math.max(g.H, 150) + 20;
  const err = errPerDay(r, drive);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Verge and foliot, period ${T.toFixed(2)} seconds`}>
        <InkText x={14} y={16} size={11.5} role="soft">the foliot, from above</InkText>
        {/* the swing limits */}
        {[1, -1].map((s) => {
          const a = (s * 40 * Math.PI) / 180;
          return <line key={s} x1={g.cx} y1={g.cy} x2={g.cx + g.L * Math.cos(a)} y2={g.cy - g.L * Math.sin(a)} style={{ ...ink('faint', 0.7), strokeDasharray: '3 3' }} />;
        })}
        <line x1={g.ends[1].x} y1={g.ends[1].y} x2={g.ends[0].x} y2={g.ends[0].y} style={ink('ink', 3)} />
        {g.weights.map((p, i) => (
          <rect key={i} x={p.x - 7} y={p.y - 7} width={14} height={14} transform={`rotate(${(-th * 180) / Math.PI} ${p.x} ${p.y})`} style={{ fill: C.brass, stroke: C.ink, strokeWidth: 1 }} />
        ))}
        <circle cx={g.cx} cy={g.cy} r={6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.4 }} />
        <InkText x={g.cx} y={g.cy + g.L * 0.55 + 22} anchor="middle" size={11} role="brass">
          {`weights at ${Math.round(r * 100)}% of the arm`}
        </InkText>

        {/* the crown wheel, unrolled */}
        <InkText x={sx0} y={sy - 12} size={11.5} role="soft">crown wheel teeth, unrolled</InkText>
        <line x1={sx0} x2={sx1} y1={rowTop} y2={rowTop} style={ink('ink', 1)} />
        <line x1={sx0} x2={sx1} y1={rowBot} y2={rowBot} style={ink('ink', 1)} />
        <path d={teeth(rowTop, 1)} style={{ ...ink('ink', 1), fill: C.deep }} />
        <path d={teeth(rowBot, -1)} style={{ ...ink('ink', 1), fill: C.deep }} />
        <line x1={mid} x2={mid} y1={rowTop - 4} y2={rowBot + 4} style={ink('ink', 2)} />
        <line x1={pTop.x1} y1={pTop.y1} x2={pTop.x2} y2={pTop.y2} style={{ ...ink('red', 3), strokeLinecap: 'round' }} />
        <line x1={pBot.x1} y1={pBot.y1} x2={pBot.x2} y2={pBot.y2} style={{ ...ink('red', 3), strokeLinecap: 'round' }} />
        {hung && (
          <InkText x={sx1} y={rowBot + 18} anchor="end" size={10.5} role="soft">
            {`one tooth per swing, ${T.toFixed(2)} s`}
          </InkText>
        )}
        <InkText x={W / 2} y={H - 6} anchor="middle" size={11.5} role={Math.abs(err) <= KEEP_TOL ? 'ink' : 'red'}>
          {!hung ? 'No weight: nothing moves.' : Math.abs(err) <= KEEP_TOL ? 'Within 15 minutes a day: as good as a foliot gets.' : `${err > 0 ? 'Gains' : 'Loses'} ${fmtSec(Math.abs(err))} a day.`}
        </InkText>
      </svg>
    </div>
  );
}

export default function VergePlate() {
  const [step, setStep] = useState(0);
  const [hung, setHung] = useState(false);
  const [r, setR] = useState(0.4);
  const [drive, setDrive] = useState(1);
  const [inRef, inView] = useInView<HTMLElement>();
  const reduced = useReducedMotion();
  const t = useClock(inView && !reduced && hung);
  const driveAt = (i: number) => (i >= 3 ? drive : 1);
  const T = period(r, driveAt(step));
  const err = errPerDay(r, driveAt(step));
  const keptNow = Math.abs(errPerDay(r, 1)) <= KEEP_TOL;
  const [kept, setKept] = useState(false);
  useEffect(() => {
    if (keptNow) setKept(true);
  }, [keptNow]);
  const tTrue = 15; // the reference: an afternoon, the clock set at noon
  const since = tTrue - 12;

  const steps: StepDef[] = [
    {
      id: 'hang',
      title: 'Hang the weight',
      text: <>A weight on a cord turns the crown wheel. Nothing else drives this clock, and nothing else times it.</>,
      controls: (
        <button type="button" className="in-btn" disabled={hung} onClick={() => setHung(true)}>
          {hung ? 'Weight hung' : 'Hang the weight'}
        </button>
      ),
      gated: true,
      done: hung,
      hint: 'Hang the weight',
    },
    {
      id: 'escape',
      title: 'Watch it escape',
      text: (
        <>
          A tooth pushes the top pallet and swings the foliot one way, until the bottom pallet meets a tooth coming the
          other way and swings it back. One tooth escapes per swing; a {CROWN_TEETH}-tooth wheel turns once every{' '}
          {CROWN_TEETH} swings.
        </>
      ),
    },
    {
      id: 'regulate',
      title: 'Move the weights',
      text: (
        <>
          The foliot&apos;s two small weights set how hard it is to turn. Slide them out and each swing takes longer. Find
          the spot where one swing takes {T_SET} s and the clock keeps time.
        </>
      ),
      controls: <Slider label="regulating weights, along the arm" value={r} min={R_MIN} max={R_MAX} step={R_STEP} onChange={setR} format={(v) => `${Math.round(v * 100)}%`} width="10rem" />,
      gated: true,
      done: kept,
      hint: `Within ${KEEP_TOL / 60} min a day`,
    },
    {
      id: 'fail',
      title: 'Where it fails',
      text: (
        <>
          Now change the push: a heavier weight, a drier pivot, a stiffer cord. The foliot has no rhythm of its own; it
          swings as fast as it is pushed, so the rate follows the drive.
        </>
      ),
      controls: <Slider label="drive, relative to the usual weight" value={drive} min={DRIVE_MIN} max={DRIVE_MAX} step={0.01} onChange={setDrive} format={(v) => `${Math.round(v * 100)}%`} width="10rem" />,
    },
  ];

  return (
      <StepPlate
        rootRef={inRef}
        n={4}
        title="Verge and foliot: a clock that ticks by itself"
        wide
        steps={steps}
        step={step}
        onStep={setStep}
        view={(i) => <Movement r={r} drive={driveAt(i)} t={t} hung={hung || i > 0} />}
        viewLabel="The escapement"
        reference={(i) => {
          const e = errPerDay(r, driveAt(i)) * (since / 24);
          const on = (hung || i > 0) && i >= 2;
          return (
            <TimeRef
              ep={3}
              t={tTrue}
              dec={0}
              dateText="at the equinox"
              read={on ? tTrue + e / 3600 : null}
              gapSec={on ? Math.round(Math.abs(e)) : null}
              gapNote={on ? `set at noon, ${since} h ago` : undefined}
              noRead={i < 2 ? 'not regulated yet' : 'stopped'}
              rate={on ? Math.max(1e-3, Math.abs(errPerDay(r, driveAt(i)))) : null}
              hi={i >= 2 ? 'clock' : null}
            />
          );
        }}
        equation={
          <Working label="Period of the foliot">
            <Eq>
              <Sym role="red">T</Sym> ∝ √(<Sym role="brass">I</Sym> ÷ <Sym>τ</Sym>), <Sym role="brass">I</Sym> = I<sub>bar</sub> + 2<Sym>m</Sym>
              <Sym role="brass">r</Sym>²
            </Eq>
            <Eq>
              <Known when={step >= 1} role="red">{`${T.toFixed(3)} s`}</Known> per swing, drive <V>{`${Math.round(driveAt(step) * 100)}%`}</V>
            </Eq>
            <Eq>
              error: <Known when={step >= 2} role="red">{`${signed(err, (x) => fmtSec(x))} a day`}</Known>
            </Eq>
          </Working>
        }
        caption={
          <>
            A schematic model, not a particular clock: the drive pushes the foliot back toward the middle with a torque of
            fixed size, so the period goes as √(I/τ). Early verge clocks are thought to have erred by one to two hours a
            day; the best, by the mid-1600s, by about 15 minutes (Wikipedia, Verge escapement). The reference reads the
            clock at {hm(tTrue)}, three hours after it was set at noon.
          </>
        }
      />
  );
}
