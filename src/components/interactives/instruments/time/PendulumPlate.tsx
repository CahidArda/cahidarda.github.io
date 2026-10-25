/**
 * Episode 5: the pendulum clock, as a procedure. 1 Set the length · 2 Watch the escapement
 * (lock, impulse, drop: a magnified close-up of the real contact) · 3 Make it beat seconds ·
 * 4 Warm the rod (where it fails on land).
 * Views: the 3D movement (./PendulumModel.tsx) and a 2D elevation with a ×9 close-up, both
 * driven by the same deadbeat solver (./deadbeat.ts). Numbers: ./pendulumGeom.ts.
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
import { AMP, PIVOT, initialState, stepMovement, type Pt } from './deadbeat';
import { escScene } from './escScene';
import { fmtSec, signed } from './fmt';
import { DL_MAX, L_MAX, L_MIN, L_STEP, SECONDS_L, errPerDay, isSeconds, period, snapL } from './pendulumGeom';
import { ARMS, Movement3D } from './PendulumModel';
import TimeRef from './TimeRef';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

/** Runs the deadbeat solver in React state (for the 2D view) while visible. */
function useMovement(L: number, run: boolean) {
  const [st, setSt] = useState(() => initialState());
  const raf = useRef(0);
  const Lref = useRef(L);
  Lref.current = L;
  useEffect(() => {
    if (!run) return;
    let last = 0;
    const f = (now: number) => {
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      // the close-up runs at a quarter speed so the reader can see lock, impulse and drop
      setSt((s) => {
        const n = stepMovement(s, dt / 4, period(Lref.current));
        return { phase: n.phase, w: n.w };
      });
      raf.current = requestAnimationFrame(f);
    };
    raf.current = requestAnimationFrame(f);
    return () => cancelAnimationFrame(raf.current);
  }, [run]);
  return st;
}

const rotPt = (p: Pt, c: Pt, a: number): Pt => {
  const k = Math.cos(a);
  const n = Math.sin(a);
  return [c[0] + (p[0] - c[0]) * k - (p[1] - c[1]) * n, c[1] + (p[0] - c[0]) * n + (p[1] - c[1]) * k];
};

function CloseUp({ L, st }: { L: number; st: { phase: number; w: number } }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const a = AMP * Math.sin(st.phase);
  const g = escScene(st.w, a, L, W);
  const pts = (p: Pt[], f = (q: Pt) => g.P(q)) => p.map((q) => { const v = f(q); return `${v.x.toFixed(1)},${v.y.toFixed(1)}`; }).join(' ');
  const c = g.contact;
  const z = g.zoom!;
  const zf = (q: Pt) => z.f(q);
  const clip = `cu-${W}`;
  const say = c.face === 'lock' ? 'Locked: the tooth rests on the locking face; the wheel does not move.' : 'Impulse: the tooth slides along the slanted face and pushes the pallet.';
  const arms = ARMS.map((arm) => arm.map((q) => rotPt(q, PIVOT, a)));
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Deadbeat escapement, ${say}`}>
        {/* main elevation */}
        <path d={`M ${g.pivot.x + Math.sin(g.arc.a0) * g.arc.r} ${g.pivot.y + Math.cos(g.arc.a0) * g.arc.r} A ${g.arc.r} ${g.arc.r} 0 0 0 ${g.pivot.x + Math.sin(g.arc.a1) * g.arc.r} ${g.pivot.y + Math.cos(g.arc.a1) * g.arc.r}`} style={ink('red', 1)} />
        <line x1={g.pivot.x} y1={g.pivot.y} x2={g.bob.x} y2={g.bob.y} style={ink('brass', 1.4)} />
        <circle cx={g.bob.x} cy={g.bob.y} r={g.bobR} style={{ ...ink('brass', 1), fill: C.paper }} />
        <polygon points={pts(g.wheel)} style={{ ...ink('ink', 0.9), fill: C.paper }} />
        <circle cx={g.centre.x} cy={g.centre.y} r={g.hubR * g.s} style={ink('ink', 0.7)} />
        {arms.map((arm, i) => (
          <polygon key={i} points={pts(arm)} style={{ ...ink('ink', 0.9), fill: C.paper }} />
        ))}
        {g.pals.map((p, i) => (
          <polygon key={i} points={pts(p)} style={{ ...ink(i === c.pallet ? 'red' : 'ink', 1), fill: C.paper }} />
        ))}
        <circle cx={g.pivot.x} cy={g.pivot.y} r={5} style={{ ...ink('ink', 1), fill: C.paper }} />
        {/* the close-up window and its source */}
        <circle cx={g.P(c.point).x} cy={g.P(c.point).y} r={z.r / g.Z} style={ink('soft', 0.8)} />
        <line x1={g.P(c.point).x} y1={g.P(c.point).y} x2={z.cx} y2={z.cy} style={{ ...ink('soft', 0.6), strokeDasharray: '3 3' }} />
        <defs>
          <clipPath id={clip}>
            <circle cx={z.cx} cy={z.cy} r={z.r} />
          </clipPath>
        </defs>
        <circle cx={z.cx} cy={z.cy} r={z.r} style={{ ...ink('ink', 1.2), fill: C.paper }} />
        <g clipPath={`url(#${clip})`}>
          <polygon points={pts(g.wheel, zf)} style={{ ...ink('ink', 1.1), fill: C.deep }} />
          {g.pals.map((p, i) => (
            <polygon key={i} points={pts(p, zf)} style={{ ...ink(i === c.pallet ? 'red' : 'ink', 1.3), fill: C.paper }} />
          ))}
          <circle cx={z.cx} cy={z.cy} r={3} style={{ fill: C.red }} />
        </g>
        <InkText x={z.cx} y={z.cy + z.r + 14} anchor="middle" size={10.5} role="soft">
          {`× ${g.Z}, at the contact`}
        </InkText>
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={11.5} role={c.face === 'lock' ? 'ink' : 'red'}>
          {say}
        </InkText>
      </svg>
    </div>
  );
}

export default function PendulumPlate() {
  const [step, setStep] = useState(0);
  const [L, setL] = useState(0.5);
  const [moved, setMoved] = useState(false);
  const [dL, setDL] = useState(0);
  const [inRef, inView] = useInView<HTMLElement>();
  const reduced = useReducedMotion();
  const st = useMovement(L, inView && !reduced && step === 1);
  const dLAt = (i: number) => (i === 3 ? dL : 0);
  const err = errPerDay(L, dLAt(step));
  const tTrue = 15;
  const since = 3;

  const lengthSlider = (
    <Slider label="pendulum length L" value={L} min={L_MIN} max={L_MAX} step={L_STEP} onChange={(v) => { setL(snapL(v)); setMoved(true); }} format={(v) => `${v.toFixed(3)} m`} width="11rem" />
  );
  const steps: StepDef[] = [
    {
      id: 'length',
      title: 'Set the length',
      text: (
        <>
          A pendulum swings at its own pace, set by its length and nothing else you can easily change: not the weight of
          the bob, not (for small swings) how far it swings.
        </>
      ),
      controls: lengthSlider,
      gated: true,
      done: moved,
      hint: 'Change the length',
    },
    {
      id: 'escape',
      title: 'Watch the escapement',
      text: (
        <>
          George Graham&apos;s deadbeat escapement (1715): a tooth lands on a locking face, an arc about the anchor&apos;s
          pivot, so the wheel stands dead still while the pendulum swings on. Near the middle of the swing the tooth slips
          onto the slanted face and gives a small push, then drops onto the other pallet. Half a tooth per swing.
        </>
      ),
    },
    {
      id: 'seconds',
      title: 'Make it beat seconds',
      text: (
        <>
          The dial is geared for one beat a second, a full swing in 2 s, which makes the 30-tooth wheel turn once a minute.
          Find the length that does it.
        </>
      ),
      controls: lengthSlider,
      gated: true,
      done: isSeconds(L),
      hint: 'Find a 2 s period',
    },
    {
      id: 'heat',
      title: 'Warm the rod',
      text: (
        <>
          On a warm day a steel rod grows. A millimetre is enough to matter: the clock that kept perfect time in the
          morning now loses about {fmtSec(Math.abs(errPerDay(SECONDS_L, 1)))} a day for each millimetre.
        </>
      ),
      controls: <Slider label="rod longer by" value={dL} min={0} max={DL_MAX} step={0.1} onChange={setDL} format={(v) => `${v.toFixed(1)} mm`} width="9rem" />,
    },
  ];

  return (
    <StepPlate
      rootRef={inRef}
      n={5}
      title="How the rod length sets the tick"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (i === 1 ? <CloseUp L={L} st={st} /> : <Movement3D L={L + dLAt(i) / 1000} />)}
      viewLabel="The movement"
      reference={(i) => {
        const on = i >= 2;
        const e = errPerDay(L, dLAt(i)) * (since / 24);
        return (
          <TimeRef
            ep={4}
            t={tTrue}
            dec={0}
            dateText="at the equinox"
            read={on ? tTrue + e / 3600 : null}
            gapSec={on ? Math.abs(e) : null}
            gapNote={on ? `set at noon, ${since} h ago` : undefined}
            noRead="not geared to seconds yet"
            rate={on && Math.abs(errPerDay(L, dLAt(i))) > 1e-3 ? Math.abs(errPerDay(L, dLAt(i))) : null}
            hi={on ? 'clock' : null}
          />
        );
      }}
      equation={
        <Working label="Period of the pendulum">
          <Eq>
            <Sym role="red">T</Sym> = 2π √(<Sym role="brass">L</Sym> ÷ <Sym>g</Sym>)
          </Eq>
          <Eq>
            <V role="red">{`${period(L + dLAt(step) / 1000).toFixed(4)} s`}</V> = 2π √(
            <V role="brass">{`${(L + dLAt(step) / 1000).toFixed(4)} m`}</V> ÷ <V>9.807 m/s²</V>)
          </Eq>
          <Eq>
            error = 86 400 s × (2 s ÷ <Sym role="red">T</Sym> − 1) ={' '}
            <Known when={step >= 2} role="red">{`${signed(err, (x) => fmtSec(x, 3))} a day`}</Known>
          </Eq>
        </Working>
      }
      caption={
        <>
          The escape wheel, anchor and pallets are drawn to the solver&apos;s geometry: the pallets span 7½ teeth, each
          locking face is an arc about the anchor&apos;s pivot, and tooth and pallet meet at a computed contact (within
          2 millionths of the wheel&apos;s radius, checked every frame). The 3D pendulum is shortened so 4 m still fits; the
          close-up runs at a quarter speed. T and the error use g = 9.80665 m/s².
        </>
      }
    />
  );
}
