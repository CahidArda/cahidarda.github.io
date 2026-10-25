/**
 * Episode 7: the quartz watch, as a procedure. 1 Squeeze the crystal · 2 Let it ring ·
 * 3 Halve it fifteen times · 4 Warm or chill it (where it fails).
 * View: the tuning-fork crystal (its deflection drawn slowed 4096 times) above the divider
 * chain. Numbers: ./quartzGeom.ts.
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
import { fmtSec, sig, signed } from './fmt';
import TimeRef from './TimeRef';
import { ASTRON_PER_DAY, F0, SHOW_SLOW, STAGES, T_MAX, T_MIN, TURNOVER, chainLayout, errPerDay, hzShift, hzText } from './quartzGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function useTime(run: boolean) {
  const [t, setT] = useState(0.1);
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

function Crystal({ step, t, stages, squeezed, temp }: { step: number; t: number; stages: number; squeezed: boolean; temp: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const ch = chainLayout(W);
  const ringing = step >= 1;
  const amp = ringing ? 5 : squeezed ? 3 : 0;
  const d = ringing ? amp * Math.sin(2 * Math.PI * (F0 / SHOW_SLOW) * t) : amp;
  const cx = W / 2;
  // the fork: two tines rising from a base, bending apart and together
  const tine = (s: 1 | -1) => `M ${cx + s * 14} 92 C ${cx + s * 14} 60, ${cx + s * (14 + d * 0.5)} 40, ${cx + s * (14 + d)} 18`;
  const pulse = (k: number) => (k <= stages ? Math.sin(2 * Math.PI * (F0 / SHOW_SLOW / 2 ** k) * t) > 0 : false);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${ch.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Quartz tuning fork and a ${STAGES}-stage divider; ${stages} stages connected`}>
        <rect x={cx - 28} y={88} width={56} height={10} style={{ ...ink('ink', 1), fill: C.paper }} />
        <path d={tine(-1)} style={ink('sky', 6)} />
        <path d={tine(1)} style={ink('sky', 6)} />
        <path d={tine(-1)} style={ink('paper', 3.4)} />
        <path d={tine(1)} style={ink('paper', 3.4)} />
        {(squeezed || ringing) && (
          <>
            <InkText x={cx - 34} y={50} anchor="end" kind="num" size={14} role="red">+</InkText>
            <InkText x={cx + 34} y={50} kind="num" size={14} role="red">−</InkText>
          </>
        )}
        <InkText x={12} y={18} size={11.5} role="soft">quartz tuning fork, about 4 mm long</InkText>
        <InkText x={12} y={34} size={10.5} role="soft">{ringing ? `ringing at ${hzText(F0 + hzShift(temp))} (shown 4096× slower)` : squeezed ? 'squeezed: a voltage appears' : 'at rest'}</InkText>
        {ch.items.map((b) => {
          const on = b.k <= stages && step >= 2;
          return (
            <g key={b.k} style={{ opacity: on || b.k === 0 ? 1 : 0.35 }}>
              <rect x={b.x} y={b.y} width={b.w} height={b.h} style={{ ...ink(b.k === STAGES && on ? 'red' : 'ink', b.k === STAGES && on ? 1.6 : 0.9), fill: pulse(b.k) && (on || (b.k === 0 && ringing)) ? C.deep : C.paper }} />
              <InkText x={b.x + b.w / 2} y={b.y + b.h / 2 + 4} kind="num" anchor="middle" size={W < 560 ? 9.5 : 10} role={b.k === STAGES && on ? 'red' : 'ink'} halo={false}>
                {hzText(b.f)}
              </InkText>
            </g>
          );
        })}
        <InkText x={W / 2} y={ch.H - 10} anchor="middle" size={11.5} role={stages === STAGES && step >= 2 ? 'red' : 'soft'}>
          {step < 2 ? 'The divider chain: each box halves the frequency.' : stages === STAGES ? 'One pulse a second: step the second hand.' : `${stages} of ${STAGES} halvings: ${hzText(F0 / 2 ** stages)}.`}
        </InkText>
      </svg>
    </div>
  );
}

export default function QuartzPlate() {
  const [step, setStep] = useState(0);
  const [squeezed, setSqueezed] = useState(false);
  const [stages, setStages] = useState(0);
  const [temp, setTemp] = useState(TURNOVER);
  const [inRef, inView] = useInView<HTMLElement>();
  const reduced = useReducedMotion();
  const t = useTime(inView && !reduced && step >= 1);
  const tempAt = (i: number) => (i === 3 ? temp : TURNOVER);
  const e = errPerDay(tempAt(step));

  const steps: StepDef[] = [
    {
      id: 'squeeze',
      title: 'Squeeze the crystal',
      text: <>Quartz is piezoelectric: bend it and a voltage appears across it; put a voltage on it and it bends.</>,
      controls: (
        <button type="button" className="in-btn" onClick={() => setSqueezed(true)} disabled={squeezed}>
          {squeezed ? 'Squeezed' : 'Squeeze it'}
        </button>
      ),
      gated: true,
      done: squeezed,
      hint: 'Squeeze the crystal',
    },
    {
      id: 'ring',
      title: 'Let it ring',
      text: (
        <>
          Feed that voltage back through an amplifier and the fork rings at its own frequency, like a bell that never stops.
          Watch crystals are cut to 32,768 Hz. In 1927 Warren Marrison and J. W. Horton built the first quartz clock at Bell
          Labs.
        </>
      ),
    },
    {
      id: 'divide',
      title: 'Halve it fifteen times',
      text: (
        <>
          32,768 is 2<sup>15</sup>. Each flip-flop in the chain halves the frequency, so after fifteen of them the chip sends
          one pulse a second to the motor that steps the hand.
        </>
      ),
      controls: <Slider label="halvings connected" value={stages} min={0} max={STAGES} step={1} onChange={setStages} format={(x) => `${x}`} width="10rem" />,
      gated: true,
      done: stages === STAGES,
      hint: 'Connect all fifteen',
    },
    {
      id: 'temp',
      title: 'Warm or chill it',
      text: (
        <>
          The fork is cut to ring truest near 25 °C, about wrist temperature. Leave the watch on a cold windowsill or a hot
          dashboard and it rings a little slow; and every crystal ages.
        </>
      ),
      controls: <Slider label="temperature" value={temp} min={T_MIN} max={T_MAX} step={1} onChange={setTemp} format={(x) => `${x} °C`} width="10rem" />,
    },
  ];

  return (
    <StepPlate
      rootRef={inRef}
      n={7}
      title="Inside a quartz watch"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Crystal step={i} t={t} stages={i >= 2 ? stages : 0} squeezed={squeezed || i > 0} temp={tempAt(i)} />}
      viewLabel="The crystal and the divider"
      reference={(i) => {
        const ei = errPerDay(tempAt(i));
        const on = i >= 2 && stages === STAGES;
        const days = 30;
        return (
          <TimeRef
            ep={6}
            t={12}
            dec={0}
            dateText="a month after setting"
            read={on ? 12 + (ei * days) / 3600 : null}
            gapSec={on ? Math.abs(ei * days) : null}
            gapNote={on ? `30 days at ${tempAt(i)} °C` : undefined}
            noRead="no second pulses yet"
            rate={on && Math.abs(ei) > 1e-6 ? Math.abs(ei) : null}
            hi={on ? 'gauge' : null}
          />
        );
      }}
      equation={
        <Working label="From frequency error to seconds a day">
          <Eq>
            <Sym role="red">error</Sym> = 86 400 s × Δ<Sym role="sky">f</Sym> ÷ <Sym role="sky">f</Sym>
          </Eq>
          <Eq>
            <Known when={step >= 3} role="red">{`${signed(e, (x) => fmtSec(x, 2))} a day`}</Known> = 86 400 s ×{' '}
            <Known when={step >= 3} role="sky">{`${sig(hzShift(tempAt(step)), 2)} Hz`}</Known> ÷ <V role="sky">32 768 Hz</V>
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em' }}>{`Seiko's Astron (1969): about 5 s a month, ${sig(ASTRON_PER_DAY, 2)} s a day.`}</span>
          </Eq>
        </Working>
      }
      caption={
        <>
          The temperature curve is the textbook one for a 32 kHz tuning-fork crystal, −0.04 parts per million per °C²
          away from 25 °C (Wikipedia, Crystal oscillator); a real watch adds its own offset and ageing. The reference shows
          the watch a month after setting.
        </>
      }
    />
  );
}
