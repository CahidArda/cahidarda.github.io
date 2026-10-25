/**
 * Episode 6: take the pendulum to sea, then swap in Harrison's balance and spring.
 * 1 Regulate it on land · 2 Put it on a ship (heave) · 3 Sail south (gravity) · 4 Into the warm
 * (heat) · 5 The bill (days vs the Longitude Act's prizes) · 6 Swap in a balance and spring.
 * View: the ship on the sea (the sea and the horizon are the world: they never move with the
 * controls, the ship rides them) with its clock, and the longitude-error chart.
 * Numbers: ./seaGeom.ts.
 */
import { useEffect, useRef, useState } from 'react';
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
  useInView,
  useMeasuredWidth,
  useReducedMotion,
  type Role,
  type StepDef,
} from '../shared';
import { fmtSec, signed, sig } from './fmt';
import TimeRef from './TimeRef';
import {
  DAYS_MAX,
  H4_DAYS,
  HEAVE_MAX,
  JAMAICA,
  M_PER_S_EQ,
  PORTSMOUTH,
  PRIZES,
  VOYAGE_DAYS,
  WAVE_PERIOD,
  Y_TICKS,
  arcminOf,
  chartFrame,
  errorAfter,
  errorCurve,
  gravity,
  kmOf,
  latOnDay,
  prizeFor,
  rateOn,
  tempAt,
  type Clock,
  type Voyage,
} from './seaGeom';

const ink = (role: Role, w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function useTime(run: boolean) {
  const [t, setT] = useState(1.3);
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

function Ship({ v, t, days, showH4 }: { v: Voyage; t: number; days: number; showH4: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const seaY = 92;
  const H = 128;
  // the ship heaves with the waves (amplitude in px proportional to the heave in m)
  const lift = -Math.sin((2 * Math.PI * t) / WAVE_PERIOD * 4) * (v.heave / HEAVE_MAX) * 12; // ×4 speed, for the eye
  const roll = Math.cos((2 * Math.PI * t) / WAVE_PERIOD * 4) * (v.heave / HEAVE_MAX) * 5;
  const cx = W * 0.32;
  const f = chartFrame(W);
  const cur = errorCurve(v, W, days);
  const pend = v.clock === 'pendulum';
  // the clock's oscillator, drawn inside the cabin
  const swing = pend
    ? Math.sin(t * Math.PI) * 0.22 + (v.heave / HEAVE_MAX) * 0.35 * Math.sin(t * 1.7 + 0.6) // kicked by the sea
    : Math.sin(t * Math.PI * 5) * 1.6; // H4: 5 beats a second, big swing
  const h4 = { ...v, clock: 'balance' as Clock };
  const h4Curve = showH4 ? errorCurve(h4, W, Math.min(days, DAYS_MAX)) : [];
  const a = arcminOf(errorAfter(v, days));
  const offTop = a > 3000;
  const phi = latOnDay(v, days);
  const wave = Array.from({ length: 41 }, (_, i) => {
    const x = (i / 40) * W;
    return `${x.toFixed(1)},${(seaY + Math.sin(i * 0.9 + t * 3) * (1 + (v.heave / HEAVE_MAX) * 3)).toFixed(1)}`;
  }).join(' ');
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H + f.H + 10}`} width="100%" style={{ display: 'block' }} role="img"
        aria-label={`Ship at ${phi.toFixed(0)} degrees north on day ${days.toFixed(0)}; the ${pend ? 'pendulum clock' : 'balance-spring clock'} is ${arcminOf(errorAfter(v, days)).toFixed(1)} arcminutes of longitude out`}>
        {/* the world: horizon and sea (fixed) */}
        <line x1={0} x2={W} y1={seaY - 22} y2={seaY - 22} style={{ ...ink('faint', 0.7), strokeDasharray: '2 4' }} />
        <polyline points={wave} style={ink('sky', 1.2)} />
        {/* the ship rides on it */}
        <g transform={`translate(${cx} ${seaY - 6 + lift}) rotate(${roll})`}>
          <path d="M -64 0 L 64 0 L 52 18 L -54 18 Z" style={{ ...ink('ink', 1.2), fill: C.paper }} />
          <line x1={-6} x2={-6} y1={0} y2={-64} style={ink('ink', 1.2)} />
          <path d="M -4 -60 L 26 -14 L -4 -14 Z" style={{ ...ink('soft', 0.9), fill: C.paper }} />
          <rect x={-46} y={-26} width={30} height={26} style={{ ...ink('ink', 1), fill: C.paper }} />
          {pend ? (
            <g transform={`rotate(${(swing * 180) / Math.PI} -31 -24)`}>
              <line x1={-31} x2={-31} y1={-24} y2={-6} style={ink('brass', 1.2)} />
              <circle cx={-31} cy={-5} r={3} style={{ fill: C.brass }} />
            </g>
          ) : (
            <g>
              <circle cx={-31} cy={-13} r={8} style={ink('brass', 1.2)} />
              <line x1={-31} y1={-13} x2={-31 + 8 * Math.sin(swing)} y2={-13 - 8 * Math.cos(swing)} style={ink('brass', 1.2)} />
            </g>
          )}
        </g>
        <InkText x={W - 12} y={18} anchor="end" size={11} role="soft">
          {`${phi.toFixed(1)}° N, day ${days.toFixed(0)}, g ${gravity(phi).toFixed(3)} m/s²`}
        </InkText>
        <InkText x={W - 12} y={33} anchor="end" size={11} role="soft">
          {`sea ${v.heave.toFixed(1)} m swell, ${tempAt(v, phi).toFixed(0)} °C`}
        </InkText>

        {/* the chart: longitude error against days at sea */}
        <g transform={`translate(0 ${H})`}>
          {PRIZES.map((p) => (
            <g key={p.pounds}>
              <line x1={f.m.l} x2={f.m.l + f.iw} y1={f.y(p.arcmin)} y2={f.y(p.arcmin)} style={{ ...ink('brass', 0.8), strokeDasharray: '4 3' }} />
            </g>
          ))}
          <InkText x={f.m.l + f.iw - 2} y={f.y(30) + 11} anchor="end" size={10} role="brass">£20,000: within 30′</InkText>
          <InkText x={f.m.l + f.iw - 2} y={f.y(60) - 4} anchor="end" size={10} role="brass">£10,000: 1°</InkText>
          {Y_TICKS.map(([val, s]) => (
            <InkText key={val} x={f.m.l - 5} y={f.y(val)} kind="num" anchor="end" baseline="central" size={9.5} role="soft" halo={false}>{s}</InkText>
          ))}
          {[0, 30, 60, 90].map((d) => (
            <InkText key={d} x={f.x(d)} y={f.H - f.m.b + 13} kind="num" anchor="middle" size={9.5} role="soft" halo={false}>{`${d}`}</InkText>
          ))}
          <InkText x={f.m.l + f.iw} y={f.H - 3} anchor="end" size={10.5} role="soft">days at sea</InkText>
          <InkText x={f.m.l} y={f.m.t - 4} size={10.5} role="soft">longitude error</InkText>
          <line x1={f.x(VOYAGE_DAYS)} x2={f.x(VOYAGE_DAYS)} y1={f.m.t} y2={f.m.t + f.ih} style={{ ...ink('faint', 0.7), strokeDasharray: '2 3' }} />
          <path d={`M ${f.m.l} ${f.m.t} V ${f.m.t + f.ih} H ${f.m.l + f.iw}`} style={ink('ink', 0.9)} />
          {h4Curve.length > 1 && (
            <polyline points={h4Curve.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={ink('sky', 1.6)} />
          )}
          {showH4 && (
            <g>
              <circle cx={f.x(H4_DAYS)} cy={f.y(1.25)} r={3.8} style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1.6 }} />
              <InkText x={f.x(H4_DAYS) - 6} y={f.y(1.25) - 9} anchor="end" size={10} role="sky">H4, Jamaica 1762</InkText>
            </g>
          )}
          {v.clock === 'pendulum' && cur.length > 1 && (
            <polyline points={cur.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={ink('red', 1.6)} />
          )}
          {v.clock === 'pendulum' && (
            <>
              <circle cx={f.x(days)} cy={f.y(a)} r={3.5} style={{ fill: C.red }} />
              {offTop && <path d={`M ${f.x(days)} ${f.m.t - 2} l -4 7 h 8 z`} style={{ fill: C.red }} />}
            </>
          )}
        </g>
      </svg>
    </div>
  );
}

export default function SeaClockPlate() {
  const [step, setStep] = useState(0);
  const [heave, setHeave] = useState(0);
  const [dest, setDest] = useState(PORTSMOUTH);
  const [heat, setHeat] = useState(false);
  const [days, setDays] = useState(20);
  const [selClock, setClock] = useState<Clock>('pendulum');
  const [inRef, inView] = useInView<HTMLElement>();
  const reduced = useReducedMotion();
  const t = useTime(inView && !reduced);
  // each step adds one wall; earlier steps preview the voyage as it was then
  const voyageAt = (i: number): Voyage => ({
    clock: i === 5 ? selClock : 'pendulum',
    heave: i >= 1 ? heave : 0,
    dest: i >= 2 ? dest : PORTSMOUTH,
    heat: i >= 3 ? heat : false,
  });
  const daysAt = (i: number) => (i >= 4 ? days : VOYAGE_DAYS);
  const v = voyageAt(step);
  const d = daysAt(step);
  const err = errorAfter(v, d);
  const r = rateOn(v, Math.min(d, VOYAGE_DAYS - 1));
  const phi = latOnDay(v, d);
  const prize = prizeFor(err);

  const steps: StepDef[] = [
    {
      id: 'land',
      title: 'Regulate it on land',
      text: (
        <>
          The best clock of 1700: a pendulum clock, set in Portsmouth to keep perfect time on a still floor. A ship that
          carries Portsmouth time can compare it with local noon and know how far west it has come: 4 minutes of time per
          degree of longitude.
        </>
      ),
    },
    {
      id: 'heave',
      title: 'Put it on a ship',
      text: (
        <>
          Every wave lifts and drops the ship, so the bob feels gravity plus the ship&apos;s acceleration. The swell kicks the
          swing too, and no two days are alike, so the error cannot even be allowed for.
        </>
      ),
      controls: <Slider label="height of the swell" value={heave} min={0} max={HEAVE_MAX} step={0.1} onChange={setHeave} format={(x) => `${x.toFixed(1)} m`} width="9rem" />,
      gated: true,
      done: heave > 0,
      hint: 'Raise the swell',
    },
    {
      id: 'south',
      title: 'Sail south',
      text: (
        <>
          Gravity is weaker toward the equator (the Earth bulges and spins). In 1672 Jean Richer took a Paris-regulated
          pendulum clock to Cayenne, at 5° N; it lost 2 minutes 28 seconds a day.
        </>
      ),
      controls: <Slider label="sail to latitude" value={dest} min={JAMAICA} max={PORTSMOUTH} step={0.1} onChange={setDest} format={(x) => `${x.toFixed(1)}° N`} width="9rem" />,
      gated: true,
      done: dest < PORTSMOUTH - 5,
      hint: 'Sail at least 5° south',
    },
    {
      id: 'heat',
      title: 'Into the warm',
      text: <>The Caribbean is warmer than the Channel. A warm steel rod is a long rod, and a long rod is a slow clock.</>,
      controls: (
        <Segmented
          label="temperature"
          options={[
            { value: 'no', label: 'same as Portsmouth' },
            { value: 'yes', label: 'warms as you sail south' },
          ]}
          value={heat ? 'yes' : 'no'}
          onChange={(x) => setHeat(x === 'yes')}
        />
      ),
    },
    {
      id: 'bill',
      title: 'The bill',
      text: (
        <>
          The Longitude Act of 1714: £20,000 for a method good to half a degree on a voyage to the West Indies, about six
          weeks. Half a degree is 2 minutes of time, after six weeks. Count the days.
        </>
      ),
      controls: <Slider label="days at sea" value={days} min={0} max={DAYS_MAX} step={1} onChange={setDays} format={(x) => `${x} d`} width="10rem" />,
    },
    {
      id: 'h4',
      title: 'Swap in a balance',
      text: (
        <>
          John Harrison&apos;s answer: no pendulum. A balance wheel and a spiral spring keep the beat; the spring, not gravity,
          pulls it back. His H4, a large watch, ticks five times a second, and on the 1761-62 Jamaica trial it was 5
          seconds slow after 81 days.
        </>
      ),
      controls: (
        <>
          <Segmented
            label="clock"
            options={[
              { value: 'pendulum', label: 'pendulum clock' },
              { value: 'balance', label: 'balance and spring' },
            ]}
            value={selClock}
            onChange={setClock}
          />
          <Slider label="days at sea" value={days} min={0} max={DAYS_MAX} step={1} onChange={setDays} format={(x) => `${x} d`} width="8rem" />
        </>
      ),
    },
  ];

  const pend = v.clock === 'pendulum';
  return (
    <StepPlate
      rootRef={inRef}
      n={6}
      title="How a pendulum fails at sea, and what replaced it"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => <Ship v={voyageAt(i)} t={t} days={daysAt(i)} showH4={i === 5} />}
      viewLabel="The ship and its clock"
      reference={(i) => {
        const vi = voyageAt(i);
        const di = daysAt(i);
        const e = errorAfter(vi, di);
        const p = latOnDay(vi, di);
        return (
          <TimeRef
            ep={i === 5 && vi.clock === 'balance' ? 5 : 4}
            t={12}
            dec={0}
            dateText={`at sea, day ${Math.round(di)}`}
            read={12 + e / 3600}
            gapSec={Math.abs(e)}
            gapNote={i === 0 ? 'on land' : undefined}
            metresPerSec={M_PER_S_EQ * Math.cos((p * Math.PI) / 180)}
            distWord="of longitude"
            rate={Math.abs(rateOn(vi, Math.min(di, VOYAGE_DAYS - 1)).total) || null}
            hi="clock"
          />
        );
      }}
      equation={
        <Working label="Rate at sea">
          {pend ? (
            <>
              <Eq>
                <Sym role="red">T</Sym> = 2π √(<Sym role="brass">L</Sym> ÷ <Sym role="sky">g</Sym>
                <sub>eff</sub>)
              </Eq>
              <Eq>
                rate ={' '}
                <Known when={step >= 1} role="sky">{signed(r.heave + r.scatter, (x) => `${sig(x, 2)} s`)}</Known> (sea){' '}
                <Known when={step >= 2} role="sky">{signed(r.gravity, (x) => `${sig(x, 2)} s`)}</Known> (g){' '}
                <Known when={step >= 3} role="brass">{signed(r.heat, (x) => `${sig(x, 2)} s`)}</Known> (heat) a day
              </Eq>
            </>
          ) : (
            <>
              <Eq>
                <Sym role="red">T</Sym> = 2π √(<Sym role="brass">I</Sym> ÷ <Sym role="brass">κ</Sym>): no <Sym role="sky">g</Sym>
              </Eq>
              <Eq>
                rate = <V role="sky">{`${H4_RATEtxt} s a day`}</V> (H4&apos;s trial average)
              </Eq>
            </>
          )}
          <Eq>
            after <V>{`${Math.round(d)} d`}</V>: <V role="red">{fmtSec(Math.abs(err))}</V> ={' '}
            <V role="red">{`${sig(arcminOf(err), 3)}′`}</V> = <V role="red">{`${sig(kmOf(err, phi), 3)} km`}</V>
          </Eq>
          <Eq>
            <span style={{ fontSize: '0.88em' }}>
              {prize ? `Wins the £${prize.pounds.toLocaleString('en-GB')} prize.` : 'Wins nothing: worse than 1° of longitude.'}
            </span>
          </Eq>
        </Working>
      }
      caption={
        <>
          My model, from the physics: heave loss a²/(16g²) for a swell with an {WAVE_PERIOD} s period, scattered ±50% day to
          day (illustrative); gravity from the WGS-84 formula; a steel rod, 11.5 millionths per °C, the sea warming from 12
          to 27 °C between Portsmouth and 18° N (an assumption). The balance-and-spring line is H4&apos;s own Jamaica result
          spread evenly over the voyage. Longitude: 1 s of time is 0.25′, 463 m at the equator times the cosine of the
          latitude.
        </>
      }
    />
  );
}

const H4_RATEtxt = (-5 / (81 + 5 / 24)).toFixed(3).replace('-', '−');
