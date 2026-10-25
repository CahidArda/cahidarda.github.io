/**
 * Plate 4 (signature): telling the time from the sun with the astrolabe, step by step.
 * 1 Hang it · 2 Sight the sun · 3 Read the altitude · 4 Find the sun's place · 5 Turn the rete
 * · 6 Read the time · 7 Istanbul clock time.
 * Instrument: the 3D model (AstrolabeModel) for the hands-on steps, the flat faces (Faces.tsx)
 * for the readings, all from plateGeom.ts. Reference: the same sky dome as every plate.
 */
import { useState } from 'react';
import { C, Eq, Known, Slider, StepPlate, Sym, Working, type StepDef } from '../shared';
import { DOY_DEFAULT, PHI, deg1, dayLabel, dmin, eotMinutes, hm, lonMinutes, sunLambda, wrap360, type MomentId } from './astro';
import { AstrolabeModel, AstrolabePoster, type ModelParams } from './AstrolabeModel';
import { BackView, FrontView } from './Faces';
import { Dome, type DomeHi } from './Dome';
import { DateMoment, isMorning, lamText, momentT } from './common';
import { PersistentPortal, Slot, usePersistentHost } from './Persist';
import { SIGHT_TOL, eclipticPoint, limbAngle, RC } from './plateGeom';
import { aliForDate } from './faceGeom';
import {
  ALI_MAX,
  ALI_MIN,
  ALI_STEP,
  RETE_STEP,
  RULE_STEP,
  clockOf,
  reteStart,
  reteState,
  sighted,
  snapAli,
  snapRete,
  snapRule,
  solution,
  world,
} from './tellGeom';

const HI: DomeHi[][] = [[], ['sun', 'h'], ['h'], ['day', 'sun'], ['h', 'day'], ['hour', 'sun'], ['hour']];
const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}°`;

export default function TellTimePlate() {
  const [doy, setDoyRaw] = useState(DOY_DEFAULT);
  const [moment, setMomentRaw] = useState<MomentId>('m1');
  const [ali, setAliRaw] = useState(0);
  const [lst, setLstRaw] = useState(() => reteStart(sunLambda(DOY_DEFAULT)));
  const [ruleH, setRuleRaw] = useState(-150);
  const [step, setStepRaw] = useState(0);
  const [shown, setShown] = useState(0);
  const [swing, setSwing] = useState(0);
  const host = usePersistentHost();

  const t = momentT(moment);
  const side = isMorning(moment) ? 'am' : 'pm';
  const sun = world(doy, t);
  const lam = sun.lam;
  const on = sighted(ali, sun.h);
  const hMeas = ali; // the reading, not the true value
  const rs = reteState(lam, lst, hMeas, side);
  const sol = solution(lam, hMeas, side);
  const ruleOn = rs.on && Math.abs(ruleH - rs.Hs) < 1e-9;
  const tRead = 12 + ruleH / 15;

  // a new date or moment: the sky changed, the old sight is no longer valid; go back to it
  const resettle = () => setStepRaw((k) => Math.min(k, 1));
  const setDoy = (d: number) => {
    setDoyRaw(d);
    setLstRaw(reteStart(sunLambda(d)));
    resettle();
  };
  const setMoment = (m: MomentId) => {
    setMomentRaw(m);
    resettle();
  };
  const setStep = (i: number) => {
    if (i === 0 && step !== 0) setSwing((s) => s + 1);
    setStepRaw(i);
  };
  const setAli = (a: number) => setAliRaw(snapAli(a, sun.h));
  const setLst = (v: number) => setLstRaw(snapRete(lam, v, hMeas, side));
  const setRule = (v: number) => setRuleRaw(snapRule(v, rs.Hs));

  const model = (i: number): ModelParams => ({
    face: i === 1 ? 1 : 0,
    parts: 5,
    lst,
    ali,
    ruleH: i >= 5 ? ruleH : null,
    rays: i === 1 ? sun.h : null,
    sunLam: i >= 4 ? lam : null,
    almH: i >= 4 ? hMeas : null,
    plumb: i === 0,
    swing,
  });
  const sunPt = eclipticPoint(lam, lst);
  const view = (i: number) => {
    if (i === 2) return <BackView ali={ali} mark="alt" zoom={2.1} focus={{ u: -1.1, v: 1.1 }} note={`the edge reads ${deg1(ali)} on the altitude scale`} noteRole="sky" />;
    if (i === 3)
      return <BackView ali={aliForDate(doy)} mark="date" zoom={1} note={`${dayLabel(doy)} on the calendar ring: the sun at ${lamText(lam)}`} noteRole="brass" />;
    if (i === 5)
      return (
        <FrontView
          phi={PHI}
          lst={lst}
          sunLam={lam}
          almH={hMeas}
          ruleH={ruleH}
          zoom={1.6}
          focus={{ u: sunPt.u * 0.6 + RC * 0.4 * Math.cos((limbAngle(ruleH) * Math.PI) / 180), v: sunPt.v * 0.6 + RC * 0.4 * Math.sin((limbAngle(ruleH) * Math.PI) / 180) }}
          note={ruleOn ? `the rule cuts the limb at ${hm(tRead)}` : 'lay the rule over the sun'}
          noteRole={ruleOn ? 'red' : 'soft'}
        />
      );
    return (
      <Slot
        host={host}
        onShow={() => setShown(i)}
        fallback={<AstrolabePoster p={model(i)} />}
      />
    );
  };

  const steps: StepDef[] = [
    {
      id: 'hang',
      title: 'Hang it',
      text: <>Hold the astrolabe by its ring and let it hang. Its own weight makes the line from the ring through the centre point straight down: the instrument's 90° is now the zenith.</>,
    },
    {
      id: 'sight',
      title: 'Sight the sun',
      text: (
        <>
          Turn to face the sun and swing the alidade on the back until the light through the top vane's hole falls on the hole in the bottom vane.
          You never look at the sun; you watch the spot of light.
        </>
      ),
      controls: <Slider label="swing the alidade" value={ali} min={ALI_MIN} max={ALI_MAX} step={ALI_STEP} onChange={setAli} format={(v) => `${v.toFixed(1)}°`} />,
      gated: true,
      done: on,
      hint: `Light on the bottom hole (within ${SIGHT_TOL.toFixed(2)}°)`,
    },
    {
      id: 'read',
      title: 'Read the altitude',
      text: <>Read the alidade's edge on the degree scale round the top of the back: <span style={{ color: C.sky }}>h = {deg1(hMeas)}</span>. That is all you measure.</>,
    },
    {
      id: 'date',
      title: "Find the sun's place",
      text: (
        <>
          Lay the alidade on today's date on the calendar ring. The zodiac ring beside it gives the sun's place on the ecliptic:{' '}
          <span style={{ color: C.brass }}>{lamText(lam)}</span>. Find that degree on the rete's ecliptic ring.
        </>
      ),
    },
    {
      id: 'turn',
      title: 'Turn the rete',
      text: (
        <>
          Turn the rete until the sun's degree on the ecliptic sits on the <span style={{ color: C.sky }}>{deg1(hMeas)}</span> height circle, on the{' '}
          {side === 'am' ? 'east (left) side: it is morning' : 'west (right) side: it is afternoon'}. The rete now shows the whole sky as it is.
        </>
      ),
      controls: (
        <Slider label="turn the rete" value={Number(lst.toFixed(2))} min={0} max={360} step={RETE_STEP} onChange={setLst} format={(v) => `${wrap360(v).toFixed(1)}°`} />
      ),
      gated: true,
      done: rs.on,
      hint: rs.sideOk ? `Sun's point on the ${deg1(hMeas)} circle (now ${deg1(rs.hPlate)})` : `Wrong side: the sun is in the ${side === 'am' ? 'east' : 'west'}`,
    },
    {
      id: 'rule',
      title: 'Read the time',
      text: (
        <>
          Lay the rule from the pin over the sun's point and read where it cuts the hour scale on the limb:{' '}
          <span style={{ color: C.red }}>{ruleOn ? hm(tRead) : '?'}</span> sun time. Each hour is 15° of turn.
        </>
      ),
      controls: <Slider label="lay the rule" value={ruleH} min={-180} max={180} step={RULE_STEP} onChange={setRule} format={(v) => hm(12 + v / 15)} />,
      gated: true,
      done: ruleOn,
      hint: 'Rule over the sun',
    },
    {
      id: 'clock',
      title: 'Istanbul clock time',
      text: (
        <>
          The astrolabe gives sun time. Istanbul's clocks run on UTC+3, set for 45° E; the city is at 29° E, so add {Math.round(lonMinutes)} minutes, and
          correct for the equation of time ({dmin(-eotMinutes(doy) / 60)} on {dayLabel(doy)}).
        </>
      ),
    },
  ];

  const kn = (i: number) => step >= i;
  return (
    <>
      <PersistentPortal host={host}>
        <AstrolabeModel p={model(shown)} label="A 3D astrolabe: mater, a plate for 41 degrees north, the rete, the rule on the front and the alidade on the back" />
      </PersistentPortal>
      <StepPlate
        n={4}
        title="Telling the time from the sun with an astrolabe"
        wide
        steps={steps}
        step={step}
        onStep={setStep}
        view={view}
        viewLabel="The astrolabe"
        reference={(i) => <Dome opts={{ dec: sun.dec, sunH: sun.H, alm: i >= 4 ? hMeas : null, showHourArc: i >= 5 }} hi={HI[i]} />}
        referenceLabel="The real sky over Istanbul (east on the left)"
        settings={<DateMoment doy={doy} setDoy={setDoy} moment={moment} setMoment={setMoment} />}
        onSkip={() => {
          const a = Math.round(sun.h * 10) / 10;
          setAliRaw(a);
          const st = solution(lam, a, side);
          if (st.H !== null) {
            setLstRaw(wrap360(st.H + sun.ra));
            setRuleRaw(st.H);
          }
        }}
        equation={
          <Working label="What the astrolabe works out for you">
            <Eq>
              cos <Sym role="red">H</Sym> = (sin <Sym role="sky">h</Sym> − sin <Sym role="soft">φ</Sym> sin <Sym role="brass">δ</Sym>) / (cos <Sym role="soft">φ</Sym> cos{' '}
              <Sym role="brass">δ</Sym>)
            </Eq>
            <Eq>
              h <Known when={kn(2) && on} role="sky">{deg1(hMeas)}</Known>, φ <span style={{ color: C.soft }}>41°</span>, δ{' '}
              <Known when={kn(3)} role="brass">{deg1(sol.dec)}</Known> (λ <Known when={kn(3)} role="brass">{lamText(lam)}</Known>)
            </Eq>
            <Eq>
              cos <Sym role="red">H</Sym> = <Known when={kn(4) && rs.on} role="red">{sol.cosH.toFixed(3)}</Known>, <Sym role="red">H</Sym> ={' '}
              <Known when={kn(4) && rs.on} role="red">{sol.H === null ? '·' : signed(sol.H)}</Known>
            </Eq>
            <Eq>
              <Sym role="red">t</Sym> = 12:00 + <Sym role="red">H</Sym> / 15° = <Known when={kn(5) && ruleOn} role="red">{sol.t === null ? '·' : hm(sol.t)}</Known>
            </Eq>
            <Eq>
              clock = <Sym role="red">t</Sym> + {Math.round(lonMinutes)} min − EoT ={' '}
              <Known when={kn(6) && ruleOn} role="red">{sol.t === null ? '·' : hm(clockOf(sol.t, doy))}</Known>
            </Eq>
          </Working>
        }
        caption={
          <>
            The rete does the trigonometry: turning it until the sun's point sits on the measured height circle solves the same equation the panel shows.
            The reference on the right is the real sky at the moment you picked; your measured time agrees with it to the minute. Sun positions and
            equation of time: USNO approximate formulae. Engraving: almucantars every 10°, azimuths every 30°, computed for 41° N.
          </>
        }
      />
    </>
  );
}
