/**
 * Plate 3 (episode 3): putting an astrolabe together, part by part, then turning the rete.
 * Instrument: the 3D model (AstrolabeModel). Reference: the same sky dome as every plate, at the
 * sun time the rete is turned to, so turning the rete visibly turns the real sky.
 */
import { useState } from 'react';
import { C, Eq, Known, Slider, StepPlate, Sym, Working, type StepDef } from '../shared';
import { DOY_DEFAULT, eclDec, hm, sunLambda, eclRA, wrap180, wrap360 } from './astro';
import { AstrolabeModel, AstrolabePoster, type ModelParams } from './AstrolabeModel';
import { Dome, type DomeHi } from './Dome';
import { PersistentPortal, Slot, usePersistentHost } from './Persist';
import { lamText } from './common';

const HI: DomeHi[][] = [['horizon'], ['horizon', 'alm'], ['day', 'sun'], [], ['sun', 'day']];
const NAMES: { ar?: string; tr?: string; ch: string }[] = [
  { ar: 'umm, "mother"', tr: 'ana gövde', ch: 'moder' },
  { ar: 'ṣafīḥa', tr: 'plaka', ch: 'tables' },
  { ar: 'ʿankabūt, "spider"', tr: 'örümcek', ch: 'riet' },
  { ar: 'quṭb and faras, "horse"', ch: 'pyn and hors' },
  { tr: 'cetvel', ch: 'label' },
];

export default function AssemblePlate() {
  const [step, setStep] = useState(0);
  const [hour, setHour] = useState(9);
  const [shown, setShown] = useState(0);
  const host = usePersistentHost();
  const doy = DOY_DEFAULT;
  const lam = sunLambda(doy);
  const ra = eclRA(lam);
  const H = (hour - 12) * 15;
  const lst = wrap360(H + ra);
  const model = (i: number): ModelParams => ({
    face: 0,
    parts: Math.min(5, i + 1),
    lst,
    ali: 30,
    ruleH: i >= 4 ? H : null,
    rays: null,
    sunLam: i >= 2 ? lam : null,
    almH: null,
    swing: 0,
  });
  const name = (i: number) => {
    const n = NAMES[i];
    const bits = [n.ar && `Arabic ${n.ar}`, n.tr && `Turkish ${n.tr}`, `Chaucer's ${n.ch}`].filter(Boolean);
    return <span style={{ color: C.soft }}> ({bits.join('; ')})</span>;
  };
  const steps: StepDef[] = [
    {
      id: 'mater',
      title: 'The mater',
      text: (
        <>
          A brass disc with a raised rim, the limb, divided into 24 hours of 15° each. It hangs from a throne and a ring, so it always hangs plumb.
          {name(0)}
        </>
      ),
    },
    {
      id: 'plate',
      title: 'The plate',
      text: (
        <>
          A plate engraved for one latitude drops into the mater: the horizon, height circles (almucantars) every 10°, and azimuth lines, all
          projected for 41° N. This is the part that belongs to the place.{name(1)}
        </>
      ),
    },
    {
      id: 'rete',
      title: 'The rete',
      text: (
        <>
          On top goes the sky: an openwork disc with the ecliptic ring (the sun's yearly path, in zodiac signs) and pointers for bright stars. The
          brass dot is today's sun, at {lamText(lam)}.{name(2)}
        </>
      ),
    },
    {
      id: 'pin',
      title: 'Pin and horse',
      text: <>A pin through the centre (the celestial pole) holds it all together; a wedge called the horse locks the pin.{name(3)}</>,
    },
    {
      id: 'turn',
      title: 'Turn the rete',
      text: (
        <>
          Now turn the rete clockwise: that is the day passing. The sun's dot rises over the horizon on the left (east), crosses the meridian at
          the top at noon and sets on the right. The rule reads the hour on the limb.{name(4)}
        </>
      ),
      controls: <Slider label="sun time (turns the rete)" value={hour} min={0} max={24} step={0.25} onChange={setHour} format={(v) => hm(v)} />,
    },
  ];
  return (
    <>
      <PersistentPortal host={host}>
        <AstrolabeModel p={model(shown)} label="A 3D astrolabe being assembled: mater, a plate for 41 degrees north, the rete, pin and rule" />
      </PersistentPortal>
      <StepPlate
        n={3}
        title="The parts of an astrolabe"
        wide
        steps={steps}
        step={step}
        onStep={setStep}
        view={(i) => <Slot host={host} onShow={() => setShown(i)} fallback={<AstrolabePoster p={model(i)} />} />}
        viewLabel="The astrolabe, front"
        reference={(i) => <Dome opts={{ dec: eclDec(lam), sunH: i >= 2 ? (i >= 4 ? wrap180(H) : -42.5) : null, alm: i === 1 ? 30 : null }} hi={HI[i]} />}
        referenceLabel="The real sky over Istanbul (east on the left)"
        equation={
          <Working label="Rete turn and sun time">
            <Eq>
              turn of the rete = <Sym role="red">H</Sym> + <Sym role="brass">α</Sym>
            </Eq>
            <Eq>
              <Known when={step >= 4} role="ink">{`${lst.toFixed(1)}°`}</Known> = <Known when={step >= 4} role="red">{`${H > 0 ? '+' : H < 0 ? '−' : ''}${Math.abs(H).toFixed(1)}°`}</Known> +{' '}
              <Known when={step >= 2} role="brass">{`${ra.toFixed(1)}°`}</Known>
            </Eq>
            <Eq>
              <span style={{ fontSize: '0.88em', color: C.soft }}>H: the sun's hour angle (15° an hour, 0 at noon). α: its right ascension today.</span>
            </Eq>
          </Working>
        }
        caption={
          <>
            Part names: Arabic from Pingree (Encyclopaedia Iranica), Turkish from the TDV İslâm Ansiklopedisi's USTURLAP figure, Middle English from
            Chaucer's Treatise (1391). Star pointers: Vega, Altair, Aldebaran, Arcturus, Capella, Regulus, Spica, at their J2000 positions.
          </>
        }
      />
    </>
  );
}
