/** Small shared bits for the astrolabe plates: the date / moment settings and the sun equation. */
import { C, Eq, Known, Segmented, Slider, Sym, Working } from '../shared';
import { MOMENTS, deg1, dayLabel, hm, signOf, type MomentId } from './astro';

export function DateMoment({
  doy,
  setDoy,
  moment,
  setMoment,
}: {
  doy: number;
  setDoy: (d: number) => void;
  moment: MomentId;
  setMoment: (m: MomentId) => void;
}) {
  return (
    <>
      <Slider label="today's date" value={doy} min={1} max={365} step={1} onChange={setDoy} format={dayLabel} />
      <Segmented
        label="when you step outside"
        options={MOMENTS.map((m) => ({ value: m.id, label: m.label }))}
        value={moment}
        onChange={setMoment}
      />
    </>
  );
}
export const momentT = (m: MomentId) => MOMENTS.find((x) => x.id === m)!.t;
export const isMorning = (m: MomentId) => momentT(m) < 12;

export const lamText = (lam: number) => {
  const s = signOf(lam);
  return `${s.deg.toFixed(0)}° ${s.sign}`;
};

/** δ from λ, then H from h, then t: the astrolabe's whole job as one worked equation. */
export function TimeWorking({
  lam,
  dec,
  h,
  phi,
  cosH,
  H,
  t,
  known,
  label = 'What the astrolabe works out',
}: {
  lam: number;
  dec: number;
  h: number;
  phi: number;
  cosH: number;
  H: number;
  t: number;
  known: { lam: boolean; h: boolean; H: boolean; t: boolean };
  label?: string;
}) {
  return (
    <Working label={label}>
      <Eq>
        <Sym role="brass">δ</Sym> = asin(sin ε · sin <Sym role="brass">λ</Sym>)
      </Eq>
      <Eq>
        <Known when={known.lam} role="brass">{deg1(dec)}</Known> = asin(sin 23.44° · sin{' '}
        <Known when={known.lam} role="brass">{`${lam.toFixed(1)}°`}</Known>)
      </Eq>
      <Eq>
        cos <Sym role="red">H</Sym> = (sin <Sym role="sky">h</Sym> − sin <Sym role="soft">φ</Sym> sin <Sym role="brass">δ</Sym>) / (cos{' '}
        <Sym role="soft">φ</Sym> cos <Sym role="brass">δ</Sym>)
      </Eq>
      <Eq>
        <Known when={known.H} role="red">{cosH.toFixed(3)}</Known> = (sin <Known when={known.h} role="sky">{deg1(h)}</Known> − sin{' '}
        <V2 role="soft">{`${phi}°`}</V2> sin <Known when={known.lam} role="brass">{deg1(dec)}</Known>) / (…)
      </Eq>
      <Eq>
        <Sym role="red">t</Sym> = 12:00 + <Sym role="red">H</Sym> / 15° ={' '}
        <Known when={known.t} role="red">{`12:00 ${H < 0 ? '−' : '+'} ${(Math.abs(H) / 15).toFixed(2)} h = ${hm(t)}`}</Known>
      </Eq>
    </Working>
  );
}
function V2({ role, children }: { role: 'soft'; children: string }) {
  return <span className="in-v" style={{ color: C[role] }}>{children}</span>;
}
