/**
 * Plate 3: two clocks in Istanbul. Geometry in ./alaturkaGeom (swept by a script). Alafranga counts from midnight, so noon is always 12.
 * Alaturka (ezanî) counts from sunset, so sunset is always 12 and noon wanders through the
 * year. A strip below plots the alaturka reading of true noon over the year.
 */
import { useMemo, useState } from 'react';
import {
  C,
  Controls,
  Eq,
  InkText,
  Plate,
  Slider,
  Sym,
  V,
  Working,
  polar,
  useMeasuredWidth,
  type Role,
} from '../shared';
import { dayLabel, declination, halfDay, hm } from './astro';
import { alaturka, dialAngle, hm12, hm12f, sinceSunset, sunriseOf, sunsetOf } from './alaturkaGeom';

function Dial({
  cx,
  cy,
  r,
  hours,
  title,
  sub,
  marks,
  handRole = 'ink',
}: {
  cx: number;
  cy: number;
  r: number;
  hours: number;
  title: string;
  sub: string;
  marks: { at: number; label: string; role: Role }[];
  handRole?: Role;
}) {
  const ang = dialAngle;
  const hand = polar(cx, cy, r * 0.62, ang(hours));
  const mHand = polar(cx, cy, r * 0.86, 90 - ((hours % 1) * 360));
  return (
    <g>
      <circle cx={cx} cy={cy} r={r + 6} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.6 }} />
      <circle cx={cx} cy={cy} r={r + 2} style={{ fill: C.paper, stroke: C.soft, strokeWidth: 0.6 }} />
      {Array.from({ length: 60 }, (_, i) => {
        const a = 90 - i * 6;
        const major = i % 5 === 0;
        const p = polar(cx, cy, r, a);
        const q = polar(cx, cy, r - (major ? 9 : 4), a);
        return (
          <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={{ stroke: major ? C.ink : C.soft, strokeWidth: major ? 1.1 : 0.6 }} />
        );
      })}
      {Array.from({ length: 12 }, (_, i) => i + 1).map((k) => {
        const p = polar(cx, cy, r - 21, 90 - k * 30);
        return (
          <InkText key={k} x={p.x} y={p.y} kind="num" anchor="middle" baseline="central" size={12} halo={false}>
            {k}
          </InkText>
        );
      })}
      {marks.map((m) => {
        const a = ang(m.at);
        const p = polar(cx, cy, r + 9, a);
        const q = polar(cx, cy, r + 18, a);
        const t = polar(cx, cy, r + 30, a);
        return (
          <g key={m.label}>
            <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={{ stroke: C[m.role], strokeWidth: 1.6 }} />
            <InkText x={t.x} y={t.y} anchor="middle" baseline="central" size={11} role={m.role}>
              {m.label}
            </InkText>
          </g>
        );
      })}
      <line x1={cx} y1={cy} x2={hand.x} y2={hand.y} style={{ stroke: C[handRole], strokeWidth: 3, strokeLinecap: 'round' }} />
      <line x1={cx} y1={cy} x2={mHand.x} y2={mHand.y} style={{ stroke: C[handRole], strokeWidth: 1.4, strokeLinecap: 'round' }} />
      <circle cx={cx} cy={cy} r={3.5} style={{ fill: C[handRole] }} />
      <InkText x={cx} y={cy + r + 52} kind="title" anchor="middle" size={15}>
        {title}
      </InkText>
      <InkText x={cx} y={cy + r + 69} anchor="middle" size={11.5} role="soft">
        {sub}
      </InkText>
    </g>
  );
}

function YearStrip({ day }: { day: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(560);
  const H = 120;
  const m = { l: 40, r: 10, t: 12, b: 24 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const lo = 4;
  const hi = 8;
  const x = (n: number) => m.l + ((n - 1) / 364) * iw;
  const y = (v: number) => m.t + ((v - lo) / (hi - lo)) * ih; // later times lower
  const d = useMemo(() => {
    let s = '';
    for (let n = 1; n <= 365; n += 2) s += `${n === 1 ? 'M' : 'L'} ${x(n).toFixed(1)} ${y(alaturka(12, n)).toFixed(1)} `;
    return s;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W]);
  const v = alaturka(12, day);
  const months = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
  const ML = 'JFMAMJJASOND';
  return (
    <div ref={ref} style={{ marginTop: '0.5rem' }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }}>
        {[4, 5, 6, 7, 8].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={m.l + iw} y1={y(t)} y2={y(t)} style={{ stroke: C.faint, strokeWidth: 0.5, opacity: 0.7 }} />
            <InkText x={m.l - 6} y={y(t)} kind="num" anchor="end" baseline="central" size={10} role="soft" halo={false}>
              {`${t}:00`}
            </InkText>
          </g>
        ))}
        {months.map((n, i) => (
          <InkText key={n} x={x(n + 14)} y={m.t + ih + 15} kind="num" anchor="middle" size={9.5} role="soft" halo={false}>
            {ML[i]}
          </InkText>
        ))}
        <path d={d} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.3 }} />
        <line x1={x(day)} x2={x(day)} y1={m.t} y2={m.t + ih} style={{ stroke: C.soft, strokeWidth: 0.9, strokeDasharray: '3 3' }} />
        <circle cx={x(day)} cy={y(v)} r={4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.5 }} />
        <InkText x={m.l + iw} y={m.t - 1} anchor="end" size={11.5} role="soft">
          noon on the alaturka clock, through the year
        </InkText>
      </svg>
    </div>
  );
}

export default function AlaturkaDial() {
  const [day, setDay] = useState(172);
  const [t, setT] = useState(12);
  const [ref, Wm] = useMeasuredWidth<HTMLDivElement>(560);
  const narrow = Wm < 480;
  const W = 560;
  const Hh = 300;
  const r = 82;
  const sunset = sunsetOf(day);
  const sunrise = sunriseOf(day);
  const at = alaturka(t, day);
  const reset = (sunsetOf(Math.min(day + 1, 365)) - sunset) * 60;
  const H0 = halfDay(declination(day));
  const dec = declination(day);
  const wrap = t < sunset; // before sunset: count from yesterday's sunset (+24 h)
  const since = sinceSunset(t, day);
  return (
    <Plate
      n={4}
      id="plate-alaturka"
      title="Clocks that counted from sunset"
      caption={
        <>
          Local solar time in Istanbul (41° N), sun&apos;s centre on the horizon, no refraction or
          time zone. Each evening the alaturka clock was set to 12 again, by however much sunset
          had moved since the day before.
        </>
      }
    >
      <div ref={ref}>
        <svg
          viewBox={narrow ? `0 0 280 ${Hh * 2 - 30}` : `0 0 ${W} ${Hh}`}
          width="100%"
          style={{ display: 'block', maxWidth: narrow ? 270 : 600, margin: '0 auto' }}
          role="img"
          aria-label={`At ${hm(t)} solar time on ${dayLabel(day)}, an alafranga clock reads ${hm12f(t)} and an alaturka clock reads ${hm12(at)}`}
        >
          <Dial
            cx={140}
            cy={130}
            r={r}
            hours={t}
            title="Alafranga"
            sub="the day starts at midnight"
            marks={[
              { at: 12, label: 'noon', role: 'soft' },
              { at: sunset, label: 'sunset', role: 'brass' },
            ]}
          />
          <g transform={narrow ? `translate(0 ${Hh - 20})` : 'translate(280 0)'}>
            <Dial
              cx={140}
              cy={130}
              r={r}
              hours={at}
              title="Alaturka"
              sub="the day starts at sunset"
              handRole="red"
              marks={[
                { at: alaturka(12, day), label: 'noon', role: 'soft' },
                { at: 12, label: 'sunset', role: 'brass' },
              ]}
            />
          </g>
        </svg>
      </div>
      <YearStrip day={day} />
      <Controls>
        <Slider label="date" value={day} min={1} max={365} step={1} onChange={setDay} format={dayLabel} width="8rem" />
        <Slider label="time since midnight (sun time)" value={t} min={0} max={23.95} step={1 / 12} onChange={setT} format={(v) => hm(v)} width="8rem" />
      </Controls>
      <Working label="Converting to alaturka time">
        <Eq>
          <Sym role="brass">sunset</Sym> = 12:00 + <Sym>H₀</Sym> ÷ 15°, cos <Sym>H₀</Sym> = −tan φ tan δ
        </Eq>
        <Eq>
          <V role="brass">{hm(sunset)}</V> = 12:00 + <V>{H0.toFixed(1)}°</V> ÷ 15°{' '}
          <span className="al-small">(φ = 41°, δ = {dec.toFixed(1)}°)</span>
        </Eq>
        <Eq>
          <Sym role="red">alaturka</Sym> = <Sym>time</Sym> − <Sym role="brass">last sunset</Sym>
        </Eq>
        <Eq>
          <V role="red">{hm(since)}</V> = <V>{hm(t)}</V>
          {wrap ? <> + <V>24:00</V></> : null} − <V role="brass">{hm(sunset)}</V>
          {since >= 12 ? (
            <>
              , shown as <V role="red">{hm12(at)}</V> on a 12-hour dial
            </>
          ) : null}
        </Eq>
      </Working>
      <p className="al-note">
        Sunrise {hm(sunrise)}. Noon reads {hm12(alaturka(12, day))} alaturka on this date. Tonight&apos;s
        reset: {Math.abs(reset) < 0.05 ? 'almost nothing' : `${Math.abs(reset).toFixed(1)} min ${reset > 0 ? 'back' : 'forward'}`}.
      </p>
      <style>{`.al-note{margin:.6rem 0 0;font-family:var(--font-display);font-style:italic;font-size:.98rem;color:var(--in-ink-soft)}.al-small{font-size:.85em;color:var(--in-ink-soft)}`}</style>
    </Plate>
  );
}
