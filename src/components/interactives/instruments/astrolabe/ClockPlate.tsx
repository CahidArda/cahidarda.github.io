/**
 * Plate 7 (episode 7, small): the astrolabe inside a clock. A clockwork turns the rete at the
 * stars' rate and the sun's hand at the sun's rate; the reader runs the clock day by day at the
 * same hour and watches the rete creep ahead of the sun. Rates: USNO GMST formula
 * (1.0027379 sidereal per solar). Geometry from plateGeom.ts (the same plate and rete).
 */
import { useState } from 'react';
import { C, Eq, InkText, Plate, Slider, Sym, V, Working, useMeasuredWidth } from '../shared';
import { PHI, dayLabel, noteSize } from './astro';
import { clockState } from './clockGeom';
import { PrimsSvg } from './Prims';
import { RC, RM, eclipticPoint, limbAngle, limbPrims, platePrims, retePrims } from './plateGeom';

function Dial({ days }: { days: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const H = Math.round(Math.max(260, Math.min(380, W * 0.75)));
  const k = (Math.min(W, H - 30) / 2 - 8) / RM;
  const f = { cx: W / 2, cy: (H - 30) / 2 + 4, k };
  const s = clockState(days);
  const sunP = eclipticPoint(s.lam, s.lst);
  const a = (limbAngle(s.H) * Math.PI) / 180;
  const note = `${dayLabel(s.doy)}, 21:00 sun time: the hand stays put, the stars do not`;
  const hand = { x: f.cx + RC * 1.05 * k * Math.cos(a), y: f.cy - RC * 1.05 * k * Math.sin(a) };
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', background: C.paper }} role="img" aria-label={`Astrolabe dial on ${dayLabel(s.doy)} at 21:00: the rete has turned ${s.turned.toFixed(0)} degrees since 21 March`}>
        <circle cx={f.cx} cy={f.cy} r={RM * k} style={{ fill: C.deep, stroke: C.ink, strokeWidth: 1.2 }} />
        <circle cx={f.cx} cy={f.cy} r={RC * k} style={{ fill: C.paper }} />
        <PrimsSvg prims={limbPrims()} f={f} scaleW={0.8} />
        <PrimsSvg prims={platePrims(PHI, { labels: false, azimuths: false })} f={f} scaleW={0.8} opacity={0.7} />
        <PrimsSvg prims={retePrims(s.lst)} f={f} scaleW={1} />
        <line x1={f.cx} y1={f.cy} x2={hand.x} y2={hand.y} style={{ stroke: C.red, strokeWidth: 2 }} />
        <circle cx={f.cx + sunP.u * k} cy={f.cy - sunP.v * k} r={5} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1 }} />
        <circle cx={f.cx} cy={f.cy} r={3} style={{ fill: C.ink }} />
        <InkText x={W / 2} y={H - 10} anchor="middle" size={noteSize(note, W)} role="soft">{note}</InkText>
      </svg>
    </div>
  );
}

export default function ClockPlate() {
  const [days, setDays] = useState(0);
  const s = clockState(days);
  return (
    <Plate
      n={7}
      title="The astrolabe dial in a clock"
      caption={
        <>
          The red hand turns once a solar day; the rete turns a little faster, once a sidereal day of 23 h 56 min 4 s (from the USNO sidereal-time
          formula). Look at the same hour each evening and the stars stand about 3 min 56 s further on: a full turn in a year. Clockmakers built this
          into the gearing, as on the dial of Prague's astronomical clock.
        </>
      }
    >
      <Dial days={days} />
      <div className="in-controls">
        <Slider label="days since 21 March" value={days} min={0} max={365} step={1} onChange={setDays} format={(v) => `${v} d`} />
      </div>
      <Working label="Why the rete runs ahead">
        <Eq>
          rete gain = <Sym>days</Sym> × 24 h × (1.0027379 − 1)
        </Eq>
        <Eq>
          <V role="red">{`${Math.floor(s.gain / 60)} h ${String(Math.round(s.gain % 60)).padStart(2, '0')} min`}</V> = <V>{`${days}`}</V> × 3.94 min ≈{' '}
          <V role="red">{`${s.turned.toFixed(0)}°`}</V> of rete turn
        </Eq>
      </Working>
    </Plate>
  );
}
