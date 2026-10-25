/**
 * Plate 4: a clock error is a distance error, once you multiply by a speed. For a navigator
 * the speed is the Earth's turn under the sun (about 464 m/s at the equator); for GPS it is
 * the speed of light. Log slider over the clock error; presets for the rungs of the post.
 */
import { useState } from 'react';
import { C, Controls, Eq, InkText, Plate, Segmented, Sym, V, Working, useHoverPreview, useMeasuredWidth } from '../shared';

import { C_LIGHT, D0, D1, EARTH_EQ, LMAX, LMIN, ROWS, distanceLayout, dropSegments, rowBlock } from './distanceGeom';

const PRESETS = [
  { id: 'verge', s: 900, label: 'verge clock, 1 day', note: 'A good verge clock: about 15 minutes a day.' },
  { id: 'h4', s: 5, label: 'Harrison H4, 81 days', note: 'Harrison’s H4 after the voyage to Jamaica: 5 seconds slow.' },
  { id: 'quartz', s: 5 / 30, label: 'quartz watch, 1 day', note: 'The 1969 Seiko Astron: 5 seconds a month, about 0.17 s a day.' },
  { id: 'gps', s: 1e-9, label: 'GPS, 1 ns', note: 'One nanosecond: what GPS clocks have to care about.' },
] as const;
type PresetId = (typeof PRESETS)[number]['id'];


function fmtTime(s: number) {
  if (s >= 60) return `${(s / 60).toFixed(s >= 600 ? 0 : 1)} min`;
  if (s >= 1) return `${s.toFixed(s >= 10 ? 0 : 1)} s`;
  if (s >= 1e-3) return `${(s * 1e3).toPrecision(2)} ms`;
  if (s >= 1e-6) return `${(s * 1e6).toPrecision(2)} µs`;
  return `${(s * 1e9).toPrecision(2)} ns`;
}
function fmtDist(m: number) {
  if (m >= 1e6) return `${Math.round(m / 1e3).toLocaleString('en-US')} km`;
  if (m >= 1e3) return `${(m / 1e3).toPrecision(3)} km`;
  if (m >= 1) return `${m.toPrecision(3)} m`;
  if (m >= 1e-2) return `${(m * 100).toPrecision(2)} cm`;
  if (m >= 1e-3) return `${(m * 1000).toPrecision(2)} mm`;
  if (m >= 1e-6) return `${(m * 1e6).toPrecision(2)} µm`;
  return `${(m * 1e9).toPrecision(2)} nm`;
}

export default function ClockToDistance() {
  const [lg, setLg] = useState(Math.log10(5));
  const [selP, setSelP] = useState<PresetId | null>('h4');
  const [hoverP, bindP] = useHoverPreview<PresetId | null>(selP);
  const preview = PRESETS.find((p) => p.id === hoverP);
  const s = preview && hoverP !== selP ? preview.s : 10 ** lg;
  const ship = s * EARTH_EQ;
  const radio = s * C_LIGHT;

  const [ref, W] = useMeasuredWidth<HTMLDivElement>(560);
  const H = 168;
  const off0 = (d: number) => d < 10 ** D0;
  const { narrow, m, iw, x, row } = distanceLayout(W);
  void D1;
  const ticks = [
    { v: 1e-2, l: '1 cm' },
    { v: 1, l: '1 m' },
    { v: 1e3, l: '1 km' },
    { v: 1e6, l: '1,000 km' },
    { v: 1e9, l: '10⁶ km' },
    { v: 1e12, l: '10⁹ km' },
  ];
  const yShip = ROWS.ship;
  const yRadio = ROWS.radio;
  const axisY = ROWS.axis;

  const radioRow = row(ROWS.radio, s * C_LIGHT, 'range, by a radio signal', fmtDist(s * C_LIGHT), x(s * EARTH_EQ));
  const bar = (y: number, d: number, role: 'red' | 'sky', label: string, avoid?: number) => {
    // shorter than the axis shows: pinned to the left end and marked
    const { xe, off, tx, anchor } = row(y, d, label, off0(d) ? `${fmtDist(d)}, below the scale` : fmtDist(d), avoid);
    return (
      <g>
        <line x1={m.l} x2={xe} y1={y} y2={y} style={{ stroke: C[role], strokeWidth: 2 }} />
        {dropSegments(xe, y - 7, axisY, role === 'red' ? rowBlock(radioRow) : undefined).map(([a, b]) => (
          <line key={a} x1={xe} x2={xe} y1={a} y2={b} style={{ stroke: C[role], strokeWidth: 0.7, strokeDasharray: '3 3' }} />
        ))}
        {off ? (
          <path d={`M ${xe + 7} ${y - 5} L ${xe} ${y} L ${xe + 7} ${y + 5}`} style={{ fill: 'none', stroke: C[role], strokeWidth: 1.5 }} />
        ) : (
          <circle cx={xe} cy={y} r={3.5} style={{ fill: C.paper, stroke: C[role], strokeWidth: 1.5 }} />
        )}
        <InkText x={tx} y={y - 10} anchor={anchor} size={narrow ? 11.5 : 12.5}>
          {label}
        </InkText>
        <InkText x={tx} y={y + 14} anchor={anchor} kind="num" size={narrow ? 12 : 13} role={role}>
          {off ? `${fmtDist(d)}, below the scale` : fmtDist(d)}
        </InkText>
      </g>
    );
  };

  return (
    <Plate
      n={4}
      id="plate-distance"
      title="How clock error becomes position error"
      caption={
        <>
          The same clock error, turned into distance by two speeds: the Earth&apos;s surface turning
          under the sun (40,075 km a day at the equator, less at higher latitudes) for a navigator
          finding longitude, and light for a GPS receiver. Log scale: each tick is a factor of a
          thousand.
        </>
      }
    >
      <Controls>
        <span className="in-field">clock error of</span>
        <Segmented
          label="Example clock errors"
          options={PRESETS.map((p) => ({ value: p.id, label: p.label }))}
          value={selP as PresetId}
          onChange={(v) => {
            setSelP(v);
            setLg(Math.log10(PRESETS.find((p) => p.id === v)!.s));
          }}
          preview={bindP as (v: PresetId) => { onMouseEnter: () => void; onMouseLeave: () => void }}
        />
      </Controls>
      <div ref={ref} style={{ marginTop: '0.9rem' }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }} role="img" aria-label={`A clock error of ${fmtTime(s)} is ${fmtDist(ship)} of longitude at the equator, or ${fmtDist(radio)} of light travel`}>
          <line x1={m.l} x2={m.l + iw} y1={axisY} y2={axisY} style={{ stroke: C.ink, strokeWidth: 1 }} />
          {ticks.map((t) => (
            <g key={t.l}>
              <line x1={x(t.v)} x2={x(t.v)} y1={axisY} y2={axisY + 5} style={{ stroke: C.ink, strokeWidth: 0.9 }} />
              {(!narrow || t.v !== 1e12) && (
                <InkText x={x(t.v)} y={axisY + 17} kind="num" anchor="middle" size={narrow ? 9.5 : 10.5} role="soft" halo={false}>
                  {t.l}
                </InkText>
              )}
            </g>
          ))}
          {bar(yShip, ship, 'red', 'longitude, by the clock')}
          {bar(yRadio, radio, 'sky', 'range, by a radio signal', x(ship))}
        </svg>
      </div>
      <div className="cd-row">
        <label className="in-field" htmlFor="cd-err">
          clock error Δt
        </label>
        <input
          id="cd-err"
          className="in-range"
          type="range"
          min={LMIN}
          max={LMAX}
          step={0.01}
          value={Math.log10(s)}
          style={{ width: '100%', maxWidth: '20rem' }}
          onChange={(e) => {
            setLg(Number(e.target.value));
            setSelP(null);
          }}
        />
        <span className="in-readout">{fmtTime(s)}</span>
      </div>
      <Working label="Clock error times speed">
        <Eq>
          <Sym role="red">longitude error</Sym> = <Sym>Δt</Sym> × <Sym>Earth&apos;s turn at the equator</Sym>
        </Eq>
        <Eq>
          <V role="red">{fmtDist(ship)}</V> = <V>{fmtTime(s)}</V> × <V>463.8 m/s</V>
        </Eq>
        <Eq>
          <Sym role="sky">range error</Sym> = <Sym>Δt</Sym> × <Sym>c</Sym>
        </Eq>
        <Eq>
          <V role="sky">{fmtDist(radio)}</V> = <V>{fmtTime(s)}</V> × <V>299,792,458 m/s</V>
        </Eq>
      </Working>
      <p className="cd-note">{(PRESETS.find((p) => p.id === hoverP) ?? { note: ' ' }).note}</p>
      <style>{`.cd-row{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;margin-top:.6rem}.cd-note{margin:.4rem 0 0;min-height:1.5em;font-family:var(--font-display);font-style:italic;font-size:.98rem;color:var(--in-ink-soft)}`}</style>
    </Plate>
  );
}
