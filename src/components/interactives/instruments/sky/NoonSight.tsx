/**
 * Plate 4: latitude from the noon sun. A cut through the Earth along your meridian: the sun's
 * parallel rays come in at its declination δ (set by the date), you measure its altitude h at
 * noon, and the angle at the centre between the equator and you is φ = 90° − h + δ.
 * Declination: USNO "approximate solar coordinates" (about 1′ near the year 2000).
 */
import { useState } from 'react';
import { C, Controls, InkText, Plate, Readout, Slider, arcPath, polar, useMeasuredWidth } from '../shared';

const RAD = Math.PI / 180;
const YEAR = 2026;
const ISTANBUL = 41.0136;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Day of year (1..365) → Date at local noon in Istanbul (≈ 09:50 UTC). */
const dateOf = (doy: number) => new Date(Date.UTC(YEAR, 0, doy, 9, 50));
const label = (doy: number) => {
  const d = dateOf(doy);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
};

/** Solar declination in degrees (USNO approximate solar coordinates). */
export function declination(date: Date) {
  const D = date.getTime() / 86400000 + 2440587.5 - 2451545.0;
  const g = (357.529 + 0.98560028 * D) * RAD;
  const q = 280.459 + 0.98564736 * D;
  const L = (q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) * RAD;
  const e = (23.439 - 0.00000036 * D) * RAD;
  return Math.asin(Math.sin(e) * Math.sin(L)) / RAD;
}

export const dms = (v: number, pos = 'N', neg = 'S') => {
  const a = Math.abs(v);
  let d = Math.floor(a);
  let m = Math.round((a - d) * 60);
  if (m === 60) {
    d += 1;
    m = 0;
  }
  return `${d}°${String(m).padStart(2, '0')}′${pos ? ` ${v >= 0 ? pos : neg}` : ''}`;
};

const DOY_PRESET = 298; // 25 October 2026

export default function NoonSight() {
  const [doy, setDoy] = useState(DOY_PRESET);
  const dec0 = declination(dateOf(DOY_PRESET));
  const [h, setH] = useState(Math.round((90 - ISTANBUL + dec0) * 60) / 60);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const narrow = W < 520;
  const dec = declination(dateOf(doy));
  const phi = 90 - h + dec;
  const valid = phi <= 90;

  const HH = narrow ? 320 : 340;
  const Re = narrow ? Math.min(W * 0.3, 104) : 118;
  const cx = narrow ? W * 0.4 : W * 0.38;
  const cy = HH * 0.56;
  const phiD = Math.max(-89, Math.min(89.5, phi));
  const obs = polar(cx, cy, Re, phiD);
  const zen = polar(cx, cy, Re + (narrow ? 70 : 92), phiD);
  const sunDir = dec; // direction towards the sun, from anywhere (rays are parallel)
  const L = narrow ? 64 : 84;
  const hor1 = polar(obs.x, obs.y, L, phiD - 90); // horizon, pointing south (towards the sun)
  const hor2 = polar(obs.x, obs.y, L * 0.7, phiD + 90);
  const sunAt = polar(obs.x, obs.y, L * 1.6, sunDir);
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const rayYs = [-0.55, 0, 0.55];
  const ist = polar(cx, cy, Re, ISTANBUL);
  const ist2 = polar(cx, cy, Re + 9, ISTANBUL);

  return (
    <Plate
      n={4}
      title="Latitude from the noon sun"
      kicker="sun due south at noon"
      caption={
        <>
          A slice through the Earth along your meridian. The date fixes where the sun stands over the
          Earth (its declination δ); the sextant gives its height above your horizon at noon (h). The
          angle you are standing at, measured from the equator, follows. Each arc minute of error in
          h moves you one nautical mile, 1.85 km, north or south.
        </>
      }
    >
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${HH}`} width="100%" style={{ display: 'block' }} role="img"
          aria-label={`Meridian section: declination ${dec.toFixed(1)}, altitude ${h.toFixed(1)}, latitude ${phi.toFixed(1)} degrees`}>
          {/* sunlight from the right, tilted by the declination */}
          {rayYs.map((o) => {
            const base = { x: cx + Math.sin(sunDir * RAD) * o * Re * 1.2, y: cy + Math.cos(sunDir * RAD) * o * Re * 1.2 };
            const far = polar(base.x, base.y, W, sunDir);
            const near = polar(base.x, base.y, Re * 1.25, sunDir);
            return <line key={o} x1={far.x} y1={far.y} x2={near.x} y2={near.y} style={{ stroke: C.red, strokeWidth: 0.7, opacity: 0.45 }} />;
          })}
          {(() => {
            const p = polar(cx, cy, W - cx - 14, sunDir);
            return <InkText x={W - 10} y={p.y - 8} anchor="end" size={12} role="soft">to the sun</InkText>;
          })()}
          {/* Earth */}
          <circle cx={cx} cy={cy} r={Re} style={{ ...ink, fill: C.paper, strokeWidth: 1.3 }} />
          <line x1={cx - Re - 10} y1={cy} x2={cx + Re + 16} y2={cy} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />
          <InkText x={cx - Re + 6} y={cy - 6} size={11} role="soft">equator</InkText>
          <line x1={cx} y1={cy - Re - 8} x2={cx} y2={cy + Re + 8} style={{ stroke: C.faint, strokeWidth: 0.7 }} />
          <InkText x={cx + 4} y={cy - Re - 12} size={11} role="soft">N pole</InkText>
          {/* Istanbul tick */}
          <line x1={ist.x} y1={ist.y} x2={ist2.x} y2={ist2.y} style={{ stroke: C.soft, strokeWidth: 1 }} />
          {/* centre: φ and δ */}
          <line x1={cx} y1={cy} x2={zen.x} y2={zen.y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />
          {(() => {
            const s = polar(cx, cy, Re * 0.95, sunDir);
            return <line x1={cx} y1={cy} x2={s.x} y2={s.y} style={{ stroke: C.red, strokeWidth: 0.9, strokeDasharray: '3 3' }} />;
          })()}
          <path d={arcPath(cx, cy, Re * 0.42, 0, phiD)} style={{ ...ink, stroke: C.ink }} />
          {(() => {
            const p = polar(cx, cy, Re * 0.42 + 13, phiD / 2);
            return <InkText x={p.x} y={p.y} anchor="middle" baseline="central" size={13}>φ</InkText>;
          })()}
          {Math.abs(dec) > 1.5 && (
            <>
              <path d={arcPath(cx, cy, Re * 0.66, 0, sunDir)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1 }} />
              {(() => {
                const p = polar(cx, cy, Re * 0.66 + 11, sunDir / 2);
                return <InkText x={p.x} y={p.y} anchor="middle" baseline="central" size={13} role="red">δ</InkText>;
              })()}
            </>
          )}
          <circle cx={cx} cy={cy} r={2} style={{ fill: C.ink }} />
          {/* the observer, the horizon, the zenith, the sun's altitude */}
          <line x1={hor2.x} y1={hor2.y} x2={hor1.x} y2={hor1.y} style={{ ...ink, stroke: C.sky, strokeWidth: 1.4 }} />
          <line x1={obs.x} y1={obs.y} x2={sunAt.x} y2={sunAt.y} style={{ stroke: C.red, strokeWidth: 1.4 }} />
          <path d={arcPath(obs.x, obs.y, L * 0.62, phiD - 90, sunDir)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.1 }} />
          {(() => {
            const p = polar(obs.x, obs.y, L * 0.62 + 12, (phiD - 90 + sunDir) / 2);
            return <InkText x={p.x} y={p.y} anchor="middle" baseline="central" size={13} role="red">h</InkText>;
          })()}
          <circle cx={obs.x} cy={obs.y} r={3.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
          <InkText x={zen.x + 6} y={zen.y} size={11} role="soft">zenith</InkText>
          <InkText x={hor1.x + 4} y={hor1.y + 14} size={11} role="soft">horizon</InkText>
          {/* the working, in the margin */}
          <g>
            <InkText x={W - 10} y={HH - 58} anchor="end" kind="num" size={12} role="soft">{`90° − ${dms(h, '')}`}</InkText>
            <InkText x={W - 10} y={HH - 40} anchor="end" kind="num" size={12} role="soft">{`${dec >= 0 ? '+' : '−'} ${dms(Math.abs(dec), '')} (δ)`}</InkText>
            <line x1={W - (narrow ? 120 : 140)} y1={HH - 32} x2={W - 10} y2={HH - 32} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
            <InkText x={W - 10} y={HH - 14} anchor="end" kind="num" size={14} role={valid ? 'ink' : 'red'}>
              {valid ? `φ = ${dms(phi)}` : 'sun north of you'}
            </InkText>
          </g>
        </svg>
      </div>
      <Controls>
        <Slider label="date" value={doy} min={1} max={365} step={1} onChange={setDoy} format={label} />
        <Slider label="noon altitude h" value={h} min={10} max={89} step={1 / 60} onChange={setH} format={(v) => dms(v, '')} />
      </Controls>
      <Controls>
        <Readout label="declination δ">{dms(dec)}</Readout>
        <Readout label="latitude φ">{valid ? dms(phi) : '·'}</Readout>
        <button
          type="button"
          className="in-seg"
          style={{ padding: '0.35rem 0.7rem', fontFamily: 'var(--font-mono)', fontSize: '0.66rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: C.ink, background: 'transparent' }}
          onClick={() => {
            setDoy(DOY_PRESET);
            setH(Math.round((90 - ISTANBUL + dec0) * 60) / 60);
          }}
        >
          Istanbul, 25 Oct
        </button>
      </Controls>
    </Plate>
  );
}
