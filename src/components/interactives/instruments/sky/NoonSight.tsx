/**
 * Plate: latitude from the noon sun. A cut through the Earth along your meridian: the sun's
 * parallel rays come in at its declination δ (set by the date), you measure its altitude h at
 * noon, and the angle at the centre between the equator and you is φ = 90° − h + δ.
 * Geometry and declination live in ./noonGeom.ts (swept by the verification script).
 */
import { useState } from 'react';
import { C, Controls, Eq, InkText, Plate, Slider, Sym, V, Working, arcPath, polar, useMeasuredWidth } from '../shared';
import { DOY_PRESET, H_MAX, ISTANBUL, clampH, dateLabel, dateOf, declination, dms, hMin, noonScene, sdm } from './noonGeom';

const RAD = Math.PI / 180;
const dec0 = declination(dateOf(DOY_PRESET));
const H0 = Math.round((90 - ISTANBUL + dec0) * 60) / 60;

export default function NoonSight() {
  const [doy, setDoy] = useState(DOY_PRESET);
  const [hRaw, setH] = useState(H0);
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(640);
  const dec = declination(dateOf(doy));
  const lo = hMin(dec);
  const h = clampH(hRaw, dec); // the slider and the drawing always agree
  const s = noonScene(h, dec, W);
  const { HH, phi, Re, cx, cy, obs, zen, horS, horN, sunAt, labels: L } = s;
  const ink = { fill: 'none', stroke: C.ink, strokeWidth: 1.1 };
  const rayYs = [-0.55, 0, 0.55];
  const ist = polar(cx, cy, Re, ISTANBUL);
  const ist2 = polar(cx, cy, Re + 9, ISTANBUL);
  const nearIst = Math.abs(phi - ISTANBUL) < 0.5;

  return (
    <Plate
      n={5}
      title="Latitude from the noon sun"
      kicker="sun due south at noon"
      caption={
        <>
          A slice through the Earth along your meridian. The date fixes where the sun stands over the
          Earth (its declination δ); the sextant gives its height above your horizon at noon (h). Your
          latitude φ, the angle from the equator, follows. Each arcminute of error in h moves you one
          nautical mile, 1.85 km, north or south. The small tick on the rim is Istanbul.
        </>
      }
    >
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${HH}`} width="100%" style={{ display: 'block' }} role="img"
          aria-label={`Meridian section: declination ${dec.toFixed(1)}, altitude ${h.toFixed(1)}, latitude ${phi.toFixed(1)} degrees`}>
          {/* sunlight from the right, tilted by the declination */}
          {rayYs.map((o) => {
            const base = { x: cx + Math.sin(dec * RAD) * o * Re * 1.2, y: cy + Math.cos(dec * RAD) * o * Re * 1.2 };
            const far = polar(base.x, base.y, W, dec);
            const near = polar(base.x, base.y, Re * 1.25, dec);
            return <line key={o} x1={far.x} y1={far.y} x2={near.x} y2={near.y} style={{ stroke: C.red, strokeWidth: 0.7, opacity: 0.45 }} />;
          })}
          <InkText x={W - 10} y={14} anchor="end" size={12} role="soft">sunlight →</InkText>
          {/* Earth */}
          <circle cx={cx} cy={cy} r={Re} style={{ ...ink, fill: C.paper, strokeWidth: 1.3 }} />
          <line x1={cx - Re - 10} y1={cy} x2={cx + Re + 16} y2={cy} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />
          <InkText x={L.equator.x} y={L.equator.y} size={11} role="soft">{L.equator.text}</InkText>
          <line x1={cx} y1={cy - Re - 8} x2={cx} y2={cy + Re + 8} style={{ stroke: C.faint, strokeWidth: 0.7 }} />
          <InkText x={L.pole.x} y={L.pole.y} size={11} role="soft">{L.pole.text}</InkText>
          <line x1={ist.x} y1={ist.y} x2={ist2.x} y2={ist2.y} style={{ stroke: C.soft, strokeWidth: 1 }} />
          {/* centre: φ (ink) and δ (brass) */}
          <line x1={cx} y1={cy} x2={zen.x} y2={zen.y} style={{ stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '5 4' }} />
          {(() => {
            const p = polar(cx, cy, Re * 0.95, dec);
            return <line x1={cx} y1={cy} x2={p.x} y2={p.y} style={{ stroke: C.brass, strokeWidth: 0.9, strokeDasharray: '3 3' }} />;
          })()}
          <path d={arcPath(cx, cy, s.phiArcR, 0, phi)} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.4 }} />
          <InkText x={L.phi.x} y={L.phi.y} anchor="middle" baseline="central" size={14} role="red">φ</InkText>
          {Math.abs(dec) > 1.5 && (
            <>
              <path d={arcPath(cx, cy, s.decArcR, 0, dec)} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1.3 }} />
              <InkText x={L.dec.x} y={L.dec.y} anchor="middle" baseline="central" size={14} role="brass">δ</InkText>
            </>
          )}
          <circle cx={cx} cy={cy} r={2} style={{ fill: C.ink }} />
          {/* the observer, the horizon, the zenith, the sun's altitude h (sky) */}
          <line x1={horN.x} y1={horN.y} x2={horS.x} y2={horS.y} style={{ ...ink, strokeWidth: 1.4 }} />
          <line x1={obs.x} y1={obs.y} x2={sunAt.x} y2={sunAt.y} style={{ stroke: C.red, strokeWidth: 1.2, opacity: 0.8 }} />
          <path d={arcPath(obs.x, obs.y, s.hArcR, phi - 90, dec)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.5 }} />
          <InkText x={L.h.x} y={L.h.y} anchor="middle" baseline="central" size={14} role="sky">h</InkText>
          <circle cx={obs.x} cy={obs.y} r={3.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
          <InkText x={L.zenith.x} y={L.zenith.y} size={11} role="soft">zenith</InkText>
          <InkText x={L.horizon.x} y={L.horizon.y} size={11} role="soft">horizon</InkText>
        </svg>
      </div>
      <Controls>
        <Slider label="date" value={doy} min={1} max={365} step={1} onChange={setDoy} format={dateLabel} />
        <Slider label="sun's height at noon h" value={h} min={lo} max={H_MAX} step={1 / 60} onChange={setH} format={(v) => dms(v, '')} />
        <button
          type="button"
          className="in-seg"
          style={{ padding: '0.35rem 0.7rem', fontFamily: 'var(--font-mono)', fontSize: '0.66rem', letterSpacing: '0.12em', textTransform: 'uppercase', color: C.ink, background: 'transparent' }}
          onClick={() => {
            setDoy(DOY_PRESET);
            setH(H0);
          }}
        >
          Istanbul, 25 Oct
        </button>
      </Controls>
      <Working label="Latitude from the noon altitude">
        <Eq>
          <Sym role="red">φ</Sym> = 90° − <Sym role="sky">h</Sym> + <Sym role="brass">δ</Sym>
        </Eq>
        <Eq>
          <V role="red">{dms(phi)}</V> = 90° − <V role="sky">{dms(h, '')}</V> + (<V role="brass">{sdm(dec)}</V>)
        </Eq>
        <Eq>
          <span style={{ fontSize: '0.9em' }}>
            δ on {dateLabel(doy)}: the sun stands {dms(Math.abs(dec), '')} {dec >= 0 ? 'north' : 'south'} of the equator
            {nearIst ? '; that latitude is Istanbul' : ''}
          </span>
        </Eq>
      </Working>
    </Plate>
  );
}
