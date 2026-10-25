/**
 * Episode 8 (small plate): positions from satellites, RTK. A base receiver sits on a known
 * point (A); a rover anywhere within 30 km gets its position relative to the base to
 * 8 mm + 1 ppm horizontally and 15 mm + 1 ppm vertically (Trimble SPS855 spec). The reader
 * puts the rover on each point of the site and reads the error. No baseline, no line of sight.
 * Drawn on the stable site reference; the error circles are far too small to see at site
 * scale, so they are drawn ×EX (said on the drawing).
 */
import { useState } from 'react';
import { C, Eq, Plate, Segmented, Sym, V, Working } from '../shared';
import { SiteRef } from './SiteRef';
import { SITE, FAR_X } from './site';

export const RTK = { h0: 0.008, v0: 0.015, ppm: 1e-6 };
export const rtkErr = (dist: number) => ({ h: RTK.h0 + RTK.ppm * dist, v: RTK.v0 + RTK.ppm * dist });
const POINTS = [
  { id: 'C', name: 'Far quay, C', x: FAR_X, y: 0, z: SITE.FAR_Z },
  { id: 'M', name: 'Minaret finial', x: 0, y: 0, z: SITE.HILL_Z + SITE.MINARET_H },
  { id: 'far', name: '25 km away', x: 25_000, y: 0, z: 30 },
] as const;
type Id = (typeof POINTS)[number]['id'];
const EX = 2000;

export default function GpsPlate() {
  const [id, setId] = useState<Id>('C');
  const p = POINTS.find((q) => q.id === id)!;
  const d = Math.hypot(p.x - SITE.SHORE_X, p.y, p.z - SITE.QUAY_Z);
  const e = rtkErr(d);
  const mm = (m: number) => `${(m * 1000).toFixed(1)} mm`;
  return (
    <Plate
      n={13}
      title="Positions from satellites, with a base station"
      caption={
        <>
          RTK: a base receiver on a known point and a rover share the same satellite signals and cancel most of their errors.
          Accuracy from a survey receiver’s specification: 8 mm + 1 ppm horizontally, 15 mm + 1 ppm vertically, up to 30 km.
          Circles drawn ×{EX}.
        </>
      }
    >
      <Segmented label="where the rover stands" options={POINTS.map((q) => ({ value: q.id, label: q.name }))} value={id} onChange={setId} />
      <div style={{ marginTop: '0.6rem' }}>
        <SiteRef
          hi={[]}
          best={`rover ${d >= 10_000 ? `${(d / 1000).toFixed(0)} km` : `${d.toFixed(0)} m`} out: ±${mm(e.h)} across, ±${mm(e.v)} up`}
          label={`Base at A, rover at ${p.name}`}
          overlay={(g) => {
            const base = g.names.pA;
            const inPlan = p.x <= SITE.END_X;
            const r = g.names[id === 'M' ? 'pM' : 'pC'];
            const sec = id === 'M' ? g.names.tip : g.names.C;
            const rp = e.h * g.k * EX;
            const rv = e.v * g.kz * EX;
            return (
              <g>
                <circle cx={base.x} cy={base.y} r={6} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.4 }} />
                {inPlan ? (
                  <>
                    <line x1={base.x} y1={base.y} x2={r.x} y2={r.y} style={{ stroke: C.sky, strokeWidth: 0.8, strokeDasharray: '3 3' }} />
                    <circle cx={r.x} cy={r.y} r={rp} style={{ fill: C.red, opacity: 0.2, stroke: C.red, strokeWidth: 1 }} />
                    <ellipse cx={sec.x} cy={sec.y} rx={rp} ry={rv} style={{ fill: C.red, opacity: 0.2, stroke: C.red, strokeWidth: 1 }} />
                  </>
                ) : (
                  <path d={`M ${g.W - 26} ${base.y - 6} l 14 6 l -14 6`} style={{ fill: 'none', stroke: C.red, strokeWidth: 1.6 }} />
                )}
              </g>
            );
          }}
        />
      </div>
      <Working label="RTK accuracy">
        <Eq>
          <Sym role="red">σ</Sym>
          <sub>across</sub> = 8 mm + 1 ppm · <Sym role="sky">d</Sym>, <Sym role="red">σ</Sym>
          <sub>up</sub> = 15 mm + 1 ppm · <Sym role="sky">d</Sym>
        </Eq>
        <Eq>
          <V role="red">{mm(e.h)}</V> and <V role="red">{mm(e.v)}</V> at <V role="sky">{d >= 10_000 ? `${(d / 1000).toFixed(1)} km` : `${d.toFixed(0)} m`}</V>
        </Eq>
      </Working>
    </Plate>
  );
}
