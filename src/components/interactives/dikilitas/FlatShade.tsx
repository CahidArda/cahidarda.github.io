/**
 * How the obelisk gets its printed look: no lighting model, one flat colour per face.
 * Left: the shaft seen from above, with a fixed light and the viewer. Right: what the viewer
 * sees. Both panels share one SVG coordinate system (skill 4.1); the turn is driven by a
 * slider or a slow auto-turn while in view, and a static pose under reduced motion.
 */
import { useEffect, useState } from 'react';
import { Widget, useInView, useReducedMotion } from '../dataplat/shared';

const INK = 'var(--color-dk-ink)';
const GRANITE = [201, 143, 120];
// light direction in camera space (x right, y up, z toward the viewer), as in the app
const L = (() => {
  const v = [-0.55, 0.5, 1];
  const m = Math.hypot(...v);
  return v.map((c) => c / m);
})();
const tintOf = (nx: number, nz: number) => 0.7 + 0.36 * Math.max(0, nx * L[0] + nz * L[2]);
const rgb = (k: number) => `rgb(${GRANITE.map((c) => Math.round(Math.min(255, c * k))).join(',')})`;

// plan view: centre and half-diagonal of the square section
const PX = 150;
const PZ = 140;
const PR = 62;
// elevation: centre line, base, top of shaft, apex, horizontal scale from plan units
const EX = 470;
const EB = 270;
const ET = 70;
const EA = 30;
const TAPER = 0.7;

export default function FlatShade() {
  const reduced = useReducedMotion();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [deg, setDeg] = useState(28);
  const [held, setHeld] = useState(false);

  useEffect(() => {
    if (reduced || held || !inView) return;
    const id = setInterval(() => setDeg((d) => (d + 0.6) % 360), 40);
    return () => clearInterval(id);
  }, [reduced, held, inView]);

  const th = (deg * Math.PI) / 180;
  // corners of the square section, in plan units (x right, z toward the viewer = down)
  const corners = [0, 1, 2, 3].map((k) => {
    const a = th + Math.PI / 4 + (k * Math.PI) / 2;
    return { x: Math.cos(a), z: Math.sin(a) };
  });
  const faces = [0, 1, 2, 3].map((k) => {
    const a = corners[k];
    const b = corners[(k + 1) % 4];
    const mx = (a.x + b.x) / 2;
    const mz = (a.z + b.z) / 2;
    const m = Math.hypot(mx, mz);
    const nx = mx / m;
    const nz = mz / m;
    return {
      a,
      b,
      nx,
      nz,
      visible: nz > 0.001,
      k: tintOf(nx, nz),
      name: ['I', 'II', 'III', 'IV'][k],
    };
  });

  const lamp = { x: PX + L[0] * 118, z: PZ + L[2] * 118 * 0.92 };
  const sx = (x: number) => EX + x * 78;

  return (
    <Widget
      title="One flat colour per face"
      kicker={held ? 'turn: manual' : reduced ? 'static' : 'turning'}
      rootRef={viewRef}
    >
      <svg
        viewBox="0 0 640 300"
        className="block h-auto w-full"
        role="img"
        aria-label="The obelisk seen from above with a fixed light, and the flat colours its visible faces get"
      >
        {/* labels */}
        <text
          x={PX}
          y={20}
          textAnchor="middle"
          className="font-mono"
          style={{ fill: 'var(--color-muted)', fontSize: 11 }}
        >
          from above
        </text>
        <text
          x={EX}
          y={20}
          textAnchor="middle"
          className="font-mono"
          style={{ fill: 'var(--color-muted)', fontSize: 11 }}
        >
          what you see
        </text>
        <line
          x1={320}
          y1={30}
          x2={320}
          y2={285}
          style={{ stroke: 'var(--color-line)', strokeDasharray: '3 4' }}
        />

        {/* light ray and lamp */}
        <line
          x1={lamp.x}
          y1={lamp.z}
          x2={PX + L[0] * PR * 0.95}
          y2={PZ + L[2] * PR * 0.95}
          style={{ stroke: 'var(--color-accent)', strokeWidth: 1.5, strokeDasharray: '4 3' }}
        />
        <circle
          cx={lamp.x}
          cy={lamp.z}
          r={9}
          style={{ fill: 'var(--color-paper)', stroke: 'var(--color-accent)', strokeWidth: 2 }}
        />
        <text
          x={lamp.x - 14}
          y={lamp.z + 4}
          textAnchor="end"
          className="font-mono"
          style={{ fill: 'var(--color-accent)', fontSize: 11 }}
        >
          light
        </text>

        {/* viewer */}
        <path d={`M ${PX - 10} 280 L ${PX} 266 L ${PX + 10} 280 Z`} style={{ fill: INK }} />
        <text
          x={PX + 16}
          y={279}
          className="font-mono"
          style={{ fill: 'var(--color-muted)', fontSize: 11 }}
        >
          you
        </text>

        {/* plan: the square section, visible faces drawn in their colour */}
        <polygon
          points={corners.map((c) => `${PX + c.x * PR},${PZ + c.z * PR}`).join(' ')}
          style={{ fill: 'var(--color-paper)', stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        {faces.map((f, i) => (
          <g key={i}>
            <line
              x1={PX + f.a.x * PR}
              y1={PZ + f.a.z * PR}
              x2={PX + f.b.x * PR}
              y2={PZ + f.b.z * PR}
              style={{
                stroke: f.visible ? rgb(f.k) : 'var(--color-line-strong)',
                strokeWidth: f.visible ? 8 : 2,
                strokeLinecap: 'round',
              }}
            />
            <text
              x={PX + f.nx * (PR * 0.71 + 24)}
              y={PZ + f.nz * (PR * 0.71 + 24) + 4}
              textAnchor="middle"
              className="font-mono"
              style={{
                fill: f.visible ? INK : 'var(--color-muted)',
                fontSize: 11,
                fontWeight: f.visible ? 600 : 400,
              }}
            >
              {f.name}
            </text>
          </g>
        ))}

        {/* elevation: each visible face is one flat polygon, outlined in ink */}
        {faces
          .filter((f) => f.visible)
          .map((f, i) => {
            const x0 = f.a.x;
            const x1 = f.b.x;
            const shaft = `${sx(x0)},${EB} ${sx(x1)},${EB} ${sx(x1 * TAPER)},${ET} ${sx(x0 * TAPER)},${ET}`;
            const cap = `${sx(x0 * TAPER)},${ET} ${sx(x1 * TAPER)},${ET} ${EX},${EA}`;
            const mid = (x0 + x1) / 2;
            return (
              <g key={i}>
                <polygon
                  points={shaft}
                  style={{ fill: rgb(f.k), stroke: INK, strokeWidth: 1.5, strokeLinejoin: 'round' }}
                />
                <polygon
                  points={cap}
                  style={{
                    fill: rgb(f.k * 0.97),
                    stroke: INK,
                    strokeWidth: 1.5,
                    strokeLinejoin: 'round',
                  }}
                />
                {Math.abs(x1 - x0) > 0.3 && (
                  <>
                    <text
                      x={sx(mid)}
                      y={176}
                      textAnchor="middle"
                      className="font-mono"
                      style={{ fill: INK, fontSize: 12, fontWeight: 600 }}
                    >
                      {f.name}
                    </text>
                    <text
                      x={sx(mid)}
                      y={194}
                      textAnchor="middle"
                      className="font-mono"
                      style={{ fill: INK, fontSize: 11 }}
                    >
                      {f.k.toFixed(2)}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        <line
          x1={EX - 120}
          y1={EB}
          x2={EX + 120}
          y2={EB}
          style={{ stroke: INK, strokeWidth: 1.5 }}
        />
      </svg>

      <label className="mt-3 flex items-center gap-3 font-mono text-[0.7rem] text-muted">
        turn
        <input
          type="range"
          min={0}
          max={359}
          value={Math.round(deg)}
          onChange={(e) => {
            setHeld(true);
            setDeg(+e.target.value);
          }}
          className="w-full"
          style={{ accentColor: 'var(--color-accent)' }}
          aria-label="Turn the obelisk"
        />
        {held && !reduced && (
          <button
            type="button"
            onClick={() => setHeld(false)}
            className="border px-2 py-0.5"
            style={{ borderColor: 'var(--color-line)' }}
          >
            spin
          </button>
        )}
      </label>
      <p className="mt-3 text-sm leading-snug text-ink">
        Each face's colour is the granite colour times one number, 0.7 + 0.36 × max(0, n · l), where
        n is the face's direction and l the light's. Turn it and only those numbers change: no
        shading inside a face, no shadows, no texture of stone. The ink outlines do the rest.
      </p>
    </Widget>
  );
}
