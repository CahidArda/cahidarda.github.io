/**
 * Plate 2: the sine quadrant (rubʿ al-mujayyab, rubu tahtası), drawn after the 1848 study
 * sheet: a quarter circle filled with a 60 × 60 grid, a graduated rim, and one red line (the
 * plumb thread). Drag the thread (or use the slider) to the sun's altitude; the drawing shows
 * where 60·sin h is read off the grid, and the panel follows the muvakkit from sin h to the hour.
 */
import { useRef, useState } from 'react';
import type { PointerEvent as RPointerEvent } from 'react';
import {
  ArcScale,
  C,
  Controls,
  InkText,
  Plate,
  Segmented,
  Slider,
  arcPath,
  polar,
  useHoverPreview,
} from '../shared';
import {
  PHI,
  cosd,
  dayLabel,
  declination,
  hm,
  hourAngle,
  noonAltitude,
  sind,
} from './astro';

const W = 440;
const H = 440;
const BAND = 22; // width of the numbered bands along the two straight edges
const AX = 30 + BAND; // apex (centre of the quarter circle)
const AY = 22 + BAND;
const R = 350; // radius of the gridded quarter
const U = R / 60; // one "part"

/** Board coordinates: thread at altitude h hangs at angle (h − 90°) in maths convention. */
const threadEnd = (h: number, r = R) => polar(AX, AY, r, h - 90);

export default function SineQuadrant() {
  const [h, setH] = useState(30);
  const [day, setDay] = useState(80); // 21 Mar
  const [selHalf, setHalf] = useState<'am' | 'pm'>('am');
  const [half, bindHalf] = useHoverPreview(selHalf);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef(false);

  const dec = declination(day);
  const hMax = noonAltitude(dec);
  const hEff = Math.min(h, hMax);
  const tooHigh = h > hMax + 0.05;

  const x60 = 60 * sind(hEff);
  const a = sind(PHI) * sind(dec);
  const b = cosd(PHI) * cosd(dec);
  const cosH = (sind(hEff) - a) / b;
  const Hdeg = hourAngle(hEff, dec) ?? 0;
  const fromNoon = Hdeg / 15;
  const clock = half === 'am' ? 12 - fromNoon : 12 + fromNoon;

  const setFromPointer = (e: RPointerEvent<SVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const p = pt.matrixTransform(ctm.inverse());
    const ang = (Math.atan2(p.x - AX, p.y - AY) * 180) / Math.PI; // from the down edge
    setH(Math.round(Math.max(0, Math.min(90, ang)) * 2) / 2);
  };

  const tip = threadEnd(hEff);
  const bead = threadEnd(hEff, R * 0.62);
  const xRead = AX + x60 * U;
  const yRead = AY + 60 * cosd(hEff) * U;
  const xOffset = AX + 60 * a * U;

  // grid lines clipped to the quarter circle
  const grid: { d: string; major: boolean }[] = [];
  for (let k = 1; k < 60; k++) {
    const t = k * U;
    const len = Math.sqrt(R * R - t * t);
    const major = k % 5 === 0;
    grid.push({ d: `M ${AX + t} ${AY} V ${AY + len} M ${AX} ${AY + t} H ${AX + len}`, major });
  }

  return (
    <Plate
      n={2}
      label="Levha"
      id="plate-quadrant"
      title="The sine quadrant"
      kicker="drag the red thread"
      caption={
        <>
          Drawn after the 1848 study sheet. The thread hangs from the apex while the top edge is
          aimed at the sun, so the angle it makes with the left edge is the sun&apos;s altitude.
          Where it crosses the rim, the vertical grid line runs up to 60·sin h on the top scale.
        </>
      }
    >
      <div className="sq-split">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          role="img"
          aria-label={`Sine quadrant with the thread at ${hEff.toFixed(1)} degrees`}
          style={{ display: 'block', maxWidth: 460, margin: '0 auto', userSelect: 'none' }}
        >
          {/* outer border bands with numbers (top and left edges) */}
          <path
            d={`M ${AX - BAND} ${AY - BAND} H ${AX + R + 6} V ${AY} M ${AX - BAND} ${AY - BAND} V ${AY + R + 6} H ${AX}`}
            style={{ fill: 'none', stroke: C.ink, strokeWidth: 1 }}
          />
          <path
            d={`M ${AX - BAND + 3} ${AY - BAND + 3} H ${AX + R + 3} M ${AX - BAND + 3} ${AY - BAND + 3} V ${AY + R + 3}`}
            style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.6 }}
          />
          {Array.from({ length: 12 }, (_, i) => (i + 1) * 5).map((v) => (
            <g key={v}>
              <InkText x={AX + v * U} y={AY - BAND / 2 + 1} kind="num" anchor="middle" baseline="central" size={9} role="soft" halo={false}>
                {v}
              </InkText>
              <InkText x={AX - BAND / 2} y={AY + v * U} kind="num" anchor="middle" baseline="central" size={9} role="soft" halo={false}>
                {v}
              </InkText>
            </g>
          ))}
          {/* sights on the top edge */}
          {[AX + 10, AX + R - 4].map((x) => (
            <rect key={x} x={x - 4} y={AY - BAND - 9} width={8} height={9} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 0.9 }} />
          ))}

          {/* the grid */}
          {grid.map((g, i) => (
            <path
              key={i}
              d={g.d}
              style={{ fill: 'none', stroke: g.major ? C.soft : C.faint, strokeWidth: g.major ? 0.55 : 0.35 }}
            />
          ))}
          {/* the two sine semicircles with diameters on the edges */}
          <path d={`M ${AX} ${AY} A ${R / 2} ${R / 2} 0 0 0 ${AX + R} ${AY}`} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.7 }} />
          <path d={`M ${AX} ${AY} A ${R / 2} ${R / 2} 0 0 1 ${AX} ${AY + R}`} style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.7 }} />
          {/* edges of the quarter */}
          <path
            d={`M ${AX + R} ${AY} H ${AX} V ${AY + R}`}
            style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }}
          />
          {/* graduated rim */}
          <path d={arcPath(AX, AY, R, -90, 0)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
          <ArcScale
            cx={AX}
            cy={AY}
            r={R + 2}
            from={-90}
            to={0}
            major={10}
            minor={1}
            labelEvery={10}
            value={(v) => v + 90}
            tick={8}
            size={9.5}
            rim={false}
          />
          <path d={arcPath(AX, AY, R + 26, -90, 0)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.8 }} />

          {/* reading: from where the thread meets the rim, straight up to the top scale */}
          <path
            d={`M ${tip.x} ${tip.y} V ${AY}`}
            style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1, strokeDasharray: '4 2.5' }}
          />
          <path
            d={`M ${tip.x} ${tip.y} H ${AX}`}
            style={{ fill: 'none', stroke: C.soft, strokeWidth: 0.8, strokeDasharray: '2 3' }}
          />
          <circle cx={xRead} cy={AY} r={3.2} style={{ fill: C.ink }} />
          <InkText x={xRead} y={AY + 15} kind="num" anchor={xRead > AX + R - 40 ? 'end' : 'middle'} size={11.5}>
            {x60.toFixed(1)}
          </InkText>
          <InkText x={AX + 8} y={yRead - 6} kind="num" size={10} role="soft">
            {(60 * cosd(hEff)).toFixed(1)} = 60·cos h
          </InkText>
          {/* the muvakkit's offset for this date: 60·sin φ·sin δ along the same scale */}
          {Math.abs(a) > 0.005 && (
            <g>
              <path d={`M ${xOffset} ${AY - 3} V ${AY + 34}`} style={{ stroke: C.brass, strokeWidth: 1 }} />
              <InkText x={xOffset + (a > 0 ? 4 : 4)} y={AY + 44} size={10.5} role="brass">
                sin φ sin δ
              </InkText>
            </g>
          )}

          {/* the thread */}
          <line x1={AX} y1={AY} x2={tip.x} y2={tip.y} style={{ stroke: C.red, strokeWidth: 1.6 }} />
          <circle cx={bead.x} cy={bead.y} r={3.6} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.3 }} />
          <circle cx={AX} cy={AY} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          {(() => {
            const bob = threadEnd(hEff, R + 40);
            return (
              <g>
                <line x1={tip.x} y1={tip.y} x2={bob.x} y2={bob.y} style={{ stroke: C.red, strokeWidth: 1.2 }} />
                <path
                  d={`M ${bob.x} ${bob.y} m -4 0 l 4 12 l 4 -12 z`}
                  transform={`rotate(${-hEff} ${bob.x} ${bob.y})`}
                  style={{ fill: C.paper, stroke: C.red, strokeWidth: 1 }}
                />
              </g>
            );
          })()}
          {/* angle label at the apex */}
          <path d={arcPath(AX, AY, 70, -90, hEff - 90)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.8 }} />
          <InkText x={polar(AX, AY, 84, (hEff - 180) / 2).x + 4} y={polar(AX, AY, 84, (hEff - 180) / 2).y + 4} kind="num" anchor="start" baseline="central" size={11}>
            {`h = ${hEff.toFixed(1)}°`}
          </InkText>

          {/* hit area for dragging (touch: only here, so the page still scrolls elsewhere) */}
          <line
            x1={AX}
            y1={AY}
            x2={threadEnd(hEff, R + 50).x}
            y2={threadEnd(hEff, R + 50).y}
            style={{ stroke: 'transparent', strokeWidth: 30, cursor: 'grab', touchAction: 'none' }}
            onPointerDown={(e) => {
              drag.current = true;
              (e.target as Element).setPointerCapture(e.pointerId);
              setFromPointer(e);
            }}
            onPointerMove={(e) => drag.current && setFromPointer(e)}
            onPointerUp={() => (drag.current = false)}
            onPointerCancel={() => (drag.current = false)}
          />

          {/* inset: how the board is held */}
          <Inset h={hEff} />
        </svg>

        <ol className="sq-steps">
          <li>
            <span className="sq-k">thread</span>
            <span>
              Aim the top edge at the sun. The thread sits at <b>{hEff.toFixed(1)}°</b>, and the
              grid gives <b>{x60.toFixed(1)}</b> parts of 60: sin h = <b>{(x60 / 60).toFixed(3)}</b>.
            </span>
          </li>
          <li>
            <span className="sq-k">date</span>
            <span>
              On {dayLabel(day)} the sun&apos;s declination is δ = {dec.toFixed(1)}°. For 41° the
              board already carries sin φ sin δ = <b>{a.toFixed(3)}</b>
              {Math.abs(a) > 0.005 ? ' (the brass mark)' : ''}.
            </span>
          </li>
          <li>
            <span className="sq-k">divide</span>
            <span>
              Subtract, then scale by cos φ cos δ = {b.toFixed(3)} with a pair of similar triangles:
              cos H = <b>{cosH.toFixed(3)}</b>.
            </span>
          </li>
          <li>
            <span className="sq-k">hour</span>
            <span>
              Read cos H back off the rim: H = <b>{Hdeg.toFixed(1)}°</b>, and at 15° an hour that is{' '}
              <b>
                {Math.floor(fromNoon)} h {Math.round((fromNoon % 1) * 60)} min
              </b>{' '}
              from noon. The time is <b className="sq-time">{hm(clock)}</b> local solar time.
            </span>
          </li>
          {tooHigh && (
            <li className="sq-warn">
              The sun never climbs above {hMax.toFixed(1)}° at Istanbul on {dayLabel(day)}, so the
              thread stops at noon.
            </li>
          )}
        </ol>
      </div>

      <Controls>
        <Slider label="altitude" value={h} min={0} max={90} step={0.5} onChange={setH} format={(v) => `${v.toFixed(1)}°`} width="8rem" />
        <Slider label="date" value={day} min={1} max={365} step={1} onChange={setDay} format={dayLabel} width="8rem" />
        <Segmented
          label="Morning or afternoon"
          options={[
            { value: 'am', label: 'morning' },
            { value: 'pm', label: 'afternoon' },
          ]}
          value={selHalf}
          onChange={setHalf}
          preview={bindHalf}
        />
      </Controls>
      <style>{CSS}</style>
    </Plate>
  );
}

/** Small sketch in the empty corner: the board held up, the sight edge on the sun, the
 *  thread plumb. The apex is the sun end of the sight edge; the other edges fall away from it. */
function Inset({ h }: { h: number }) {
  const s = 40;
  const r = (h * Math.PI) / 180;
  const ox = W - 78;
  const oy = H - 100;
  const X = { x: Math.cos(r), y: Math.sin(r) }; // sight edge, away from the sun
  const Y = { x: -Math.sin(r), y: Math.cos(r) }; // the other straight edge
  const B = { x: ox + X.x * s, y: oy + X.y * s };
  const D = { x: ox + Y.x * s, y: oy + Y.y * s };
  const ground = oy + s + 14;
  return (
    <g>
      <line x1={ox - 52} x2={ox + 56} y1={ground} y2={ground} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
      <InkText x={ox + 56} y={ground + 12} anchor="end" size={10} role="soft">
        horizon
      </InkText>
      <path
        d={`M ${ox} ${oy} L ${B.x} ${B.y} A ${s} ${s} 0 0 1 ${D.x} ${D.y} Z`}
        style={{ fill: C.paper, stroke: C.ink, strokeWidth: 0.9 }}
      />
      <line x1={ox} y1={oy} x2={ox} y2={oy + s + 4} style={{ stroke: C.red, strokeWidth: 1.1 }} />
      <line
        x1={B.x}
        y1={B.y}
        x2={ox - X.x * 30}
        y2={oy - X.y * 30}
        style={{ stroke: C.brass, strokeWidth: 0.9, strokeDasharray: '3 2' }}
      />
      <circle cx={ox - X.x * 37} cy={oy - X.y * 37} r={5} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1 }} />
      <InkText x={ox - X.x * 37} y={oy - X.y * 37 - 10} anchor="middle" size={10} role="soft">
        sun
      </InkText>
    </g>
  );
}

const CSS = `
.sq-split { display: grid; grid-template-columns: 1fr; gap: 1rem; }
@media (min-width: 46rem) { .sq-split { grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); align-items: center; } }
.sq-steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.55rem; counter-reset: sq; }
.sq-steps li { display: grid; grid-template-columns: 4.2rem 1fr; gap: 0.6rem; font-family: var(--font-display); font-size: 0.98rem; line-height: 1.4; color: var(--in-ink-soft); border-bottom: 1px dotted var(--in-ink-faint); padding-bottom: 0.45rem; }
.sq-steps li b { font-family: var(--font-mono); font-weight: 400; font-size: 0.86rem; color: var(--in-ink); font-variant-numeric: tabular-nums; }
.sq-steps .sq-k { font-family: var(--font-mono); font-size: 0.64rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--in-ink-soft); padding-top: 0.25rem; }
.sq-steps li::before { content: none; }
.sq-steps .sq-time { color: var(--in-red); font-size: 1rem; }
.sq-steps .sq-warn { display: block; font-style: italic; color: var(--in-red); }
`;
