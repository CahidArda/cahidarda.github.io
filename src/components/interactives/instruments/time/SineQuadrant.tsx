/**
 * Plate 1: the sine quadrant (rubʿ al-mujayyab, rubu tahtası), drawn after the 1848 study
 * sheet. Drag the thread to the sun's altitude; the board then works the hour out the way a
 * muvakkit did, in parts of 60: read 60·sin h, shift it by the date term (subtract when the
 * sun is north of the equator, add when south), divide by the date's divisor with the thread
 * and bead (similar triangles), and read the hour angle off the rim. Geometry: ./quadrantGeom.
 */
import { useRef, useState } from 'react';
import type { PointerEvent as RPointerEvent } from 'react';
import {
  ArcScale,
  C,
  Controls,
  Eq,
  InkText,
  Plate,
  Segmented,
  Slider,
  Sym,
  V,
  Working,
  arcPath,
  polar,
  useHoverPreview,
} from '../shared';
import { dayLabel, hm } from './astro';
import { AX, AY, BAND, H, R, ROW, U, W, maxAltitude, quadrant, quadrantLabels, threadAt } from './quadrantGeom';

const p60 = (v: number) => (60 * v).toFixed(1);

export default function SineQuadrant() {
  const [hRaw, setHRaw] = useState(30);
  const [day, setDayRaw] = useState(80); // 21 Mar
  const [selHalf, setHalf] = useState<'am' | 'pm'>('am');
  const [half, bindHalf] = useHoverPreview(selHalf);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef(false);

  const hMax = maxAltitude(day);
  // the slider, the thread and the numbers all use the same clamped value
  const setH = (v: number) => setHRaw(Math.max(0, Math.min(hMax, v)));
  const setDay = (d: number) => {
    setDayRaw(d);
    setHRaw((v) => Math.min(v, maxAltitude(d)));
  };
  const q = quadrant(hRaw, day);
  const h = q.h;
  const atNoon = h >= hMax - 0.05;
  const fromNoon = q.Hdeg / 15;
  const clock = half === 'am' ? 12 - fromNoon : 12 + fromNoon;
  const lab = quadrantLabels(q);

  const setFromPointer = (e: RPointerEvent<SVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const p = pt.matrixTransform(ctm.inverse());
    const ang = (Math.atan2(p.x - AX, p.y - AY) * 180) / Math.PI; // from the left edge
    setH(Math.round(Math.max(0, Math.min(90, ang)) * 10) / 10);
  };

  // grid lines clipped to the quarter circle
  const grid: { d: string; major: boolean }[] = [];
  for (let k = 1; k < 60; k++) {
    const t = k * U;
    const len = Math.sqrt(R * R - t * t);
    const major = k % 5 === 0;
    grid.push({ d: `M ${AX + t} ${AY} V ${AY + len} M ${AX} ${AY + t} H ${AX + len}`, major });
  }

  const tip = q.tip;
  const { bead, divTip, shift } = q;
  const sub = q.a >= 0;

  return (
    <Plate
      n={1}
      label="Levha"
      id="plate-quadrant"
      title="Telling the time with a sine quadrant"
      kicker="drag the blue thread"
      caption={
        <>
          Drawn after the 1848 study sheet; all numbers are in parts of 60, as on the board. Aim
          the top edge at the sun and the thread hangs at the sun&apos;s altitude h. Its rim point
          sits under 60·sin h on the top scale. Shift that by the date term (brass: subtract when
          the sun is north of the equator, add when it is south), then set the thread again so
          the bead, placed at the divisor, lies on the shifted line: the rim now gives cos H, and
          the rim arc gives 90° − H.
        </>
      }
    >
      <div className="sq-split">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          role="img"
          aria-label={`Sine quadrant with the thread at ${h.toFixed(1)} degrees, giving ${hm(clock)} local solar time`}
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
          <path d={`M ${AX} ${AY} A ${R / 2} ${R / 2} 0 0 0 ${AX + R} ${AY}`} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.7 }} />
          <path d={`M ${AX} ${AY} A ${R / 2} ${R / 2} 0 0 1 ${AX} ${AY + R}`} style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.7 }} />
          <path d={`M ${AX + R} ${AY} H ${AX} V ${AY + R}`} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
          <path d={arcPath(AX, AY, R, -90, 0)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1 }} />
          <ArcScale cx={AX} cy={AY} r={R + 2} from={-90} to={0} major={10} minor={1} labelEvery={10} value={(v) => v + 90} tick={8} size={9.5} rim={false} />
          <path d={arcPath(AX, AY, R + 26, -90, 0)} style={{ fill: 'none', stroke: C.ink, strokeWidth: 0.8 }} />

          {/* step 1: from the thread's rim point straight up to 60·sin h */}
          <path d={`M ${tip.x} ${tip.y} V ${AY}`} style={{ fill: 'none', stroke: C.ink, strokeWidth: 1.1, strokeDasharray: '4 2.5' }} />
          <circle cx={q.xRead} cy={ROW.sin} r={3.2} style={{ fill: C.ink }} />
          <InkText x={lab.sin.x} y={lab.sin.y} kind="num" anchor="middle" size={lab.sin.size}>
            {lab.sin.s}
          </InkText>

          {/* step 2: the date term, shifted along a row of the grid (brass) */}
          {Math.abs(q.a) > 0.002 && (
            <g>
              <path
                d={`M ${shift[0]} ${ROW.date} H ${shift[1]}`}
                style={{ fill: 'none', stroke: C.brass, strokeWidth: 2 }}
              />
              {shift.map((x, i) => (
                <line key={i} x1={x} x2={x} y1={ROW.date - 5} y2={ROW.date + 5} style={{ stroke: C.brass, strokeWidth: 1.2 }} />
              ))}
              <InkText x={lab.date.x} y={lab.date.y} kind="num" anchor="middle" size={lab.date.size} role="brass">
                {lab.date.s}
              </InkText>
            </g>
          )}
          {/* the difference, carried down to the bead */}
          <path d={`M ${q.xAbs} ${ROW.date} V ${bead.y}`} style={{ fill: 'none', stroke: C.brass, strokeWidth: 1, strokeDasharray: '2 2.5' }} />

          {/* step 3: the dividing thread through the bead (red), and its reading */}
          <line x1={AX} y1={AY} x2={divTip.x} y2={divTip.y} style={{ stroke: C.red, strokeWidth: 1.1, strokeDasharray: '6 3' }} />
          <path d={`M ${divTip.x} ${divTip.y} V ${AY}`} style={{ fill: 'none', stroke: C.red, strokeWidth: 1, strokeDasharray: '4 2.5' }} />
          <circle cx={q.xCos} cy={ROW.cos} r={3.2} style={{ fill: C.red }} />
          <InkText x={lab.cos.x} y={lab.cos.y} kind="num" anchor="middle" size={lab.cos.size} role="red">
            {lab.cos.s}
          </InkText>
          <circle cx={bead.x} cy={bead.y} r={4} style={{ fill: C.paper, stroke: C.soft, strokeWidth: 1.4 }} />
          <circle cx={divTip.x} cy={divTip.y} r={3.4} style={{ fill: C.red }} />

          {/* the thread */}
          <line x1={AX} y1={AY} x2={tip.x} y2={tip.y} style={{ stroke: C.sky, strokeWidth: 1.8 }} />
          <circle cx={AX} cy={AY} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
          {(() => {
            const bob = threadAt(h, R + 40);
            return (
              <g>
                <line x1={tip.x} y1={tip.y} x2={bob.x} y2={bob.y} style={{ stroke: C.sky, strokeWidth: 1.3 }} />
                <path
                  d={`M ${bob.x} ${bob.y} m -4 0 l 4 12 l 4 -12 z`}
                  transform={`rotate(${-h} ${bob.x} ${bob.y})`}
                  style={{ fill: C.paper, stroke: C.sky, strokeWidth: 1 }}
                />
              </g>
            );
          })()}
          {/* the altitude angle at the apex */}
          <path d={arcPath(AX, AY, 62, -90, h - 90)} style={{ fill: 'none', stroke: C.sky, strokeWidth: 1.1 }} />
          <InkText x={polar(AX, AY, 70, Math.min(h, 80) / 2 - 90).x + 2} y={polar(AX, AY, 70, Math.min(h, 80) / 2 - 90).y + 4} kind="num" anchor="start" size={11} role="sky">
            h
          </InkText>

          {/* hit area for dragging (touch: only here, so the page still scrolls elsewhere) */}
          <line
            x1={AX}
            y1={AY}
            x2={threadAt(h, R + 50).x}
            y2={threadAt(h, R + 50).y}
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

          <Inset h={h} />
        </svg>

        <div className="sq-work">
          <Working label="The muvakkit's working, in parts of 60">
            <Eq>
              <Sym role="sky">h</Sym> = <V role="sky">{h.toFixed(1)}°</V>, so 60·sin <Sym role="sky">h</Sym> = <V>{p60(q.sinh)}</V>
            </Eq>
            <Eq>
              cos <Sym role="red">H</Sym> = (sin <Sym role="sky">h</Sym> − <Sym role="brass">sin φ sin δ</Sym>) ÷ <Sym role="soft">cos φ cos δ</Sym>
            </Eq>
            <Eq>
              <V role="red">{p60(q.cosH)}</V> = (<V>{p60(q.sinh)}</V> − <V role="brass">{sub ? p60(q.a) : `(${p60(q.a)})`}</V>) ÷{' '}
              <V role="soft">{p60(q.b)}</V> × 60
            </Eq>
            <Eq>
              <Sym role="red">H</Sym> = <V role="red">{q.Hdeg.toFixed(1)}°</V> ={' '}
              <V role="red">
                {Math.floor(fromNoon)} h {String(Math.round((fromNoon % 1) * 60)).padStart(2, '0')} min
              </V>{' '}
              from noon
            </Eq>
          </Working>
          <p className="sq-note">
            On {dayLabel(day)} the sun is {Math.abs(q.dec).toFixed(1)}° {q.dec >= 0 ? 'north' : 'south'} of the
            equator, so the board {sub ? 'subtracts' : 'adds'} the date term. The time is{' '}
            <b>{hm(clock)}</b> local solar time{half === 'am' ? ', morning' : ', afternoon'}.
            {q.diff < 0 && ' The sun is lower than it stands at six o\'clock, so cos H is negative and H passes 90°.'}
            {atNoon && ` ${hMax.toFixed(1)}° is the noon altitude on this date; the sun climbs no higher.`}
          </p>
        </div>
      </div>

      <Controls>
        <Slider
          label="sun's altitude h"
          value={h}
          min={0}
          max={hMax}
          step={0.1}
          onChange={setH}
          format={(v) => `${v.toFixed(1)}°${v >= hMax - 0.05 ? ' (noon)' : ''}`}
          width="8rem"
        />
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
      <line x1={ox} y1={oy} x2={ox} y2={oy + s + 4} style={{ stroke: C.sky, strokeWidth: 1.1 }} />
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
.sq-work .in-working { margin-top: 0; }
.sq-note { margin: 0.7rem 0 0; font-family: var(--font-display); font-style: italic; font-size: 0.98rem; line-height: 1.45; color: var(--in-ink-soft); }
.sq-note b { font-family: var(--font-mono); font-style: normal; font-weight: 400; color: var(--in-red); }
`;
