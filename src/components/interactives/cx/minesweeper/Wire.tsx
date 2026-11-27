/**
 * CX-12.2 "a wire made of mines". Kaye's wire (fig. 6) feeding his NOT gate (fig. 9) and out
 * again. The numbers leave exactly two ways to fill the grey squares: mines on every x (the
 * wire carries TRUE) or on every x′ (FALSE). Flip the input and the forced mines move down the
 * wire like a signal; past the gate the pattern is shifted, so the output reads the opposite.
 * Every grey square is labelled with Kaye's own letter. One clock: a front sweeps left to
 * right at constant speed, each square swaps its mine as the front passes it.
 */
import { useEffect, useRef, useState } from 'react';
import { BigStat, GT, Specimen, T, useFrame, useInView, useReducedMotion } from '../../gt/shared';
import { useCompact } from '../../gt/useCompact';
import { isHidden, mineAt, notGate } from './kaye';

const CS = 30;
const PADX = 18;
const TOP = 40;
const SPEED = 15; // columns per second
const FADE = 0.16; // seconds for one square to swap
const valColor = (v: boolean) => (v ? GT.opt : GT.b);
const word = (v: boolean) => (v ? 'TRUE' : 'FALSE');

type Sq = { s: number; on: boolean; cv: boolean };

export default function Wire() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact<HTMLDivElement>(560);
  const side = compact ? 5 : 8;
  const rows = notGate(side);
  const cols = rows[0].length;
  const gate = side; // first gate column
  const W = PADX * 2 + cols * CS;
  const H = TOP + 5 * CS + 8;

  const [x, setX] = useState(true); // the input
  const [front, setFront] = useState(cols + 4); // column the change has reached
  const sqRef = useRef<Map<string, Sq>>(new Map());
  const [, force] = useState(0);
  const touched = useRef(false);

  // the value a square reflects: input for the left wire, NOT input from the gate centre on
  const regionVal = (c: number, v: boolean) => (c < gate + 1 ? v : !v);
  const sqKey = (r: number, c: number) => `${side}:${r},${c}`;

  // squares settle instantly when the layout or reduced-motion changes
  const settle = (v: boolean) => {
    const m = new Map<string, Sq>();
    rows.forEach((row, r) =>
      [...row].forEach((t, c) => {
        if (t === 'x' || t === 'y') {
          const on = mineAt(t, v);
          m.set(sqKey(r, c), { s: on ? 1 : 0, on, cv: regionVal(c, v) });
        }
      }),
    );
    sqRef.current = m;
  };
  if (!sqRef.current.has(sqKey(2, 0))) settle(x);
  useEffect(() => {
    settle(x);
    setFront(cols + 4);
    force((k) => k + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [side, reduced]);

  const flip = () => {
    const nx = !x;
    setX(nx);
    if (reduced) {
      settle(nx);
      setFront(cols + 4);
    } else setFront(0);
  };

  // auto-demo: one flip shortly after the figure comes into view
  useEffect(() => {
    if (!inView || reduced || touched.current) return;
    const id = setTimeout(() => {
      if (!touched.current) {
        touched.current = true;
        flip();
      }
    }, 1100);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);

  const animating =
    front < cols + 4 || [...sqRef.current.values()].some((q) => q.s !== (q.on ? 1 : 0));
  useFrame((dt) => {
    const f = Math.min(cols + 4, front + dt * SPEED);
    rows.forEach((row, r) =>
      [...row].forEach((t, c) => {
        const q = sqRef.current.get(sqKey(r, c));
        if (!q) return;
        if (f >= c + 0.5) {
          const on = mineAt(t, x);
          if (on !== q.on) {
            q.on = on;
            if (on) q.cv = regionVal(c, x);
          }
        }
        const target = q.on ? 1 : 0;
        const step = dt / FADE;
        q.s = q.s < target ? Math.min(target, q.s + step) : Math.max(target, q.s - step);
      }),
    );
    setFront(f);
    force((k) => k + 1);
  }, animating && !reduced);

  const outDone = front >= cols; // the output label updates when the change arrives
  const out = outDone ? !x : x; // until it arrives: the old output, NOT(old input) = x

  const cx = (c: number) => PADX + c * CS;
  const cy = (r: number) => TOP + r * CS;
  const midGate = cx(gate) + 1.5 * CS;

  return (
    <Specimen
      code="CX-12.2"
      title="a wire made of mines"
      kicker="Kaye 2000, figs. 6 and 9"
      rootRef={rootRef}
      hint="tap the strip to flip the input"
      footer="only 2 ways to fill the grey squares"
    >
      <div ref={boxRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full cursor-pointer select-none outline-none"
          role="button"
          tabIndex={0}
          aria-label={`Minesweeper wire into a NOT gate. Input ${word(x)}, output ${word(out)}. Tap to flip the input.`}
          onClick={() => {
            touched.current = true;
            flip();
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              touched.current = true;
              flip();
            }
          }}
        >
          {/* labels on the wire itself */}
          <T x={cx(0)} y={14} anchor="start" size={10.5} color={GT.ink}>
            input →
          </T>
          <T x={midGate} y={14} size={10.5} color={GT.ink}>
            NOT gate
          </T>
          <line
            x1={cx(gate)}
            y1={22}
            x2={cx(gate + 3)}
            y2={22}
            style={{ stroke: GT.ink, strokeWidth: 1 }}
          />
          <T x={cx(cols)} y={14} anchor="end" size={10.5} color={GT.ink}>
            output →
          </T>

          {rows.map((row, r) =>
            [...row].map((t, c) => {
              const X0 = cx(c);
              const Y0 = cy(r);
              const k = `${r},${c}`;
              if (isHidden(t) || t === '*') {
                const q = sqRef.current.get(sqKey(r, c));
                const s = t === '*' ? 1 : (q?.s ?? 0);
                const col = t === '*' ? GT.ink : valColor(q?.cv ?? true);
                return (
                  <g key={k}>
                    <rect
                      x={X0 + 1}
                      y={Y0 + 1}
                      width={CS - 2}
                      height={CS - 2}
                      style={{ fill: GT.line, stroke: GT.lineStrong, strokeWidth: 1 }}
                    />
                    {t !== '*' && (
                      <g style={{ opacity: 1 - s }}>
                        <T x={X0 + 4} y={Y0 + 8} anchor="start" size={7.5} color={GT.muted}>
                          {t === 'x' ? 'x' : 'x′'}
                        </T>
                      </g>
                    )}
                    {s > 0.001 && (
                      <g
                        transform={`translate(${X0 + CS / 2} ${Y0 + CS / 2 + 1}) scale(${0.35 + 0.65 * s})`}
                        style={{ opacity: s }}
                      >
                        <Mine color={col} />
                      </g>
                    )}
                  </g>
                );
              }
              return (
                <g key={k}>
                  <rect
                    x={X0 + 1}
                    y={Y0 + 1}
                    width={CS - 2}
                    height={CS - 2}
                    style={{ fill: GT.paper, stroke: GT.line, strokeWidth: 1 }}
                  />
                  <T
                    x={X0 + CS / 2}
                    y={Y0 + CS / 2 + 1}
                    size={14}
                    weight={600}
                    color={t === '0' ? GT.muted : GT.ink}
                  >
                    {t}
                  </T>
                </g>
              );
            }),
          )}
        </svg>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
        <BigStat
          label="input"
          value={word(x)}
          color={valColor(x)}
          sub={x ? 'mines on every x' : 'mines on every x′'}
        />
        <BigStat
          label="output"
          value={word(out)}
          color={valColor(out)}
          sub={out ? 'mines on the x′ past the gate' : 'mines on the x past the gate'}
        />
        <div className="col-span-2 sm:col-span-1">
          <BigStat label="ways to fill" value={2} sub="the numbers force the rest" />
        </div>
      </div>
    </Specimen>
  );
}

function Mine({ color }: { color: string }) {
  const r = 5.5;
  return (
    <g>
      {[0, 45, 90, 135].map((a) => {
        const dx = Math.cos((a * Math.PI) / 180) * (r + 2.8);
        const dy = Math.sin((a * Math.PI) / 180) * (r + 2.8);
        return (
          <line
            key={a}
            x1={-dx}
            y1={-dy}
            x2={dx}
            y2={dy}
            style={{ stroke: color, strokeWidth: 1.7, strokeLinecap: 'round' }}
          />
        );
      })}
      <circle r={r} style={{ fill: color }} />
    </g>
  );
}
