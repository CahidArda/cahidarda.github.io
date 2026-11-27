import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';
import { Reserve } from '../../gt/Reserve';
import { SIZES, sci, sup } from './model';

/**
 * CX-15.2: how long each letter of the witness for the prime 2 is, as a number of digits on
 * a log scale. The view starts close in (where a, o and x already have 55 to 111 digits)
 * and zooms out until r, u, d and c arrive with about 10^58 digits each. A last row: for
 * the prime 3, the letter a alone has more than 10^1168 digits, far off the chart.
 */

type View = 'close' | 'far';
const NEAR = 2.4; // axis max (log10 of digits) when close in
const FAR = 62; // axis max when zoomed out
const PRIME3 = 1168; // log10 of the digits of a in any witness for 3 (lower bound)
const ROWS = [
  ...SIZES.map((s) => ({ ...s, L: Math.log10(s.digits), three: false })),
  { label: 'a, for 3', letters: 'a', digits: NaN, exact: false, L: PRIME3, three: true },
];
const TICKS: { L: number; text: string; tier: number }[] = [
  { L: 0, text: '1', tier: 2 },
  { L: 1, text: '10', tier: 0 },
  { L: 2, text: '100', tier: 0 },
  ...[10, 20, 30, 40, 50, 60].map((L) => ({ L, text: `10${sup(L)}`, tier: 1 })),
];
const ease = (t: number) => t * t * (3 - 2 * t);
const digitsText = (r: (typeof ROWS)[number]) =>
  r.three
    ? `over 10${sup(PRIME3)} digits`
    : r.exact
      ? `${r.label === '16 letters' ? 'at most ' : ''}${r.digits} digits`
      : `${sci(r.digits)} digits`;

export default function WitnessSize() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const W = compact ? 380 : 560;
  const [svgRef, k] = useBoost(W, 9, 8.5);
  const LW = compact ? 74 : 96; // label column
  const X0 = LW + 8;
  const X1 = W - 12;
  const RH = 30;
  const TOP = 34;
  const H = TOP + ROWS.length * RH + 26;

  // SSR and the demo start close in; reduced motion shows the far view at once
  const [sel, setSel] = useState<View>('close');
  const [view, bindView] = useHoverPreview(sel);
  // z: 0 = close, 1 = far; one clock moves it toward the active view
  const [z, setZ] = useState(0);
  const zRef = useRef(0);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) return;
    timer.current = setTimeout(() => setSel('far'), 1400);
    setArmed(true);
  }, [inView, reduced]);
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    setSel('far');
    zRef.current = 1;
    setZ(1);
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const target = view === 'far' ? 1 : 0;
  useEffect(() => {
    if (reduced) {
      zRef.current = target;
      setZ(target);
    }
  }, [target, reduced]);
  useFrame(
    (dt) => {
      const d = target - zRef.current;
      const step = dt / 1.5; // a full zoom takes 1.5 s
      const v = Math.abs(d) <= step ? target : zRef.current + Math.sign(d) * step;
      zRef.current = v;
      setZ(v);
    },
    armed && inView && !reduced && z !== target,
  );

  const maxL = Math.exp(Math.log(NEAR) + (Math.log(FAR) - Math.log(NEAR)) * ease(z));
  const sx = (L: number) => X0 + (clamp(L, 0, maxL * 1.2) / maxL) * (X1 - X0);
  const pxPerUnit = (X1 - X0) / maxL;

  const choose = (v: View) => {
    if (timer.current) clearTimeout(timer.current);
    setArmed(true);
    setSel(v);
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-15.2"
      title="how big are the witnesses"
      kicker="prime 2"
      hint="hover or tap to zoom between the small letters and all of them"
      footer="digits, on a log scale"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="Number of digits of each letter in the witness that 2 is prime: 16 letters have at most 2 digits; a, m, y and v have 55; o has 58; x has 111; r and u about 7 times 10 to the 57; d about 2.8 times 10 to the 58; c about 4.9 times 10 to the 58. For the prime 3, a alone has more than 10 to the 1168 digits."
        >
          <defs>
            <clipPath id="cx15-plot">
              <rect x={X0} y={0} width={X1 - X0 + 1} height={H} />
            </clipPath>
          </defs>
          <T x={X0} y={14} anchor="start" size={9 * k} color={GT.ink}>
            how many digits each letter has
          </T>
          {/* gridlines and tick labels */}
          <g clipPath="url(#cx15-plot)">
            {TICKS.map((t) => {
              const spacing = t.tier === 2 ? 99 : pxPerUnit * (t.tier === 0 ? 1 : 10);
              const op = clamp((spacing - 26) / 24, 0, 1);
              if (op <= 0) return null;
              const x = sx(t.L);
              return (
                <g key={t.L} style={{ opacity: op }}>
                  <line
                    x1={x}
                    x2={x}
                    y1={TOP - 6}
                    y2={TOP + ROWS.length * RH}
                    style={{ stroke: GT.line }}
                  />
                  <T x={x} y={TOP + ROWS.length * RH + 12 * k} size={9 * k}>
                    {t.text}
                  </T>
                </g>
              );
            })}
          </g>
          {ROWS.map((r, i) => {
            const y = TOP + i * RH;
            const end = sx(r.L);
            const off = r.L > maxL;
            const w = Math.max(1.5, end - X0);
            const color = r.three ? GT.b : r.exact ? GT.a : GT.eq;
            const labelInside = end > X1 - 130 * k;
            return (
              <g key={r.label}>
                <T x={LW} y={y + RH / 2} anchor="end" size={10 * k} color={GT.ink} font="display">
                  {r.label}
                </T>
                <g clipPath="url(#cx15-plot)">
                  <rect
                    x={X0}
                    y={y + 7}
                    width={w}
                    height={RH - 14}
                    style={{ fill: color, opacity: r.three ? 0.85 : 1 }}
                  />
                  {off && (
                    <path
                      d={`M${X1 - 10} ${y + 5} l6 ${(RH - 10) / 2} l-6 ${(RH - 10) / 2}`}
                      style={{ fill: 'none', stroke: GT.paperRaised, strokeWidth: 3 }}
                    />
                  )}
                </g>
                <T
                  x={labelInside ? Math.min(end, X1) - 8 : end + 6}
                  y={y + RH / 2}
                  anchor={labelInside ? 'end' : 'start'}
                  size={9.5 * k}
                  weight={700}
                  color={labelInside ? GT.paper : color}
                >
                  {r.three ? `${digitsText(r)}, off the chart` : digitsText(r)}
                </T>
              </g>
            );
          })}
          <line
            x1={X0}
            x2={X0}
            y1={TOP - 6}
            y2={TOP + ROWS.length * RH}
            style={{ stroke: GT.lineStrong }}
          />
        </svg>
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <BigStat
            label="counting up to a"
            value={`2.5×10${sup(38)}`}
            unit="years"
            sub="at a billion numbers a second"
          />
          <BigStat
            label="digits in c"
            value={sci(SIZES[SIZES.length - 1].digits)}
            color={GT.eq}
            sub="one letter of the certificate"
          />
        </div>
        <Seg
          options={[
            { value: 'close', label: 'the small ones' },
            { value: 'far', label: 'all of them' },
          ]}
          value={view}
          onChange={choose}
          bind={bindView}
        />
      </div>
      <Reserve
        active={view === 'far' ? 1 : 0}
        items={[
          <Caption key="c">
            Close in: 16 letters are tiny, but a already has 55 digits, and x has 111.
          </Caption>,
          <Caption key="f">
            Zoomed out: even the digit counts of r, u, d and c are 58 or 59 digits long. For the
            next prime, 3, the letter a alone has more digits than that by far.
          </Caption>,
        ]}
      />
    </Specimen>
  );
}
