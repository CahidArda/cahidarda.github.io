import { useEffect, useRef, useState } from 'react';
import { Caption, GT, Specimen, T, useFrame, useInView, useReducedMotion } from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';

/**
 * CX-04.3 · what experts bet. William Gasarch's three polls on "Does P = NP?", counts from the
 * table in section 2.1 of "The Third P =? NP Poll" (SIGACT News 50(1), 2019). "Other" lumps
 * together independent of ZFC, don't know, don't care and other answers.
 */
const POLLS = [
  { year: 2002, ne: 61, eq: 9, other: 30 },
  { year: 2012, ne: 126, eq: 12, other: 13 },
  { year: 2019, ne: 109, eq: 15, other: 0 },
];
const total = (p: (typeof POLLS)[number]) => p.ne + p.eq + p.other;
const pct = (a: number, n: number) => Math.round((100 * a) / n);

export default function Poll() {
  const [wrapRef, compact] = useCompact(460);
  const W = compact ? 360 : 520;
  const BX = compact ? 40 : 52;
  const BR = compact ? W - 58 : W - 74;
  const ROW = 46;
  const TOP = 34;
  const BH = 26;
  const H = TOP + ROW * (POLLS.length - 1) - 14 + BH + 8;

  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [g, setG] = useState(0); // 0..1 growth of the bars
  const gRef = useRef(0);
  const [run, setRun] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    setRun(true);
  }, [inView]);
  useEffect(() => {
    if (reduced) {
      gRef.current = 1;
      setG(1);
      setRun(false);
    }
  }, [reduced]);
  useFrame((dt) => {
    const v = Math.min(1, gRef.current + Math.max(0, dt) / 0.9);
    gRef.current = v;
    setG(v);
    if (v >= 1) setRun(false);
  }, run && inView);
  const e = g * g * (3 - 2 * g);

  const active = hover ?? POLLS.length - 1;
  const captions = POLLS.map((p) => {
    const n = total(p);
    return `${p.year}: ${p.ne} of ${n} answered P ≠ NP, ${p.eq} answered P = NP${p.other ? `, ${p.other} gave another answer` : ', and nobody gave another answer'}.`;
  });

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-04.3"
      title="what experts bet"
      kicker="Gasarch's P =? NP polls"
      hint="hover a year for the counts"
      footer="SIGACT News 50(1), 2019"
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="Share of poll respondents answering P is not NP: 61% in 2002, 83% in 2012, 88% in 2019"
        >
          {/* direct labels over the first bar */}
          <T x={BX} y={12} anchor="start" size={compact ? 9.5 : 10.5} weight={700} color={GT.a}>
            P ≠ NP
          </T>
          <T x={BR} y={12} anchor="end" size={compact ? 9.5 : 10.5} weight={700} color={GT.eq}>
            P = NP
          </T>
          {POLLS.map((p, i) => {
            const n = total(p);
            const y = TOP + i * ROW - 14;
            const span = (BR - BX) * e;
            const w1 = (span * p.ne) / n;
            const w2 = (span * p.eq) / n;
            const w3 = (span * p.other) / n;
            const on = hover === i;
            return (
              <g
                key={p.year}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                style={{ cursor: 'default' }}
              >
                <rect x={0} y={y - 6} width={W} height={BH + 12} style={{ fill: 'transparent' }} />
                <T
                  x={BX - 8}
                  y={y + BH / 2}
                  anchor="end"
                  size={compact ? 10 : 11}
                  weight={on ? 700 : 400}
                  color={GT.ink}
                >
                  {p.year}
                </T>
                <rect x={BX} y={y} width={w1} height={BH} style={{ fill: GT.a }} />
                <rect x={BX + w1} y={y} width={w3} height={BH} style={{ fill: GT.line }} />
                <rect x={BX + w1 + w3} y={y} width={w2} height={BH} style={{ fill: GT.eq }} />
                <g style={{ opacity: Math.max(0, (g - 0.6) / 0.4) }}>
                  <T
                    x={BX + 8}
                    y={y + BH / 2 + 1}
                    anchor="start"
                    size={compact ? 11 : 12}
                    weight={700}
                    color={GT.paperRaised}
                  >
                    {`${pct(p.ne, n)}%`}
                  </T>
                  {w3 > 70 && (
                    <T
                      x={BX + w1 + w3 / 2}
                      y={y + BH / 2 + 1}
                      size={compact ? 9.5 : 10.5}
                      color={GT.inkSoft}
                    >
                      {`${pct(p.other, n)}% other`}
                    </T>
                  )}
                  <T
                    x={BR + 6}
                    y={y + BH / 2 + 1}
                    anchor="start"
                    size={compact ? 9.5 : 10.5}
                    weight={700}
                    color={GT.eq}
                  >
                    {`${pct(p.eq, n)}%`}
                  </T>
                </g>
                {on && (
                  <rect
                    x={BX - 0.5}
                    y={y - 0.5}
                    width={BR - BX + 1}
                    height={BH + 1}
                    style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.5 }}
                  />
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <Reserve
        items={captions.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={active}
      />
    </Specimen>
  );
}
