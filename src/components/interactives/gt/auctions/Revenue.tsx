import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Slider,
  Specimen,
  T,
  mulberry32,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { expectedRevenue, histogram, mean, simulate } from './logic';
import { useBoost } from '../braess/useBoost';

const COUNT = 2000;
const BINS = 20;
const FILL_S = 2.4; // seconds to stream all auctions in
const r2 = (x: number) => Math.round(x * 100) / 100;

/* histogram of seller revenue (the one chart) */
const HW = 520;
const HH = 236;
const HL = 14;
const HR = 14;
const HT = 34;
const HB = 40;
const CH = 6.1; // approx. width of one 10px mono character
const hx = (x: number) => r2(HL + x * (HW - HL - HR));

const dollars = (x: number) => `$${Math.round(x * 100)}`;

/** Step outline of a histogram (heights already in px). */
function stepPath(h: number[], base: number) {
  let d = `M ${hx(0)} ${base}`;
  h.forEach((y, i) => {
    d += ` L ${hx(i / BINS)} ${r2(base - y)} L ${hx((i + 1) / BINS)} ${r2(base - y)}`;
  });
  return d + ` L ${hx(1)} ${base}`;
}

export default function Revenue() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [svgRef, kb] = useBoost(HW, 10, 9);
  const [n, setN] = useState(3);
  const [seed, setSeed] = useState(1);
  const [shown, setShown] = useState(COUNT); // SSR + reduced motion: the full run
  const started = useRef(false);

  const sim = useMemo(() => simulate(n, COUNT, mulberry32(seed * 101 + n)), [n, seed]);

  const replay = () => {
    if (reduced) {
      setShown(COUNT);
      return;
    }
    setShown(0);
  };

  useEffect(() => {
    if (reduced) {
      // static poster: the full run
      started.current = true;
      setShown(COUNT);
      return;
    }
    if (started.current || !inView) return;
    // short delay so a reduced-motion preference (read after mount) can cancel the replay
    const id = setTimeout(() => {
      started.current = true;
      setShown(0);
    }, 300);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => setShown((s) => Math.min(COUNT, s + (COUNT * dt) / FILL_S)),
    !reduced && inView && shown < COUNT,
  );

  const k = Math.floor(shown);
  const sec = sim.second.slice(0, k);
  const fir = sim.first.slice(0, k);
  const mSec = mean(sec);
  const mFir = mean(fir);
  const theory = expectedRevenue(n);

  // scale bars to the full run so the histogram grows in place
  const full = [histogram(sim.second, BINS), histogram(sim.first, BINS)];
  const top = Math.max(...full[0], ...full[1]);
  const base = HH - HB;
  const scale = (base - HT) / top;
  const hs = histogram(sec, BINS).map((c) => c * scale);
  const hf = histogram(fir, BINS).map((c) => c * scale);

  // direct labels sit just above the bars they would otherwise cover
  const fullH = [full[0].map((c) => c * scale), full[1].map((c) => c * scale)];
  const clearY = (x0: number, x1: number) => {
    let m = 0;
    for (let i = 0; i < BINS; i++) {
      const a = hx(i / BINS);
      const b = hx((i + 1) / BINS);
      if (b < x0 || a > x1) continue;
      m = Math.max(m, fullH[0][i], fullH[1][i]);
    }
    return r2(base - m - 8 * kb);
  };
  const narrow = kb > 1.15;
  const secLabel = narrow ? 'second price' : 'second price: bid = value';
  const firLabel = narrow
    ? `first price: ${n - 1}/${n} × value`
    : `first price: bid ${n - 1}/${n} × value`;
  // place each label in the lowest free spot just above the bars, never across the
  // expected-value line and never on top of the other label
  const lineX = hx(theory);
  const textH = 11 * kb;
  const place = (
    w: number,
    avoid: { x0: number; x1: number; y: number } | null,
    crossLine = false,
  ): { x: number; y: number } => {
    let best: { x: number; y: number } | null = null;
    for (let x = hx(0); x + w <= hx(1); x += 4) {
      if (!crossLine && x - 6 < lineX && x + w + 6 > lineX) continue;
      const y = Math.min(clearY(x, x + w), base - 10);
      if (y - textH < HT - 4) continue;
      if (avoid && x < avoid.x1 + 8 && x + w > avoid.x0 - 8 && Math.abs(y - avoid.y) < textH + 2)
        continue;
      if (!best || y > best.y + 0.5) best = { x, y };
    }
    return best ?? (crossLine ? { x: hx(0), y: HT + 4 } : place(w, avoid, true));
  };
  const secW = secLabel.length * CH * kb;
  const firW = firLabel.length * CH * kb;
  const secPos = place(secW, null);
  const firPos = place(firW, { x0: secPos.x, x1: secPos.x + secW, y: secPos.y });

  return (
    <Specimen
      code="GT-04.2"
      title="shade or not"
      kicker={`${COUNT.toLocaleString('en-US')} auctions · values from $0 to $100`}
      rootRef={rootRef}
      hint="change the number of bidders, or run the auctions again"
      footer={`n = ${n} bidders`}
    >
      {/* seller revenue histogram */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${HW} ${HH}`}
        className="block w-full"
        role="img"
        aria-label={`Histogram of seller revenue over ${k} auctions. Average second price ${dollars(mSec)}, first price ${dollars(mFir)}, theory ${dollars(theory)}.`}
      >
        <path d={stepPath(hs, base) + ' Z'} style={{ fill: GT.aSoft, opacity: 0.85 }} />
        <path d={stepPath(hf, base) + ' Z'} style={{ fill: GT.bSoft, opacity: 0.7 }} />
        <path d={stepPath(hs, base)} style={{ fill: 'none', stroke: GT.a, strokeWidth: 1.6 }} />
        <path d={stepPath(hf, base)} style={{ fill: 'none', stroke: GT.b, strokeWidth: 1.6 }} />
        <line
          x1={hx(0)}
          y1={base}
          x2={hx(1)}
          y2={base}
          style={{ stroke: GT.ink, strokeWidth: 1.2 }}
        />
        {/* theory mean */}
        <line
          x1={hx(theory)}
          y1={HT - 18}
          x2={hx(theory)}
          y2={base}
          style={{ stroke: GT.eq, strokeWidth: 2 }}
        />
        <T
          x={hx(theory) + (theory > 0.6 ? -6 : 6)}
          y={HT - 12}
          anchor={theory > 0.6 ? 'end' : 'start'}
          color={GT.eq}
          size={12 * Math.min(kb, 1.4)}
          weight={600}
        >
          {`both average ${dollars(theory)}`}
        </T>
        {/* direct labels instead of a legend */}
        <g
          style={{
            stroke: GT.paperRaised,
            strokeWidth: 4,
            strokeLinejoin: 'round',
            paintOrder: 'stroke',
          }}
        >
          <T x={secPos.x} y={secPos.y} anchor="start" color={GT.a} size={10 * kb} weight={600}>
            {secLabel}
          </T>
          <T x={firPos.x} y={firPos.y} anchor="start" color={GT.b} size={10 * kb} weight={600}>
            {firLabel}
          </T>
        </g>
        {[0, 0.5, 1].map((t) => (
          <T
            key={t}
            x={hx(t)}
            y={base + 13 * kb}
            size={10 * kb}
            anchor={t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}
          >
            {dollars(t)}
          </T>
        ))}
        <T x={HW / 2} y={HH - 8} size={10.5 * Math.min(kb, 1.3)} color={GT.ink}>
          seller's revenue per auction →
        </T>
      </svg>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="flex gap-8">
          <BigStat
            label="second price"
            value={k ? dollars(mSec) : '...'}
            color={GT.a}
            sub="avg revenue"
          />
          <BigStat
            label="first price"
            value={k ? dollars(mFir) : '...'}
            color={GT.b}
            sub="avg revenue"
          />
        </div>
        <div className="flex items-end gap-4">
          <div className="w-44">
            <Slider
              label="bidders"
              value={n}
              min={2}
              max={10}
              onChange={(x) => {
                setN(x);
                replay();
              }}
            />
          </div>
          <Btn
            primary
            onClick={() => {
              setSeed((s) => s + 1);
              replay();
            }}
          >
            run again
          </Btn>
        </div>
      </div>
      <Caption>
        Same average, different spread. Shading lowers each bid by just enough to cancel out paying
        your own price.
      </Caption>
    </Specimen>
  );
}
