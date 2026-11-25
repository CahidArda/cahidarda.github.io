import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
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

const COUNT = 2000;
const BINS = 20;
const FILL_S = 2.4; // seconds to stream all auctions in
const r2 = (x: number) => Math.round(x * 100) / 100;

/* left panel: the bidding rule, value -> bid, both $0..$100 */
const BW = 240;
const BH = 220;
const BL = 40;
const BT = 12;
const BP = 168; // plot side
const bx = (v: number) => r2(BL + v * BP);
const by = (b: number) => r2(BT + BP - b * BP);

/* right panel: histogram of seller revenue */
const HW = 480;
const HH = 230;
const HL = 14;
const HR = 14;
const HT = 34;
const HB = 44;
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

  const slope = (n - 1) / n;

  return (
    <Specimen
      code="GT-04.2"
      title="shade or not"
      kicker={`${COUNT.toLocaleString('en-US')} auctions · values from $0 to $100`}
      rootRef={rootRef}
      hint="change the number of bidders, or run the auctions again"
      footer={`n = ${n} bidders`}
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="w-56">
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

      <div className="mt-4 grid grid-cols-3 gap-3">
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
        <BigStat
          label="theory"
          value={dollars(theory)}
          color={GT.eq}
          sub="(n − 1)/(n + 1)"
          align="right"
        />
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] md:items-end">
        {/* the bidding rule */}
        <svg
          viewBox={`0 0 ${BW} ${BH}`}
          className="mx-auto block w-full max-w-[20rem]"
          role="img"
          aria-label={`Bids by value. Second price: bid equals value. First price: bid is ${n - 1}/${n} of value.`}
        >
          <rect
            x={BL}
            y={BT}
            width={BP}
            height={BP}
            style={{ fill: 'none', stroke: GT.lineStrong }}
          />
          <polygon
            points={`${bx(0)},${by(0)} ${bx(1)},${by(1)} ${bx(1)},${by(slope)}`}
            style={{ fill: GT.bSoft }}
          />
          <line
            x1={bx(0)}
            y1={by(0)}
            x2={bx(1)}
            y2={by(1)}
            style={{ stroke: GT.a, strokeWidth: 2, strokeDasharray: '5 3' }}
          />
          <line
            x1={bx(0)}
            y1={by(0)}
            x2={bx(1)}
            y2={by(slope)}
            style={{ stroke: GT.b, strokeWidth: 2.5 }}
          />
          <T x={bx(0.62)} y={by(0.62) - 12} anchor="end" color={GT.a} size={10}>
            bid = value
          </T>
          <T
            x={bx(0.98)}
            y={r2((by(1) + by(slope)) / 2)}
            anchor="end"
            color={GT.b}
            size={10}
            weight={600}
          >
            shade
          </T>
          <T x={bx(0.66)} y={by(slope * 0.66) + 14} anchor="start" color={GT.b} size={10}>
            {`${n - 1}/${n} × value`}
          </T>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <T x={bx(t)} y={BT + BP + 12} size={9.5}>
                {dollars(t)}
              </T>
              <T x={BL - 6} y={by(t)} anchor="end" size={9.5}>
                {dollars(t)}
              </T>
            </g>
          ))}
          <T x={BL + BP / 2} y={BH - 8} size={10.5} color={GT.ink}>
            your value →
          </T>
        </svg>

        {/* seller revenue histogram */}
        <svg
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
            x={hx(theory) + (theory > 0.7 ? -5 : 5)}
            y={HT - 14}
            anchor={theory > 0.7 ? 'end' : 'start'}
            color={GT.eq}
            size={13}
            weight={600}
          >
            {`expected ${dollars(theory)}`}
          </T>
          {k > 0 && (
            <>
              <line
                x1={hx(mSec)}
                y1={base}
                x2={hx(mSec)}
                y2={base + 7}
                style={{ stroke: GT.a, strokeWidth: 2.5 }}
              />
              <line
                x1={hx(mFir)}
                y1={base}
                x2={hx(mFir)}
                y2={base + 7}
                style={{ stroke: GT.b, strokeWidth: 2.5 }}
              />
            </>
          )}
          {[0, 0.5, 1].map((t) => (
            <T
              key={t}
              x={hx(t)}
              y={base + 18}
              size={12}
              anchor={t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}
            >
              {dollars(t)}
            </T>
          ))}
          <T x={HW / 2} y={HH - 6} size={13} color={GT.ink}>
            seller's revenue per auction →
          </T>
        </svg>
      </div>

      <div className="mt-2 flex flex-wrap gap-3">
        <Chip color={GT.a}>second price: everyone bids their value</Chip>
        <Chip color={GT.b}>first price: everyone shades</Chip>
      </div>
      <Caption>
        First-price revenue bunches tighter, second-price revenue spreads wider, and the averages
        match. Shading lowers each bid by exactly enough, on average, to cancel out paying your own
        price.
      </Caption>
    </Specimen>
  );
}
