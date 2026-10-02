import { useState } from 'react';
import { NAV, PRICE, useInView, useReducedMotion, WARN, Widget } from './shared';

// Why a NAV mark is not an exit price.
//
// This is a TOY MODEL and is labelled as one in the figure. It assumes the crudest possible
// order book: to sell a position worth s of a stock's free float you walk linearly down the
// book, so your average realised price is P*(1-s). Real impact is neither linear nor
// permanent. The point is only the shape: the gap between the mark and the exit grows
// faster than the position does, because the mark revalues the WHOLE holding at a price
// only the last share traded at.
//
// The anchor at 40% is not a claim about any specific fund. It is where the chart stops
// being a curiosity.

const VB = { w: 380, h: 170 };
const PAD = { l: 34, r: 10, t: 12, b: 26 };

const realisable = (s: number) => 1 - s; // average price as a fraction of the mark

export default function MarkVsExit() {
  const reduced = useReducedMotion();
  const [s, setS] = useState(0.35);
  const [viewRef] = useInView<HTMLDivElement>();

  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const SMAX = 0.6;
  const px = (v: number) => PAD.l + (v / SMAX) * w;
  const y = (v: number) => PAD.t + h - v * h; // v in 0..1

  const curve = Array.from({ length: 61 }, (_, i) => {
    const v = (i / 60) * SMAX;
    return `${px(v)},${y(realisable(v))}`;
  }).join(' ');

  const gapPct = Math.round((1 - realisable(s)) * 100);

  return (
    <Widget title="A mark is not an exit price" kicker="illustrative model, not fund data" rootRef={viewRef}>
      <label className="flex flex-col gap-1.5">
        <span className="font-mono text-[0.7rem] text-ink-soft">
          share of the stock’s free float the fund owns:{' '}
          <strong className="text-ink">{Math.round(s * 100)}%</strong>
        </span>
        <input
          type="range"
          min={0}
          max={SMAX}
          step={0.01}
          value={s}
          onChange={(e) => setS(parseFloat(e.target.value))}
          className="w-full"
          style={{ accentColor: 'var(--color-accent)' }}
          aria-label="Share of the stock's free float owned by the fund"
        />
      </label>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="mt-3 w-full"
        style={{ maxHeight: 190 }}
        role="img"
        aria-label="A curve showing that the average price realised on exit falls as the fund owns more of the free float"
      >
        {/* the mark: a flat line at 100%, which is what the NAV assumes you can get */}
        <line
          x1={PAD.l}
          y1={y(1)}
          x2={VB.w - PAD.r}
          y2={y(1)}
          style={{ stroke: NAV, strokeWidth: 2, strokeDasharray: '4 3' }}
        />
        <text
          x={VB.w - PAD.r}
          y={y(1) - 4}
          textAnchor="end"
          style={{ fill: NAV, fontFamily: 'var(--font-mono)', fontSize: 9 }}
        >
          what the NAV marks
        </text>

        {/* the exit: what you actually average on the way out */}
        <polyline
          points={curve}
          fill="none"
          style={{ stroke: PRICE, strokeWidth: 2 }}
        />

        {/* the gap at the selected position */}
        <line
          x1={px(s)}
          y1={y(1)}
          x2={px(s)}
          y2={y(realisable(s))}
          style={{ stroke: WARN, strokeWidth: 2, transition: reduced ? 'none' : 'all 150ms' }}
        />
        <circle
          cx={px(s)}
          cy={y(realisable(s))}
          r={4}
          style={{ fill: WARN, transition: reduced ? 'none' : 'all 150ms' }}
        />

        {/* axes */}
        <line
          x1={PAD.l}
          y1={y(0)}
          x2={VB.w - PAD.r}
          y2={y(0)}
          style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        {[0, 0.2, 0.4, 0.6].map((v) => (
          <text
            key={v}
            x={px(v)}
            y={VB.h - 8}
            textAnchor="middle"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
          >
            {Math.round(v * 100)}%
          </text>
        ))}
        {[0, 0.5, 1].map((v) => (
          <text
            key={v}
            x={PAD.l - 5}
            y={y(v) + 3}
            textAnchor="end"
            style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
          >
            {Math.round(v * 100)}
          </text>
        ))}
      </svg>

      <p className="mt-1 min-h-[3.2rem] text-sm text-ink-soft">
        At {Math.round(s * 100)}% of the float, a full exit averages{' '}
        <strong style={{ color: WARN }}>{gapPct}% below</strong> the price the fund is marking its
        position at. The NAV is not wrong. It is just answering a different question: what the last
        share traded at, rather than what all of them would.
      </p>

      <p className="mt-2 border-t border-line pt-2 font-mono text-[0.66rem] leading-relaxed text-muted">
        Toy model: average fill = last price × (1 − float share), i.e. a linear walk down the book.
        Real price impact is neither linear nor fully permanent. van der Beck, Bouchaud and
        Villamaina estimate roughly half of a day’s impact reverts within 5 to 10 days.
      </p>
    </Widget>
  );
}
