/**
 * The precision ladder (series landing): how small an error people could hold, from shadow
 * clocks to optical lattice clocks (time) and from the gnomon to Gaia (angle). Log scale,
 * better = higher. The red-ochre staircase is the best-so-far. Hover, focus or tap a rung to
 * read the instrument, the date, the figure and its source.
 */
import { useMemo, useState } from 'react';
import {
  C,
  Controls,
  InkText,
  Plate,
  Segmented,
  useHoverPreview,
  useMeasuredWidth,
} from './shared';
import {
  ANGLE,
  ANGLE_TICKS,
  TIME,
  TIME_TICKS,
  fmtAngle,
  fmtTime,
  type Rung,
} from './ladderData';

type Mode = 'time' | 'angle';

const CFG: Record<
  Mode,
  {
    rungs: Rung[];
    ticks: { v: number; label: string }[];
    lo: number;
    hi: number;
    fmt: (v: number) => string;
    unit: string;
    start: number;
  }
> = {
  time: {
    rungs: TIME,
    ticks: TIME_TICKS,
    lo: -14,
    hi: 4,
    fmt: fmtTime,
    unit: 'error per day',
    start: 3,
  },
  angle: {
    rungs: ANGLE,
    ticks: ANGLE_TICKS,
    lo: -6,
    hi: 4,
    fmt: fmtAngle,
    unit: 'error in angle',
    start: 3,
  },
};

const Y0 = -1500;
const YB = 1500; // the axis is compressed before this year
const Y1 = 2040;
const SPLIT = 0.3; // share of the width given to 1500 BCE .. 1500 CE

function yearLabel(y: number) {
  return y < 0 ? `${-y} BCE` : `${y}`;
}

export default function PrecisionLadder() {
  const [sel, setSel] = useState<Mode>('time');
  const [mode, bindMode] = useHoverPreview(sel);
  const [pick, setPick] = useState<Record<Mode, number>>({
    time: CFG.time.start,
    angle: CFG.angle.start,
  });
  const [hover, setHover] = useState<number | null>(null);
  const [wrapRef, W] = useMeasuredWidth<HTMLDivElement>(640);

  const cfg = CFG[mode];
  const narrow = W < 520;
  const H = narrow ? 340 : 400;
  const m = { l: narrow ? 46 : 58, r: narrow ? 14 : 22, t: 22, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;

  const x = (yr: number) => {
    if (yr <= YB) return m.l + ((yr - Y0) / (YB - Y0)) * SPLIT * iw;
    return m.l + SPLIT * iw + ((yr - YB) / (Y1 - YB)) * (1 - SPLIT) * iw;
  };
  // smaller error = higher on the page; the ladder climbs
  const y = (v: number) => m.t + ((Math.log10(v) - cfg.lo) / (cfg.hi - cfg.lo)) * ih;

  const pts = useMemo(
    () => cfg.rungs.map((r) => ({ r, x: x(r.year), y: y(r.value) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, W, H],
  );

  // best-so-far staircase
  const stair = useMemo(() => {
    let best = Infinity;
    let d = '';
    pts.forEach((p, i) => {
      if (i === 0) d = `M ${p.x} ${p.y}`;
      else {
        d += ` H ${p.x}`;
        if (p.y < best) d += ` V ${p.y}`;
      }
      best = Math.min(best, p.y);
    });
    d += ` H ${m.l + iw}`;
    return d;
  }, [pts, m.l, iw]);

  const active = hover ?? pick[mode];
  const cur = cfg.rungs[Math.min(active, cfg.rungs.length - 1)];

  const xTicks = narrow ? [-1000, 1500, 1700, 1900, 2000] : [-1000, 500, 1500, 1600, 1700, 1800, 1900, 2000];

  const choose = (i: number) => setPick((p) => ({ ...p, [mode]: i }));

  return (
    <Plate
      n={1}
      wide
      title="The precision ladder"
      kicker="log scale · hover a rung"
      caption={
        <>
          Each rung is the best error an instrument could hold; the red line is the best so far.
          Hollow marks are order-of-magnitude estimates. The years before 1500 are squeezed into
          the dashed stretch on the left.
        </>
      }
    >
      <Controls>
        <Segmented
          label="Quantity"
          options={[
            { value: 'time', label: 'Time' },
            { value: 'angle', label: 'Angle' },
          ]}
          value={sel}
          onChange={(v) => {
            setSel(v);
            setHover(null);
          }}
          preview={bindMode}
        />
        <span
          className="font-mono text-[0.68rem] tracking-[0.12em] uppercase"
          style={{ color: C.soft }}
        >
          {cfg.unit}
        </span>
      </Controls>

      <div ref={wrapRef} style={{ marginTop: '0.75rem' }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          role="img"
          aria-label={`Log-scale chart of ${cfg.unit} for instruments from 1500 BCE to today`}
          style={{ display: 'block', overflow: 'visible' }}
          onMouseLeave={() => setHover(null)}
        >
          {/* decade grid */}
          {Array.from({ length: cfg.hi - cfg.lo + 1 }, (_, k) => cfg.lo + k).map((d) => (
            <line
              key={d}
              x1={m.l}
              x2={m.l + iw}
              y1={y(10 ** d)}
              y2={y(10 ** d)}
              style={{ stroke: C.faint, strokeWidth: 0.5, opacity: 0.45 }}
            />
          ))}
          {/* named ticks */}
          {cfg.ticks.map((t) => (
            <g key={t.label}>
              <line
                x1={m.l - 5}
                x2={m.l + iw}
                y1={y(t.v)}
                y2={y(t.v)}
                style={{ stroke: C.faint, strokeWidth: 0.8 }}
              />
              <InkText
                x={m.l - 8}
                y={y(t.v)}
                kind="num"
                anchor="end"
                baseline="central"
                size={narrow ? 9.5 : 10.5}
                role="soft"
                halo={false}
              >
                {t.label}
              </InkText>
            </g>
          ))}
          {/* axes */}
          <path
            d={`M ${m.l} ${m.t - 6} V ${m.t + ih}`}
            style={{ stroke: C.ink, strokeWidth: 1, fill: 'none' }}
          />
          <path
            d={`M ${m.l} ${m.t + ih} H ${x(YB)}`}
            style={{ stroke: C.ink, strokeWidth: 1, strokeDasharray: '4 3', fill: 'none' }}
          />
          <path
            d={`M ${x(YB)} ${m.t + ih} H ${m.l + iw}`}
            style={{ stroke: C.ink, strokeWidth: 1, fill: 'none' }}
          />
          {/* break mark where the scale changes */}
          <path
            d={`M ${x(YB) - 5} ${m.t + ih + 5} l 4 -10 M ${x(YB) - 1} ${m.t + ih + 5} l 4 -10`}
            style={{ stroke: C.ink, strokeWidth: 0.9, fill: 'none' }}
          />
          {xTicks.map((yr) => (
            <g key={yr}>
              <line
                x1={x(yr)}
                x2={x(yr)}
                y1={m.t + ih}
                y2={m.t + ih + 4}
                style={{ stroke: C.ink, strokeWidth: 0.8 }}
              />
              <InkText
                x={x(yr)}
                y={m.t + ih + 16}
                kind="num"
                anchor="middle"
                size={narrow ? 9 : 10}
                role="soft"
                halo={false}
              >
                {yearLabel(yr)}
              </InkText>
            </g>
          ))}
          {/* "better" arrow, bottom right (empty in both modes: late and imprecise) */}
          <g>
            <path
              d={`M ${m.l + iw - 4} ${m.t + ih - 8} V ${m.t + ih - 40} M ${m.l + iw - 7.5} ${m.t + ih - 33} L ${m.l + iw - 4} ${m.t + ih - 40} L ${m.l + iw - 0.5} ${m.t + ih - 33}`}
              style={{ stroke: C.soft, strokeWidth: 0.8, fill: 'none' }}
            />
            <InkText x={m.l + iw - 11} y={m.t + ih - 30} anchor="end" size={12} role="soft">
              more precise
            </InkText>
          </g>

          {/* the ladder: best-so-far */}
          <path
            key={mode}
            d={stair}
            style={{ stroke: C.red, strokeWidth: 1.6, fill: 'none', strokeLinejoin: 'miter' }}
          />

          {/* rungs */}
          {pts.map((p, i) => {
            const on = i === active;
            const labelRight = p.x < m.l + iw * 0.8;
            const s = on ? 5.5 : 4;
            const diamond = `M ${p.x} ${p.y - s} L ${p.x + s} ${p.y} L ${p.x} ${p.y + s} L ${p.x - s} ${p.y} Z`;
            const dy = labelDy(mode, i, narrow);
            return (
              <g
                key={`${mode}-${i}`}
                tabIndex={0}
                role="button"
                aria-label={`${p.r.name}, ${p.r.era}: ${cfg.fmt(p.r.value)}`}
                aria-pressed={pick[mode] === i}
                onMouseEnter={() => setHover(i)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                onClick={() => choose(i)}
                style={{ cursor: 'pointer', outline: 'none' }}
              >
                <circle cx={p.x} cy={p.y} r={14} style={{ fill: 'transparent' }} />
                <line
                  x1={p.x - 9}
                  x2={p.x + 9}
                  y1={p.y}
                  y2={p.y}
                  style={{ stroke: C.ink, strokeWidth: 1 }}
                />
                <path
                  d={diamond}
                  style={{
                    fill: p.r.oom ? C.paper : on ? C.red : C.ink,
                    stroke: on ? C.red : C.ink,
                    strokeWidth: 1.1,
                    strokeDasharray: p.r.oom ? '2 1.5' : undefined,
                  }}
                />
                {(!narrow || on) && (
                  <InkText
                    x={p.x + (labelRight ? 12 : -12)}
                    y={p.y + dy}
                    anchor={labelRight ? 'start' : 'end'}
                    baseline="central"
                    size={narrow ? 11.5 : 12.5}
                    role={on ? 'red' : 'ink'}
                  >
                    {p.r.short}
                  </InkText>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* the reading */}
      <div
        aria-live="polite"
        style={{
          marginTop: '0.9rem',
          paddingTop: '0.7rem',
          borderTop: `1px solid ${C.soft}`,
          minHeight: '7.2rem',
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) auto',
          columnGap: '1.25rem',
          rowGap: '0.25rem',
        }}
      >
        <div>
          <div
            className="font-mono text-[0.66rem] tracking-[0.16em] uppercase"
            style={{ color: C.red }}
          >
            {cur.era}
            {cur.oom ? ' · order of magnitude' : ''}
          </div>
          <div
            className="font-display text-[1.25rem] leading-snug italic"
            style={{ color: C.ink }}
          >
            {cur.name}
          </div>
        </div>
        <div
          className="self-end text-right font-mono text-[1.35rem] tabular-nums"
          style={{ color: C.ink }}
        >
          {cfg.fmt(cur.value)}
          <div
            className="text-[0.62rem] tracking-[0.12em] uppercase"
            style={{ color: C.soft }}
          >
            {mode === 'time' ? 'per day' : 'of arc'}
          </div>
        </div>
        <div
          className="text-[0.92rem] leading-normal"
          style={{ color: C.soft, gridColumn: '1 / -1' }}
        >
          {cur.figure}{' '}
          <a
            href={cur.src}
            target="_blank"
            rel="noopener"
            className="font-mono text-[0.7rem] whitespace-nowrap"
            style={{ color: C.red, textDecoration: 'underline', textUnderlineOffset: '0.2em' }}
          >
            {cur.srcLabel}
          </a>
        </div>
      </div>
    </Plate>
  );
}

/** Nudge a few labels up/down where neighbours would collide. */
function labelDy(mode: Mode, i: number, narrow: boolean): number {
  const k = narrow ? 0.8 : 1;
  if (mode === 'angle') {
    if (i === 0) return 10 * k; // gnomon below Ptolemy
    if (i === 1) return -9 * k;
    if (i === 5) return 9 * k; // theodolite below the meridian circle
    if (i === 6) return -9 * k;
  }
  if (mode === 'time') {
    if (i === 0) return 10 * k; // sundial sits on the same level as the verge
    if (i === 4) return -6 * k; // H4
  }
  return 0;
}
