import { useEffect, useId, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Slider,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../shared';
import { payoff, type Format } from './logic';

/* One square plot: x = your bid, y = the highest rival bid, both 0..100 dollars. */
const S = 340;
const L = 46;
const R = 12;
const TOP = 12;
const B = 40;
const PW = S - L - R;
const PH = S - TOP - B;
const X = (b: number) => Math.round((L + (b / 100) * PW) * 100) / 100;
const Y = (r: number) => Math.round((TOP + PH - (r / 100) * PH) * 100) / 100;
const poly = (pts: [number, number][]) => pts.map(([b, r]) => `${X(b)},${Y(r)}`).join(' ');

const FORMATS: { value: Format; label: string }[] = [
  { value: 'second', label: 'second price' },
  { value: 'first', label: 'first price' },
];

// the auto-demo: (your bid, rival bid) stops, shown in the second-price auction with value 60
const DEMO: [number, number][] = [
  [60, 40],
  [78, 70],
  [32, 46],
  [60, 46],
];
const DEMO_MS = 2100;

function outcome(format: Format, v: number, b: number, r: number) {
  const g = payoff(format, v, b, r);
  if (b > r) {
    const pay = format === 'second' ? r : b;
    if (g > 0) return `You win and pay $${pay}. Gain $${g}.`;
    if (g === 0) return `You win and pay $${pay}, exactly your value. Gain $0.`;
    return `You win but pay $${pay}, more than it is worth to you. Lose $${-g}.`;
  }
  if (r < v)
    return format === 'second'
      ? `You lose and gain $0. Bidding your value would have won $${v - r}.`
      : `You lose and gain $0. A bid between $${r} and $${v} would have won at a profit.`;
  return `You lose. Gain $0, and winning would have cost more than $${v}.`;
}

export default function BidPlot() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const uid = useId().replace(/:/g, '');

  const [sel, setSel] = useState<Format>('second');
  const [format, bindFormat] = useHoverPreview(sel);
  const [v, setV] = useState(60);
  const [pt, setPt] = useState<[number, number]>([60, 40]);
  const [show, setShow] = useState<[number, number]>([60, 40]);
  const [step, setStep] = useState(0);
  const [demo, setDemo] = useState(false);
  const touched = useRef(false);
  const started = useRef(false);

  useEffect(() => {
    if (started.current || !inView || reduced) return;
    const id = setTimeout(() => {
      started.current = true;
      if (!touched.current) setDemo(true);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useEffect(() => {
    if (!demo || !inView) return;
    const id = setTimeout(() => {
      if (step + 1 >= DEMO.length) {
        setDemo(false);
        return;
      }
      setStep(step + 1);
      setPt(DEMO[step + 1]);
    }, DEMO_MS);
    return () => clearTimeout(id);
  }, [demo, step, inView]);

  useFrame(
    (dt) =>
      setShow((s) => [
        s[0] + (pt[0] - s[0]) * Math.min(1, dt * 7),
        s[1] + (pt[1] - s[1]) * Math.min(1, dt * 7),
      ]),
    !reduced && (Math.abs(show[0] - pt[0]) > 0.05 || Math.abs(show[1] - pt[1]) > 0.05),
  );
  useEffect(() => {
    if (reduced) setShow(pt);
  }, [reduced, pt]);

  const takeOver = () => {
    touched.current = true;
    setDemo(false);
  };
  const placeAt = (e: React.PointerEvent<SVGSVGElement>) => {
    const p = svgPoint(svgRef.current!, e.clientX, e.clientY);
    const b = Math.round(clamp(((p.x - L) / PW) * 100, 0, 100));
    const r = Math.round(clamp(((TOP + PH - p.y) / PH) * 100, 0, 100));
    setPt([b, r]);
    setShow([b, r]);
  };
  const dragging = useRef(false);
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    takeOver();
    dragging.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    placeAt(e);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging.current) placeAt(e);
  };
  const onUp = () => (dragging.current = false);
  const onKey = (e: React.KeyboardEvent) => {
    const m: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    };
    const d = m[e.key];
    if (!d) return;
    e.preventDefault();
    takeOver();
    const k = e.shiftKey ? 10 : 2;
    const next: [number, number] = [
      clamp(pt[0] + d[0] * k, 0, 100),
      clamp(pt[1] + d[1] * k, 0, 100),
    ];
    setPt(next);
    setShow(next);
  };

  const [b, r] = pt;
  const gain = payoff(format, v, b, r);
  const gColor = gain > 0 ? GT.opt : gain < 0 ? GT.b : GT.muted;
  const second = format === 'second';

  // regions (in dollars), see logic.payoff
  const profit: [number, number][] = second
    ? [
        [0, 0],
        [100, 0],
        [100, v],
        [v, v],
      ]
    : [
        [0, 0],
        [v, 0],
        [v, v],
      ];
  const loss: [number, number][] = second
    ? [
        [v, v],
        [100, v],
        [100, 100],
      ]
    : [
        [v, 0],
        [100, 0],
        [100, 100],
        [v, v],
      ];
  const missed: [number, number][] = [
    [0, 0],
    [v, v],
    [0, v],
  ];
  const gP = `${uid}-p`;
  const gL = `${uid}-l`;

  return (
    <Specimen
      code="GT-04.1"
      title="your bid vs your value"
      kicker="sealed bids · one item"
      rootRef={rootRef}
      hint="drag anywhere on the plot (or focus it and use arrow keys)"
      footer={second ? 'you pay the rival bid' : 'you pay your own bid'}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Seg
          options={FORMATS}
          value={format}
          onChange={(f) => (takeOver(), setSel(f))}
          bind={bindFormat}
          label="format"
        />
        <div className="w-44">
          <Slider
            label="your value"
            value={v}
            min={20}
            max={90}
            onChange={(x) => {
              takeOver();
              setV(x);
            }}
            format={(x) => `$${x}`}
            color={GT.eq}
          />
        </div>
      </div>

      <div className="mt-4 grid gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,15rem)] md:items-center">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${S} ${S}`}
          className="mx-auto block w-full max-w-[26rem] touch-none select-none"
          role="img"
          tabIndex={0}
          aria-label={`Payoff map, ${second ? 'second' : 'first'}-price auction, your value $${v}. Your bid $${b}, rival bid $${r}. ${outcome(
            format,
            v,
            b,
            r,
          )}`}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onKeyDown={onKey}
          style={{ outline: 'none', cursor: 'crosshair' }}
        >
          <defs>
            <linearGradient
              id={gP}
              gradientUnits="userSpaceOnUse"
              x1={second ? X(0) : X(0)}
              y1={second ? Y(0) : Y(0)}
              x2={second ? X(0) : X(v)}
              y2={second ? Y(v) : Y(0)}
            >
              <stop offset="0" style={{ stopColor: GT.opt, stopOpacity: 0.55 }} />
              <stop offset="1" style={{ stopColor: GT.opt, stopOpacity: 0.06 }} />
            </linearGradient>
            <linearGradient
              id={gL}
              gradientUnits="userSpaceOnUse"
              x1={X(v)}
              y1={Y(v)}
              x2={second ? X(v) : X(100)}
              y2={second ? Y(100) : Y(v)}
            >
              <stop offset="0" style={{ stopColor: GT.b, stopOpacity: 0.08 }} />
              <stop offset="1" style={{ stopColor: GT.b, stopOpacity: 0.6 }} />
            </linearGradient>
            <pattern
              id={`${uid}-h`}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)"
            >
              <line
                x1="0"
                y1="0"
                x2="0"
                y2="6"
                style={{ stroke: GT.muted, strokeWidth: 1.2, opacity: 0.55 }}
              />
            </pattern>
          </defs>

          <rect
            x={L}
            y={TOP}
            width={PW}
            height={PH}
            style={{ fill: GT.paperRaised, stroke: GT.lineStrong }}
          />
          <polygon points={poly(profit)} style={{ fill: `url(#${gP})` }} />
          <polygon points={poly(loss)} style={{ fill: `url(#${gL})` }} />
          <polygon points={poly(missed)} style={{ fill: `url(#${uid}-h)` }} />

          {/* tie diagonal: above it the rival outbids you */}
          <line
            x1={X(0)}
            y1={Y(0)}
            x2={X(100)}
            y2={Y(100)}
            style={{ stroke: GT.muted, strokeWidth: 1, strokeDasharray: '3 3' }}
          />
          <T x={X(78)} y={Y(86)} anchor="end" size={9}>
            tie
          </T>

          {/* region labels */}
          {v >= 30 && (
            <T
              x={X(second ? Math.min(78, v + 22) : v * 0.62)}
              y={Y(second ? v * 0.3 : v * 0.18)}
              color={GT.opt}
              size={10.5}
              weight={600}
            >
              win, gain
            </T>
          )}
          {v <= 72 && (
            <T
              x={X(second ? 98 : (v + 100) / 2 + 4)}
              y={Y(second ? v + (100 - v) * 0.45 : (v + 100) / 4)}
              anchor={second ? 'end' : 'middle'}
              color={GT.b}
              size={10.5}
              weight={600}
            >
              overpay
            </T>
          )}
          {v >= 35 && (
            <T x={X(v * 0.28)} y={Y(v * 0.68)} color={GT.inkSoft} size={10.5} weight={600}>
              missed
            </T>
          )}

          {/* honest bid: the vertical line bid = value */}
          <line
            x1={X(v)}
            y1={TOP}
            x2={X(v)}
            y2={TOP + PH}
            style={{ stroke: GT.eq, strokeWidth: 2 }}
          />
          <T
            x={X(v) + (v > 70 ? -4 : 4)}
            y={TOP + 9}
            anchor={v > 70 ? 'end' : 'start'}
            color={GT.eq}
            size={10}
            weight={600}
          >
            bid = value
          </T>

          {/* the reader's point */}
          <line
            x1={X(show[0])}
            y1={Y(show[1])}
            x2={X(show[0])}
            y2={TOP + PH}
            style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
          />
          <line
            x1={L}
            y1={Y(show[1])}
            x2={X(show[0])}
            y2={Y(show[1])}
            style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 3' }}
          />
          <circle
            cx={X(show[0])}
            cy={Y(show[1])}
            r={7}
            style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 2 }}
          />
          <circle cx={X(show[0])} cy={Y(show[1])} r={2.5} style={{ fill: GT.ink }} />

          {/* axes */}
          {[0, 50, 100].map((t) => (
            <g key={t}>
              <T x={X(t)} y={TOP + PH + 12} size={9.5}>
                ${t}
              </T>
              <T x={L - 6} y={Y(t)} anchor="end" size={9.5}>
                ${t}
              </T>
            </g>
          ))}
          <T x={L + PW / 2} y={S - 8} size={10.5} color={GT.ink}>
            your bid →
          </T>
          <g transform={`translate(12 ${TOP + PH / 2}) rotate(-90)`}>
            <T x={0} y={0} size={10.5} color={GT.ink}>
              highest rival bid →
            </T>
          </g>
        </svg>

        <div className="flex flex-col gap-4">
          <BigStat
            label="your gain"
            value={`${gain < 0 ? '−' : ''}$${Math.abs(gain)}`}
            color={gColor}
            size="lg"
            sub={`bid $${b} · rival $${r} · value $${v}`}
          />
          <p className="font-mono text-[0.72rem] leading-snug text-ink-soft">
            {outcome(format, v, b, r)}
          </p>
          <div className="flex flex-col gap-1.5">
            <Chip color={GT.opt}>win, gain (darker = more)</Chip>
            <Chip color={GT.b}>win, but pay more than value</Chip>
            <Chip color={GT.muted}>lose a deal worth winning</Chip>
            <Chip color={GT.eq}>bid exactly your value</Chip>
          </div>
        </div>
      </div>
      <Caption>
        {second
          ? 'The line bid = value never touches red or grey. No other bid does better, whatever the rival does.'
          : 'Now the line bid = value earns $0 everywhere. To gain anything you must bid below your value.'}
      </Caption>
    </Specimen>
  );
}
