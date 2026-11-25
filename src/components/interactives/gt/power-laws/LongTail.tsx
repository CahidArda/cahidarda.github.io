import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { useNarrow } from './useNarrow';
import { copyingModelMulti, gauss, rng } from './model';

/**
 * Rank 1,000 pages by in-links (copying model, 3 links per new page, p = 0.3) and 1,000 men
 * by height. Drag the cut: what share of the total does the top x% hold?
 */
const N = 1000;
const M = { l: 40, r: 12 };
const TOP = { y: 22, h: 150 };
const BOT = { y: 216, h: 56 };
const H = 296;

const links = (() => copyingModelMulti(N, 3, 0.3, 2).sort((a, b) => b - a))();
const heights = (() => {
  const r = rng(5);
  return Array.from({ length: N }, () => 175 + 7 * gauss(r)).sort((a, b) => b - a);
})();
const sum = (a: number[], m = a.length) => a.slice(0, m).reduce((x, y) => x + y, 0);
const LT = sum(links);
const HT = sum(heights);

/** Area path of a ranked series. */
function areaPath(vals: number[], y0: number, h: number, vmax: number, PW: number) {
  const step = PW / vals.length;
  let d = `M${M.l},${y0 + h}`;
  vals.forEach((v, i) => {
    const y = y0 + h - (v / vmax) * h;
    d += `L${(M.l + i * step).toFixed(2)},${y.toFixed(2)}L${(M.l + (i + 1) * step).toFixed(2)},${y.toFixed(2)}`;
  });
  return d + `L${M.l + PW},${y0 + h}Z`;
}

const zeroAt = links.findIndex((v) => v === 0);

type Pick = '0.01' | '0.1' | '0.2' | '0.5';

export default function LongTail() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [cut, setCut] = useState(0.01);
  const [demoOn, setDemoOn] = useState(false);
  const demoT = useRef(0);
  const started = useRef(false);
  const dragging = useRef(false);
  const narrow = useNarrow(rootRef);
  const W = narrow ? 380 : 600;
  const PW = W - M.l - M.r;
  const linkPath = useMemo(() => areaPath(links, TOP.y, TOP.h, links[0], PW), [PW]);
  const heightPath = useMemo(() => areaPath(heights, BOT.y, BOT.h, heights[0], PW), [PW]);

  useEffect(() => {
    if (started.current || !inView || reduced) return;
    started.current = true;
    demoT.current = 0;
    setDemoOn(true);
  }, [inView, reduced]);

  // demo: sweep the cut from half the pages down to the top 1%
  useFrame((dt) => {
    demoT.current = Math.min(1, demoT.current + dt / 2.6);
    const e = 1 - Math.pow(1 - demoT.current, 3);
    setCut(0.5 + (0.01 - 0.5) * e);
    if (demoT.current >= 1) setDemoOn(false);
  }, demoOn && inView);

  const m = Math.max(1, Math.round(cut * N));
  const linkShare = useMemo(() => sum(links, m) / LT, [m]);
  const heightShare = useMemo(() => sum(heights, m) / HT, [m]);
  const cx = M.l + (m / N) * PW;

  const setFromEvent = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const { x } = svgPoint(svgRef.current, e.clientX, e.clientY);
    setCut(clamp((x - M.l) / PW, 0.002, 0.95));
  };
  const stopDemo = () => setDemoOn(false);

  const pct = (v: number) => (v < 0.1 ? `${(v * 100).toFixed(1)}%` : `${Math.round(v * 100)}%`);
  const cutLabel = m / N < 0.1 ? `${((m / N) * 100).toFixed(1)}%` : `${Math.round((m / N) * 100)}%`;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-10.3"
      title="the long tail"
      kicker="1,000 pages · 1,000 men, ranked"
      hint="drag the cut across the ranking"
      footer={
        <span className="inline-flex flex-wrap gap-x-3 gap-y-1">
          <Chip color={GT.a}>links</Chip>
          <Chip color={GT.b}>height</Chip>
        </span>
      }
    >
      <div className="mb-4 grid grid-cols-2 gap-3">
        <BigStat
          label={`top ${cutLabel} of pages`}
          value={pct(linkShare)}
          color={GT.a}
          sub="of all links"
          size="lg"
        />
        <BigStat
          label={`top ${cutLabel} of men`}
          value={pct(heightShare)}
          color={GT.b}
          sub="of all height"
          size="lg"
        />
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full touch-none select-none"
        style={{ cursor: 'ew-resize' }}
        role="img"
        aria-label={`The top ${cutLabel} of pages hold ${pct(linkShare)} of links; the top ${cutLabel} of men hold ${pct(heightShare)} of total height`}
        onPointerDown={(e) => {
          stopDemo();
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          setFromEvent(e);
        }}
        onPointerMove={(e) => dragging.current && setFromEvent(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        <defs>
          <clipPath id="gt10-3-head">
            <rect x={M.l} y={0} width={cx - M.l} height={H} />
          </clipPath>
        </defs>
        <T x={M.l + 14} y={TOP.y - 12} anchor="start" size={9.5} color={GT.inkSoft}>
          links to each page, most-linked first
        </T>
        <T x={W - M.r} y={TOP.y - 12} anchor="end" size={9.5}>
          max {links[0]}
        </T>
        <path d={linkPath} style={{ fill: GT.a, opacity: 0.22 }} />
        <path d={linkPath} clipPath="url(#gt10-3-head)" style={{ fill: GT.a }} />
        <line
          x1={M.l}
          x2={W - M.r}
          y1={TOP.y + TOP.h}
          y2={TOP.y + TOP.h}
          style={{ stroke: GT.lineStrong }}
        />
        {zeroAt > 0 && (
          <>
            <line
              x1={M.l + (zeroAt / N) * PW}
              x2={W - M.r}
              y1={TOP.y + TOP.h - 10}
              y2={TOP.y + TOP.h - 10}
              style={{ stroke: GT.muted }}
            />
            <T x={(M.l + (zeroAt / N) * PW + W - M.r) / 2} y={TOP.y + TOP.h - 20} size={9.5}>
              {Math.round(((N - zeroAt) / N) * 100)}% of pages: no links at all
            </T>
          </>
        )}

        <T x={M.l + 14} y={BOT.y - 12} anchor="start" size={9.5} color={GT.inkSoft}>
          height of each man, tallest first
        </T>
        <path d={heightPath} style={{ fill: GT.b, opacity: 0.22 }} />
        <path d={heightPath} clipPath="url(#gt10-3-head)" style={{ fill: GT.b }} />
        <line
          x1={M.l}
          x2={W - M.r}
          y1={BOT.y + BOT.h}
          y2={BOT.y + BOT.h}
          style={{ stroke: GT.lineStrong }}
        />

        {/* the cut */}
        <line
          x1={cx}
          x2={cx}
          y1={TOP.y - 4}
          y2={BOT.y + BOT.h + 4}
          style={{ stroke: GT.eq, strokeWidth: 2 }}
        />
        <rect x={cx - 6} y={BOT.y + BOT.h + 4} width={12} height={12} style={{ fill: GT.eq }} />
        <T x={clamp(cx, M.l + 30, W - M.r - 30)} y={H - 2} size={9.5} color={GT.eq} baseline="auto">
          top {cutLabel}
        </T>
        <T x={M.l - 6} y={TOP.y + TOP.h} anchor="end" size={9}>
          0
        </T>
        <T x={M.l - 6} y={BOT.y + BOT.h} anchor="end" size={9}>
          0
        </T>
      </svg>
      <div className="mt-3">
        <Seg<Pick>
          label="jump to"
          value={
            (['0.01', '0.1', '0.2', '0.5'] as Pick[]).find(
              (v) => Math.abs(Number(v) - m / N) < 1e-9,
            ) ?? ('' as Pick)
          }
          onChange={(v) => {
            stopDemo();
            setCut(Number(v));
          }}
          options={[
            { value: '0.01', label: 'top 1%' },
            { value: '0.1', label: '10%' },
            { value: '0.2', label: '20%' },
            { value: '0.5', label: '50%' },
          ]}
        />
      </div>
      <Caption>
        Heights fill the chart like a block: the top 1% of men hold about 1% of the height. Links
        pile into a needle at the left and a long, thin tail to the right.
      </Caption>
    </Specimen>
  );
}
