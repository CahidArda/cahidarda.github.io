import { useMemo, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Slider,
  Specimen,
  T,
  useHoverPreview,
  useTween,
  useReducedMotion,
} from '../shared';
import { copyingModel, fitSlope, gauss, logBinned, predictedAlpha, rng } from './model';

/**
 * Two quantities on one chart, both measured in "multiples of the average":
 *  - heights of 100,000 men (illustrative normal: mean 175 cm, sd 7 cm)
 *  - in-links of 100,000 pages grown by the copying model (mean in-links is 1)
 * Toggle linear vs log-log. On log-log the power law is a straight line; the bell plunges.
 */
const N = 100_000;
const SD = 7 / 175; // height sd as a fraction of the mean

const W = 600;
const H = 300;
const M = { l: 52, r: 14, t: 14, b: 38 };
const PW = W - M.l - M.r;
const PH = H - M.t - M.b;

type Mode = 'linear' | 'loglog';

function useSim(p: number) {
  return useMemo(() => {
    const { indeg } = copyingModel(N, p, 7);
    let max = 0;
    let total = 0;
    for (let i = 0; i < N; i++) {
      total += indeg[i];
      if (indeg[i] > max) max = indeg[i];
    }
    const mean = total / N;
    // exact counts for k = 1..30 (linear view)
    const exact = new Array(31).fill(0);
    for (let i = 0; i < N; i++) if (indeg[i] <= 30) exact[indeg[i]]++;
    // log-binned density (log view), and the fit on the clean part of the tail
    const binned = logBinned(indeg, 1, 1.5);
    const c = p / (1 - p);
    const fitPts = binned.filter((q) => q.count >= 10 && q.x >= Math.max(3, 8 * c));
    const fit = fitSlope(fitPts);
    // scale both curves so their peak = 1 (y is "how common, relative to the most common")
    const peak = exact[1] / N; // k = 1 is the most common non-zero value
    return { exact, binned, fit, peak, max, mean };
  }, [p]);
}

const tallest = (() => {
  const r = rng(11);
  let m = 0;
  for (let i = 0; i < N; i++) m = Math.max(m, 1 + SD * gauss(r));
  return m;
})();

const bell = (x: number) => Math.exp(-((x - 1) ** 2) / (2 * SD * SD));

export default function BellVsPower() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Mode>('loglog');
  const [mode, bindMode] = useHoverPreview<Mode>(sel);
  const [p, setP] = useState(0.2);
  const sim = useSim(p);
  const log = mode === 'loglog';
  const morph = useTween(log ? 1 : 0, reduced, 8);

  // axes
  const LX = [Math.log10(0.5), Math.log10(30000)];
  const LY = [-7, 0.3];
  const linX = [0, 20];
  const linY = [0, 1.08];
  const sxLog = (x: number) => M.l + ((Math.log10(x) - LX[0]) / (LX[1] - LX[0])) * PW;
  const syLog = (y: number) =>
    M.t + (1 - (Math.log10(Math.max(y, 1e-9)) - LY[0]) / (LY[1] - LY[0])) * PH;
  const sxLin = (x: number) => M.l + ((x - linX[0]) / (linX[1] - linX[0])) * PW;
  const syLin = (y: number) => M.t + (1 - (y - linY[0]) / (linY[1] - linY[0])) * PH;
  // blend the two projections so the switch is a visible morph, not a cut
  const sx = (x: number) => sxLin(x) + (sxLog(x) - sxLin(x)) * morph;
  const sy = (y: number) => syLin(y) + (syLog(y) - syLin(y)) * morph;
  const inPlot = (x: number, y: number) =>
    x >= M.l - 0.5 && x <= W - M.r + 0.5 && y >= M.t - 0.5 && y <= H - M.b + 0.5;

  // bell curve path (heights)
  const bellPts: [number, number][] = [];
  for (let i = 0; i <= 240; i++) {
    const x = 0.75 + (0.5 * i) / 240;
    const y = bell(x);
    if (y < 1e-8) continue;
    bellPts.push([sx(x), sy(y)]);
  }
  const bellPath = bellPts
    .filter(([x, y]) => inPlot(x, y))
    .map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join('');

  // power-law points: exact counts on linear, log bins on log-log (blend picks which set)
  const pts = log
    ? sim.binned.map((q) => ({ x: q.x, y: q.y / sim.peak }))
    : sim.exact.slice(1).map((c, i) => ({ x: i + 1, y: c / N / sim.peak }));

  // fitted line on log-log
  const alpha = -sim.fit.slope;
  const fitY = (x: number) =>
    (Math.pow(10, sim.fit.intercept) * Math.pow(x, sim.fit.slope)) / sim.peak;
  const fx0 = 1;
  const fx1 = Math.min(20000, sim.max * 2);

  const xTicks = log ? [1, 10, 100, 1000, 10000] : [0, 5, 10, 15, 20];
  const yTicks = log ? [1, 1e-2, 1e-4, 1e-6] : [0, 0.5, 1];
  const fmtY = (v: number) => (log ? (v === 1 ? '1' : `10${sup(Math.log10(v))}`) : String(v));
  const fmtX = (v: number) => (v >= 1000 ? `${v / 1000}k` : String(v));
  const pred = predictedAlpha(p);

  return (
    <Specimen
      code="GT-10.2"
      title="bell vs power"
      kicker="100,000 men · 100,000 pages"
      hint="switch the axes; drag p to tilt the line"
      footer={
        <span className="inline-flex flex-wrap gap-x-3 gap-y-1">
          <Chip color={GT.b}>heights</Chip>
          <Chip color={GT.a}>links to a page</Chip>
        </span>
      }
    >
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <BigStat
          label="tallest man"
          value={`${tallest.toFixed(2)}×`}
          color={GT.b}
          sub="the average height"
        />
        <BigStat
          label="most-linked page"
          value={`${sim.max.toLocaleString('en-US')}×`}
          color={GT.a}
          sub="the average page"
        />
        <BigStat
          label="slope on log-log"
          value={`−${alpha.toFixed(2)}`}
          color={GT.eq}
          sub={`predicted −${pred.toFixed(2)}`}
        />
      </div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Seg<Mode>
          label="axes"
          value={sel}
          onChange={setSel}
          bind={bindMode}
          options={[
            { value: 'linear', label: 'linear' },
            { value: 'loglog', label: 'log-log' },
          ]}
        />
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        role="img"
        aria-label="Distribution of heights and of in-links"
      >
        <defs>
          <clipPath id="gt10-2-clip">
            <rect x={M.l} y={M.t} width={PW} height={PH} />
          </clipPath>
        </defs>
        {/* grid + ticks */}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line
              x1={log ? sxLog(v) : sxLin(v)}
              x2={log ? sxLog(v) : sxLin(v)}
              y1={M.t}
              y2={H - M.b}
              style={{ stroke: GT.line }}
            />
            <T x={log ? sxLog(v) : sxLin(v)} y={H - M.b + 12} size={10}>
              {fmtX(v)}
            </T>
          </g>
        ))}
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line
              x1={M.l}
              x2={W - M.r}
              y1={log ? syLog(v) : syLin(v)}
              y2={log ? syLog(v) : syLin(v)}
              style={{ stroke: GT.line }}
            />
            <T x={M.l - 6} y={log ? syLog(v) : syLin(v)} anchor="end" size={10}>
              {fmtY(v)}
            </T>
          </g>
        ))}
        <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} style={{ stroke: GT.lineStrong }} />
        <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} style={{ stroke: GT.lineStrong }} />
        <T x={M.l + PW / 2} y={H - 8} size={10} color={GT.inkSoft}>
          value, in multiples of the average
        </T>
        <g transform={`translate(12 ${M.t + PH / 2}) rotate(-90)`}>
          <T x={0} y={0} size={10} color={GT.inkSoft}>
            how common
          </T>
        </g>

        <g clipPath="url(#gt10-2-clip)">
          {/* fitted line */}
          {morph > 0.6 && (
            <line
              x1={sx(fx0)}
              y1={sy(fitY(fx0))}
              x2={sx(fx1)}
              y2={sy(fitY(fx1))}
              strokeDasharray="5 4"
              style={{ stroke: GT.eq, strokeWidth: 1.5, opacity: (morph - 0.6) / 0.4 }}
            />
          )}
          <path d={bellPath} style={{ fill: 'none', stroke: GT.b, strokeWidth: 2.2 }} />
          {pts.map((q, i) => (
            <rect
              key={i}
              x={sx(q.x) - 3}
              y={sy(q.y) - 3}
              width={6}
              height={6}
              style={{ fill: GT.a }}
            />
          ))}
        </g>
        {/* annotations */}
        {log ? (
          <>
            <T x={sxLog(1.35)} y={syLog(1e-5)} anchor="start" size={10} color={GT.b}>
              bell: falls off a cliff
            </T>
            <T x={sxLog(60)} y={syLog(fitY(60)) - 16} anchor="start" size={10} color={GT.a}>
              power law: a straight line
            </T>
          </>
        ) : (
          <>
            <T x={sxLin(1.6)} y={syLin(0.92)} anchor="start" size={10} color={GT.b}>
              heights: everyone near 1×
            </T>
            <T x={sxLin(6)} y={syLin(0.3)} anchor="start" size={10} color={GT.a}>
              links: most pages tiny, a long tail beyond 20×
            </T>
          </>
        )}
      </svg>
      <div className="mt-3 max-w-sm">
        <Slider
          label="p: chance a new link is picked at random"
          value={p}
          min={0.05}
          max={0.5}
          step={0.05}
          onChange={setP}
          format={(v) => v.toFixed(2)}
        />
      </div>
      <Caption>
        Each new page links to a random page with probability p, otherwise it copies an existing
        link. Copying more often (smaller p) flattens the line and makes giants.
      </Caption>
    </Specimen>
  );
}

function sup(n: number) {
  const map: Record<string, string> = {
    '-': '⁻',
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
  };
  return String(Math.round(n))
    .split('')
    .map((c) => map[c] ?? c)
    .join('');
}
