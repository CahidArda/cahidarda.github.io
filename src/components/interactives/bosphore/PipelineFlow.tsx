/**
 * How the Bosphore 1819 labels were made. One SVG coordinate system (skill 4.1), edges
 * anchored to box edges (4.2), a token that walks the pipeline behind the nodes (4.6), a
 * caption per step at reading pace (5), hover a node to jump and pause (9.2), and a static
 * poster under reduced motion. Reuses the data-platform primitives so it reads as house style.
 */
import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  Edge,
  SvgNode,
  Widget,
  bottomOf,
  leftOf,
  rightOf,
  topOf,
  useGlide,
  useInView,
  useReducedMotion,
  type Box,
  type NState,
} from '../dataplat/shared';

const INK = 'var(--color-bos-ink)';
const AGENT = 'var(--color-bos-agent)';
const STEP_MS = 3000;

const N: Record<'scan' | 'tiles' | 'merge' | 'overlay' | 'review' | 'app', Box> = {
  scan: { cx: 58, cy: 150, w: 92, h: 46 },
  tiles: { cx: 186, cy: 150, w: 92, h: 46 },
  merge: { cx: 452, cy: 150, w: 92, h: 46 },
  overlay: { cx: 580, cy: 80, w: 92, h: 46 },
  review: { cx: 580, cy: 220, w: 92, h: 46 },
  app: { cx: 708, cy: 150, w: 92, h: 46 },
};
const AGENTS: Box[] = Array.from({ length: 6 }, (_, i) => ({
  cx: 318,
  cy: 45 + i * 42,
  w: 78,
  h: 30,
}));

type StepId = 'scan' | 'tiles' | 'agents' | 'merge' | 'overlay' | 'review' | 'merge2' | 'app';
const STEPS: { id: StepId; label: string; at: { x: number; y: number }; caption: string }[] = [
  {
    id: 'scan',
    label: 'scan',
    at: { x: N.scan.cx, y: N.scan.cy },
    caption: 'The scan: 12,509 by 7,749 pixels from Wikimedia Commons, public domain, never copied into the repo.',
  },
  {
    id: 'tiles',
    label: 'tiles',
    at: { x: N.tiles.cx, y: N.tiles.cy },
    caption: 'Cut into 54 overlapping tiles, each rendered with a red grid labelled in full-image pixels, so a box can be read off without arithmetic.',
  },
  {
    id: 'agents',
    label: 'OCR',
    at: { x: AGENTS[0].cx, y: 150 },
    caption: 'Fourteen batches, six Claude Opus 5.5 subagents at a time, four tiles each. Every label becomes one JSON record, uncertain: true whenever it is a guess.',
  },
  {
    id: 'merge',
    label: 'merge',
    at: { x: N.merge.cx, y: N.merge.cy },
    caption: 'Validate against the schema, drop the duplicates the overlaps produce, convert boxes to fractions of the image, write labels.json.',
  },
  {
    id: 'overlay',
    label: 'overlay',
    at: { x: N.overlay.cx, y: N.overlay.cy },
    caption: 'Draw every box back onto the map: green when confident, orange when uncertain, the id written next to it.',
  },
  {
    id: 'review',
    label: 'review',
    at: { x: N.review.cx, y: N.review.cy },
    caption: 'Twelve reviewer subagents compare each overlay with the clean tile: fix boxes, add misses, correct readings, unify spellings.',
  },
  {
    id: 'merge2',
    label: 'merge again',
    at: { x: N.merge.cx, y: N.merge.cy },
    caption: 'Merge again, then one global pass: same place, same spelling everywhere; consistent Arabic letterforms; no entry missing a gloss.',
  },
  {
    id: 'app',
    label: 'app',
    at: { x: N.app.cx, y: N.app.cy },
    caption: '387 labels, 156 of them flagged uncertain, fetched by the viewer as one 47 KB file.',
  },
];

const ORDER: StepId[] = STEPS.map((s) => s.id);
const nodeStep: Record<string, number> = {
  scan: 0,
  tiles: 1,
  agents: 2,
  merge: 3,
  overlay: 4,
  review: 5,
  app: 7,
};

function stateOf(node: string, step: number): NState {
  const idx = nodeStep[node];
  const activeNow = ORDER[step] === node || (node === 'merge' && ORDER[step] === 'merge2');
  if (activeNow) return 'active';
  return step > idx ? 'wait' : 'off';
}

export default function PipelineFlow() {
  const reduced = useReducedMotion();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [step, setStep] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (reduced) {
      setStep(STEPS.length - 1);
      return;
    }
    if (paused || !inView) return;
    const id = setInterval(() => setStep((s) => (s + 1) % STEPS.length), STEP_MS);
    return () => clearInterval(id);
  }, [reduced, paused, inView]);

  const pos = useGlide(STEPS[step].at, reduced);
  const cur = STEPS[step];
  const on = (k: StepId) => cur.id === k;

  const jump = (node: string) => {
    setPaused(true);
    setStep(nodeStep[node]);
  };
  const resume = () => setPaused(false);

  return (
    <Widget title="From scan to labels.json" kicker={`step ${step + 1} / ${STEPS.length}`} rootRef={viewRef}>
      <svg viewBox="0 0 760 300" className="block h-auto w-full" role="img" aria-label="The label pipeline: scan, tiles, OCR subagents, merge, overlay, review, merge again, app">
        <ArrowDefs />

        {/* edges, anchored to box edges */}
        <Edge from={rightOf(N.scan)} to={leftOf(N.tiles)} on={on('tiles')} />
        {AGENTS.map((a, i) => (
          <Edge key={`in${i}`} from={rightOf(N.tiles)} to={leftOf(a)} on={on('agents')} />
        ))}
        {AGENTS.map((a, i) => (
          <Edge key={`out${i}`} from={rightOf(a)} to={leftOf(N.merge)} on={on('merge')} />
        ))}
        <Edge from={topOf(N.merge)} to={leftOf(N.overlay)} on={on('overlay')} />
        <Edge from={bottomOf(N.overlay)} to={topOf(N.review)} on={on('review')} />
        <Edge from={leftOf(N.review)} to={bottomOf(N.merge)} on={on('merge2')} />
        <Edge from={rightOf(N.merge)} to={leftOf(N.app)} on={on('app')} />

        {/* the travelling token, drawn before the nodes so it passes behind them */}
        <circle cx={pos.x} cy={pos.y} r={7} style={{ fill: 'var(--color-accent)' }} />

        <g onMouseEnter={() => jump('scan')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.scan} title="scan" sub="13 MB jpeg" color={INK} state={stateOf('scan', step)} />
        </g>
        <g onMouseEnter={() => jump('tiles')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.tiles} title="54 tiles" sub="px grid" color={INK} state={stateOf('tiles', step)} />
        </g>
        {AGENTS.map((a, i) => (
          <g key={i} onMouseEnter={() => jump('agents')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
            <SvgNode n={a} title={`agent ${i + 1}`} color={AGENT} state={stateOf('agents', step)} titleSize={11} />
          </g>
        ))}
        <g onMouseEnter={() => jump('merge')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.merge} title="merge" sub="dedupe" color={INK} state={stateOf('merge', step)} />
        </g>
        <g onMouseEnter={() => jump('overlay')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.overlay} title="overlay" sub="boxes drawn" color={INK} state={stateOf('overlay', step)} />
        </g>
        <g onMouseEnter={() => jump('review')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.review} title="review" sub="12 subagents" color={AGENT} state={stateOf('review', step)} />
        </g>
        <g onMouseEnter={() => jump('app')} onMouseLeave={resume} style={{ cursor: 'pointer' }}>
          <SvgNode n={N.app} title="labels.json" sub="387 labels" color={INK} state={stateOf('app', step)} />
        </g>
      </svg>

      <div className="mt-3 flex flex-wrap gap-1" role="tablist" aria-label="Pipeline steps">
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={i === step}
            onClick={() => {
              setPaused(true);
              setStep(i);
            }}
            className="border px-2 py-0.5 font-mono text-[0.66rem] transition-colors"
            style={{
              borderColor: i === step ? 'var(--color-accent)' : 'var(--color-line)',
              color: i === step ? 'var(--color-accent)' : 'var(--color-muted)',
            }}
          >
            {i + 1}. {s.label}
          </button>
        ))}
        {paused && !reduced && (
          <button
            type="button"
            onClick={resume}
            className="ml-auto border px-2 py-0.5 font-mono text-[0.66rem]"
            style={{ borderColor: 'var(--color-line)', color: 'var(--color-muted)' }}
          >
            resume
          </button>
        )}
      </div>
      <p className="mt-3 min-h-[3.5rem] text-sm leading-snug text-ink" aria-live="polite">
        {cur.caption}
      </p>
    </Widget>
  );
}
