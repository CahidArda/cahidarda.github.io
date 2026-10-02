import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  type Box,
  Edge,
  FLOW,
  NAV,
  type NState,
  PRICE,
  SvgNode,
  useHoverPreview,
  useInView,
  useReducedMotion,
  WARN,
  Widget,
} from './shared';

// The mechanism, as one closed loop. Four nodes on a diamond so the circuit is visibly a
// circuit and not a pipeline: money in, buy the thin stock, price rises, NAV rises, the
// published return rises, which pulls the next tranche of money in.
//
// The same four nodes run in reverse under the "unwind" tab, with the arrowheads swapped
// (skill §4.3) and recoloured to the warn hue, because that is the entire argument: the
// loop has no preferred direction, and what inflates is what collapses.

const VB = { w: 440, h: 300 };

const NODE: Record<string, Box> = {
  flow: { cx: 220, cy: 36, w: 150, h: 46 },
  stock: { cx: 372, cy: 150, w: 124, h: 50 },
  nav: { cx: 220, cy: 264, w: 150, h: 46 },
  ret: { cx: 68, cy: 150, w: 124, h: 50 },
};

type Mode = 'inflate' | 'unwind';

interface Step {
  /** which node is lit */
  at: keyof typeof NODE;
  caption: string;
}

const INFLATE: Step[] = [
  {
    at: 'flow',
    caption:
      'A saver sees a fund near the top of the TEFAS return table and buys in. The fund now holds cash it must deploy.',
  },
  {
    at: 'stock',
    caption:
      'It buys more of what it already owns: a stock with very little free float. In a thin order book, a large buy does not find a seller at the last price. It walks the price up.',
  },
  {
    at: 'nav',
    caption:
      "The fund marks its whole existing position at the new, higher last-traded price. Its NAV jumps, and the jump is far larger than the money that came in, because the new price revalues every share it already held.",
  },
  {
    at: 'ret',
    caption:
      'That NAV becomes a published return. It is arithmetically correct and it is visible to everyone on the platform. It ranks the fund higher, which brings the next saver. Round again.',
  },
];

const UNWIND: Step[] = [
  {
    at: 'ret',
    caption:
      'Something stops the inflows. A rule change, a bad headline, a competitor paying more. The return stops being extraordinary.',
  },
  {
    at: 'nav',
    caption:
      'Redemptions arrive. The fund must pay in cash, so it has to sell. Every unit it redeems at the marked NAV is a claim on a price it has not yet tested.',
  },
  {
    at: 'stock',
    caption:
      'Selling into the same thin book walks the price back down, and faster than it went up, because the buyers who lifted it were the fund itself. The marks were never exit prices.',
  },
  {
    at: 'flow',
    caption:
      'The falling NAV is published too. It triggers more redemptions, which force more selling. The loop is running backwards, and whoever is still holding units owns the residue.',
  },
];

export default function ReflexiveLoop() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Mode>('inflate');
  const [mode, bindMode] = useHoverPreview(sel);
  const [step, setStep] = useState(0);
  const [paused, setPaused] = useState(false);
  const [viewRef, inView] = useInView<HTMLDivElement>();

  const steps = mode === 'inflate' ? INFLATE : UNWIND;
  const warn = mode === 'unwind';

  // Restart the walk whenever the tab changes, so a hover-preview plays that scenario
  // from its first step rather than dropping the reader mid-cycle.
  useEffect(() => setStep(0), [mode]);

  useEffect(() => {
    if (reduced || paused || !inView) return;
    const id = setInterval(() => setStep((s) => (s + 1) % 4), 3000);
    return () => clearInterval(id);
  }, [reduced, paused, inView, mode]);

  const hoverNode = (i: number) => {
    setPaused(true);
    setStep(i);
  };
  const leaveNode = () => setPaused(false);

  const cur = steps[step];
  // A node is active ONLY while the token is at it (skill §4.4); everything else on the
  // loop is 'wait' so the reader can see the circuit without it looking lit end to end.
  const stateOf = (k: keyof typeof NODE): NState => (cur.at === k ? 'active' : 'wait');

  // The edge that is live is the one LEAVING the current node, in the current direction.
  const order: (keyof typeof NODE)[] =
    mode === 'inflate' ? ['flow', 'stock', 'nav', 'ret'] : ['ret', 'nav', 'stock', 'flow'];
  const nextOf = (k: keyof typeof NODE) => order[(order.indexOf(k) + 1) % 4];
  const liveFrom = cur.at;
  const liveTo = nextOf(cur.at);

  // Each leg is drawn from its source edge to its target edge. Swapping the endpoints on
  // the unwind tab is what makes the arrowheads reverse (skill §4.3).
  const legs: [keyof typeof NODE, keyof typeof NODE][] =
    mode === 'inflate'
      ? [
          ['flow', 'stock'],
          ['stock', 'nav'],
          ['nav', 'ret'],
          ['ret', 'flow'],
        ]
      : [
          ['ret', 'nav'],
          ['nav', 'stock'],
          ['stock', 'flow'],
          ['flow', 'ret'],
        ];

  // Anchor to the corner of each box facing the other box, so no line runs under a node.
  const anchor = (from: keyof typeof NODE, to: keyof typeof NODE) => {
    const a = NODE[from];
    const b = NODE[to];
    const dx = b.cx - a.cx;
    const dy = b.cy - a.cy;
    if (Math.abs(dx) > Math.abs(dy)) {
      return { x: a.cx + (dx > 0 ? a.w / 2 : -a.w / 2), y: a.cy + Math.sign(dy) * (a.h / 4) };
    }
    return { x: a.cx + Math.sign(dx) * (a.w / 4), y: a.cy + (dy > 0 ? a.h / 2 : -a.h / 2) };
  };

  return (
    <Widget
      title="The loop"
      kicker={mode === 'inflate' ? 'running forward' : 'running backward'}
      rootRef={viewRef}
    >
      <div className="mb-3 flex flex-wrap gap-2">
        {(['inflate', 'unwind'] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setSel(m)}
            aria-pressed={mode === m}
            {...bindMode(m)}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor:
                mode === m ? (m === 'unwind' ? WARN : 'var(--color-accent)') : 'var(--color-line)',
              color:
                mode === m ? (m === 'unwind' ? WARN : 'var(--color-accent)') : 'var(--color-ink-soft)',
            }}
          >
            {m === 'inflate' ? 'inflation' : 'unwind'}
          </button>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 340 }}
        role="img"
        aria-label="A four-node loop: inflows buy a thin-float stock, the price rises, the fund's net asset value rises, the published return rises, and that attracts the next inflow"
      >
        <ArrowDefs />

        {legs.map(([a, b]) => (
          <Edge
            key={`${a}-${b}`}
            from={anchor(a, b)}
            to={anchor(b, a)}
            on={liveFrom === a && liveTo === b}
            warn={warn}
            dim={!(liveFrom === a && liveTo === b)}
          />
        ))}

        <g onMouseEnter={() => hoverNode(order.indexOf('flow'))} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode
            n={NODE.flow}
            title={warn ? 'Redemptions' : 'Investor inflows'}
            sub={warn ? 'money leaves' : 'money arrives'}
            color={warn ? WARN : FLOW}
            state={stateOf('flow')}
          />
        </g>
        <g onMouseEnter={() => hoverNode(order.indexOf('stock'))} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode
            n={NODE.stock}
            title="Thin-float stock"
            sub={warn ? 'price falls' : 'price rises'}
            color={PRICE}
            state={stateOf('stock')}
          />
        </g>
        <g onMouseEnter={() => hoverNode(order.indexOf('nav'))} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode
            n={NODE.nav}
            title="Fund NAV"
            sub="marked at last price"
            color={NAV}
            state={stateOf('nav')}
          />
        </g>
        <g onMouseEnter={() => hoverNode(order.indexOf('ret'))} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode
            n={NODE.ret}
            title="Published return"
            sub="on the platform"
            color={NAV}
            state={stateOf('ret')}
          />
        </g>
      </svg>

      <p className="mt-2 min-h-[5rem] text-sm text-ink-soft sm:min-h-[4rem]">
        <span className="font-mono text-xs text-muted">
          {step + 1}/4 ·{' '}
        </span>
        {cur.caption}
      </p>

      <p className="mt-2 border-t border-line pt-2 font-mono text-[0.68rem] leading-relaxed text-muted">
        Nothing in the forward loop requires anyone to lie. Every mark is the last traded price,
        which is what a fund is supposed to use. Hover a box to stop on that step.
      </p>
    </Widget>
  );
}
