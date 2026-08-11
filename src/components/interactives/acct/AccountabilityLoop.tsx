/**
 * The accountability bypass: work flows You -> Bot -> your tools, and when a run
 * goes wrong the loss routes around the bot, and around its vendor, back to you.
 * Auto-plays through four steps; hovering a box jumps to its step and pauses.
 */
import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  BOT,
  Edge,
  LIAB,
  SvgNode,
  TOOL,
  Widget,
  YOU,
  bottomOf,
  leftOf,
  rightOf,
  useInView,
  useReducedMotion,
  type Box,
  type NState,
} from './shared';

const YOU_N: Box = { cx: 70, cy: 130, w: 96, h: 44 };
const BOT_N: Box = { cx: 230, cy: 58, w: 104, h: 44 };
const TOOLS_N: Box = { cx: 390, cy: 130, w: 116, h: 44 };
const VENDOR_N: Box = { cx: 230, cy: 204, w: 132, h: 44 };

/* Liability arc: tools' bottom edge, under the vendor box, into your bottom edge. */
const LIAB_ARC = `M ${bottomOf(TOOLS_N).x} ${bottomOf(TOOLS_N).y}
  C 400 250, 300 258, 230 258
  C 160 258, 60 250, ${bottomOf(YOU_N).x} ${bottomOf(YOU_N).y + 4}`;

const CAPTIONS = [
  'You show the bot the workflow once. It watches, remembers, and saves it as a routine.',
  'It signs into your tools with your accounts. In every audit log, the actor is you.',
  'One run goes wrong: say a refund script fires in live mode. The money is real.',
  'The loss routes around the bot, and around its vendor, back to you. Recourse is capped at what you paid, or $100.',
];

const STEP_MS = 3400;

export default function AccountabilityLoop() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [phase, setPhase] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (reduced) {
      setPhase(3);
      return;
    }
    if (paused || !inView) return;
    const id = setInterval(() => setPhase((p) => (p + 1) % CAPTIONS.length), STEP_MS);
    return () => clearInterval(id);
  }, [reduced, paused, inView]);

  const hoverNode = (p: number) => {
    setPaused(true);
    setPhase(p);
  };

  const youState: NState = phase === 0 || phase === 3 ? 'active' : 'wait';
  const botState: NState = phase === 1 ? 'active' : phase === 3 ? 'off' : 'wait';
  const toolState: NState = phase === 0 ? 'off' : phase === 2 ? 'active' : 'wait';

  return (
    <Widget
      title="The accountability bypass"
      kicker="hover a box to inspect a step"
      rootRef={rootRef}
    >
      <svg
        viewBox="0 0 460 268"
        role="img"
        aria-label="Diagram: work flows from you to the bot to your tools; when a run fails, the loss routes around the bot and its vendor, back to you."
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        <ArrowDefs />

        {/* work edges */}
        <Edge from={rightOf(YOU_N)} to={leftOf(BOT_N)} on={phase === 0} dim={phase > 0} />
        <Edge from={rightOf(BOT_N)} to={leftOf(TOOLS_N)} on={phase === 1} dim={phase !== 1} />

        {/* liability arc, under the vendor box, only on the last step */}
        <path
          d={LIAB_ARC}
          fill="none"
          markerEnd={phase === 3 ? 'url(#acct-arrow-liab)' : undefined}
          style={{
            stroke: LIAB,
            strokeWidth: 2.2,
            opacity: phase === 3 ? 1 : 0,
            transition: 'opacity 300ms',
          }}
        />
        {phase === 3 && (
          <text
            x={230}
            y={247}
            textAnchor="middle"
            style={{ fill: LIAB, fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 600 }}
          >
            consequences, costs, liabilities
          </text>
        )}

        <g
          onMouseEnter={() => hoverNode(0)}
          onMouseLeave={() => setPaused(false)}
          style={{ cursor: 'pointer' }}
        >
          <SvgNode
            n={YOU_N}
            title="You"
            sub="account holder"
            color={phase === 3 ? LIAB : YOU}
            state={youState}
          />
        </g>
        <g
          onMouseEnter={() => hoverNode(1)}
          onMouseLeave={() => setPaused(false)}
          style={{ cursor: 'pointer' }}
        >
          <SvgNode n={BOT_N} title="Bot" sub="your teammate" color={BOT} state={botState} />
        </g>
        <g
          onMouseEnter={() => hoverNode(2)}
          onMouseLeave={() => setPaused(false)}
          style={{ cursor: 'pointer' }}
        >
          <SvgNode
            n={TOOLS_N}
            title="Your tools"
            sub="logged in as you"
            color={phase >= 2 ? LIAB : TOOL}
            state={toolState}
          />
        </g>
        <g
          onMouseEnter={() => hoverNode(3)}
          onMouseLeave={() => setPaused(false)}
          style={{ cursor: 'pointer' }}
        >
          <SvgNode
            n={VENDOR_N}
            title="Vendor"
            sub="disclaims all liability"
            color={BOT}
            state="off"
          />
        </g>
      </svg>

      <div className="mt-3 flex items-start gap-3">
        <div className="flex shrink-0 items-center gap-1.5 pt-1">
          {CAPTIONS.map((_, i) => (
            <button
              key={i}
              aria-label={`step ${i + 1}`}
              onClick={() => {
                setPhase(i);
                setPaused(true);
              }}
              style={{
                width: 8,
                height: 8,
                background: i === phase ? 'var(--color-ink)' : 'var(--color-line-strong)',
                border: 'none',
                padding: 0,
                cursor: 'pointer',
              }}
            />
          ))}
        </div>
        <p className="m-0 min-h-[3.4em] font-mono text-[0.72rem] leading-snug text-ink-soft">
          {CAPTIONS[phase]}
        </p>
      </div>
    </Widget>
  );
}
