import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  type Box,
  Edge,
  leftOf,
  rightOf,
  SvgNode,
  useHoverPreview,
  useReducedMotion,
  Widget,
} from './shared';

// Each message is parsed; valid ones go to the sinks, unparseable "poison" ones
// are routed to the dead-letter topic instead of being dropped. Watch one message
// at a time make the trip; the counters show nothing is lost.

const VB = { w: 420, h: 210 };
const NODE: Record<string, Box> = {
  producer: { cx: 48, cy: 105, w: 74, h: 44 },
  proc: { cx: 192, cy: 105, w: 116, h: 52 },
  sinks: { cx: 354, cy: 50, w: 108, h: 46 },
  dlq: { cx: 354, cy: 160, w: 108, h: 46 },
};
const VALID = 'var(--color-dp-lake)';
const POISON = 'var(--color-dp-log)';

type Rate = 'low' | 'high';
const PERIOD: Record<Rate, number> = { low: 6, high: 3 }; // every Nth message is poison

export default function DlqSplit() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Rate>('low');
  const [rate, bindRate] = useHoverPreview(sel);
  const [phase, setPhase] = useState(0); // 0 born · 1 at processor · 2 routed
  const [count, setCount] = useState(0);
  const [stats, setStats] = useState({ ok: 0, dlq: 0 });

  const isPoison = count % PERIOD[rate] === PERIOD[rate] - 1;

  useEffect(() => {
    if (reduced) {
      setPhase(2);
      setStats({ ok: 19, dlq: 1 });
      return;
    }
    const id = setInterval(() => {
      setPhase((p) => {
        if (p === 2) {
          setCount((c) => c + 1);
          return 0;
        }
        return p + 1;
      });
    }, 1100);
    return () => clearInterval(id);
  }, [reduced]);

  // Tally once when the message reaches its destination (phase 2).
  useEffect(() => {
    if (reduced || phase !== 2) return;
    setStats((s) => (isPoison ? { ...s, dlq: s.dlq + 1 } : { ...s, ok: s.ok + 1 }));
  }, [phase, reduced]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Widget title="Poison messages are dead-lettered, not dropped" kicker="parse · split · route">
      <div className="mb-3 flex flex-wrap gap-2">
        {(['low', 'high'] as Rate[]).map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => setSel(r)}
            aria-pressed={rate === r}
            {...bindRate(r)}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: rate === r ? 'var(--color-accent)' : 'var(--color-line)',
              color: rate === r ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            poison rate {r}
          </button>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 250 }}
        role="img"
        aria-label="Messages flow from the producer to the processor, where valid ones go to the sinks and poison ones go to the dead-letter topic"
      >
        <ArrowDefs />
        <Edge from={rightOf(NODE.producer)} to={leftOf(NODE.proc)} on={phase === 1} />
        <Edge
          from={rightOf(NODE.proc)}
          to={leftOf(NODE.sinks)}
          on={phase === 2 && !isPoison}
          dim={phase === 2 && isPoison}
        />
        <Edge
          from={rightOf(NODE.proc)}
          to={leftOf(NODE.dlq)}
          on={phase === 2 && isPoison}
          dim={phase === 2 && !isPoison}
        />

        <SvgNode
          n={NODE.producer}
          title="Producer"
          sub="messages"
          color="var(--color-dp-producer)"
          state={phase === 0 ? 'active' : 'wait'}
        />
        <SvgNode
          n={NODE.proc}
          title="Stream processor"
          sub="parse + split"
          color="var(--color-dp-stream)"
          state={phase === 1 ? 'active' : 'wait'}
        />
        <SvgNode
          n={NODE.sinks}
          title="Sinks"
          sub="valid events"
          color={VALID}
          state={phase === 2 && !isPoison ? 'active' : 'wait'}
        />
        <SvgNode
          n={NODE.dlq}
          title="events.dlq"
          sub="dead letter"
          color={POISON}
          state={phase === 2 && isPoison ? 'active' : 'wait'}
        />
      </svg>

      <div className="mt-2 flex gap-2">
        <span className="flex-1 border border-line px-2 py-1.5 font-mono text-[0.7rem]">
          <span className="text-muted">to sinks </span>
          <strong style={{ color: VALID }}>{stats.ok}</strong>
        </span>
        <span className="flex-1 border border-line px-2 py-1.5 font-mono text-[0.7rem]">
          <span className="text-muted">dead-lettered </span>
          <strong style={{ color: POISON }}>{stats.dlq}</strong>
        </span>
      </div>
      <p className="mt-2 text-sm text-ink-soft">
        Every message is accounted for: valid events reach the sinks, and a poison message lands on
        the dead-letter topic where it can be inspected and replayed, instead of vanishing.
      </p>
    </Widget>
  );
}
