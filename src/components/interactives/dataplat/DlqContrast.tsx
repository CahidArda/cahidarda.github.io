import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  type Box,
  bottomOf,
  Edge,
  leftOf,
  rightOf,
  SvgNode,
  topOf,
  useHoverPreview,
  useReducedMotion,
  Widget,
} from './shared';

// Both a log and a queue can dead-letter a poison message, but the OWNER of the
// decision differs. Two tabs, one flow each. The "to DLQ" arrow hangs off the box
// that decides: the CONSUMER in the log (it routes poison to a dead-letter topic),
// the QUEUE in SQS (the broker redrives after maxReceiveCount, no consumer code).

type Tab = 'log' | 'queue';
const VB = { w: 460, h: 196 };
const LOG = 'var(--color-dp-log)';
const QUEUE = 'var(--color-accent)';
const MAXRC = 3;

const SRC: Box = { cx: 110, cy: 56, w: 150, h: 46 };
const CONS: Box = { cx: 330, cy: 56, w: 150, h: 46 };
const DLQ_LOG: Box = { cx: 330, cy: 150, w: 150, h: 46 }; // under the consumer
const DLQ_Q: Box = { cx: 110, cy: 150, w: 150, h: 46 }; // under the queue

const TABS: { key: Tab; label: string }[] = [
  { key: 'log', label: 'Log (Redpanda)' },
  { key: 'queue', label: 'Queue (SQS)' },
];

export default function DlqContrast() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Tab>('log');
  const [tab, bindTab] = useHoverPreview(sel);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (reduced) return;
    const id = setInterval(() => setTick((t) => t + 1), 1100);
    return () => clearInterval(id);
  }, [reduced]);

  const isLog = tab === 'log';
  // log: topic -> consumer -> dlq (3 steps); queue: queue, receive 1/2/3, redrive (5)
  const lp = reduced ? 2 : tick % 3;
  const qp = reduced ? 4 : tick % 5;
  const onCons = qp === 1 || qp === 2 || qp === 3;
  const redrive = qp === 4;
  const rc = qp === 0 ? 0 : Math.min(MAXRC, qp);

  return (
    <Widget
      title="Two ways to dead-letter"
      kicker={isLog ? 'the consumer decides' : 'the broker decides'}
    >
      <div className="mb-3 flex flex-wrap gap-2">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setSel(t.key)}
            aria-pressed={tab === t.key}
            {...bindTab(t.key)}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: tab === t.key ? 'var(--color-accent)' : 'var(--color-line)',
              color: tab === t.key ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 240 }}
        role="img"
        aria-label={
          isLog
            ? 'Log: the consumer inspects a message and routes poison to a dead-letter topic.'
            : 'Queue: the broker redrives a message to a dead-letter queue after three failed receives.'
        }
      >
        <ArrowDefs />

        {isLog ? (
          <>
            <Edge from={rightOf(SRC)} to={leftOf(CONS)} on={lp === 1} />
            <Edge from={bottomOf(CONS)} to={topOf(DLQ_LOG)} on={lp === 2} />
            <text
              x={CONS.cx + 12}
              y={(CONS.cy + DLQ_LOG.cy) / 2}
              dominantBaseline="middle"
              style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
            >
              consumer routes
            </text>
            <SvgNode
              n={SRC}
              title="events"
              sub="topic"
              color={LOG}
              state={lp === 0 ? 'active' : 'wait'}
            />
            <SvgNode
              n={CONS}
              title="consumer"
              sub="inspects + routes"
              color={LOG}
              state={lp === 1 ? 'active' : 'wait'}
            />
            <SvgNode
              n={DLQ_LOG}
              title="events.dlq"
              sub="dead-letter topic"
              color={LOG}
              state={lp === 2 ? 'active' : 'wait'}
            />
          </>
        ) : (
          <>
            <Edge from={rightOf(SRC)} to={leftOf(CONS)} on={onCons} />
            <Edge from={bottomOf(SRC)} to={topOf(DLQ_Q)} on={redrive} />
            <text
              x={SRC.cx + 12}
              y={(SRC.cy + DLQ_Q.cy) / 2}
              dominantBaseline="middle"
              style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
            >
              broker redrives
            </text>
            <SvgNode
              n={SRC}
              title="queue"
              sub="main"
              color={QUEUE}
              state={qp === 0 ? 'active' : 'wait'}
            />
            <SvgNode
              n={CONS}
              title="consumer"
              sub={`receives ${rc}/${MAXRC}`}
              color={QUEUE}
              state={onCons ? 'active' : 'wait'}
            />
            <SvgNode
              n={DLQ_Q}
              title="DLQ"
              sub="dead-letter queue"
              color={QUEUE}
              state={redrive ? 'active' : 'wait'}
            />
          </>
        )}
      </svg>

      <p className="mt-2 min-h-[3.4rem] text-sm text-ink-soft">
        {isLog ? (
          <>
            The log's <strong>consumer</strong> inspects each message and writes poison to a
            dead-letter topic itself, leaving the original on the log to replay.
          </>
        ) : (
          <>
            The queue's <strong>broker</strong> redrives a message to its dead-letter queue on its
            own once it has been received and failed <code>maxReceiveCount</code> times, with no
            routing code in the consumer.
          </>
        )}
      </p>
    </Widget>
  );
}
