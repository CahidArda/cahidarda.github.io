import { useEffect, useState } from 'react';
import {
  ArrowDefs,
  type Box,
  COMPONENT_BY_ID,
  Edge,
  leftOf,
  type NState,
  rightOf,
  SvgNode,
  useGlide,
  useReducedMotion,
  Widget,
} from './shared';

// One SVG coordinate system shared by every node and connector, so nothing drifts.
const VB = { w: 460, h: 262 };
const NODE: Record<string, Box> = {
  producer: { cx: 48, cy: 131, w: 78, h: 46 },
  log: { cx: 168, cy: 131, w: 92, h: 50 },
  stream: { cx: 288, cy: 131, w: 90, h: 50 },
};

type StoreId = 'row' | 'col' | 'lake';
const STORE_META: Record<StoreId, { title: string; sub: string; desc: string }> = {
  row: { title: 'Postgres', sub: 'row · OLTP', desc: 'point lookups by key: current order status' },
  col: { title: 'ClickHouse', sub: 'column · OLAP', desc: 'scan-heavy analytics over few columns' },
  lake: { title: 'Iceberg', sub: 'lakehouse', desc: 'cheap durable history, ACID + time travel' },
};

const PHASES = 4;

export default function DataFlow({ stores = ['row', 'col', 'lake'] }: { stores?: StoreId[] }) {
  const reduced = useReducedMotion();
  const [phase, setPhase] = useState(reduced ? 3 : 0);
  // Hovering a node jumps to the step where it is active and pauses the auto-play;
  // moving off resumes from that step.
  const [paused, setPaused] = useState(false);
  const hoverNode = (p: number) => {
    setPaused(true);
    setPhase(p);
  };
  const leaveNode = () => setPaused(false);

  useEffect(() => {
    if (reduced) {
      setPhase(3);
      return;
    }
    if (paused) return;
    const id = setInterval(() => setPhase((p) => (p + 1) % PHASES), 3200);
    return () => clearInterval(id);
  }, [reduced, paused]);

  const producerState: NState = phase === 0 ? 'active' : 'wait';
  const logState: NState = phase === 1 ? 'active' : phase >= 2 ? 'wait' : 'off';
  const streamState: NState = phase === 2 ? 'active' : phase === 3 ? 'wait' : 'off';
  const storeState: NState = phase === 3 ? 'active' : 'off';

  const tokenNode = phase === 0 ? NODE.producer : phase === 1 ? NODE.log : NODE.stream;
  const token = useGlide({ x: tokenNode.cx, y: tokenNode.cy }, reduced);

  // Store boxes are laid out vertically, centred on the stream's row, so 1, 2, or
  // 3 stores all stay balanced.
  const SPACING = 83;
  const storeBox = (i: number): Box => ({
    cx: 414,
    cy: 131 + (i - (stores.length - 1) / 2) * SPACING,
    w: 86,
    h: 44,
  });

  const multi = stores.length > 1;
  const caption = [
    'A producer builds one order event for a tenant.',
    'The producer opens a TCP connection to Redpanda and pushes the event (the Kafka "produce" protocol). The broker appends it to the log.',
    'Spark is a consumer: it polls the log over the same Kafka protocol and pulls new records. The broker never pushes; the reader asks.',
    multi
      ? 'Spark connects out to each store and writes the record: Postgres over its TCP wire protocol, ClickHouse over HTTP, Iceberg to object storage through the REST catalog.'
      : `Spark opens a connection to ${STORE_META[stores[0]].title} and writes the record over its TCP wire protocol.`,
  ];

  return (
    <Widget title="One event, end to end" kicker={`produce · log · process · ${multi ? 'tee' : 'store'}`}>
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 320 }}
        role="img"
        aria-label="An event flows from a producer to the Redpanda log, into Spark, and on to the stores"
      >
        <ArrowDefs />

        <Edge from={rightOf(NODE.producer)} to={leftOf(NODE.log)} on={phase === 1} />
        <Edge from={rightOf(NODE.log)} to={leftOf(NODE.stream)} on={phase === 2} />
        {stores.map((id, i) => (
          <Edge key={id} from={rightOf(NODE.stream)} to={leftOf(storeBox(i))} on={phase === 3} dim={phase < 3} />
        ))}

        {!reduced && phase < 3 && (
          <circle cx={token.x} cy={token.y} r={6} style={{ fill: 'var(--color-accent)' }} />
        )}

        <g onMouseEnter={() => hoverNode(0)} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode n={NODE.producer} title="Producers" sub="events" color={COMPONENT_BY_ID.producer.color} state={producerState} />
        </g>
        <g onMouseEnter={() => hoverNode(1)} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode n={NODE.log} title="Redpanda" sub="event log" color={COMPONENT_BY_ID.log.color} state={logState} />
        </g>
        <g onMouseEnter={() => hoverNode(2)} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
          <SvgNode n={NODE.stream} title="Spark" sub="streaming" color={COMPONENT_BY_ID.stream.color} state={streamState} />
        </g>
        {stores.map((id, i) => (
          <g key={id} onMouseEnter={() => hoverNode(3)} onMouseLeave={leaveNode} style={{ cursor: 'pointer' }}>
            <SvgNode n={storeBox(i)} title={STORE_META[id].title} sub={STORE_META[id].sub} color={COMPONENT_BY_ID[id].color} state={storeState} />
          </g>
        ))}
      </svg>

      <p className="mt-2 min-h-[3.6rem] text-sm text-ink-soft">
        <span className="font-mono text-xs text-muted">{phase + 1}/4 · </span>
        {caption[phase]}
      </p>

      <div className={`mt-1 grid grid-cols-1 gap-1.5 ${multi ? 'sm:grid-cols-3' : ''}`}>
        {stores.map((id) => (
          <div key={id} className="flex items-start gap-2 border border-line px-2 py-1.5">
            <span className="mt-0.5 inline-block h-2.5 w-2.5 shrink-0" style={{ background: COMPONENT_BY_ID[id].color }} />
            <span className="font-mono text-[0.66rem] leading-tight text-ink-soft">
              <strong className="text-ink">{STORE_META[id].title}</strong>
              <br />
              {STORE_META[id].desc}
            </span>
          </div>
        ))}
      </div>
    </Widget>
  );
}
