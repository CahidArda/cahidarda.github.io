import { useEffect, useState } from 'react';
import { Widget, useReducedMotion } from './shared';

// Redpanda keeps appending offsets to the log. The stream processor commits its
// progress to a checkpoint on a PersistentVolumeClaim. When the pod is killed, the
// log keeps growing and the backlog builds, but a fresh pod mounts the SAME
// checkpoint and resumes from the last committed offset: it drains the backlog with
// no gap (every offset processed) and no double-processing (it never re-reads before
// the checkpoint).

const W = 18; // visible offsets (a scrolling window of the log)
const RUN = 8; // ticks running before the kill
const DOWN = 6; // ticks down before a fresh pod starts
const PAD = 12;
const VB = { w: 460, h: 104 };
const STEP = (VB.w - 2 * PAD) / W;
const CELL = STEP - 3;
const CY = 38;
const CH = 30;

const PROCESSED = 'var(--color-dp-lake)'; // teal: committed + processed
const PENDING = 'var(--color-dp-stream)'; // amber: appended, not yet processed
const DEAD = 'var(--color-dp-log)'; // red: processor down

type Status = 'running' | 'down' | 'recovering';
type S = { head: number; proc: number; status: Status; timer: number };

const INIT: S = { head: 14, proc: 13, status: 'running', timer: 0 };
const POSTER: S = { head: 22, proc: 16, status: 'down', timer: 3 };

function step(s: S): S {
  let { head, proc, status, timer } = s;
  head += 1; // the log always appends
  timer += 1;
  if (status === 'running') {
    proc = head - 1; // keep pace, one event in flight
    if (timer >= RUN) return { head, proc, status: 'down', timer: 0 };
  } else if (status === 'down') {
    if (timer >= DOWN) return { head, proc, status: 'recovering', timer: 0 }; // proc frozen
  } else {
    proc = Math.min(head - 1, proc + 3); // drain the backlog fast
    if (proc >= head - 1) return { head, proc, status: 'running', timer: 0 };
  }
  return { head, proc, status, timer };
}

export default function CheckpointRecovery() {
  const reduced = useReducedMotion();
  const [s, setS] = useState<S>(INIT);
  useEffect(() => {
    if (reduced) {
      setS(POSTER);
      return;
    }
    const id = setInterval(() => setS(step), 700);
    return () => clearInterval(id);
  }, [reduced]);

  const { head, proc, status } = s;
  const first = head - W + 1;
  const iProc = proc - first; // index of last processed cell
  const markerX = Math.max(PAD, Math.min(VB.w - PAD, PAD + iProc * STEP + CELL));
  const lag = head - proc;

  const note =
    status === 'down'
      ? {
          c: DEAD,
          t: 'Pod killed. The log keeps appending and the backlog grows, but nothing past the checkpoint is processed.',
        }
      : status === 'recovering'
        ? {
            c: PENDING,
            t: 'A fresh pod mounted the same checkpoint and is draining the backlog from the last committed offset: no gap, no re-processing.',
          }
        : {
            c: PROCESSED,
            t: 'The processor is keeping pace; only the newest offset is still in flight.',
          };

  return (
    <Widget
      title="Killed and recovered from the checkpoint"
      kicker="redpanda appends · processor commits offsets"
    >
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="w-full"
        style={{ maxHeight: 130 }}
        role="img"
        aria-label="The Redpanda log keeps appending offsets; when the stream processor pod is killed the backlog grows, then a fresh pod resumes from the checkpoint and drains it with no gap."
      >
        <text
          x={PAD}
          y={18}
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
        >
          Redpanda log, offsets appended →
        </text>
        <text
          x={VB.w - PAD}
          y={18}
          textAnchor="end"
          style={{ fill: note.c, fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 700 }}
        >
          {status === 'down'
            ? `processor DOWN · backlog ${lag}`
            : status === 'recovering'
              ? 'recovering'
              : 'processor running'}
        </text>

        {Array.from({ length: W }, (_, i) => {
          const o = first + i;
          const done = o <= proc;
          return (
            <rect
              key={o}
              x={PAD + i * STEP}
              y={CY}
              width={CELL}
              height={CH}
              style={{
                fill: done ? PROCESSED : PENDING,
                opacity: done ? 0.85 : 0.5,
                stroke: o === head ? 'var(--color-ink)' : 'transparent',
                strokeWidth: 1.5,
                transition: 'fill 200ms, opacity 200ms',
              }}
            />
          );
        })}

        {/* the checkpoint / processed frontier */}
        <line
          x1={markerX}
          y1={CY - 6}
          x2={markerX}
          y2={CY + CH + 6}
          style={{
            stroke: status === 'running' ? PROCESSED : 'var(--color-accent)',
            strokeWidth: 2,
          }}
        />
        <polygon
          points={`${markerX - 4},${CY + CH + 6} ${markerX + 4},${CY + CH + 6} ${markerX},${CY + CH + 12}`}
          style={{ fill: status === 'running' ? PROCESSED : 'var(--color-accent)' }}
        />
        <text
          x={markerX}
          y={CY + CH + 22}
          textAnchor="middle"
          style={{ fill: 'var(--color-ink-soft)', fontFamily: 'var(--font-mono)', fontSize: 8.5 }}
        >
          {status === 'running' ? 'processed' : 'checkpoint (resume here)'}
        </text>
      </svg>

      <p className="mt-2 min-h-[2.4rem] text-sm" style={{ color: 'var(--color-ink-soft)' }}>
        {note.t}
      </p>
    </Widget>
  );
}
