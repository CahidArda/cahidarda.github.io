import { useEffect, useRef, useState } from 'react';
import { useHoverPreview, Widget, useReducedMotion } from './shared';

// Consumer lag = how far the processor is behind the log. The producer appends at
// a steady rate; the consumer drains at a rate you control. When draining is
// slower than appending, the backlog (lag) climbs; speed it back up and it drains.
// This is the phase-2 health signal, drawn like its Grafana panel.

const N = 48; // samples shown
const MAXLAG = 240; // y-axis cap (records)
const PRODUCE = 12; // events appended per tick
const SEED = 180; // a backlog already built up, so a healthy consumer visibly drains it
const VB = { w: 340, h: 130 };
const PAD = { l: 4, r: 4, t: 10, b: 4 };

type Mode = 'healthy' | 'slow' | 'stalled';
// cap = records the consumer drains per tick. Above PRODUCE it catches up; below, lag grows.
const MODES: { key: Mode; label: string; cap: number }[] = [
  { key: 'healthy', label: 'healthy', cap: 24 }, // drains faster than it fills -> catches up
  { key: 'slow', label: 'slow', cap: 8 }, // drains some, but keeps falling behind
  { key: 'stalled', label: 'stalled', cap: 0 }, // drains nothing
];

// A static poster (SSR + reduced motion): a backlog that has ramped up to SEED.
const POSTER = Array.from({ length: N }, (_, i) => Math.min(SEED, Math.round((i / (N * 0.55)) * SEED)));

function lagColor(lag: number): string {
  if (lag < 60) return 'var(--color-dp-lake)'; // teal: keeping up
  if (lag < 150) return 'var(--color-dp-stream)'; // amber: falling behind
  return 'var(--color-dp-log)'; // red: backed up
}

export default function ConsumerLag() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Mode>('healthy');
  const [mode, bindMode] = useHoverPreview(sel);
  // Seed with the poster so the chart renders server-side; the live interval then
  // appends real samples and the poster scrolls off.
  const [samples, setSamples] = useState<number[]>(POSTER);
  // Lag is a single value clamped to [0, MAXLAG]; each tick it moves by PRODUCE - cap,
  // so it always drains under "healthy" no matter how high it climbed (no runaway backlog).
  const lagRef = useRef(SEED);
  const modeRef = useRef<Mode>(mode);
  modeRef.current = mode;

  useEffect(() => {
    if (reduced) {
      setSamples(POSTER);
      return;
    }
    const id = setInterval(() => {
      const cap = MODES.find((m) => m.key === modeRef.current)!.cap;
      lagRef.current = Math.max(0, Math.min(MAXLAG, lagRef.current + PRODUCE - cap));
      setSamples((s) => [...s, lagRef.current].slice(-N));
    }, 450);
    return () => clearInterval(id);
  }, [reduced]);

  const lag = samples[samples.length - 1] ?? 0;
  const w = VB.w - PAD.l - PAD.r;
  const h = VB.h - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (i / (N - 1)) * w;
  const y = (v: number) => PAD.t + h - (v / MAXLAG) * h;
  const pts = samples.map((v, i) => `${x(i + (N - samples.length))},${y(v)}`).join(' ');
  const area = samples.length > 1 ? `${PAD.l},${y(0)} ${pts} ${x(N - 1)},${y(0)}` : '';

  return (
    <Widget title="Consumer lag" kicker="produce 12/tick · you set the drain rate">
      <div className="mb-3 flex flex-wrap gap-2">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setSel(m.key)}
            aria-pressed={mode === m.key}
            {...bindMode(m.key)}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: mode === m.key ? 'var(--color-accent)' : 'var(--color-line)',
              color: mode === m.key ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            consumer {m.label}
          </button>
        ))}
      </div>

      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[0.7rem] text-muted">lag (records behind the log)</span>
        <span className="font-mono text-lg font-semibold" style={{ color: lagColor(lag) }}>
          {Math.round(lag)}
        </span>
      </div>

      <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="mt-1 w-full" style={{ maxHeight: 150 }} role="img"
        aria-label="A time series of consumer lag rising when the consumer is slow and draining when it is healthy">
        {/* threshold guides */}
        {[60, 150].map((t) => (
          <line key={t} x1={PAD.l} y1={y(t)} x2={VB.w - PAD.r} y2={y(t)}
            style={{ stroke: 'var(--color-line)', strokeWidth: 1, strokeDasharray: '2 3' }} />
        ))}
        {area && <polygon points={area} style={{ fill: lagColor(lag), opacity: 0.14 }} />}
        {samples.length > 1 && (
          <polyline points={pts} fill="none"
            style={{ stroke: lagColor(lag), strokeWidth: 2, transition: 'stroke 300ms' }} />
        )}
      </svg>

      <p className="mt-1 min-h-[2.4rem] text-sm text-ink-soft">
        {lag < 60
          ? 'Draining as fast as it fills: lag stays low, the platform is keeping up.'
          : lag < 150
            ? 'The consumer is behind. Lag is climbing, but recoverable.'
            : 'The consumer is stalled or too slow. Lag is piling up. Speed it back to "healthy" and watch it drain (this is what backpressure and a scaled consumer buy you in phase 4).'}
      </p>
    </Widget>
  );
}
