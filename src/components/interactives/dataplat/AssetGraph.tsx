import { useEffect, useState } from 'react';
import { type Box, leftOf, rightOf, SvgNode, useReducedMotion, Widget } from './shared';

// The nightly rollup as a Dagster asset graph with a BLOCKING data-quality gate.
// Run it clean and the gate passes, promoting the rollup. Inject a broken contract
// and the gate fails, stopping the run before promotion: the bad data never lands.

const VB = { w: 470, h: 150 };
const NODE: Record<string, Box> = {
  snapshot: { cx: 66, cy: 70, w: 112, h: 46 },
  rollup: { cx: 206, cy: 70, w: 120, h: 46 },
  promoted: { cx: 414, cy: 70, w: 100, h: 46 },
};
const GATE = { cx: 320, cy: 70, hw: 30, hh: 32 };
const PASS = 'var(--color-dp-lake)';
const FAIL = 'var(--color-dp-log)';
const EVAL = 'var(--color-dp-stream)';

type Mode = 'clean' | 'drift';

export default function AssetGraph() {
  const reduced = useReducedMotion();
  const [mode, setMode] = useState<Mode>('clean');
  const [phase, setPhase] = useState(0); // 0 snapshot · 1 rollup · 2 gate · 3 result

  useEffect(() => {
    if (reduced) {
      setPhase(3);
      return;
    }
    setPhase(0);
    const id = setInterval(() => setPhase((p) => (p >= 3 ? 3 : p + 1)), 900);
    return () => clearInterval(id);
  }, [mode, reduced]);

  const passed = mode === 'clean';
  const gateColor = phase < 2 ? 'var(--color-line-strong)' : phase === 2 ? EVAL : passed ? PASS : FAIL;
  const promotedState = phase === 3 && passed ? 'active' : 'off';

  const gateL = { x: GATE.cx - GATE.hw, y: GATE.cy };
  const gateR = { x: GATE.cx + GATE.hw, y: GATE.cy };
  const diamond = `${GATE.cx},${GATE.cy - GATE.hh} ${gateR.x},${GATE.cy} ${GATE.cx},${GATE.cy + GATE.hh} ${gateL.x},${GATE.cy}`;

  const edge = (a: { x: number; y: number }, b: { x: number; y: number }, on: boolean, color: string) => (
    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
      style={{ stroke: on ? color : 'var(--color-line-strong)', strokeWidth: on ? 2.5 : 1, strokeDasharray: on ? undefined : '3 3', transition: 'stroke 250ms' }} />
  );

  const caption =
    phase < 2
      ? 'Reading events and computing the per-tenant daily rollup.'
      : phase === 2
        ? 'The blocking quality check evaluates the data contract.'
        : passed
          ? 'Check passed: the validated rollup is promoted to the served table.'
          : 'Check FAILED (null tenant / negative revenue). The run stops here. Nothing is promoted, so the warehouse is never poisoned.';

  return (
    <Widget title="Data-quality gate" kicker="snapshot → rollup → [gate] → promote">
      <div className="mb-3 flex flex-wrap gap-2">
        {(['clean', 'drift'] as Mode[]).map((m) => (
          <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{ borderColor: mode === m ? 'var(--color-accent)' : 'var(--color-line)', color: mode === m ? 'var(--color-accent)' : 'var(--color-ink-soft)' }}>
            {m === 'clean' ? 'clean run' : 'inject drift'}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="w-full" style={{ maxHeight: 170 }} role="img"
        aria-label="A Dagster asset graph where a blocking quality check between the rollup and the promotion either passes or fails">
        {edge(rightOf(NODE.snapshot), leftOf(NODE.rollup), phase >= 1, PASS)}
        {edge(rightOf(NODE.rollup), gateL, phase >= 2, phase === 2 ? EVAL : gateColor)}
        {edge(gateR, leftOf(NODE.promoted), phase === 3 && passed, PASS)}

        <SvgNode n={NODE.snapshot} title="snapshot" sub="events" color={EVAL} state={phase === 0 ? 'active' : 'wait'} titleSize={11} />
        <SvgNode n={NODE.rollup} title="daily rollup" sub="per tenant/day" color={EVAL} state={phase === 1 ? 'active' : phase >= 1 ? 'wait' : 'off'} titleSize={11} />

        {/* the gate: a diamond that turns amber (evaluating), green (pass) or red (fail) */}
        <polygon points={diamond} style={{ fill: phase === 3 ? gateColor : 'var(--color-paper)', stroke: gateColor, strokeWidth: 2, transition: 'fill 250ms, stroke 250ms' }} />
        <text x={GATE.cx} y={GATE.cy + 1} textAnchor="middle" dominantBaseline="middle"
          style={{ fill: phase === 3 ? 'var(--color-paper)' : 'var(--color-ink-soft)', fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700 }}>
          {phase < 3 ? '?' : passed ? '✓' : '✕'}
        </text>
        <text x={GATE.cx} y={GATE.cy + GATE.hh + 11} textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 8 }}>gate</text>

        <SvgNode n={NODE.promoted} title="promoted" sub="served table" color={PASS} state={promotedState} titleSize={11} />
      </svg>

      <p className="mt-2 min-h-[2.4rem] text-sm text-ink-soft">{caption}</p>
    </Widget>
  );
}
