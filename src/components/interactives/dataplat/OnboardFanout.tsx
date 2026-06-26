import { useEffect, useState } from 'react';
import { type Box, leftOf, rightOf, SvgNode, useReducedMotion, Widget } from './shared';

// One command turns a tenant name into everything the tenant needs. Watch the
// single onboarding command fan out into the provisioned resources, on a
// stopwatch, versus the manual runbook it replaces.

const VB = { w: 444, h: 224 };
const CMD: Box = { cx: 66, cy: 112, w: 104, h: 54 };
const RES = [
  { name: 'namespace', group: 'isolation', color: 'var(--color-dp-producer)' },
  { name: 'resource quota', group: 'isolation', color: 'var(--color-dp-producer)' },
  { name: 'network policy', group: 'isolation', color: 'var(--color-dp-producer)' },
  { name: 'rbac role', group: 'isolation', color: 'var(--color-dp-producer)' },
  { name: 'kafka topic', group: 'ingest', color: 'var(--color-dp-log)' },
  { name: 'warehouse schema', group: 'storage', color: 'var(--color-dp-row)' },
  { name: 'starter dashboard', group: 'observability', color: 'var(--color-accent)' },
];
const BOX = { x: 270, w: 166, h: 22 };
const ys = RES.map((_, i) => 22 + i * 28);
const TOTAL_S = 6;

export default function OnboardFanout() {
  const reduced = useReducedMotion();
  const [lit, setLit] = useState(0); // how many resources provisioned so far

  useEffect(() => {
    if (reduced) {
      setLit(RES.length);
      return;
    }
    const id = setInterval(() => setLit((n) => (n >= RES.length ? 0 : n + 1)), 600);
    return () => clearInterval(id);
  }, [reduced]);

  const seconds = ((lit / RES.length) * TOTAL_S).toFixed(1);

  return (
    <Widget title="One command, a whole tenant" kicker="onboard-tenant.sh <name>">
      <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="w-full" style={{ maxHeight: 250 }} role="img"
        aria-label="A single onboarding command fanning out into namespace, quota, network policy, rbac, kafka topic, warehouse schema, and dashboard">
        {/* fan-out connectors */}
        {RES.map((r, i) => {
          const on = i < lit;
          const a = rightOf(CMD);
          const b = { x: BOX.x, y: ys[i] + BOX.h / 2 };
          return (
            <line key={r.name} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              style={{ stroke: on ? r.color : 'var(--color-line-strong)', strokeWidth: on ? 2 : 1, strokeDasharray: on ? undefined : '3 3', opacity: on ? 1 : 0.5, transition: 'stroke 200ms, opacity 200ms' }} />
          );
        })}

        <SvgNode n={CMD} title="onboard" sub="one command" color="var(--color-accent)" state="active" titleSize={12} />

        {RES.map((r, i) => {
          const on = i < lit;
          return (
            <g key={r.name} style={{ opacity: on ? 1 : 0.45, transition: 'opacity 200ms' }}>
              <rect x={BOX.x} y={ys[i]} width={BOX.w} height={BOX.h}
                style={{ fill: on ? r.color : 'var(--color-paper)', stroke: on ? r.color : 'var(--color-line-strong)', strokeWidth: 1.5, transition: 'fill 200ms' }} />
              <text x={BOX.x + 8} y={ys[i] + BOX.h / 2 + 1} dominantBaseline="middle"
                style={{ fill: on ? 'var(--color-paper)' : 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600 }}>
                {r.name}
              </text>
              <text x={BOX.x + BOX.w - 6} y={ys[i] + BOX.h / 2 + 1} textAnchor="end" dominantBaseline="middle"
                style={{ fill: on ? 'var(--color-paper)' : 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 7, opacity: 0.85 }}>
                {r.group}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="mt-2 flex items-center gap-2">
        <span className="font-mono text-[0.7rem] text-muted">provisioned</span>
        <span className="font-mono text-base font-semibold text-ink">{lit}/{RES.length}</span>
        <span className="font-mono text-[0.7rem] text-muted">in</span>
        <span className="font-mono text-base font-semibold" style={{ color: 'var(--color-accent)' }}>{seconds}s</span>
      </div>
      <div className="mt-2 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        <span className="border border-line px-2 py-1.5 font-mono text-[0.66rem] text-ink-soft">
          one command: <strong className="text-ink">~{TOTAL_S}s</strong>, declarative, repeatable
        </span>
        <span className="border border-line px-2 py-1.5 font-mono text-[0.66rem] text-ink-soft">
          manual runbook: <strong className="text-ink">~20 min</strong>, 10 hand-applied objects
        </span>
      </div>
    </Widget>
  );
}
