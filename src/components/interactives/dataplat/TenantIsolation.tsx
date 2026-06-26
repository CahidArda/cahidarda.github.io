import { useState } from 'react';
import { type Box, leftOf, rightOf, SvgNode, Widget } from './shared';

// Pick an action the retail tenant might attempt; see whether it is allowed and
// which isolation control decides. Cross-tenant access is denied by RBAC and by
// NetworkPolicy; only the explicitly allow-listed path (the event log) is open.

const VB = { w: 430, h: 210 };
const NODE: Record<string, Box> = {
  retail: { cx: 72, cy: 105, w: 104, h: 52 },
  marketplace: { cx: 300, cy: 46, w: 150, h: 44 },
  log: { cx: 300, cy: 105, w: 150, h: 44 },
  quota: { cx: 300, cy: 164, w: 150, h: 44 },
};
const ALLOW = 'var(--color-dp-lake)';
const DENY = 'var(--color-dp-log)';

type Scn = { label: string; target: keyof typeof NODE; allow: boolean; by: string };
const SCN: Scn[] = [
  { label: "Read another tenant's data", target: 'marketplace', allow: false, by: 'RBAC: the retail identity has no role outside its own namespace' },
  { label: "Reach another tenant's pod", target: 'marketplace', allow: false, by: 'NetworkPolicy: deny-by-default blocks cross-tenant traffic' },
  { label: 'Publish to the event log', target: 'log', allow: true, by: 'NetworkPolicy: the shared log is the one allow-listed egress' },
  { label: 'Outgrow its resource budget', target: 'quota', allow: false, by: 'ResourceQuota: the namespace is capped, containing the hog' },
];

export default function TenantIsolation() {
  const [i, setI] = useState(0);
  const s = SCN[i];
  const tgt = NODE[s.target];
  const v = s.allow ? ALLOW : DENY;
  const from = rightOf(NODE.retail);
  const to = leftOf(tgt);

  return (
    <Widget title="What can a tenant do?" kicker="isolation: rbac · netpol · quota">
      <div className="mb-3 flex flex-wrap gap-2">
        {SCN.map((sc, idx) => (
          <button key={idx} type="button" onClick={() => setI(idx)} aria-pressed={i === idx}
            className="border-2 px-2 py-1 font-mono text-[0.68rem] tracking-wide transition-colors"
            style={{ borderColor: i === idx ? 'var(--color-accent)' : 'var(--color-line)', color: i === idx ? 'var(--color-accent)' : 'var(--color-ink-soft)' }}>
            {sc.label}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="w-full" style={{ maxHeight: 240 }} role="img"
        aria-label="The retail tenant attempts an action against another tenant, the event log, or its quota, and the result is allowed or denied">
        <defs>
          <marker id="ti-allow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" style={{ fill: ALLOW }} />
          </marker>
          <marker id="ti-deny" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" style={{ fill: DENY }} />
          </marker>
        </defs>

        {/* the attempt */}
        <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} markerEnd={`url(#${s.allow ? 'ti-allow' : 'ti-deny'})`}
          style={{ stroke: v, strokeWidth: 2.5, strokeDasharray: s.allow ? undefined : '5 4', transition: 'stroke 200ms' }} />
        {/* a deny "block" tick across the line midpoint */}
        {!s.allow && (
          <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 4} textAnchor="middle"
            style={{ fill: DENY, fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700 }}>✕</text>
        )}

        <SvgNode n={NODE.retail} title="tenant-retail" sub="the actor" color="var(--color-accent)" state="wait" titleSize={11} />
        <SvgNode n={NODE.marketplace} title="tenant-marketplace" sub="another tenant" color={v} state={s.target === 'marketplace' ? 'active' : 'off'} titleSize={11} />
        <SvgNode n={NODE.log} title="event log" sub="shared platform" color={v} state={s.target === 'log' ? 'active' : 'off'} titleSize={11} />
        <SvgNode n={NODE.quota} title="resource quota" sub="its own budget" color={v} state={s.target === 'quota' ? 'active' : 'off'} titleSize={11} />
      </svg>

      <div className="mt-2 flex items-center gap-2">
        <span className="border-2 px-2 py-0.5 font-mono text-[0.72rem] font-semibold"
          style={{ borderColor: v, color: 'var(--color-paper)', background: v }}>
          {s.allow ? 'ALLOWED' : 'DENIED'}
        </span>
        <span className="text-sm text-ink-soft">{s.by}</span>
      </div>
    </Widget>
  );
}
