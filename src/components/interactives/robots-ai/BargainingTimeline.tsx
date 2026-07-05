import { useState } from 'react';
import { AI, ROBOT, WARN, Widget } from './shared';

// 200 years of labour meeting new machines. The recurring pattern: workers bargained over
// the TERMS of transition and rarely stopped the technology. Colour-coded by outcome, with
// the recent hardening toward outright bans at the bottom. Vertical so it reflows on mobile.

type Outcome = 'accepted' | 'struck' | 'resisted';
const OUTCOME: Record<Outcome, { label: string; color: string }> = {
  accepted: { label: 'accepted for terms', color: ROBOT },
  struck: { label: 'struck, then settled', color: WARN },
  resisted: { label: 'resisted outright', color: AI },
};

interface Node {
  year: string;
  title: string;
  outcome: Outcome;
  detail: string;
}

const NODES: Node[] = [
  {
    year: '1810s',
    title: 'The Luddites',
    outcome: 'resisted',
    detail:
      'Skilled textile workers smashed the frames. Not technophobia: mechanisation destroyed their bargaining position, and they knew it. The same structure as today, 200 years early. They were crushed by the state.',
  },
  {
    year: '1950s',
    title: 'UMW & the continuous miner',
    outcome: 'accepted',
    detail:
      'John L. Lewis backed the continuous mining machine, betting on fewer but secure, better-paid jobs. The machines came; the secure jobs largely did not.',
  },
  {
    year: '1960',
    title: 'ILWU Mechanization & Modernization Agreement',
    outcome: 'accepted',
    detail:
      'West Coast dockworkers accepted containerisation in exchange for benefits. It created a two-tier "A-men / B-men" workforce that foreshadowed modern two-tier labour, and containerisation went on to eliminate most longshore work.',
  },
  {
    year: '1964',
    title: 'East Coast longshore strike',
    outcome: 'struck',
    detail:
      'A rank-and-file strike closed ports for 18 days and forced LBJ to invoke Taft-Hartley. Leadership settled for reduced gang sizes anyway. The work still went.',
  },
  {
    year: '1970s',
    title: 'UAW: automation for security',
    outcome: 'accepted',
    detail:
      'Through mid-century the UAW accepted automation in exchange for job security. By the late 1970s the trade had reversed into concessions.',
  },
  {
    year: '2023',
    title: 'Teamsters / UPS driverless-truck ban',
    outcome: 'resisted',
    detail:
      'The UPS contract banned driverless trucks for its duration. Not a bargain over terms: a temporary stop.',
  },
  {
    year: '2024',
    title: 'ILA total-automation-ban demand',
    outcome: 'resisted',
    detail:
      'The International Longshoremen’s Association threatened to shut East and Gulf ports demanding a total ban on automating cranes, gates and container moves. Full automation framed as existential, not tradeable. The hardening is new.',
  },
];

export default function BargainingTimeline() {
  const [sel, setSel] = useState(0);
  const n = NODES[sel];

  return (
    <Widget title="200 years of bargaining, not stopping" kicker="click a node to read the case">
      <p className="mb-4 text-sm text-ink-soft">
        Labour has met new machines many times. It shaped the terms, struck over the pace, and only
        recently started demanding outright bans. It rarely stopped the technology.
      </p>

      <div className="flex flex-wrap gap-3">
        {(Object.keys(OUTCOME) as Outcome[]).map((k) => (
          <span
            key={k}
            className="inline-flex items-center gap-1.5 font-mono text-[0.66rem] text-ink-soft"
          >
            <span
              aria-hidden
              style={{
                width: 10,
                height: 10,
                background: OUTCOME[k].color,
                display: 'inline-block',
              }}
            />
            {OUTCOME[k].label}
          </span>
        ))}
      </div>

      <div className="mt-4 flex gap-3">
        {/* spine + nodes */}
        <ol className="relative flex flex-1 flex-col gap-0" style={{ margin: 0, padding: 0 }}>
          <span
            aria-hidden
            className="absolute bottom-2 left-[5px] top-2 w-px"
            style={{ background: 'var(--color-line-strong)' }}
          />
          {NODES.map((node, i) => (
            <li key={node.title} className="relative">
              <button
                type="button"
                onClick={() => setSel(i)}
                aria-pressed={i === sel}
                className="flex w-full items-center gap-3 py-1.5 text-left transition-opacity"
                style={{ opacity: i === sel ? 1 : 0.62 }}
              >
                <span
                  aria-hidden
                  className="relative z-10 shrink-0 rounded-full"
                  style={{
                    width: 11,
                    height: 11,
                    background: i === sel ? OUTCOME[node.outcome].color : 'var(--color-paper)',
                    border: `2px solid ${OUTCOME[node.outcome].color}`,
                  }}
                />
                <span className="w-12 shrink-0 font-mono text-[0.68rem] text-muted">
                  {node.year}
                </span>
                <span
                  className="font-mono text-[0.78rem]"
                  style={{
                    color: i === sel ? 'var(--color-ink)' : 'var(--color-ink-soft)',
                    fontWeight: i === sel ? 600 : 400,
                  }}
                >
                  {node.title}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>

      <div className="mt-4 border-t border-line pt-3">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            style={{
              width: 10,
              height: 10,
              background: OUTCOME[n.outcome].color,
              display: 'inline-block',
            }}
          />
          <span className="font-mono text-sm font-semibold text-ink">
            {n.year} · {n.title}
          </span>
        </div>
        <p className="mt-2 text-sm text-ink-soft">{n.detail}</p>
      </div>
    </Widget>
  );
}
