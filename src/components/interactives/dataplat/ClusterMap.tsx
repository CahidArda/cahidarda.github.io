import { useEffect, useState } from 'react';
import { Widget } from './shared';

// The cluster, as of a given phase. Overlays are cumulative, so the cluster only
// grows: each phase adds namespaces and workloads onto the same cluster. Embed
// <ClusterMap phase={N} /> in a phase's article and the reader can scrub 0..N to
// see exactly what that phase introduced (highlighted) on top of what came before.

type Group = 'app' | 'store' | 'obs' | 'orch' | 'queue' | 'gitops' | 'misc';
type Workload = { name: string; phase: number; removed?: number };
type Ns = { ns: string; phase: number; isolation?: boolean; workloads: Workload[] };

// One fixed colour per workload, reusing the cast the reader learned in the
// data-flow diagram (redpanda red, postgres blue, clickhouse gold, lake teal).
const COLOR: Record<string, string> = {
  redpanda: 'var(--color-dp-log)',
  postgres: 'var(--color-dp-row)',
  clickhouse: 'var(--color-dp-col)',
  minio: 'var(--color-dp-lake)',
  'iceberg-rest': 'var(--color-dp-lake)',
  producer: 'var(--color-dp-producer)',
  'stream-processor': 'var(--color-dp-stream)',
  api: 'var(--color-accent)',
  elasticmq: 'var(--color-dp-log)',
  dagster: 'var(--color-accent)',
};
const colorFor = (n: string) => COLOR[n] ?? 'var(--color-muted)';

const NAMESPACES: Ns[] = [
  {
    ns: 'platform',
    phase: 0,
    workloads: [
      { name: 'placeholder', phase: 0, removed: 7 },
      { name: 'redpanda', phase: 1 },
      { name: 'postgres', phase: 1 },
      { name: 'producer', phase: 1 },
      { name: 'stream-processor', phase: 1 },
      { name: 'api', phase: 1 },
      { name: 'clickhouse', phase: 3 },
      { name: 'minio', phase: 3 },
      { name: 'iceberg-rest', phase: 3 },
      { name: 'elasticmq', phase: 4 },
      { name: 'dagster', phase: 6 },
    ],
  },
  {
    ns: 'observability',
    phase: 2,
    workloads: [
      { name: 'prometheus', phase: 2 },
      { name: 'grafana', phase: 2 },
      { name: 'loki', phase: 2 },
      { name: 'alloy', phase: 2 },
      { name: 'kafka-lag-exporter', phase: 2 },
      { name: 'alertmanager', phase: 4 },
    ],
  },
  { ns: 'tenant-retail', phase: 5, isolation: true, workloads: [] },
  { ns: 'tenant-marketplace', phase: 5, isolation: true, workloads: [] },
  { ns: 'tenant-subscriptions', phase: 5, isolation: true, workloads: [] },
  { ns: 'argocd', phase: 7, workloads: [{ name: 'argocd', phase: 7 }, { name: 'app-of-apps', phase: 7 }] },
];

// What each phase adds to the cluster, summarised for the scrubber.
const PHASES: { title: string; adds: string }[] = [
  { title: 'Foundation', adds: 'the cluster, the platform and observability namespaces, shared config, and a placeholder service' },
  { title: 'Ingest', adds: 'the data spine: a producer, the Redpanda log, the Spark stream-processor, Postgres, and the API' },
  { title: 'Observability', adds: 'metrics, dashboards, and logs: Prometheus, Grafana, Loki, Alloy, and the consumer-lag exporter' },
  { title: 'Columnar', adds: 'the column store and lakehouse: ClickHouse, MinIO, and the Iceberg REST catalog' },
  { title: 'Resilience', adds: 'Alertmanager, and an SQS-compatible queue (ElasticMQ) to contrast a queue DLQ with the log DLQ' },
  { title: 'Multi-tenancy', adds: 'a namespace per tenant, each with its own ResourceQuota, NetworkPolicy, and RBAC' },
  { title: 'Orchestration', adds: 'Dagster, running the nightly batch rollup behind a blocking data-quality gate' },
  { title: 'Self-service', adds: 'ArgoCD for GitOps (its pruning finally removes the phase-0 placeholder)' },
];

export default function ClusterMap({ phase = 7 }: { phase?: number }) {
  const [p, setP] = useState(phase);
  useEffect(() => setP(phase), [phase]);

  const namespaces = NAMESPACES.filter((n) => n.phase <= p);

  return (
    <Widget title={`The cluster at phase ${p}: ${PHASES[p].title}`} kicker="cumulative · scrub to replay the growth">
      <div className="mb-3 flex items-center gap-2">
        <span className="font-mono text-[0.7rem] text-muted">phase</span>
        {Array.from({ length: phase + 1 }, (_, i) => i).map((i) => (
          <button key={i} type="button" onClick={() => setP(i)} aria-pressed={p === i}
            className="h-6 w-6 border-2 font-mono text-[0.7rem] transition-colors"
            style={{ borderColor: p === i ? 'var(--color-accent)' : 'var(--color-line)', color: p === i ? 'var(--color-accent)' : 'var(--color-ink-soft)' }}>
            {i}
          </button>
        ))}
      </div>

      <div className="border-2 border-line-strong p-2.5">
        <div className="mb-2 flex items-center justify-between">
          <span className="label">k3d cluster: data-platform</span>
          <span className="font-mono text-[0.6rem] text-muted">{namespaces.length} namespaces</span>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {namespaces.map((n) => {
            const fresh = n.phase === p;
            const wls = n.workloads.filter((w) => w.phase <= p && (w.removed == null || w.removed > p));
            return (
              <div key={n.ns} className="border p-2" style={{ borderColor: fresh ? 'var(--color-accent)' : 'var(--color-line)' }}>
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="font-mono text-[0.72rem] font-semibold text-ink">{n.ns}</span>
                  {fresh && <span className="font-mono text-[0.55rem] uppercase tracking-wider" style={{ color: 'var(--color-accent)' }}>new</span>}
                </div>
                {n.isolation ? (
                  <div className="flex flex-wrap gap-1">
                    {['quota', 'netpol', 'rbac'].map((b) => (
                      <span key={b} className="border border-line-strong px-1.5 py-0.5 font-mono text-[0.6rem] text-ink-soft">{b}</span>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {wls.map((w) => {
                      const isNew = w.phase === p;
                      return (
                        <span key={w.name} className="flex items-center gap-1 border px-1.5 py-0.5 font-mono text-[0.62rem]"
                          style={{ borderColor: isNew ? 'var(--color-accent)' : 'var(--color-line)', color: isNew ? 'var(--color-ink)' : 'var(--color-ink-soft)', opacity: isNew ? 1 : 0.85 }}>
                          <span className="inline-block h-2 w-2 shrink-0" style={{ background: colorFor(w.name) }} />
                          {w.name}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-2 border-l-2 pl-2.5" style={{ borderColor: 'var(--color-accent)' }}>
        <p className="font-mono text-[0.72rem] font-semibold text-ink">
          Phase {p}: {PHASES[p].title}
        </p>
        <p className="mt-0.5 text-sm text-ink-soft">
          {p === 0 ? 'Stands up ' : 'Adds '}{PHASES[p].adds}.{' '}
          {p > 0 && 'Everything else (in '}
          {p > 0 && <span className="text-muted">muted</span>}
          {p > 0 && ') was already there: one cluster, growing a layer at a time.'}
        </p>
      </div>
    </Widget>
  );
}
