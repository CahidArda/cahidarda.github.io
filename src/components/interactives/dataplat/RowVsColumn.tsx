import { useState } from 'react';
import { useHoverPreview, Widget } from './shared';

// A tiny orders table. The point is physical layout, so we show the SAME five rows
// laid out two ways and count which bytes each engine must actually read.
const COLS = ['order_id', 'tenant', 'status', 'amount'] as const;
type Col = (typeof COLS)[number];
const ROWS: Record<Col, string>[] = [
  { order_id: 'ord-1', tenant: 'retail', status: 'paid', amount: '120' },
  { order_id: 'ord-2', tenant: 'marketplace', status: 'placed', amount: '80' },
  { order_id: 'ord-3', tenant: 'retail', status: 'shipped', amount: '300' },
  { order_id: 'ord-4', tenant: 'subscriptions', status: 'paid', amount: '45' },
  { order_id: 'ord-5', tenant: 'marketplace', status: 'cancelled', amount: '210' },
];
const TARGET_ROW = 2; // ord-3
const KEY_COL: Col = 'order_id'; // the column the point lookup filters on

type Query = 'point' | 'scan';
const QUERIES: Record<Query, { label: string; sql: string; cols: Col[]; rows: 'one' | 'all' }> = {
  point: {
    label: 'Point lookup',
    sql: "SELECT * FROM orders WHERE order_id = 'ord-3'",
    cols: [...COLS],
    rows: 'one',
  },
  scan: {
    label: 'Analytics scan',
    sql: 'SELECT tenant, sum(amount) FROM orders GROUP BY tenant',
    cols: ['tenant', 'amount'],
    rows: 'all',
  },
};

type Cell = 'used' | 'wasted' | 'skipped';

// What a ROW store must read: it can only fetch whole rows. So it reads every row the
// query touches, and within those rows it gets all columns whether it needs them or not.
function rowStoreCell(q: Query, r: number, c: Col): Cell {
  const Q = QUERIES[q];
  const rowTouched = Q.rows === 'all' || r === TARGET_ROW;
  if (!rowTouched) return 'skipped';
  return Q.cols.includes(c) ? 'used' : 'wasted';
}

// What a COLUMN store must read: it fetches whole columns. For a scan it reads only
// the needed columns (every row, nothing wasted). For a point lookup it has no row
// index, so it scans the key column to find the matching row (wasting the
// non-matches), then fetches that row's other fields from their segments.
function colStoreCell(q: Query, r: number, c: Col): Cell {
  const Q = QUERIES[q];
  if (Q.rows === 'all') {
    return Q.cols.includes(c) ? 'used' : 'skipped';
  }
  if (c === KEY_COL) return r === TARGET_ROW ? 'used' : 'wasted'; // scan to find the row
  if (!Q.cols.includes(c)) return 'skipped';
  return r === TARGET_ROW ? 'used' : 'skipped';
}

function tally(fn: (r: number, c: Col) => Cell) {
  let read = 0;
  let wasted = 0;
  for (let r = 0; r < ROWS.length; r++)
    for (const c of COLS) {
      const s = fn(r, c);
      if (s !== 'skipped') read++;
      if (s === 'wasted') wasted++;
    }
  return { read, wasted };
}

const cellStyle = (s: Cell, color: string): React.CSSProperties =>
  s === 'used'
    ? { background: color, color: 'var(--color-paper)', borderColor: color }
    : s === 'wasted'
      ? {
          background: 'transparent',
          color: 'var(--color-muted)',
          borderColor: 'var(--color-line-strong)',
          opacity: 0.5,
          textDecoration: 'line-through',
        }
      : {
          background: 'transparent',
          color: 'var(--color-muted)',
          borderColor: 'var(--color-line)',
          opacity: 0.25,
        };

export default function RowVsColumn() {
  const [sel, setSel] = useState<Query>('scan');
  const [q, bindQ] = useHoverPreview(sel);
  const ROW_COLOR = 'var(--color-dp-row)';
  const COL_COLOR = 'var(--color-dp-col)';
  const rowT = tally((r, c) => rowStoreCell(q, r, c));
  const colT = tally((r, c) => colStoreCell(q, r, c));

  return (
    <Widget title="Row vs column: who reads less?" kicker="same 20 cells, two layouts">
      <div className="mb-3 flex flex-wrap gap-2">
        {(Object.keys(QUERIES) as Query[]).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setSel(k)}
            aria-pressed={q === k}
            {...bindQ(k)}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: q === k ? 'var(--color-accent)' : 'var(--color-line)',
              color: q === k ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {QUERIES[k].label}
          </button>
        ))}
      </div>

      <p className="mb-3 font-mono text-[0.72rem] text-ink-soft">{QUERIES[q].sql}</p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* ROW STORE: physical order is row-major; each row is one contiguous block. */}
        <Panel
          name="Row store"
          sub="Postgres · row-major"
          color={ROW_COLOR}
          tally={rowT}
          blocks={QUERIES[q].rows === 'all' ? `${ROWS.length} row blocks` : '1 row block'}
        >
          {/* One shared grid (not independent flex rows) so the columns line up and a
              long value like "subscriptions" sizes its column for every row, not just its own. */}
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: 'repeat(4, minmax(min-content, 1fr))' }}
          >
            {COLS.map((c) => (
              <span
                key={`h-${c}`}
                className="mb-0.5 text-center font-mono text-[0.58rem] text-muted"
              >
                {c}
              </span>
            ))}
            {ROWS.flatMap((row, r) =>
              COLS.map((c) => (
                <CellBox
                  key={`${r}-${c}`}
                  v={row[c]}
                  style={cellStyle(rowStoreCell(q, r, c), ROW_COLOR)}
                />
              )),
            )}
          </div>
        </Panel>

        {/* COLUMN STORE: physical order is column-major; each column is one contiguous segment. */}
        <Panel
          name="Column store"
          sub="ClickHouse · column-major"
          color={COL_COLOR}
          tally={colT}
          blocks={`${QUERIES[q].cols.length} column segment${QUERIES[q].cols.length > 1 ? 's' : ''}`}
        >
          <div className="flex gap-1">
            {COLS.map((c) => (
              <div key={c} className="flex flex-1 flex-col gap-1">
                <span className="mb-0.5 text-center font-mono text-[0.58rem] text-muted">{c}</span>
                {ROWS.map((row, r) => (
                  <CellBox key={r} v={row[c]} style={cellStyle(colStoreCell(q, r, c), COL_COLOR)} />
                ))}
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <p className="mt-3 text-sm text-ink-soft">
        {q === 'scan' ? (
          <>
            The scan needs two columns across every row. The row store still reads all 20 cells (it
            can only fetch whole rows) and throws half away. The column store reads just the
            <span style={{ color: COL_COLOR }} className="font-medium">
              {' '}
              tenant
            </span>{' '}
            and
            <span style={{ color: COL_COLOR }} className="font-medium">
              {' '}
              amount
            </span>{' '}
            segments: 10 cells, nothing wasted. This is why analytics lives on a column store.
          </>
        ) : (
          <>
            The point lookup wants one whole row. The row store jumps straight to it by primary key
            and reads one contiguous block: 4 cells, nothing wasted. The column store has no row
            index, so it scans the whole
            <span style={{ color: COL_COLOR }} className="font-medium">
              {' '}
              order_id
            </span>{' '}
            column to find the row (the non-matches are wasted). That scan also tells it the match
            is the 3rd row, so it seeks straight to that slot in each other column (one cell each,
            no re-scan). More cells, out of four separate places. This is why current-state lookups
            live on a row store.
          </>
        )}
      </p>
    </Widget>
  );
}

function Panel({
  name,
  sub,
  color,
  tally,
  blocks,
  children,
}: {
  name: string;
  sub: string;
  color: string;
  tally: { read: number; wasted: number };
  blocks: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-line p-2.5">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="font-mono text-xs font-semibold" style={{ color }}>
          {name}
        </span>
        <span className="font-mono text-[0.58rem] text-muted">{sub}</span>
      </div>
      <div className="flex flex-col gap-1">{children}</div>
      <div className="mt-2 font-mono text-[0.62rem] text-ink-soft">
        read <strong style={{ color }}>{tally.read}</strong>/20 cells
        {tally.wasted > 0 && <span className="text-muted"> · {tally.wasted} wasted</span>} ·{' '}
        {blocks}
      </div>
    </div>
  );
}

function CellBox({ v, style }: { v: string; style: React.CSSProperties }) {
  return (
    <span
      className="flex-1 border px-1 py-1 text-center font-mono text-[0.6rem] transition-all"
      style={style}
    >
      {v}
    </span>
  );
}
