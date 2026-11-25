/**
 * The series roadmap, drawn as a metro map. Three lines (strategy, networks, crowds),
 * one station per page, interchanges where a page belongs to two lines. Hover or focus a
 * station to read what it is about; click to go there. A small train runs each line.
 * On narrow screens the map is redrawn as a vertical strip so labels stay legible.
 */
import { useEffect, useRef, useState } from 'react';
import { GT, Specimen, useFrame, useInView, useReducedMotion } from './shared';

export interface Station {
  slug: string;
  n: number;
  title: string;
  description: string;
}

type LineId = 'strategy' | 'network' | 'crowd' | 'story';
type Line = { id: LineId; name: string; color: string; stops: string[] };
const BASE_LINES: Line[] = [
  {
    id: 'strategy',
    name: 'strategy line',
    color: GT.a,
    stops: ['nash', 'prisoners-dilemma', 'hotelling', 'auctions', 'braess'],
  },
  {
    id: 'network',
    name: 'network line',
    color: GT.opt,
    stops: ['braess', 'pagerank', 'small-worlds', 'cascades'],
  },
  {
    id: 'crowd',
    name: 'crowd line',
    color: GT.b,
    stops: ['cascades', 'segregation', 'power-laws'],
  },
];

/* Desktop geometry: blue along the top, green down the right, red along the bottom. */
const W = 640;
const H_BASE = 380;
const H_STORY = 480;
type Pos = { x: number; y: number; label: 'above' | 'below' | 'left' };
const BASE_POS: Record<string, Pos> = {
  nash: { x: 60, y: 80, label: 'above' },
  'prisoners-dilemma': { x: 175, y: 80, label: 'above' },
  hotelling: { x: 290, y: 80, label: 'above' },
  auctions: { x: 405, y: 80, label: 'above' },
  braess: { x: 560, y: 80, label: 'above' },
  pagerank: { x: 560, y: 175, label: 'left' },
  'small-worlds': { x: 560, y: 245, label: 'left' },
  cascades: { x: 560, y: 315, label: 'below' },
  segregation: { x: 380, y: 315, label: 'below' },
  'power-laws': { x: 200, y: 315, label: 'below' },
};
const SHORT: Record<string, string> = {
  nash: 'Nash',
  'prisoners-dilemma': 'Dilemma',
  hotelling: 'Huddling',
  auctions: 'Auctions',
  braess: 'Braess',
  pagerank: 'PageRank',
  'small-worlds': 'Six degrees',
  cascades: 'Cascades',
  'power-laws': 'Power laws',
  segregation: 'Segregation',
};
const INTERCHANGE = new Set(['braess', 'cascades']);

/** Example ("story") pages hang off station 10 on a fourth line along the bottom. */
function buildMap(order: string[], bySlug: Record<string, Station>) {
  const stories = order.filter((s) => (bySlug[s]?.n ?? 0) > 10);
  const pos: Record<string, Pos> = { ...BASE_POS };
  const lines: Line[] = [...BASE_LINES];
  if (stories.length) {
    pos['power-laws'] = { ...pos['power-laws'], label: 'left' };
    stories.forEach((s, i) => (pos[s] = { x: 200 + i * 150, y: 420, label: 'below' }));
    lines.push({ id: 'story', name: 'story line', color: GT.alt, stops: ['power-laws', ...stories] });
  }
  return { pos, lines, stories, H: stories.length ? H_STORY : H_BASE };
}
function shortTitle(slug: string, s: Station) {
  return SHORT[slug] ?? s.title.split(':')[0];
}
function stationTag(s: Station) {
  return s.n > 10 ? `E${s.n - 10}` : String(s.n).padStart(2, '0');
}
function polyLen(pts: { x: number; y: number }[]) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return L;
}
function pointAt(pts: { x: number; y: number }[], d: number) {
  for (let i = 1; i < pts.length; i++) {
    const seg = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    if (d <= seg) {
      const t = d / seg;
      return { x: pts[i - 1].x + (pts[i].x - pts[i - 1].x) * t, y: pts[i - 1].y + (pts[i].y - pts[i - 1].y) * t };
    }
    d -= seg;
  }
  return pts[pts.length - 1];
}
function linesOf(lines: Line[], slug: string) {
  return lines.filter((l) => l.stops.includes(slug));
}

export default function Roadmap({ stations }: { stations: Station[] }) {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const bySlug = Object.fromEntries(stations.map((s) => [s.slug, s]));
  const order = [...stations].sort((a, b) => a.n - b.n).map((s) => s.slug);

  const { pos: POS, lines: LINES, stories, H } = buildMap(order, bySlug);
  const linePoints = (stops: string[]) => stops.map((x) => POS[x]);
  const [sel, setSel] = useState<string>(order[0] ?? 'nash');
  const [touched, setTouched] = useState(false);
  const [narrow, setNarrow] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Layout switch by container width.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setNarrow(e.contentRect.width < 520));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Tour the stations until the reader takes over.
  useEffect(() => {
    if (reduced || touched || !inView) return;
    const id = setInterval(() => {
      setSel((cur) => order[(order.indexOf(cur) + 1) % order.length]);
    }, 2600);
    return () => clearInterval(id);
  }, [reduced, touched, inView, order.join()]);

  // Trains: one per line, ping-ponging along it.
  const [clock, setClock] = useState(0);
  useFrame((dt) => setClock((c) => c + dt), !reduced && inView && !narrow);

  const cur = bySlug[sel];
  const pick = (slug: string) => {
    setTouched(true);
    setSel(slug);
  };

  return (
    <Specimen
      code="GT-00"
      title="the route map"
      kicker={stories.length ? `10 stations · ${stories.length} ${stories.length === 1 ? 'story' : 'stories'}` : '10 stations · 3 lines'}
      hint="hover a station to preview · click to ride"
      rootRef={rootRef}
    >
      <div ref={wrapRef}>
        {narrow ? (
          <NarrowMap order={order} bySlug={bySlug} sel={sel} pick={pick} lines={LINES} />
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label="Series route map">
            {/* lines */}
            {LINES.map((l) => {
              const pts = linePoints(l.stops);
              return (
                <polyline
                  key={l.id}
                  points={pts.map((p) => `${p.x},${p.y}`).join(' ')}
                  style={{ fill: 'none', stroke: l.color, strokeWidth: 7, strokeLinejoin: 'round', strokeLinecap: 'round' }}
                />
              );
            })}
            {/* trains (behind stations) */}
            {!reduced &&
              LINES.map((l, i) => {
                const pts = linePoints(l.stops);
                const L = polyLen(pts);
                const speed = 38;
                const raw = (clock * speed + i * 140) % (2 * L);
                const d = raw < L ? raw : 2 * L - raw;
                const p = pointAt(pts, d);
                return (
                  <rect
                    key={l.id}
                    x={p.x - 7}
                    y={p.y - 4}
                    width={14}
                    height={8}
                    style={{ fill: GT.paperRaised, stroke: l.color, strokeWidth: 2 }}
                  />
                );
              })}
            {/* poster title in the open middle of the map */}
            <text x={60} y={190} style={{ fill: GT.ink, fontFamily: GT.display, fontSize: 38 }}>
              The Logic
            </text>
            <text x={60} y={232} style={{ fill: GT.ink, fontFamily: GT.display, fontSize: 38 }}>
              of Crowds
            </text>
            <text
              x={62}
              y={258}
              style={{ fill: GT.muted, fontFamily: GT.mono, fontSize: 9.5, letterSpacing: '0.14em' }}
            >
              A ROUTE MAP IN TEN STOPS
            </text>
            {/* line names */}
            <text x={60} y={36} style={{ fill: GT.a, fontFamily: GT.mono, fontSize: 9, letterSpacing: '0.18em' }}>
              STRATEGY LINE
            </text>
            <text x={604} y={200} transform="rotate(90 604 200)" textAnchor="middle" style={{ fill: GT.opt, fontFamily: GT.mono, fontSize: 9, letterSpacing: '0.18em' }}>
              NETWORK LINE
            </text>
            <text x={stories.length ? 410 : 200} y={stories.length ? 299 : 364} style={{ fill: GT.b, fontFamily: GT.mono, fontSize: 9, letterSpacing: '0.18em' }}>
              CROWD LINE
            </text>
            {stories.length > 0 && (
              <text
                x={224}
                y={392}
                style={{ fill: GT.alt, fontFamily: GT.mono, fontSize: 9, letterSpacing: '0.18em' }}
              >
                STORY LINE
              </text>
            )}
            {/* stations */}
            {order.map((slug) => {
              const p = POS[slug];
              const s = bySlug[slug];
              if (!p || !s) return null;
              const on = slug === sel;
              const inter = INTERCHANGE.has(slug);
              const r = inter ? 13 : 11;
              const lx = p.label === 'left' ? p.x - 22 : p.x;
              const ly = p.label === 'above' ? p.y - 26 : p.label === 'below' ? p.y + 30 : p.y + 1;
              const anchor = p.label === 'left' ? 'end' : 'middle';
              return (
                <a
                  key={slug}
                  href={`/articles/logic-of-crowds/${slug}`}
                  className="gt-station"
                  aria-label={`${s.n}. ${s.title}`}
                  style={{ textDecoration: 'none' }}
                >
                  <g
                    onMouseEnter={() => pick(slug)}
                    onFocus={() => pick(slug)}
                    style={{ cursor: 'pointer' }}
                  >
                    <circle cx={p.x} cy={p.y} r={r + 10} style={{ fill: 'transparent' }} />
                    {on && (
                      <circle cx={p.x} cy={p.y} r={r + 6} style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }} />
                    )}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={r}
                      style={{
                        fill: on ? GT.ink : GT.paperRaised,
                        stroke: GT.ink,
                        strokeWidth: inter ? 3.5 : 2.5,
                        transition: 'fill 200ms',
                      }}
                    />
                    <text
                      x={p.x}
                      y={p.y + 0.5}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      style={{
                        fill: on ? GT.paper : GT.ink,
                        fontFamily: GT.mono,
                        fontSize: 9.5,
                        fontWeight: 700,
                        fontStyle: 'normal',
                        textDecoration: 'none',
                      }}
                    >
                      {stationTag(s)}
                    </text>
                    <text
                      x={lx}
                      y={ly}
                      textAnchor={anchor}
                      dominantBaseline="middle"
                      style={{
                        fill: on ? GT.eq : GT.ink,
                        fontFamily: GT.display,
                        fontSize: 15,
                        fontStyle: 'normal',
                        textDecoration: 'none',
                        transition: 'fill 200ms',
                      }}
                    >
                      {shortTitle(slug, s)}
                    </text>
                  </g>
                </a>
              );
            })}
          </svg>
        )}

        {/* the departure board */}
        {cur && (
          <a
            href={`/articles/logic-of-crowds/${cur.slug}`}
            className="mt-4 grid grid-cols-[auto_1fr] items-start gap-x-4 border border-line-strong bg-paper p-4 transition-colors hover:border-ink"
          >
            <span className="font-display text-5xl leading-none tabular-nums text-accent">
              {stationTag(cur)}
            </span>
            <span className="min-w-0">
              <span className="flex flex-wrap items-center gap-2">
                {linesOf(LINES, cur.slug).map((l) => (
                  <span key={l.id} className="inline-flex items-center gap-1.5 font-mono text-[0.62rem] uppercase tracking-[0.14em] text-muted">
                    <span className="inline-block h-1.5 w-4" style={{ background: l.color }} />
                    {l.name}
                  </span>
                ))}
              </span>
              <span className="mt-1 block font-display text-xl leading-tight text-ink">{cur.title}</span>
              <span className="mt-1 block text-[0.88rem] leading-snug text-ink-soft">{cur.description}</span>
              <span className="mt-2 block font-mono text-[0.68rem] text-accent">ride →</span>
            </span>
          </a>
        )}
      </div>
    </Specimen>
  );
}

function NarrowMap({
  order,
  bySlug,
  sel,
  pick,
  lines,
}: {
  order: string[];
  bySlug: Record<string, Station>;
  sel: string;
  pick: (s: string) => void;
  lines: Line[];
}) {
  return (
    <ol className="relative">
      {order.map((slug, i) => {
        const s = bySlug[slug];
        const ls = linesOf(lines, slug);
        const on = slug === sel;
        return (
          <li key={slug} className="relative flex items-stretch gap-3">
            <div className="relative flex w-9 shrink-0 justify-center">
              <span
                className="absolute w-[3px]"
                style={{
                  background: GT.ink,
                  top: i === 0 ? '50%' : 0,
                  bottom: i === order.length - 1 ? '50%' : 0,
                }}
              />
              <span
                className="relative z-10 my-2 flex h-7 w-7 items-center justify-center rounded-full border-2 font-mono text-[0.62rem] font-bold"
                style={{
                  background: on ? GT.ink : GT.paperRaised,
                  color: on ? GT.paper : GT.ink,
                  borderColor: GT.ink,
                }}
              >
                {stationTag(s)}
              </span>
            </div>
            <a
              href={`/articles/logic-of-crowds/${slug}`}
              onMouseEnter={() => pick(slug)}
              onFocus={() => pick(slug)}
              className="flex min-w-0 flex-1 items-center py-2"
            >
              <span className="min-w-0">
                <span className="flex gap-1">
                  {ls.map((l) => (
                    <span key={l.id} className="inline-block h-1.5 w-4" style={{ background: l.color }} />
                  ))}
                </span>
                <span
                  className={`block font-display text-lg leading-tight ${on ? 'text-accent' : 'text-ink'}`}
                >
                  {s.title}
                </span>
              </span>
            </a>
          </li>
        );
      })}
    </ol>
  );
}
