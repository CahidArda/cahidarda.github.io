/**
 * A month of the Current as swimlanes: one row per category on a shared date axis. Each
 * day with links in a category gets one square marker, with the count printed on it when
 * there are several. Hovering (or focusing) a marker shows a tooltip listing that day's
 * links in the category, each with a link to the piece and a jump to its note. Clicking the
 * marker itself scrolls to the first of them in the list below (rendered by
 * `CurrentGroup` / `CurrentList`) and flashes it.
 *
 * Layout: x is a percentage of the lane width (reflows with the column), y is in px (a
 * marker stays the same size at any width), so markers and tooltip share one frame.
 */
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  categoryColor,
  dayNumber,
  dayToIso,
  formatItemDate,
  formatShortDate,
  hasDay,
  itemId,
  sourceSuffix,
  type Current,
  type CurrentItem,
} from '@/lib/current';

interface Props {
  data: Current;
  /** First day of the axis (ISO). Defaults to the earliest item. */
  from?: string;
  /** Last day of the axis (ISO). Defaults to the latest item. */
  to?: string;
  title?: string;
}

/** All of one category's links on one day. */
interface Marker {
  key: string;
  category: string;
  catIndex: number;
  /** Row on the timeline (categories with nothing on the axis get no row). */
  lane: number;
  date: string;
  items: CurrentItem[];
  /** 0..1 along the axis. */
  x: number;
}

const LANE = 36; // row height, px
const LABEL = 22; // from the top of a lane to its line, leaves room for the lane label
const SQ = 12; // marker size, px
const HIDE_DELAY = 140; // ms, so the pointer can travel from marker into the tooltip

export default function CurrentTimeline({ data, from, to, title = 'Timeline' }: Props) {
  const { markers, lanes, start, span, count } = useMemo(
    () => layout(data, from, to),
    [data, from, to],
  );
  const ticks = useMemo(() => weekTicks(start, span), [start, span]);

  const [hover, setHover] = useState<number | null>(null);
  const hideTimer = useRef<number | undefined>(undefined);

  const show = (i: number) => {
    window.clearTimeout(hideTimer.current);
    setHover(i);
  };
  const hideSoon = () => {
    window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setHover(null), HIDE_DELAY);
  };
  useEffect(() => () => window.clearTimeout(hideTimer.current), []);

  // Escape closes the tooltip.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setHover(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const height = lanes.length * LANE;
  const active = hover === null ? null : markers[hover];
  const lineY = (i: number) => i * LANE + LABEL;
  const laneTop = (m: Marker) => lineY(m.lane) - SQ / 2;

  return (
    <div className="fx not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        <span className="font-mono text-[0.65rem] text-muted">{count} links</span>
      </div>

      <div className="px-4 pt-4 pb-4 sm:px-6">
        <div className="relative" style={{ height: height + 22 }}>
          {/* lanes: label above, dashed line through the markers */}
          {lanes.map((g, i) => (
            <div key={g.category}>
              <div
                className="absolute left-0 flex items-center gap-1.5 font-mono text-[0.62rem] leading-none text-ink-soft"
                style={{ top: i * LANE }}
              >
                <span
                  aria-hidden
                  className="inline-block h-2 w-2"
                  style={{ background: categoryColor(g.catIndex) }}
                />
                {g.category} <span className="text-muted">{g.count}</span>
              </div>
              <div
                className="absolute right-0 left-0 border-t border-dashed border-line"
                style={{ top: lineY(i) }}
              />
            </div>
          ))}
          {/* axis */}
          <div
            className="absolute right-0 left-0 border-t border-line-strong"
            style={{ top: height }}
          />
          {ticks.map((t, i) => (
            <div
              key={t.iso}
              className={`absolute -translate-x-1/2 font-mono text-[0.62rem] text-muted ${
                i % 2 === 1 ? 'hidden sm:block' : ''
              }`}
              style={{ left: `${t.x * 100}%`, top: height }}
            >
              <div className="mx-auto h-1.5 w-px bg-line-strong" />
              <div className="mt-0.5 whitespace-nowrap">{formatShortDate(t.iso)}</div>
            </div>
          ))}

          {/* markers */}
          {markers.map((m, i) => {
            const n = m.items.length;
            const label =
              n === 1
                ? `${m.items[0].title}, ${formatItemDate(m.date)}. Jump to it in the list.`
                : `${n} ${m.category} links on ${formatItemDate(m.date)}. Jump to the first in the list.`;
            return (
              <button
                key={m.key}
                type="button"
                aria-label={label}
                aria-describedby={hover === i ? 'cur-tip' : undefined}
                onMouseEnter={() => show(i)}
                onMouseLeave={hideSoon}
                onFocus={() => show(i)}
                onBlur={hideSoon}
                onClick={() => jumpTo(m.items[0])}
                className="group absolute"
                style={{
                  width: SQ,
                  height: SQ,
                  left: `calc(${m.x * 100}% - ${SQ / 2}px)`,
                  top: laneTop(m),
                }}
              >
                <span
                  className="flex h-full w-full items-center justify-center border font-mono text-[0.5rem] leading-none font-bold text-paper transition-transform duration-150 group-hover:scale-125 group-focus-visible:scale-125"
                  style={{
                    background: categoryColor(m.catIndex),
                    borderColor: hover === i ? 'var(--ink)' : 'var(--paper-raised)',
                  }}
                >
                  {n > 1 && <span aria-hidden>{n}</span>}
                </span>
              </button>
            );
          })}

          {/* tooltip */}
          {active && (
            <div
              id="cur-tip"
              role="tooltip"
              onMouseEnter={() => show(hover!)}
              onMouseLeave={hideSoon}
              className="absolute z-20 w-max max-w-[min(22rem,85vw)]"
              style={tooltipPos(active.x, laneTop(active))}
            >
              {/* inner box animates; the outer one owns the positioning transform */}
              <div className="fx-fly-in border border-line bg-paper px-3 py-2 text-xs leading-snug shadow-sm">
                <div className="mb-1 flex items-center gap-1.5 font-mono text-[0.62rem] whitespace-nowrap text-muted">
                  <span
                    aria-hidden
                    className="inline-block h-2 w-2"
                    style={{ background: categoryColor(active.catIndex) }}
                  />
                  {active.category} · {formatItemDate(active.date)}
                </div>
                <ul className={active.items.length > 1 ? 'space-y-1.5' : ''}>
                  {active.items.map((item) => (
                    <li key={itemId(item)} className="flex items-baseline justify-between gap-2">
                      <span>
                        {/* opens the piece in a new tab and brings its note into view here */}
                        <a
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={() => jumpTo(item)}
                          className="font-medium text-ink underline decoration-line-strong underline-offset-2 hover:text-accent hover:decoration-accent"
                        >
                          {item.title}
                        </a>
                        <span className="text-muted">{sourceSuffix(item)}</span>
                      </span>
                      {active.items.length > 1 && (
                        <button
                          type="button"
                          onClick={() => jumpTo(item)}
                          className="shrink-0 font-mono text-[0.6rem] text-muted hover:text-accent"
                        >
                          note ↓
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function jumpTo(item: CurrentItem) {
  const el = document.getElementById(itemId(item));
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  history.replaceState(null, '', `#${el.id}`);
  el.classList.remove('cur-flash');
  void el.offsetWidth; // restart the animation on repeated clicks
  el.classList.add('cur-flash');
}

/**
 * One marker per category per day, placed on the axis. Items without a full date, or dated
 * outside the axis (an older piece saved this month), stay in the list but off the timeline.
 */
function layout(data: Current, from?: string, to?: string) {
  const days = data.flatMap((g) =>
    g.items.filter((i) => hasDay(i.date)).map((i) => dayNumber(i.date!)),
  );
  const start = from ? dayNumber(from) : Math.min(...days);
  const end = to ? dayNumber(to) : Math.max(...days);
  const span = Math.max(1, end - start);

  const byKey = new Map<string, Marker>();
  const lanes: { category: string; catIndex: number; count: number }[] = [];
  data.forEach((g, catIndex) => {
    const placed = g.items.filter(
      (item) => hasDay(item.date) && dayNumber(item.date) >= start && dayNumber(item.date) <= end,
    );
    if (!placed.length) return;
    const lane = lanes.push({ category: g.category, catIndex, count: placed.length }) - 1;
    for (const item of placed) {
      const day = dayNumber(item.date!);
      const key = `${catIndex}:${day}`;
      const m = byKey.get(key);
      if (m) m.items.push(item);
      else
        byKey.set(key, {
          key,
          category: g.category,
          catIndex,
          lane,
          date: item.date!,
          items: [item],
          x: (day - start) / span,
        });
    }
  });
  const markers = [...byKey.values()].sort((a, b) => a.x - b.x || a.catIndex - b.catIndex);
  const count = lanes.reduce((n, l) => n + l.count, 0);
  return { markers, lanes, start, span, count };
}

/** One tick per week from the axis start. */
function weekTicks(start: number, span: number) {
  const ticks = [];
  for (let d = 0; d <= span; d += 7) ticks.push({ iso: dayToIso(start + d), x: d / span });
  return ticks;
}

/** Places the tooltip above a marker, kept inside the frame near the axis ends. */
function tooltipPos(x: number, top: number): CSSProperties {
  const t = top - 8;
  if (x < 0.2) return { left: `calc(${x * 100}% - 12px)`, top: t, transform: 'translateY(-100%)' };
  if (x > 0.8)
    return { right: `calc(${(1 - x) * 100}% - 12px)`, top: t, transform: 'translateY(-100%)' };
  return { left: `${x * 100}%`, top: t, transform: 'translate(-50%, -100%)' };
}
