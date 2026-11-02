/**
 * A month of the Current on one date axis. Each item is a marker coloured by its category;
 * items from the same day stack upward. Hovering (or focusing) a marker shows a tooltip
 * with the category, date and a link to the piece itself. Clicking a marker scrolls to
 * the item in the list below (rendered by `CurrentGroup` / `CurrentList`) and flashes it,
 * so the description is right there.
 *
 * Layout: x is a percentage of the axis width (reflows with the column), y is in px (a
 * marker stays the same size at any width), so markers and tooltip share one frame.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  categoryColor,
  dayNumber,
  dayToIso,
  formatItemDate,
  formatShortDate,
  itemId,
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

interface Marker {
  item: CurrentItem;
  category: string;
  catIndex: number;
  /** 0..1 along the axis. */
  x: number;
  /** Stack position within its day, 0 = on the axis. */
  level: number;
}

const DOT = 11; // marker diameter, px
const STEP = 14; // vertical spacing between stacked markers, px
const AXIS_PAD = 18; // space between the lowest marker and the axis line, px
const HIDE_DELAY = 140; // ms, so the pointer can travel from marker into the tooltip

export default function CurrentTimeline({ data, from, to, title = 'Timeline' }: Props) {
  const { markers, start, span, maxLevel } = useMemo(
    () => layout(data, from, to),
    [data, from, to],
  );
  const ticks = useMemo(() => weekTicks(start, span), [start, span]);

  const [hover, setHover] = useState<number | null>(null);
  const [focusCat, setFocusCat] = useState<number | null>(null);
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

  const jumpTo = (m: Marker) => {
    const el = document.getElementById(itemId(m.item));
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    history.replaceState(null, '', `#${el.id}`);
    el.classList.remove('cur-flash');
    void el.offsetWidth; // restart the animation on repeated clicks
    el.classList.add('cur-flash');
  };

  const plotHeight = AXIS_PAD + (maxLevel + 1) * STEP + 8;
  const active = hover === null ? null : markers[hover];

  return (
    <div className="fx not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{title}</span>
        <span className="font-mono text-[0.65rem] text-muted">{markers.length} links</span>
      </div>

      <div className="px-4 pt-5 pb-4 sm:px-6">
        <div className="relative" style={{ height: plotHeight + 22 }}>
          {/* axis */}
          <div
            className="absolute right-0 left-0 border-t border-line-strong"
            style={{ top: plotHeight }}
          />
          {ticks.map((t, i) => (
            <div
              key={t.iso}
              className={`absolute -translate-x-1/2 font-mono text-[0.62rem] text-muted ${
                i % 2 === 1 ? 'hidden sm:block' : ''
              }`}
              style={{ left: `${t.x * 100}%`, top: plotHeight }}
            >
              <div className="mx-auto h-1.5 w-px bg-line-strong" />
              <div className="mt-0.5 whitespace-nowrap">{formatShortDate(t.iso)}</div>
            </div>
          ))}

          {/* markers */}
          {markers.map((m, i) => {
            const dim = focusCat !== null && focusCat !== m.catIndex;
            return (
              <button
                key={itemId(m.item)}
                type="button"
                aria-label={`${m.item.title}, ${formatItemDate(m.item.date)}. Jump to it in the list.`}
                aria-describedby={hover === i ? 'cur-tip' : undefined}
                onMouseEnter={() => show(i)}
                onMouseLeave={hideSoon}
                onFocus={() => show(i)}
                onBlur={hideSoon}
                onClick={() => jumpTo(m)}
                className="absolute rounded-full border transition-[transform,opacity] duration-150 hover:scale-125 focus-visible:scale-125"
                style={{
                  width: DOT,
                  height: DOT,
                  left: `calc(${m.x * 100}% - ${DOT / 2}px)`,
                  top: plotHeight - AXIS_PAD - m.level * STEP - DOT / 2,
                  background: categoryColor(m.catIndex),
                  borderColor: hover === i ? 'var(--ink)' : 'var(--paper-raised)',
                  opacity: dim ? 0.2 : 1,
                }}
              />
            );
          })}

          {/* tooltip */}
          {active && (
            <div
              id="cur-tip"
              role="tooltip"
              onMouseEnter={() => show(hover!)}
              onMouseLeave={hideSoon}
              className="absolute z-20 w-60"
              style={{
                bottom: `calc(100% - ${plotHeight - AXIS_PAD - active.level * STEP - DOT / 2 - 10}px)`,
                ...tooltipX(active.x),
              }}
            >
              {/* inner box animates; the outer one owns the centring transform */}
              <div className="fx-fly-in border border-line bg-paper px-3 py-2 text-xs leading-snug shadow-sm">
                <div className="mb-1 flex items-center gap-1.5 font-mono text-[0.62rem] text-muted">
                  <span
                    aria-hidden
                    className="inline-block h-2 w-2"
                    style={{ background: categoryColor(active.catIndex) }}
                  />
                  {active.category} · {formatItemDate(active.item.date)}
                </div>
                <a
                  href={active.item.link}
                  className="font-medium text-ink underline decoration-line-strong underline-offset-2 hover:text-accent hover:decoration-accent"
                >
                  {active.item.title}
                </a>
                <div className="mt-1 font-mono text-[0.6rem] text-muted">
                  click the marker to read the note
                </div>
              </div>
            </div>
          )}
        </div>

        {/* legend: hover a category to pick out its markers */}
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
          {data.map((g, i) => (
            <li key={g.category}>
              <span
                onMouseEnter={() => setFocusCat(i)}
                onMouseLeave={() => setFocusCat(null)}
                className="inline-flex cursor-default items-center gap-1.5 font-mono text-[0.7rem] text-ink-soft"
              >
                <span
                  aria-hidden
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ background: categoryColor(i) }}
                />
                {g.category} <span className="text-muted">{g.items.length}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Places every item on the axis and stacks same-day items. */
function layout(data: Current, from?: string, to?: string) {
  const all = data.flatMap((g, catIndex) =>
    g.items.map((item) => ({ item, category: g.category, catIndex, day: dayNumber(item.date) })),
  );
  const days = all.map((a) => a.day);
  const start = from ? dayNumber(from) : Math.min(...days);
  const end = to ? dayNumber(to) : Math.max(...days);
  const span = Math.max(1, end - start);

  // Same day: order by category so stacks read the same way as the legend.
  all.sort((a, b) => a.day - b.day || a.catIndex - b.catIndex);
  const perDay = new Map<number, number>();
  const markers: Marker[] = all.map(({ item, category, catIndex, day }) => {
    const clamped = Math.min(Math.max(day, start), end);
    const level = perDay.get(clamped) ?? 0;
    perDay.set(clamped, level + 1);
    return { item, category, catIndex, x: (clamped - start) / span, level };
  });
  const maxLevel = Math.max(0, ...markers.map((m) => m.level));
  return { markers, start, span, maxLevel };
}

/** One tick per week from the axis start, plus the last day. */
function weekTicks(start: number, span: number) {
  const ticks = [];
  for (let d = 0; d <= span; d += 7) ticks.push({ iso: dayToIso(start + d), x: d / span });
  return ticks;
}

/** Keeps the tooltip inside the frame near the axis ends. */
function tooltipX(x: number): React.CSSProperties {
  if (x < 0.2) return { left: `calc(${x * 100}% - 12px)` };
  if (x > 0.8) return { right: `calc(${(1 - x) * 100}% - 12px)` };
  return { left: `${x * 100}%`, transform: 'translateX(-50%)' };
}
