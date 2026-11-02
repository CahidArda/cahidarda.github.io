/**
 * "The Current" link posts. Each month's links live in a JSON file under
 * `src/data/current/` as a list of groups (a category title plus its items). The same data
 * feeds the static list components (`CurrentGroup`, `CurrentList`) and the timeline island
 * (`CurrentTimeline`), so the anchor ids and category colours below are shared by both.
 * The build-time validator lives in `current-schema.ts` so the island does not bundle zod.
 */
export interface CurrentItem {
  title: string;
  link: string;
  /**
   * Short summary. Supports `[text](url)` links and `code` spans. A list renders one bullet
   * per entry.
   */
  description: string | string[];
  /**
   * When the linked piece was published: YYYY-MM-DD, or YYYY-MM / YYYY when that is all that
   * is known. Only full dates go on the timeline.
   */
  date?: string;
  /** Publishing company or org, shown after the title: "Claude Haiku 5.5 (Anthropic)". */
  source?: string;
  /** Person who wrote it, shown before the date: "Nick Cammarata, 15 Oct 2026". */
  author?: string;
}

export interface CurrentGroup {
  category: string;
  items: CurrentItem[];
}

export type Current = CurrentGroup[];

/** Number of category colours defined in global.css (`--cur-1` .. `--cur-N`). */
const PALETTE_SIZE = 8;

/** One fixed colour per category, assigned by its position in the file. */
export function categoryColor(index: number): string {
  return `var(--cur-${(index % PALETTE_SIZE) + 1})`;
}

/** Lowercase, hyphenated, punctuation dropped. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/** Anchor id of an item in the rendered list; the timeline scrolls to it. */
export function itemId(item: CurrentItem): string {
  return `cur-${slugify(item.title)}`;
}

/** The "(Anthropic)" after a title, left out when the title already names the source. */
export function sourceSuffix(item: CurrentItem): string {
  const { source, title } = item;
  if (!source || title.toLowerCase().includes(source.toLowerCase())) return '';
  return ` (${source})`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * "2026-10-21" -> "21 Oct 2026", "2026-09" -> "Sep 2026", "2001" -> "2001". Parsed by hand
 * so no timezone can shift the day.
 */
export function formatItemDate(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number);
  if (!m) return `${y}`;
  if (!d) return `${MONTHS[m - 1]} ${y}`;
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

/** True for a full YYYY-MM-DD date, the only kind the timeline can place. */
export function hasDay(date: string | undefined): date is string {
  return !!date && /^\d{4}-\d{2}-\d{2}$/.test(date);
}

/** "Nick Cammarata, 15 Oct 2026": author and date, whichever exist. */
export function byline(item: CurrentItem): string {
  return [item.author, item.date && formatItemDate(item.date)].filter(Boolean).join(', ');
}

/** "2026-10-21" -> "Oct 21", for axis ticks. */
export function formatShortDate(iso: string): string {
  const [, m, d] = iso.split('-').map(Number);
  return `${MONTHS[m - 1]} ${d}`;
}

/** Whole days since the epoch for an ISO date (UTC), for laying items out on an axis. */
export function dayNumber(iso: string): number {
  return Math.round(Date.parse(`${iso}T00:00:00Z`) / 86_400_000);
}

/** Inverse of `dayNumber`. */
export function dayToIso(day: number): string {
  return new Date(day * 86_400_000).toISOString().slice(0, 10);
}

const escapeHtml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/**
 * The bold linked title plus its source, then the comma and italic byline when there is
 * one, as one HTML string. Rendered with `set:html` so formatting the markup can never put a
 * space before the comma.
 */
export function titleHtml(item: CurrentItem): string {
  const href = escapeHtml(item.link);
  const by = byline(item);
  return (
    `<strong><a href="${href}">${escapeHtml(item.title)}</a></strong>` +
    escapeHtml(sourceSuffix(item)) +
    (by ? `, <em>${escapeHtml(by)}</em>` : '')
  );
}

/** A description as a list of bullet texts. */
export function descriptionParts(item: CurrentItem): string[] {
  return typeof item.description === 'string' ? [item.description] : item.description;
}

/** Renders a description's `[text](url)` links and `code` spans to HTML; escapes the rest. */
export function descriptionHtml(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
}
