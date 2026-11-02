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
  /** Short summary. Supports `[text](url)` links and `code` spans. */
  description: string;
  /** ISO date (YYYY-MM-DD): when the linked piece was published. */
  date: string;
  /** Optional byline shown before the date, e.g. "Kevin Kelly". */
  source?: string;
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

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "2026-10-21" -> "21 Oct 2026". Parsed by hand so no timezone can shift the day. */
export function formatItemDate(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
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

/** Renders a description's `[text](url)` links and `code` spans to HTML; escapes the rest. */
export function descriptionHtml(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
}
