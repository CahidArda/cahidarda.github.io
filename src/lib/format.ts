const longFormatter = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
});

const monthYearFormatter = new Intl.DateTimeFormat('en-US', {
  year: 'numeric',
  month: 'long',
});

/** "June 1, 2026" */
export function formatDate(date: Date): string {
  return longFormatter.format(date);
}

/** "June 2026" - for publications and roundups. */
export function formatMonthYear(date: Date): string {
  return monthYearFormatter.format(date);
}

/** "2026-06-01" - machine-readable, for <time datetime>. */
export function isoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}
