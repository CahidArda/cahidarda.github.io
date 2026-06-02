/**
 * The tag registry that drives badges, tooltips, and tag-page headers.
 * Keys here must stay in sync with the `tags` enum in src/content.config.ts.
 */
export const tags = {
  blog: { label: 'Blog', blurb: 'Essays, opinions, and how-tos.' },
  publication: { label: 'Publications', blurb: 'Peer-reviewed papers and academic work.' },
  accessions: { label: 'Accessions', blurb: 'Monthly roundup of things I enjoyed reading.' },
  repository: { label: 'Repositories', blurb: 'Open-source projects and experiments on GitHub.' },
} as const;

export type TagKey = keyof typeof tags;

export const tagKeys = Object.keys(tags) as TagKey[];

/** All tag links resolve to the single index page with the tag pre-selected. */
export const tagHref = (tag: TagKey): string => `/articles?tag=${tag}`;
