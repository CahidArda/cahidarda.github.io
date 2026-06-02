/**
 * Single source of truth for the persistent profile area (sidebar / mobile header).
 * `intro` drives the landing page lede.
 */
export const profile = {
  name: 'Cahid Arda Öz',
  tagline: 'DX - Software Engineer at Upstash',
  // The employer mention in `tagline` is rendered as a link in the sidebar.
  employer: { name: 'Upstash', href: 'https://upstash.com' },
  location: 'Istanbul, Turkey',
  // Shown next to the Publications tag on publication pages (not in the sidebar).
  orcid: 'https://orcid.org/0009-0001-6049-3869',
  // Landing-page intro. TODO(owner): expand / rewrite in your own voice.
  intro:
    'Software engineer at Upstash, Boğaziçi University graduate. I write about ' +
    'developer tooling, the occasional paper, and Redis - plus a monthly ' +
    'roundup of things I enjoyed reading.',
} as const;

export type Profile = typeof profile;
