import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';
import { profile } from '../data/profile';

export async function GET(context: APIContext) {
  // Local articles only - external entries (Upstash, etc.) link out elsewhere.
  const articles = (
    await getCollection('articles', ({ data }) => !data.draft && !data.externalUrl)
  ).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

  return rss({
    title: `${profile.name} - Articles`,
    description: profile.intro,
    site: context.site ?? 'https://cahidarda.com',
    // @astrojs/rss appends a trailing slash to item links by default.
    trailingSlash: false,
    items: articles.map((entry) => ({
      title: entry.data.title,
      description: entry.data.description,
      pubDate: entry.data.date,
      // No trailing slash: `build.format: 'file'` emits `articles/foo.html`, so
      // `/articles/foo/` has no `index.html` to serve on GitHub Pages.
      link: `/articles/${entry.id}`,
      categories: entry.data.tags,
    })),
  });
}
