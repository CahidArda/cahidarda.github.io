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
    site: context.site ?? 'https://cahidarda.github.io',
    items: articles.map((entry) => ({
      title: entry.data.title,
      description: entry.data.description,
      pubDate: entry.data.date,
      link: `/articles/${entry.id}/`,
      categories: entry.data.tags,
    })),
  });
}
