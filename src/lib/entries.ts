import { getCollection } from 'astro:content';
import type { ImageMetadata } from 'astro';
import type { TagKey } from '../data/tags';
import { getExternalPosts } from './upstash-feed';
import { getProjects } from './github';

/**
 * A normalized item for the unified content list - covers local articles,
 * external posts (Upstash), and GitHub repositories alike. ArticleCard renders this.
 */
export interface ListEntry {
  title: string;
  description: string;
  date: Date;
  tags: TagKey[];
  href: string;
  external: boolean;
  source?: string;
  heroImage?: ImageMetadata;
  stars?: number;
}

// Show all owned repos (forks/archived are already filtered out in getProjects).
const REPO_LIMIT = 100;

export async function getEntries(): Promise<ListEntry[]> {
  // Series member pages are hidden from the index; only the landing page (which
  // sets seriesLanding, not series) and standalone articles appear.
  const articles = await getCollection('articles', ({ data }) => !data.draft && !data.series);
  const articleEntries: ListEntry[] = articles.map((entry) => ({
    title: entry.data.title,
    description: entry.data.description,
    date: entry.data.date,
    tags: entry.data.tags,
    href: entry.data.externalUrl ?? `/articles/${entry.id}`,
    external: Boolean(entry.data.externalUrl),
    source: entry.data.source,
    heroImage: entry.data.heroImage,
  }));

  // External posts (Upstash) - link-out cards, sourced from my Upstash author feed
  // at build time (with a committed fallback list if the feed is unreachable).
  const externalEntries: ListEntry[] = (await getExternalPosts()).map((post) => ({
    title: post.title,
    description: post.description,
    date: new Date(post.date),
    tags: post.tags ?? ['blog'],
    href: post.url,
    external: true,
    source: post.source,
  }));

  // Repos are sorted by creation time; their entry date IS the creation date,
  // so they interleave with articles by recency in the unified list.
  const repoEntries: ListEntry[] = (await getProjects(REPO_LIMIT)).map((repo) => ({
    title: repo.name,
    description: repo.description ?? '',
    date: repo.createdAt,
    tags: ['repository'],
    href: repo.url,
    external: true,
    source: 'GitHub',
    stars: repo.stars,
  }));

  return [...articleEntries, ...externalEntries, ...repoEntries].sort(
    (a, b) => b.date.valueOf() - a.date.valueOf(),
  );
}

/** Blog posts without a `source` live on this site. */
export const SITE_SOURCE = 'site';

/** URL-safe key for a publication source ("Upstash" -> "upstash"). */
export const sourceKey = (source?: string): string =>
  source ? source.toLowerCase().replace(/[^a-z0-9]+/g, '-') : SITE_SOURCE;

export interface BlogSource {
  key: string;
  label: string;
  count: number;
}

/**
 * Where blog posts were published: this site first, then each external source by
 * post count. Drives the Blog source menu in the sidebar and on the Index.
 */
export async function getBlogSources(): Promise<BlogSource[]> {
  const blog = (await getEntries()).filter((e) => e.tags.includes('blog'));
  const external = [...new Set(blog.map((e) => e.source).filter((s): s is string => !!s))]
    .map((label) => ({
      key: sourceKey(label),
      label,
      count: blog.filter((e) => e.source === label).length,
    }))
    .sort((a, b) => b.count - a.count);
  const site = {
    key: SITE_SOURCE,
    label: 'This site',
    count: blog.filter((e) => !e.source).length,
  };
  return [site, ...external].filter((s) => s.count > 0);
}
