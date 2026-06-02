import { getCollection } from 'astro:content';
import type { ImageMetadata } from 'astro';
import type { TagKey } from '../data/tags';
import { externalPosts } from '../data/external-posts';
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
  const articles = await getCollection('articles', ({ data }) => !data.draft);
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

  // External posts (Upstash, etc.) - link-out cards, maintained in one data file.
  const externalEntries: ListEntry[] = externalPosts.map((post) => ({
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
