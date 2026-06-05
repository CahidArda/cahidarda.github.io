import type { ExternalPost } from '../data/external-posts';
import { externalPosts as fallbackPosts } from '../data/external-posts';

/**
 * Build-time fetch of my Upstash author feed. This is the source of truth for the
 * Upstash link-out cards: publish a post on Upstash and it shows up here on the next
 * build, no edit needed. Every failure mode (offline, non-200, unexpected/empty XML)
 * resolves to the committed `externalPosts` list so a build never breaks - and a feed
 * outage can't blank the section.
 */

const FEED_URL = 'https://upstash.com/blog/author/arda/feed.xml';
const SOURCE = 'Upstash';

// Fetch once per build and reuse - the index and command palette both ask for entries.
let cache: Promise<ExternalPost[]> | null = null;

// Minimal entity decode. Fields are CDATA-wrapped (so usually literal), but decode the
// common five in case a field is ever emitted as plain escaped text.
function decodeEntities(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#3?9;|&#x27;/gi, "'")
    .replace(/&amp;/g, '&');
}

// Pull one tag's text out of an <item> block, unwrapping CDATA. Returns '' if absent.
function tag(item: string, name: string): string {
  const m = item.match(new RegExp(`<${name}[^>]*>([\\s\\S]*?)</${name}>`));
  if (!m) return '';
  const inner = m[1].trim();
  const cdata = inner.match(/^<!\[CDATA\[([\s\S]*?)\]\]>$/);
  return decodeEntities((cdata ? cdata[1] : inner).trim());
}

function parseFeed(xml: string): ExternalPost[] {
  // Drop the heavy HTML bodies first so per-item field extraction stays unambiguous.
  const cleaned = xml.replace(/<content:encoded>[\s\S]*?<\/content:encoded>/g, '');

  const posts: ExternalPost[] = [];
  for (const m of cleaned.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const item = m[1];
    const title = tag(item, 'title');
    const url = tag(item, 'link');
    const pubDate = tag(item, 'pubDate');
    const description = tag(item, 'description');

    const date = new Date(pubDate);
    if (!title || !url || Number.isNaN(date.valueOf())) continue;

    posts.push({
      title,
      description,
      date: date.toISOString().slice(0, 10), // YYYY-MM-DD
      url,
      source: SOURCE,
    });
  }

  return posts.sort((a, b) => b.date.localeCompare(a.date));
}

async function fetchExternalPosts(): Promise<ExternalPost[]> {
  try {
    const res = await fetch(FEED_URL, {
      headers: { 'User-Agent': 'cahidarda.github.io-build', Accept: 'application/rss+xml' },
    });
    if (!res.ok) {
      console.warn(`[upstash-feed] fetch failed: ${res.status} ${res.statusText}; using fallback`);
      return fallbackPosts;
    }
    const posts = parseFeed(await res.text());
    if (posts.length === 0) {
      console.warn('[upstash-feed] parsed 0 items; using fallback');
      return fallbackPosts;
    }
    return posts;
  } catch (err) {
    console.warn(
      '[upstash-feed] fetch error:',
      err instanceof Error ? err.message : err,
      '; using fallback',
    );
    return fallbackPosts;
  }
}

export async function getExternalPosts(): Promise<ExternalPost[]> {
  cache ??= fetchExternalPosts();
  return cache;
}
