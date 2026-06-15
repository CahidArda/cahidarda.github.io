import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

// Single-file sitemap served directly at /sitemap.xml (a urlset, not an index).
// We generate it ourselves instead of using @astrojs/sitemap so there's no
// sitemap-index.xml / sitemap-0.xml split and no /sitemap.xml -> index redirect.
export async function GET(context: APIContext) {
  const site = context.site ?? new URL('https://cahidarda.github.io');

  // Local article pages only - drafts and external (link-out) entries have no page.
  const articles = await getCollection(
    'articles',
    ({ data }) => !data.draft && !data.externalUrl,
  );

  // Static routes plus one page per article.
  const paths = ['/', '/articles/', ...articles.map((entry) => `/articles/${entry.id}/`)];

  const urls = paths
    .map((path) => `  <url><loc>${new URL(path, site).href}</loc></url>`)
    .join('\n');

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;

  return new Response(body, {
    headers: { 'Content-Type': 'application/xml' },
  });
}
