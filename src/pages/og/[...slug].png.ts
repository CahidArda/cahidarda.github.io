import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { articleOgSvg, renderOgPng } from '../../lib/og';
import { tags, type TagKey } from '../../data/tags';

// One static OG image per local article: /og/<slug>.png
export async function getStaticPaths() {
  const entries = await getCollection('articles', ({ data }) => !data.draft && !data.externalUrl);
  return entries.map((entry) => ({
    params: { slug: entry.id },
    props: {
      title: entry.data.title,
      description: entry.data.description,
      date: entry.data.date,
      slug: entry.id,
      tag: entry.data.tags[0] as TagKey,
    },
  }));
}

export const GET: APIRoute = async ({ props }) => {
  const { title, description, date, slug, tag } = props as {
    title: string;
    description: string;
    date: Date;
    slug: string;
    tag: TagKey;
  };
  const png = await renderOgPng(
    articleOgSvg({ title, description, date, slug, kicker: tags[tag].label }),
  );
  return new Response(new Uint8Array(png), {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
};
