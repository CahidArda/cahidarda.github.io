import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const articles = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/articles' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      date: z.coerce.date(),
      updated: z.coerce.date().optional(),
      tags: z.array(z.enum(['blog', 'publication', 'accessions'])).default(['blog']),
      draft: z.boolean().default(false),
      heroImage: image().optional(), // optimized via astro:assets

      // external posts (e.g. Upstash) - card links out, no local page
      externalUrl: z.string().url().optional(),
      source: z.string().optional(), // e.g. "Upstash"

      // publication-only metadata
      venue: z.string().optional(),
      authors: z.array(z.string()).optional(),
      doi: z.string().optional(),
      pdf: z.string().optional(),
    }),
});

export const collections = { articles };
