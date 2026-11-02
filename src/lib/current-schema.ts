/** Build-time validation for "The Current" JSON files (see `current.ts`). */
import { z } from 'astro/zod';
import type { Current } from './current';

const itemSchema = z.object({
  title: z.string(),
  // An absolute URL, or a path to a page on this site.
  link: z.union([z.url(), z.string().regex(/^\/[^/]/)]),
  description: z.union([z.string(), z.array(z.string()).min(1)]),
  date: z
    .string()
    .regex(/^\d{4}(-\d{2}(-\d{2})?)?$/)
    .optional(),
  source: z.string().optional(),
  author: z.string().optional(),
});

const schema = z.array(z.object({ category: z.string(), items: z.array(itemSchema) }));

/** Validates a month's JSON. Throws at build time on a malformed file. */
export function parseCurrent(data: unknown): Current {
  return schema.parse(data);
}
