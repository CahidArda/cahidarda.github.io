/** Build-time validation for "The Current" JSON files (see `current.ts`). */
import { z } from 'astro/zod';
import type { Current } from './current';

const itemSchema = z.object({
  title: z.string(),
  link: z.url(),
  description: z.string(),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  source: z.string().optional(),
  author: z.string().optional(),
});

const schema = z.array(z.object({ category: z.string(), items: z.array(itemSchema) }));

/** Validates a month's JSON. Throws at build time on a malformed file. */
export function parseCurrent(data: unknown): Current {
  return schema.parse(data);
}
