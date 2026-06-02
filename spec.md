# Build Spec - Personal Website Rebuild (Astro)

**Owner:** Cahid Arda Öz · **Repo:** `cahidarda.github.io` (GitHub user page)
**For:** an autonomous coding agent. Execute the stages in order. Each stage has acceptance criteria; do not advance until they pass (`npm run build` must succeed at every stage boundary).

---

## 0. Goals & constraints

Rebuild the existing Jekyll/AcademicPages site as a modern, fast, content-first Astro site.

- **Stack:** Astro 6, TypeScript (strict), Tailwind CSS v4, MDX. React islands **only** where genuine interactivity is required.
- **Hosting:** static output → GitHub Pages via the official Astro Action. (No SSR/middleware; content negotiation is out of scope for now.)
- **Design north star:** the Hermes Agent site - editorial/brutalist. Bold display serif, letter-spaced uppercase labels, hairline-bordered grid, monospace accents, generous whitespace, restrained palette. Take it as *inspiration*, produce something original; do not copy assets or copy.
- **Layout rule (every page):** a persistent profile area showing the personal image + social links, alongside the page content. Sidebar on desktop, collapsing to a header on mobile.

### Inputs the owner must provide (use clearly-marked `TODO` placeholders if absent; do not invent)
- Bio / intro copy for the landing page.
- Old site source repo URL (for Stage 9). Assume `https://github.com/CahidArda/cahidarda.github.io` unless told otherwise.
- Optional `GITHUB_TOKEN` repo secret (for the GitHub integration in Stage 8).

---

## 1. Stage 0 - Project setup

Assume the scaffold already exists (created with `npm create astro@latest`, integrations `react`, `tailwind`, `mdx` added, plus `@astrojs/rss` and `@astrojs/sitemap` installed). If not, do that first.

- Configure `astro.config.mjs`: `site: 'https://cahidarda.github.io'`. No `base` (user pages serve from root). Register integrations: `react`, `mdx`, `sitemap`. Tailwind v4 is wired via the `@tailwindcss/vite` plugin.
- Add the `@/*` path alias in `tsconfig.json` → `"paths": { "@/*": ["./src/*"] }`.
- Create `src/styles/global.css` with `@import "tailwindcss";` and import it from the base layout.
- Add Prettier + `prettier-plugin-astro`.

**Acceptance:** `npm run dev` serves a blank page with Tailwind classes working; `npm run build` succeeds.

---

## 2. Information architecture

| Route | Purpose |
|---|---|
| `/` | Intro + most recent articles (all tags) |
| `/articles` | All articles, with client-side tag filter |
| `/articles/[...slug]` | Article detail (local entries only) |
| `/tags/[tag]` | Articles for one tag. `/tags/publication` is the de-facto "papers" page |
| `/rss.xml` | Feed of local articles |
| `/404` | Not found |

External entries (e.g. Upstash posts) appear in listings but link out - they get **no** `/articles/[slug]` page.

---

## 3. Data layer (single sources of truth)

Create plain TS modules under `src/data/`:

- `profile.ts` - `{ name, tagline, location, imagePath, intro }`.
- `socials.ts` - array of `{ label, href, icon }`. Seed from the current site: Email (`mailto:`), X/Twitter, LinkedIn, GitHub, ORCID, Wikipedia, RSS (`/rss.xml`).
- `tags.ts` - the tag registry that drives badges, tooltips, and tag-page headers:

```ts
// src/data/tags.ts
export const tags = {
  blog:        { label: 'Blog',        blurb: 'Longer-form writing.' },
  publication: { label: 'Publication', blurb: 'Peer-reviewed papers and academic work.' },
  accessions:  { label: 'Accessions',  blurb: 'Monthly roundup of things I read, with notes.' },
} as const;

export type TagKey = keyof typeof tags;
```

---

## 4. Content collections (Astro 6 API)

One collection, `articles`. Tags do the categorization; optional fields carry publication/external metadata.

```ts
// src/content.config.ts
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
      heroImage: image().optional(),          // optimized via astro:assets

      // external posts (e.g. Upstash) - card links out, no local page
      externalUrl: z.string().url().optional(),
      source: z.string().optional(),           // e.g. "Upstash"

      // publication-only metadata
      venue: z.string().optional(),
      authors: z.array(z.string()).optional(),
      doi: z.string().optional(),
      pdf: z.string().optional(),
    }),
});

export const collections = { articles };
```

Notes for the agent: in Astro 6 the config file is `src/content.config.ts`; the entry slug is `entry.id` (not `.slug`); render with `const { Content } = await render(entry)`. Use `getCollection('articles', ({ data }) => !data.draft)` to filter drafts in production.

---

## 5. Layout & components

- `src/layouts/BaseLayout.astro` - `<html>` shell, `<head>` (meta, OG, canonical, RSS `<link>`), imports `global.css`. Renders `ProfileSidebar` + `<slot />` in a two-column grid (sidebar fixed/left on `md+`, stacked header on mobile). This is the persistent profile + socials frame.
- `src/layouts/ArticleLayout.astro` - wraps `BaseLayout`, adds title/date/tags header and a `CitationBlock` when publication metadata exists.
- Components (`src/components/`):
  - `ProfileSidebar.astro` - personal image (`astro:assets <Image>`), name, tagline, location, `SocialLinks`.
  - `SocialLinks.astro` - maps `socials.ts`.
  - `ArticleCard.astro` - handles **both** local and external entries. If `externalUrl` is set, the card links out and shows the `source`; otherwise links to `/articles/[id]`. Shows `TagBadge`s.
  - `TagBadge.astro` - label + tooltip from `tags.ts` `blurb`. Tooltip is **CSS-only** (`:hover`/`:focus-within`, accessible, zero JS).
  - `CitationBlock.astro` - renders `authors`, `venue`, `date`, `doi`/`pdf` links for publications.
  - `Prose.astro` - typographic wrapper for rendered markdown.

**Acceptance:** every page renders the sidebar with image + socials; tag badges show tooltips on hover and keyboard focus.

---

## 6. Pages

- `index.astro` - `profile.intro` + the N most recent non-draft articles (sorted by `date` desc), via `ArticleCard`.
- `articles/index.astro` - full list + a `TagFilter` island.
- `articles/[...slug].astro` - `getStaticPaths` over local entries (exclude `externalUrl` and `draft`), uses `ArticleLayout`.
- `tags/[tag].astro` - `getStaticPaths` from `tags.ts` keys; header pulls `label` + `blurb`; lists matching articles.
- `rss.xml.ts` - `@astrojs/rss`, local articles only.
- `404.astro`.

---

## 7. React islands (keep minimal)

Only these, each hydrated with the narrowest directive:

- `CommandPalette.tsx` - ⌘K / Ctrl-K to jump to articles & tags. `client:idle`.
- `TagFilter.tsx` - filters the articles index in place. `client:visible`.
- `CopyButton.tsx` - copy code blocks / email. `client:visible`.
- (Optional) `ThemeToggle.tsx` - light/dark. `client:load`.

Everything else stays static `.astro`. If a future component needs shadcn/ui, isolate the whole interactive unit inside one `.tsx` island (Astro islands don't share React context across separate calls).

---

## 8. GitHub integration (optional, build-time)

Add a custom content loader or a `src/lib/github.ts` fetched in page frontmatter at build time:
- Repos: `https://api.github.com/users/CahidArda/repos?sort=updated&per_page=6`.
- Render a "Projects" section (landing or sidebar) from the result. Use `.optional()` liberally - many repos lack a description.
- Unauthenticated = 60 req/hr (fine for builds). If `GITHUB_TOKEN` is present, use it for higher limits / pinned repos (GraphQL). Skip the *activity/events* feed by default - it looks stale between deploys. If wanted, add a nightly `cron` GitHub Action to trigger rebuilds.

**Acceptance:** build works both with and without a token; missing fields never crash the build.

---

## 9. Styling system (Hermes-inspired)

- Tailwind v4 theme tokens in `global.css` via `@theme`: a tight neutral palette + one ink/accent color, a display serif, a body face, and a mono face.
- Fonts: load via Fontsource or local `woff2`. Suggested directions (owner picks): a strong display serif (e.g. Fraunces / a Didone) for headings, a clean grotesk or the system stack for body, a mono (e.g. JetBrains Mono) for code and labels. Use letter-spaced uppercase for nav/section labels.
- Visual motifs to capture the reference: hairline (1px) borders forming a grid, uppercase tracked labels, a mono "terminal"-style block for any code/CLI snippet, lots of negative space. Do **not** reproduce Hermes's exact colors/textures - derive an original palette.

---

## 10. Mock content (do this BEFORE real content)

Generate enough fake entries to exercise every code path, in `src/content/articles/`:

- **3 blog posts** - `tags: ['blog']`, varied dates, one with a `heroImage`, lorem body.
- **2 publications** - `tags: ['publication']`, with `venue`, `authors`, `date`, and one with `doi`, one with `pdf`. Lorem abstract as body.
- **1 accessions post** - `tags: ['accessions']`, title **"The Current - June 2026"**, body = a few fake links with one-line commentary each (mirrors the intended monthly format).
- **1 external entry** - `tags: ['blog']`, `externalUrl` + `source: 'Upstash'`, **no body** (verifies link-out cards).

**Acceptance:** landing shows recent mix; `/tags/publication` renders citations; `/tags/accessions` renders the roundup; the external card links out with no detail page; RSS validates; `npm run build` clean.

---

## 11. Deploy (GitHub Pages)

- Add `.github/workflows/deploy.yml` using `withastro/action` + `actions/deploy-pages`.
- Confirm `site` is set; user page → no `base`.
- Enable Pages → "GitHub Actions" source.

**Acceptance:** push to `main` deploys; site loads at `https://cahidarda.github.io` with working nav, RSS, and sitemap.

---

## 12. FINAL STAGE - Migrate real content from the current site

The current site is Jekyll + AcademicPages (a Minimal Mistakes fork). Goal: move publications (and any blog posts/portfolio) + images into the new `articles` collection, then retire the mock content.

### 12.1 Inventory
Clone the old repo into a scratch dir (do **not** nest it in the new project). AcademicPages stores content as Markdown in collection folders:
- `_publications/` → papers (map to `tags: ['publication']`)
- `_posts/` → blog posts (map to `tags: ['blog']`)
- `_portfolio/`, `_talks/`, `_teaching/` → review; migrate to `blog` or drop, owner's call
- `images/` (and sometimes `assets/`) → image files, including `images/profile.png`

List every content file and every image actually referenced before transforming anything.

### 12.2 Front-matter mapping
For each old file, translate Jekyll front matter → the new schema:

| AcademicPages field | New field |
|---|---|
| `title` | `title` |
| `date` | `date` |
| `excerpt` / `description` | `description` |
| `tags` / `categories` | fold into the `tags` enum (don't pass arbitrary tags - coerce to `blog`/`publication`/`accessions`) |
| `venue` | `venue` (publications) |
| `paperurl` / `paper_url` | `pdf` |
| `citation` | parse `authors` from it; capture `doi` if present |
| body | keep, after cleanup (12.3) |

Set `draft: false` once validated. Filename → slug becomes `entry.id`; keep slugs stable where they matter for inbound links (preserve old publication permalinks if any are linked externally - add redirects if they change).

### 12.3 Body cleanup
- Remove Jekyll/Liquid syntax: `{% include ... %}`, `{{ site.* }}`, `{% link %}`, etc. Replace with plain Markdown or Astro equivalents.
- Convert image references (12.4).
- Verify Markdown renders; fix any front-matter-only entries (publications with no body get a short abstract or none).

### 12.4 Images
- Move referenced images out of the old `images/` into the new project. Two patterns - pick per use:
  - **Co-locate** an article's images in `src/content/articles/<slug>/` and reference them with **relative paths** in the Markdown; with the `image()` schema helper + `astro:assets` they get optimized.
  - Shared/non-content images → `src/assets/` (imported & optimized) or `public/` (served as-is, e.g. PDFs, favicons).
- `images/profile.png` → wire into `profile.ts` (via `src/assets` so `<Image>` optimizes it).
- Rewrite every old image URL (`/images/...`) to the new location. Confirm no broken references remain.
- Copy non-image assets (paper PDFs) referenced by `pdf:` into `public/papers/` and point `pdf` at them, or keep external URLs as-is.

### 12.5 Swap & verify
- Delete the mock entries from Stage 10 (keep **one** real or freshly-written "The Current - June 2026" accessions post, or leave the mock until the owner writes the first real one - flag this for the owner).
- Add the **Upstash external entries** as `externalUrl` stubs (no body), `source: 'Upstash'`:
  - Fast, Cost-Effective MCPs with Redis - 2025-10-05 - `https://upstash.com/blog/mcp-with-redis`
  - Storing Time Series Data in Redis - 2025-09-03 - `https://upstash.com/blog/redis-timeseries`
  - Caching Drizzle Queries with Upstash Redis - 2025-07-10 - `https://upstash.com/blog/drizzle-integration`
  - Introducing Workflow Agents - 2025-02-17 - `https://upstash.com/blog/workflow-agents`
  - Four Ways to Reduce Your Vercel Serverless Costs - 2024-10-07 - `https://upstash.com/blog/vercel-cost`
  - Upstash Ratelimit in LangChain - 2024-06-10 - `https://upstash.com/blog/ratelimit-langchain`
  - DegreeGuru: Build a RAG Chatbot - 2024-03-05 - `https://upstash.com/blog/degree-guru`

**Acceptance (definition of done):** all real publications appear under `/tags/publication` with correct citations and working PDF/DOI links; any blog posts migrated; all images render and are optimized; Upstash entries link out; no mock content remains; RSS + sitemap valid; `npm run build` clean; deployed site verified.

---

## Suggested commit/PR sequence
Stages 1–7 (skeleton + components + islands) → Stage 10 (mock content, full visual pass) → Stage 9 (styling polish) → Stage 8 (GitHub) → Stage 11 (deploy) → Stage 12 (migration). Open a PR per stage so the owner can review the design before real content lands.