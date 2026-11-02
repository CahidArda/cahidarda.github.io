# CLAUDE.md

Personal website of Cahid Arda Öz. Editorial/brutalist Astro site that deploys static to
GitHub Pages. Content is a single `articles` collection; tags do the categorization.

## Stack

- **Astro 6** (static output), **TypeScript** (strict), **Tailwind v4** (via `@tailwindcss/vite`), **MDX**.
- **React islands only where there is real interactivity**, hydrated with `client:visible`.
- **pnpm** (`packageManager` pinned). Node ≥ 22.12.
- Path alias `@/*` → `./src/*`.

## Commands

```bash
pnpm dev       # local dev server
pnpm build     # production build → dist/ (run before every commit)
pnpm check     # astro check (types) — must be 0 errors
pnpm format    # prettier --write .
```

## Layout

- `src/content/articles/**` — posts (`.md` / `.mdx`). Schema in `src/content.config.ts`.
- `src/components/interactives/<topic>/` — React island widgets, one file per widget, plus a
  shared `shared.tsx` design module per topic.
- `src/styles/global.css` — design tokens (CSS vars, light/dark), Tailwind `@theme`, `.prose`
  typography, and per-topic widget styles.
- `src/layouts/`, `src/lib/og.ts` — article layout; per-slug OG images are generated at build.
- `src/data/current/YYYY-MM.json` — links for a monthly "The Current" post (tag `accessions`): a list
  of `{ category, items: [{ title, link, description, date?, source?, author? }] }` (`source` = company,
  shown as "Title (Anthropic)"; `author` = person, shown before the date; `date` may be YYYY-MM or
  YYYY, and only full dates inside the month go on the timeline; `description` may be a list of
  bullets; `link` may be a site path like `/articles/x`). Every Current post since May uses this. Render with
  `src/components/current/`: `CurrentGroup` (one category, put a `## Category` heading above it so
  it reaches the TOC), `CurrentList` (all categories), `CurrentTimeline` (React island, one swimlane per category, one marker per day with a count;
  hover lists the links, click jumps to the item). Types/helpers in `src/lib/current.ts`.

## Conventions

- Math is already wired (`remark-math` + `rehype-katex` in `astro.config.mjs`; KaTeX CSS imported
  once in `ArticleLayout.astro`). Don't re-add it.
- OG images auto-generate per slug (`/og/<slug>.png`). Don't hand-author them.
- Keep the editorial/brutalist look: reuse existing tokens, typography, and the tag system.
- **Article prose & widget text: no em-dashes** (use `.`, `:`, `,`, or parentheses), and do a quick
  signs-of-AI-writing pass. Cite quotes to the exact source document (link the PDF, not a landing page)
  and fact-check quotes/equations/benchmark numbers against primary sources before publishing. Reusable
  helpers: `EqTip.astro` (equation hover-tooltip), `TweetEmbed.astro`. See the skill for details.
- Don't commit/push to the default branch; branch first. Only open a PR when asked.

## Writing interactive articles, charts, and diagrams

This repo has hard-won lessons about KaTeX-in-MDX, SVG/canvas alignment, animation direction,
and keeping widgets consistent. **Before building a post with charts/diagrams/animations, read
the skill:**

→ `.claude/skills/interactive-articles/SKILL.md`
