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

## Conventions

- Math is already wired (`remark-math` + `rehype-katex` in `astro.config.mjs`; KaTeX CSS imported
  once in `ArticleLayout.astro`). Don't re-add it.
- OG images auto-generate per slug (`/og/<slug>.png`). Don't hand-author them.
- Keep the editorial/brutalist look: reuse existing tokens, typography, and the tag system.
- Don't commit/push to the default branch; branch first. Only open a PR when asked.

## Writing interactive articles, charts, and diagrams

This repo has hard-won lessons about KaTeX-in-MDX, SVG/canvas alignment, animation direction,
and keeping widgets consistent. **Before building a post with charts/diagrams/animations, read
the skill:**

→ `.claude/skills/interactive-articles/SKILL.md`
