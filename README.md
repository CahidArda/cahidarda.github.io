# cahidarda.github.io

Personal website of Cahid Arda Öz - built with [Astro](https://astro.build), TypeScript, Tailwind CSS v4, and MDX. Static output deployed to GitHub Pages.

## Commands

| Command            | Action                                    |
| :----------------- | :---------------------------------------- |
| `pnpm install`     | Install dependencies                      |
| `pnpm dev`         | Start dev server at `localhost:4321`      |
| `pnpm build`       | Build to `./dist/`                        |
| `pnpm preview`     | Preview the build locally                 |
| `pnpm check`       | Type-check with `astro check`             |
| `pnpm format`      | Format with Prettier                      |

## Structure

- `src/data/` - single sources of truth (`profile`, `socials`, `tags`)
- `src/content/articles/` - the one content collection (blog, publications, accessions, external links)
- `src/components/`, `src/layouts/` - Astro UI
- `src/pages/` - routes
- `src/styles/global.css` - Tailwind v4 theme tokens

See `spec.md` for the full build specification.
