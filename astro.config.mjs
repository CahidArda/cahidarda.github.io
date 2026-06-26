// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  site: 'https://cahidarda.github.io',
  // Canonical URLs without trailing slashes. `format: 'file'` emits
  // `articles/foo.html` (served at `/articles/foo` on GitHub Pages) instead of
  // `articles/foo/index.html` (served at `/articles/foo/`).
  trailingSlash: 'never',
  build: { format: 'file' },
  // LaTeX math via KaTeX. mdx() inherits this markdown config by default.
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    // Dual Shiki themes so code follows the site's light/dark mode. The default
    // (light) token colours are applied inline; global.css swaps to the dark
    // ones under `.dark` and keeps the editorial paper background in both.
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
    },
  },
  integrations: [react(), mdx(), sitemap()],

  redirects: {
    // @astrojs/sitemap emits /sitemap-index.xml; redirect the conventional URL to it.
    '/sitemap.xml': '/sitemap-index.xml',
    // Article renamed from "primer" to "guide"; keep the old link working.
    // (The trailing-slash variant is handled by a static stub in
    // public/articles/vercel-eve-primer/index.html — `trailingSlash: 'never'`
    // makes Astro strip trailing slashes from redirect sources.)
    '/articles/vercel-eve-primer': '/articles/vercel-eve-guide',
  },

  vite: {
    plugins: [tailwindcss()],
  },
});
