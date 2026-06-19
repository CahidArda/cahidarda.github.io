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
  },
  integrations: [react(), mdx(), sitemap()],

  redirects: {
    // @astrojs/sitemap emits /sitemap-index.xml; redirect the conventional URL to it.
    '/sitemap.xml': '/sitemap-index.xml',
    // Article renamed from "primer" to "guide"; keep the old link working.
    '/articles/vercel-eve-primer': '/articles/vercel-eve-guide',
  },

  vite: {
    plugins: [tailwindcss()],
  },
});
