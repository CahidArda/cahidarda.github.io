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
  // LaTeX math via KaTeX. mdx() inherits this markdown config by default.
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [react(), mdx(), sitemap()],

  // @astrojs/sitemap emits the sitemap at /sitemap-index.xml (which references
  // /sitemap-0.xml). We don't redirect /sitemap.xml to it: on a static host
  // (GitHub Pages) an Astro redirect becomes an HTML meta-refresh page served
  // as `text/html`, which breaks validators. robots.txt points at the real
  // /sitemap-index.xml instead.

  vite: {
    plugins: [tailwindcss()],
  },
});
