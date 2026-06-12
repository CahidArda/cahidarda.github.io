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

  // @astrojs/sitemap emits /sitemap-index.xml; redirect the conventional URL to it.
  redirects: {
    '/sitemap.xml': '/sitemap-index.xml',
  },

  vite: {
    plugins: [tailwindcss()],
  },
});
