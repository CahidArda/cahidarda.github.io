// Renders every X post embedded with <TweetEmbed url="..."> to static images.
//
// Runs before every `pnpm build` and `pnpm dev`. For each post it writes a light
// and a dark WebP to public/tweets/ and a small metadata file (author, text,
// size) to src/data/tweets/. Neither folder is committed: CI keeps them between
// deploys with actions/cache, and locally they stay on disk.
//
// - A post with no render yet is always fetched.
// - In CI (CI=true) or with --refresh, every post is fetched again, so the
//   counts stay current.
// - A failed fetch never fails the build. The last good render is kept, and a
//   post that was never rendered falls back to a plain quote in TweetEmbed.
//
// Rendering is done by @xeet/core, which rebuilds X's post layout in WebAssembly.
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { loadXeet } from '@xeet/core/node';

const ROOT = path.resolve(import.meta.dirname, '..');
const CONTENT = path.join(ROOT, 'src/content');
const DATA = path.join(ROOT, 'src/data/tweets');
const IMAGES = path.join(ROOT, 'public/tweets');

// Width of the rendered post in CSS pixels; images are 2x that for sharp text.
const WIDTH = 550;
const DPR = 2;
const USER_AGENT = 'cahidarda.com-build (+https://cahidarda.com)';
// Give up on a single request after this long, so a slow X can't stall a deploy.
const REQUEST_TIMEOUT_MS = 15_000;

const refresh = process.argv.includes('--refresh') || process.env.CI === 'true';

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (/\.mdx?$/.test(entry.name)) yield full;
  }
}

/** Every post URL used in a <TweetEmbed>, keyed by post id. */
async function findPosts() {
  const posts = new Map();
  const pattern = /<TweetEmbed[^>]*?url="(https:\/\/(?:x|twitter)\.com\/[^/]+\/status\/(\d+))"/gs;
  for await (const file of walk(CONTENT)) {
    for (const [, url, id] of (await readFile(file, 'utf8')).matchAll(pattern)) {
      posts.set(id, url);
    }
  }
  return posts;
}

// Send our own name instead of xeet's default browser user agent.
const namedFetch = (input, init = {}) => {
  const headers = new Headers(init.headers);
  headers.set('user-agent', USER_AGENT);
  return fetch(input, {
    ...init,
    headers,
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
};

async function render(xeet, host, id, url) {
  const { post, warnings } = await xeet.resolve(url, { media: true }, host);

  // Render both themes in memory first, so a failure leaves the old files alone.
  const images = {};
  let size;
  for (const theme of ['light', 'dark']) {
    const { png, report } = xeet.render(post, {
      surface: 'web-timeline',
      width: WIDTH,
      theme,
      dpr: DPR,
    });
    size = report.size;
    images[theme] = await sharp(png).webp({ quality: 90 }).toBuffer();
  }

  for (const [theme, webp] of Object.entries(images)) {
    await writeFile(path.join(IMAGES, `${id}-${theme}.webp`), webp);
  }
  const meta = {
    id,
    url,
    name: post.author?.name ?? '',
    handle: post.author?.handle ?? '',
    text: post.text,
    createdAt: post.createdAt ?? null,
    width: size.width,
    height: size.height,
    renderedAt: new Date().toISOString(),
  };
  await writeFile(path.join(DATA, `${id}.json`), JSON.stringify(meta, null, 2) + '\n');
  return { meta, warnings };
}

await mkdir(DATA, { recursive: true });
await mkdir(IMAGES, { recursive: true });

const posts = await findPosts();
let rendered = 0;
let failed = 0;
const todo = [...posts].filter(([id]) => refresh || !existsSync(path.join(DATA, `${id}.json`)));

if (todo.length) {
  const xeet = await loadXeet();
  const host = xeet.whatwgHost({ fetch: namedFetch });

  for (const [id, url] of todo) {
    const hadRender = existsSync(path.join(DATA, `${id}.json`));
    try {
      const { meta, warnings } = await render(xeet, host, id, url);
      rendered++;
      console.log(`[tweets] rendered ${id} (@${meta.handle})`);
      for (const w of warnings) console.warn(`[tweets]   ${w}`);
    } catch (error) {
      failed++;
      const kept = hadRender ? 'keeping the last render' : 'will show a plain quote';
      console.warn(`[tweets] could not fetch ${url} (${error.message}); ${kept}`);
    }
  }
}

console.log(
  `[tweets] ${posts.size} posts: ${rendered} rendered, ${failed} failed, ${posts.size - todo.length} already rendered`,
);
