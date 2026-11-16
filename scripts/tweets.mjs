// Renders every X post embedded with <TweetEmbed url="..."> to static images.
//
// For each post it writes a light and a dark WebP to public/tweets/ and a small
// metadata file (author, text, size) to src/data/tweets/. Both are committed, so
// `pnpm build` never talks to X. Posts that already have metadata are skipped.
//
//   pnpm tweets            fetch and render posts that are new
//   pnpm tweets --refresh  fetch and render every post again (fresh counts)
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

const refresh = process.argv.includes('--refresh');

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
  return fetch(input, { ...init, headers });
};

const xeet = await loadXeet();
const host = xeet.whatwgHost({ fetch: namedFetch });

await mkdir(DATA, { recursive: true });
await mkdir(IMAGES, { recursive: true });

let failed = 0;
for (const [id, url] of await findPosts()) {
  const metaPath = path.join(DATA, `${id}.json`);
  if (!refresh && existsSync(metaPath)) continue;

  try {
    const { post, warnings } = await xeet.resolve(url, { media: true }, host);
    let size;
    for (const theme of ['light', 'dark']) {
      const { png, report } = xeet.render(post, {
        surface: 'web-timeline',
        width: WIDTH,
        theme,
        dpr: DPR,
      });
      size = report.size;
      await sharp(png)
        .webp({ quality: 90 })
        .toFile(path.join(IMAGES, `${id}-${theme}.webp`));
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
    await writeFile(metaPath, JSON.stringify(meta, null, 2) + '\n');
    console.log(`rendered ${id} (@${meta.handle})`);
    for (const w of warnings) console.warn(`  warning: ${w}`);
  } catch (error) {
    failed++;
    console.error(`failed ${id} ${url}: ${error.message}`);
  }
}

if (failed) process.exitCode = 1;
