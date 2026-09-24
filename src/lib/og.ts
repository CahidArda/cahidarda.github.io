/**
 * Per-article Open Graph images (1200x630), in the same style as the site's share card
 * (public/og-card.png, source in scripts/og/og-card.html): dark plate with accent corner
 * ticks, a terminal prompt line as the kicker, the title in Fraunces, the description as
 * an italic dek, an accent rule, and a mono footer. Rendered with resvg and the vendored
 * fonts in scripts/og/fonts so the build machine's font set does not matter.
 */
import { Resvg } from '@resvg/resvg-js';
import { readFile } from 'node:fs/promises';

const PAPER = '#131210';
const INK = '#f1eee4';
const SOFT = '#cbc7ba';
const MUTED = '#908b7d';
const LINE = '#322f29';
const ACCENT = '#d4794f';
const SITE = 'cahidarda.com';

const FONT_FILES = [
  'scripts/og/fonts/Fraunces-Display.ttf',
  'scripts/og/fonts/Fraunces-DisplayItalic.ttf',
  'scripts/og/fonts/JetBrainsMono-Regular.ttf',
  'scripts/og/fonts/JetBrainsMono-Medium.ttf',
];

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Greedy word wrap with a hard line cap; the last line is ellipsized when it overflows. */
function wrap(text: string, maxChars: number, maxLines: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    const trial = cur ? `${cur} ${w}` : w;
    if (trial.length > maxChars && cur) {
      lines.push(cur);
      cur = w;
    } else {
      cur = trial;
    }
  }
  if (cur) lines.push(cur);
  if (lines.length > maxLines) {
    const head = lines.slice(0, maxLines);
    let last = lines.slice(maxLines - 1).join(' ');
    if (last.length > maxChars) last = `${last.slice(0, maxChars - 1).trimEnd()}…`;
    head[maxLines - 1] = last;
    return head;
  }
  return lines;
}

export interface OgArticle {
  title: string;
  /** One-sentence description; becomes the italic dek. */
  description?: string;
  /** Tag label, e.g. "Blog"; becomes the prompt path and the footer. */
  kicker: string;
  slug: string;
  date?: Date;
}

function formatDate(d: Date): string {
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

export function articleOgSvg(a: OgArticle): string {
  const title = a.title.replace(/`/g, '').trim();
  let fontSize: number;
  let maxChars: number;
  if (title.length <= 24) {
    fontSize = 84;
    maxChars = 22;
  } else if (title.length <= 52) {
    fontSize = 64;
    maxChars = 30;
  } else {
    fontSize = 52;
    maxChars = 38;
  }
  const titleLines = wrap(title, maxChars, 3);
  const lineH = Math.round(fontSize * 1.06);

  const dekLines = a.description ? wrap(a.description.replace(/`/g, '').trim(), 66, 2) : [];
  const dekSize = 30;
  const dekH = Math.round(dekSize * 1.3);

  // Vertical layout: prompt at a fixed top, then the title block, dek and rule stacked;
  // the whole stack is nudged so long titles never collide with the footer.
  const promptY = 128;
  const blockH =
    titleLines.length * lineH + (dekLines.length ? 22 + dekLines.length * dekH : 0) + 34;
  const top = Math.max(promptY + 46, Math.round((630 - blockH) / 2) - 6);
  let y = top + Math.round(fontSize * 0.86);
  const titleSvg = titleLines
    .map(
      (l, i) =>
        `<text x="94" y="${y + i * lineH}" font-family="Fraunces" font-size="${fontSize}" letter-spacing="${(-0.03 * fontSize).toFixed(1)}" fill="${INK}">${esc(l)}</text>`,
    )
    .join('');
  y += (titleLines.length - 1) * lineH;
  let dekSvg = '';
  if (dekLines.length) {
    y += 22 + Math.round(dekSize * 0.95);
    dekSvg = dekLines
      .map(
        (l, i) =>
          `<text x="96" y="${y + i * dekH}" font-family="Fraunces" font-style="italic" font-size="${dekSize}" fill="${SOFT}">${esc(l)}</text>`,
      )
      .join('');
    y += (dekLines.length - 1) * dekH;
  }
  const ruleY = y + 30;

  const kicker = a.kicker.toLowerCase().replace(/\s+/g, '-');
  const prompt = `<tspan fill="${ACCENT}" font-weight="500">~</tspan> $ cat ${esc(kicker)}/${esc(a.slug)}.md`;
  const footRight = [a.kicker.toUpperCase(), a.date ? formatDate(a.date).toUpperCase() : '']
    .filter(Boolean)
    .join(' · ');

  const tick = (x: number, y2: number, sx: number, sy: number) =>
    `<path d="M${x} ${y2 + 16 * sy} V${y2} H${x + 16 * sx}" fill="none" stroke="${ACCENT}" stroke-width="2"/>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1200" height="630" viewBox="0 0 1200 630">
    <rect width="1200" height="630" fill="${PAPER}"/>
    <rect x="30.5" y="30.5" width="1139" height="569" fill="none" stroke="${LINE}" stroke-width="1"/>
    ${tick(30, 30, 1, 1)}${tick(1170, 30, -1, 1)}${tick(30, 600, 1, -1)}${tick(1170, 600, -1, -1)}
    <text x="94" y="${promptY}" font-family="JetBrains Mono" font-size="22" fill="${MUTED}">${prompt}</text>
    ${titleSvg}
    ${dekSvg}
    <rect x="96" y="${ruleY}" width="120" height="3" fill="${ACCENT}"/>
    <text x="94" y="556" font-family="JetBrains Mono" font-size="17" letter-spacing="4" fill="${INK}">${SITE.toUpperCase()}</text>
    <text x="1106" y="556" text-anchor="end" font-family="JetBrains Mono" font-size="17" letter-spacing="4" fill="${MUTED}">${esc(footRight)}</text>
    {{OWL}}
  </svg>`;
}

/** Rasterize an OG SVG with the vendored fonts and the owl (public/favicon.png) top-right. */
export async function renderOgPng(svg: string): Promise<Buffer> {
  const owl = await readFile('public/favicon.png');
  const owlTag = `<image x="1044" y="66" width="104" height="104" xlink:href="data:image/png;base64,${owl.toString('base64')}"/>`;
  const resvg = new Resvg(svg.replace('{{OWL}}', owlTag), {
    fitTo: { mode: 'width', value: 1200 },
    font: { fontFiles: FONT_FILES, loadSystemFonts: false, defaultFontFamily: 'Fraunces' },
  });
  return Buffer.from(resvg.render().asPng());
}
