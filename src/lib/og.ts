/**
 * Builds per-article Open Graph images (1200x630). The title is word-wrapped to fit
 * the width, the font shrinks for longer titles, and anything past the line cap is
 * ellipsized so it can never overflow. The owl (the same Apple emoji used as the
 * favicon) is composited on top as a raster so it keeps its color.
 */
import sharp from 'sharp';
import { readFile } from 'node:fs/promises';

const PAPER = '#f4f1e9';
const INK = '#1a1813';
const ACCENT = '#b14a2a';
const MUTED = '#777264';
const LINE = '#d8d3c5';
const NAME = 'Cahid Arda Öz';

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function wrap(title: string, maxChars: number, maxLines: number): string[] {
  const words = title.split(/\s+/).filter(Boolean);
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

export function articleOgSvg(title: string, kicker: string): string {
  const clean = title.replace(/`/g, '').trim();

  let fontSize: number;
  let maxChars: number;
  if (clean.length <= 24) {
    fontSize = 88;
    maxChars = 17;
  } else if (clean.length <= 48) {
    fontSize = 68;
    maxChars = 23;
  } else {
    fontSize = 54;
    maxChars = 30;
  }

  const maxLines = 4;
  const lineHeight = Math.round(fontSize * 1.16);
  const lines = wrap(clean, maxChars, maxLines);
  const blockH = lines.length * lineHeight;
  const firstBaseline = Math.round(330 - blockH / 2 + fontSize * 0.8);
  const accentY = firstBaseline + (lines.length - 1) * lineHeight + 40;

  const titleLines = lines
    .map(
      (l, i) =>
        `<text x="80" y="${firstBaseline + i * lineHeight}" font-family="Georgia, 'Times New Roman', serif" font-size="${fontSize}" fill="${INK}">${esc(l)}</text>`,
    )
    .join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
    <rect width="1200" height="630" fill="${PAPER}"/>
    <rect x="40" y="40" width="1120" height="550" fill="none" stroke="${LINE}" stroke-width="2"/>
    <text x="80" y="120" font-family="monospace" font-size="24" letter-spacing="6" fill="${MUTED}">${esc(kicker.toUpperCase())}</text>
    ${titleLines}
    <rect x="80" y="${accentY}" width="120" height="5" fill="${ACCENT}"/>
    <text x="80" y="565" font-family="monospace" font-size="22" letter-spacing="4" fill="${MUTED}">${esc(NAME.toUpperCase())}</text>
  </svg>`;
}

/** Rasterize an OG SVG and composite the Apple owl (public/favicon.png) top-right. */
export async function renderOgPng(svg: string): Promise<Buffer> {
  const owl = await sharp(await readFile('public/favicon.png')).resize(124, 124).toBuffer();
  return sharp(Buffer.from(svg))
    .composite([{ input: owl, top: 58, left: 1012 }])
    .png()
    .toBuffer();
}
