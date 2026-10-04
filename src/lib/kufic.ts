/**
 * Square-kufic (murabba kûfî) Latin lettering, generated at build time.
 *
 * Inspired by Emin Barın's square-kufic compositions in Latin letters: every letter lives on
 * one square grid, stroke and gap are both exactly one cell, words are joined along the
 * baseline, and rows are justified by stretching horizontals (the kufic keşide) so a stack of
 * words closes into a solid block. Output is a single SVG path of grid cells, so the mark is
 * static, crisp at any size, and ships no JS.
 */

type Glyph = { rows: string[]; stretch?: number | null };

// 5 rows tall. '#' = ink, '.' = ground. `stretch` is the column that repeats when a row is
// justified (null: the letter never stretches, e.g. I and T, whose only stem would thicken).
const G: Record<string, Glyph> = {
  A: { rows: ['###', '#.#', '###', '#.#', '#.#'] },
  B: { rows: ['##.', '#.#', '###', '#.#', '###'] },
  C: { rows: ['###', '#..', '#..', '#..', '###'] },
  D: { rows: ['##.', '#.#', '#.#', '#.#', '##.'], stretch: 0 },
  E: { rows: ['###', '#..', '###', '#..', '###'] },
  F: { rows: ['###', '#..', '###', '#..', '#..'] },
  G: { rows: ['###', '#..', '#.#', '#.#', '###'] },
  H: { rows: ['#.#', '#.#', '###', '#.#', '#.#'] },
  I: { rows: ['#', '#', '#', '#', '#'], stretch: null },
  K: { rows: ['#.#', '#.#', '##.', '#.#', '#.#'], stretch: null },
  L: { rows: ['#..', '#..', '#..', '#..', '###'], stretch: 2 },
  M: { rows: ['#####', '#.#.#', '#.#.#', '#.#.#', '#.#.#'], stretch: 1 },
  N: { rows: ['###', '#.#', '#.#', '#.#', '#.#'] },
  O: { rows: ['###', '#.#', '#.#', '#.#', '###'] },
  P: { rows: ['###', '#.#', '###', '#..', '#..'] },
  Q: { rows: ['###.', '#.#.', '#.#.', '#.#.', '####'], stretch: 1 },
  R: { rows: ['##.', '#.#', '##.', '#.#', '#.#'], stretch: 0 },
  S: { rows: ['###', '#..', '###', '..#', '###'] },
  T: { rows: ['###', '.#.', '.#.', '.#.', '.#.'], stretch: null },
  U: { rows: ['#.#', '#.#', '#.#', '#.#', '###'] },
  V: { rows: ['#.#', '#.#', '#.#', '#.#', '.#.'], stretch: null },
  W: { rows: ['#.#.#', '#.#.#', '#.#.#', '#.#.#', '#####'], stretch: 1 },
  X: { rows: ['#.#', '#.#', '.#.', '#.#', '#.#'], stretch: null },
  Y: { rows: ['#.#', '#.#', '###', '.#.', '.#.'], stretch: null },
  Z: { rows: ['###', '..#', '###', '#..', '###'] },
};

export const H = 5;

/** A bitmap: rows of booleans. */
type Bits = boolean[][];

const toBits = (rows: string[]): Bits => rows.map((r) => [...r].map((c) => c === '#'));

function glyph(ch: string): Glyph {
  const g = G[ch];
  if (!g) throw new Error(`kufic: no glyph for "${ch}"`);
  return g;
}

/** Repeat column `col` of a bitmap `n` extra times. */
function widen(bits: Bits, col: number, n: number): Bits {
  return bits.map((row) => [
    ...row.slice(0, col + 1),
    ...Array(n).fill(row[col]),
    ...row.slice(col + 1),
  ]);
}

/**
 * Lay out one word, justified to `width` cells (or natural width when omitted).
 * Slack goes first into baseline connectors between letters (kufic joins), then into the
 * letters' stretch columns, spread evenly from the centre out.
 */
export function word(text: string, width?: number): Bits {
  const chars = [...text.toUpperCase()];
  const glyphs = chars.map(glyph);
  let letters = glyphs.map((g) => toBits(g.rows));
  const natural = letters.reduce((s, b) => s + b[0].length, 0) + (letters.length - 1);
  let slack = width ? width - natural : 0;
  if (slack < 0) throw new Error(`kufic: "${text}" needs ${natural} cells, got ${width}`);

  // Gaps: 1 cell between letters. A gap is a joint when both neighbours touch the baseline
  // at their facing edges; joints are filled on the baseline row and can grow.
  const gaps = letters.slice(0, -1).map((b, i) => {
    const next = letters[i + 1];
    const joint = b[H - 1][b[0].length - 1] && next[H - 1][0];
    return { w: 1, joint };
  });
  const stretchable = glyphs
    .map((g, i) => ({ i, col: g.stretch === undefined ? 1 : g.stretch }))
    .filter((s) => s.col !== null) as { i: number; col: number }[];

  // Hand out slack round-robin: joints first (centre-out), then letters.
  const order = <T>(xs: T[]) => {
    const mid = (xs.length - 1) / 2;
    return [...xs].sort((a, b) => Math.abs(xs.indexOf(a) - mid) - Math.abs(xs.indexOf(b) - mid));
  };
  const joints = order(gaps.filter((g) => g.joint));
  const extra = new Map<number, number>();
  const slots: (() => void)[] = [
    ...joints.map((g) => () => (g.w += 1)),
    ...order(stretchable).map((s) => () => extra.set(s.i, (extra.get(s.i) ?? 0) + 1)),
  ];
  if (slack > 0 && slots.length === 0) throw new Error(`kufic: "${text}" cannot stretch`);
  for (let k = 0; slack > 0; k++, slack--) slots[k % slots.length]();

  letters = letters.map((b, i) => {
    const s = stretchable.find((x) => x.i === i);
    return s && extra.get(i) ? widen(b, s.col, extra.get(i)!) : b;
  });

  const out: Bits = Array.from({ length: H }, () => []);
  letters.forEach((b, i) => {
    for (let r = 0; r < H; r++) out[r].push(...b[r]);
    if (i < gaps.length) {
      for (let r = 0; r < H; r++)
        out[r].push(...Array(gaps[i].w).fill(gaps[i].joint && r === H - 1));
    }
  });
  return out;
}

export const naturalWidth = (text: string) => word(text)[0].length;

/** Stack words into a justified block (istif), one ground row between them. */
export function block(words: string[], opts: { width?: number; frame?: boolean } = {}): Bits {
  const width = opts.width ?? Math.max(...words.map(naturalWidth));
  const rows: Bits = [];
  words.forEach((w, i) => {
    if (i) rows.push(Array(width).fill(false));
    rows.push(...word(w, width));
  });
  return opts.frame ? frame(rows) : rows;
}

/** One-cell rule around a bitmap with a one-cell ground between. */
export function frame(bits: Bits): Bits {
  const w = bits[0].length + 4;
  const solid = Array(w).fill(true);
  const edge = [true, ...Array(w - 2).fill(false), true];
  return [solid, edge, ...bits.map((r) => [true, false, ...r, false, true]), edge, solid];
}

/** A single line of words separated by a square dot, for bands and friezes. */
export function band(words: string[], sep = 3): Bits {
  const parts = words.map((w) => word(w));
  const out: Bits = Array.from({ length: H }, () => []);
  parts.forEach((p, i) => {
    if (i) {
      for (let r = 0; r < H; r++) {
        out[r].push(...Array(sep).fill(false), r === 2, ...Array(sep).fill(false));
      }
    }
    for (let r = 0; r < H; r++) out[r].push(...p[r]);
  });
  return out;
}

/** SVG path data: one rectangle per horizontal run of ink, in cell units. */
export function toPath(bits: Bits): string {
  let d = '';
  bits.forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      if (!row[x]) {
        x++;
        continue;
      }
      const x0 = x;
      while (x < row.length && row[x]) x++;
      d += `M${x0} ${y}h${x - x0}v1h${x0 - x}z`;
    }
  });
  return d;
}

export const size = (bits: Bits) => ({ w: bits[0].length, h: bits.length });

/* --------------------------------------------------------------------------
   Mosaic: words fitted together like bricks into one field.
   The field is a stack of bands, each four word-lines tall. A band is a row of tiles
   separated by one ground column: a *stack* tile holds four horizontal words justified to
   the tile width; a *pillar* tile holds one to three words turned 90° and justified to
   the band height. Every tile edge lands on the grid, so the whole field interlocks with
   the same one-cell rhythm as a single word.
   -------------------------------------------------------------------------- */

const LINES = 4;
const BAND = LINES * H + (LINES - 1); // 23 cells

/** Small seeded PRNG (mulberry32) so the field is identical on every build. */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Turn a bitmap 90° clockwise (a word read top to bottom, letter tops facing right). */
function rotate(bits: Bits): Bits {
  const h = bits.length;
  const w = bits[0].length;
  return Array.from({ length: w }, (_, x) =>
    Array.from({ length: h }, (_, y) => bits[h - 1 - y][x]),
  );
}

type Tile = { kind: 'stack' | 'pillar'; w: number; words: string[] };

export interface Field {
  ink: Bits;
  /** Cells of the single highlighted word, if any (same size as ink). */
  accent: Bits;
}

export function mosaic(
  vocab: readonly string[],
  opts: {
    cols: number;
    bands: number;
    seed?: number;
    /** Highlight the one word whose centre lies nearest this cell. */
    accent?: { x: number; y: number };
  },
): Field {
  const r = rng(opts.seed ?? 1819);
  const pick = <T>(xs: readonly T[]) => xs[Math.floor(r() * xs.length)];
  const nat = new Map(vocab.map((v) => [v, naturalWidth(v)]));
  const tall = vocab.filter((v) => nat.get(v)! <= BAND); // words that fit a pillar
  const rows = opts.bands * BAND + (opts.bands - 1);
  const ink: Bits = Array.from({ length: rows }, () => Array(opts.cols).fill(false));
  const accent: Bits = Array.from({ length: rows }, () => Array(opts.cols).fill(false));
  const placed: { bits: Bits; x: number; y: number }[] = [];

  // Words for a stack of width w: lengths within 6 cells of w, so no word over-stretches.
  const stackWords = (w: number) => {
    const fit = vocab.filter((v) => nat.get(v)! <= w && nat.get(v)! >= w - 6);
    const pool = fit.length ? fit : vocab.filter((v) => nat.get(v)! <= w);
    const out: string[] = [];
    for (let i = 0; i < LINES; i++) {
      const fresh = pool.filter((p) => !out.includes(p));
      out.push(pick(fresh.length ? fresh : pool));
    }
    return out;
  };

  for (let b = 0; b < opts.bands; b++) {
    const tiles: Tile[] = [];
    let x = 0;
    let lastPillar = false;
    while (x < opts.cols) {
      const left = opts.cols - x;
      if (!lastPillar && r() < 0.32 && left >= 17) {
        const k = 1 + Math.floor(r() * 3);
        tiles.push({
          kind: 'pillar',
          w: k * H + (k - 1),
          words: Array.from({ length: k }, () => pick(tall)),
        });
        lastPillar = true;
      } else {
        const base = nat.get(pick(vocab))!;
        tiles.push({ kind: 'stack', w: base + Math.floor(r() * 4), words: [] });
        lastPillar = false;
      }
      x += tiles[tiles.length - 1].w + 1;
    }
    // Close the band exactly at `cols`: trim the overshoot, or hand slack to the last stack.
    let over = x - 1 - opts.cols;
    for (let i = tiles.length - 1; over !== 0 && i >= 0; i--) {
      const t = tiles[i];
      if (t.kind !== 'stack') continue;
      const nw = Math.max(15, t.w - over);
      over -= t.w - nw;
      t.w = nw;
    }
    if (over > 0) tiles.pop(); // pathological: drop the last tile and leave ground

    // Paint the band.
    let cx = 0;
    const y0 = b * (BAND + 1);
    for (const t of tiles) {
      if (t.kind === 'stack') t.words = stackWords(t.w);
      const parts =
        t.kind === 'stack'
          ? t.words.map((w, i) => ({ bits: word(w, t.w), dx: 0, dy: i * (H + 1), w }))
          : t.words.map((w, i) => ({ bits: rotate(word(w, BAND)), dx: i * (H + 1), dy: 0, w }));
      for (const p of parts) placed.push({ bits: p.bits, x: cx + p.dx, y: y0 + p.dy });
      cx += t.w + 1;
    }
  }
  // Pick the accent word, then paint everything onto the grid.
  const a = opts.accent;
  const dist = (q: (typeof placed)[number]) =>
    a ? Math.hypot(q.x + q.bits[0].length / 2 - a.x, q.y + q.bits.length / 2 - a.y) : 0;
  const hot = a ? placed.reduce((best, q) => (dist(q) < dist(best) ? q : best)) : null;
  for (const q of placed) {
    const target = q === hot ? accent : ink;
    q.bits.forEach((row, yy) =>
      row.forEach((on, xx) => {
        const X = q.x + xx;
        const Y = q.y + yy;
        if (on && X < opts.cols && Y < rows) target[Y][X] = true;
      }),
    );
  }
  return { ink, accent };
}
