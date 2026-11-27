/**
 * Wang tiles and the searcher Hao Wang described in 1961. A tile is [north, east, south, west]
 * edge colours. Tiles may not be rotated. Neighbours must share the colour on the touching edge.
 * Rows run downward: tile (r, c) south meets tile (r + 1, c) north; east meets the west of (r, c + 1).
 */
export type Tile = readonly [number, number, number, number];
export type Grid = number[][]; // tile indices

/**
 * Jeandel and Rao's aperiodic set of 11 tiles on 4 colours: the set W' of Figure 2 in
 * "An aperiodic set of 11 Wang tiles" (arXiv:1506.06492, Advances in Combinatorics 2021:1),
 * transcribed from the TikZ source (left/right/above/below labels) and cross-checked against
 * the second string of the authors' program anc/src/11tiles.cpp
 * ("3331 3111 3122 1102 1022 0010 0321 1323 3011 2210 2202", read as west east south north).
 */
export const JEANDEL_RAO: Tile[] = [
  [0, 0, 1, 0],
  [1, 3, 2, 0],
  [2, 0, 2, 1],
  [2, 1, 0, 1],
  [3, 3, 2, 1],
  [1, 0, 1, 3],
  [1, 1, 1, 3],
  [2, 1, 2, 3],
  [1, 3, 3, 3],
  [0, 2, 1, 2],
  [2, 2, 0, 2],
];

/** The authors' string for the same set, in west-east-south-north order. */
export const JEANDEL_RAO_SRC = '3331 3111 3122 1102 1022 0010 0321 1323 3011 2210 2202';

export function fromWESN(s: string): Tile[] {
  return s
    .trim()
    .split(/\s+/)
    .map((w) => {
      const [we, ea, so, no] = [...w].map(Number);
      return [no, ea, so, we] as const;
    });
}

/** Is a (possibly partial, -1 = empty) grid consistent? Optional wrap-around (torus). */
export function valid(tiles: Tile[], g: Grid, wrap = false): boolean {
  const R = g.length;
  const C = g[0].length;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < C; c++) {
      const t = g[r][c];
      if (t < 0) continue;
      const rr = r + 1 < R ? r + 1 : wrap ? 0 : -1;
      const cc = c + 1 < C ? c + 1 : wrap ? 0 : -1;
      if (rr >= 0 && g[rr][c] >= 0 && tiles[t][2] !== tiles[g[rr][c]][0]) return false;
      if (cc >= 0 && g[r][cc] >= 0 && tiles[t][1] !== tiles[g[r][cc]][3]) return false;
    }
  return true;
}

/**
 * Fill an R x C rectangle (or torus) by backtracking in reading order. Returns the grid or null,
 * plus the number of tile placements tried. `order` permutes the tiles tried at each cell.
 */
export function fill(
  tiles: Tile[],
  R: number,
  C: number,
  wrap = false,
  order?: (r: number, c: number) => number[],
  limit = Infinity,
): { grid: Grid | null; tries: number; exhausted: boolean } {
  const g: Grid = Array.from({ length: R }, () => Array(C).fill(-1));
  const all = tiles.map((_, i) => i);
  let tries = 0;
  const fits = (r: number, c: number, t: number) => {
    const T = tiles[t];
    if (r > 0 && tiles[g[r - 1][c]][2] !== T[0]) return false;
    if (c > 0 && tiles[g[r][c - 1]][1] !== T[3]) return false;
    if (wrap && r === R - 1 && (r === 0 ? T : tiles[g[0][c]])[0] !== T[2]) return false;
    if (wrap && c === C - 1 && (c === 0 ? T : tiles[g[r][0]])[3] !== T[1]) return false;
    return true;
  };
  const go = (k: number): boolean => {
    if (k === R * C) return true;
    if (tries > limit) return false;
    const r = Math.floor(k / C);
    const c = k % C;
    for (const t of order ? order(r, c) : all) {
      tries++;
      if (!fits(r, c, t)) continue;
      g[r][c] = t;
      if (go(k + 1)) return true;
      g[r][c] = -1;
    }
    return false;
  };
  const ok = go(0);
  return { grid: ok ? g : null, tries, exhausted: !ok && tries <= limit };
}

/** Smallest p x q block (p, q <= n) that tiles the plane by repeating, or null. */
export function periodicBlock(tiles: Tile[], n: number) {
  for (let s = 1; s <= n; s++)
    for (let p = 1; p <= s; p++)
      for (const [a, b] of p === s
        ? [[s, s]]
        : [
            [p, s],
            [s, p],
          ]) {
        const res = fill(tiles, a, b, true);
        if (res.grid) return res.grid;
      }
  return null;
}
