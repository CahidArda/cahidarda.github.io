import type { APIRoute, GetStaticPaths } from 'astro';
import { mosaic, size, toPath } from '../../lib/kufic';
import { kuficWords } from '../../data/kufic-words';

/**
 * The background mosaic, built once as two static SVGs: `field` (every word) and `accent`
 * (the one highlighted word). Pages use them as CSS masks, so colour and opacity come from
 * the theme tokens and the file is cached across the whole site.
 */
// accent: the cell that lands in the lower right of a desktop viewport once the field is
// centred and turned 45° (see .kf-layer in global.css).
export const FIELD = { cols: 240, bands: 10, seed: 1819, accent: { x: 170, y: 111 } };

const field = mosaic(kuficWords, FIELD);

export const getStaticPaths: GetStaticPaths = () => [
  { params: { layer: 'field' } },
  { params: { layer: 'accent' } },
];

export const GET: APIRoute = ({ params }) => {
  const bits = params.layer === 'accent' ? field.accent : field.ink;
  const { w, h } = size(bits);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" shape-rendering="crispEdges"><path d="${toPath(bits)}"/></svg>`;
  return new Response(svg, { headers: { 'Content-Type': 'image/svg+xml' } });
};
