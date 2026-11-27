/**
 * CX-14.3 "the hat" (static). The aperiodic monotile of Smith, Myers, Kaplan and
 * Goodman-Strauss (arXiv:2303.10798), drawn on its grid of kites, next to its mirror image.
 * Outline: `hat_outline` from the paper's validation code (anc/validate/hat.py), in the basis
 * (1, 0), (1/2, sqrt3/2). Checked: 13 corners, edges of length 1, sqrt3 and one of 2 (two unit
 * edges in a line), area 8 sqrt3 = eight kites, and exactly the union of the eight kites below.
 */
import { GT, Specimen, T } from '../../gt/shared';

const R3 = Math.sqrt(3);
const OUTLINE: [number, number][] = [
  [0, 0],
  [-1, -1],
  [0, -2],
  [2, -2],
  [2, -1],
  [4, -2],
  [5, -1],
  [4, 0],
  [3, 0],
  [2, 2],
  [0, 3],
  [0, 2],
  [-1, 2],
].map(([x, y]) => [x + y / 2, (y * R3) / 2]);

// the kite grid: hexagons of side 2 centred on the lattice (3, sqrt3), (3, -sqrt3), each cut
// into six kites by lines from the centre to the edge midpoints
const HEXES: [number, number][] = [];
for (let i = -6; i <= 6; i++)
  for (let j = -6; j <= 6; j++) HEXES.push([3 * i + 3 * j, R3 * i - R3 * j]);

// mirror image: reflect x -> -x (a symmetry of the grid), then move by the lattice vector (9, sqrt3)
const MIRROR = OUTLINE.map(([x, y]) => [-x + 9, y + R3] as [number, number]);

const VB = { x: -2.6, y: -3.3, w: 14.2, h: 7.6 };
const S = 40; // svg units per tiling unit
const P = (pts: [number, number][]) =>
  pts
    .map(
      ([x, y], i) =>
        `${i ? 'L' : 'M'}${((x - VB.x) * S).toFixed(1)} ${((y - VB.y) * S).toFixed(1)}`,
    )
    .join(' ') + ' Z';

export default function Hat() {
  const W = VB.w * S;
  const H = VB.h * S;
  const grid: string[] = [];
  for (const [cx, cy] of HEXES) {
    if (cx < VB.x - 3 || cx > VB.x + VB.w + 3 || cy < VB.y - 3 || cy > VB.y + VB.h + 3) continue;
    const v = Array.from(
      { length: 6 },
      (_, k) =>
        [cx + 2 * Math.cos((k * Math.PI) / 3), cy + 2 * Math.sin((k * Math.PI) / 3)] as [
          number,
          number,
        ],
    );
    grid.push(P(v));
    for (let k = 0; k < 6; k++) {
      const a = v[k];
      const b = v[(k + 1) % 6];
      grid.push(
        P([
          [cx, cy],
          [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
        ]).replace(' Z', ''),
      );
    }
  }
  const X = (x: number) => (x - VB.x) * S;
  const Y = (y: number) => (y - VB.y) * S;
  return (
    <Specimen
      code="CX-14.3"
      title="the hat"
      kicker="Smith et al. 2023"
      hint="eight kites from a grid of hexagons"
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block w-full max-w-[640px]"
        role="img"
        aria-label="The hat monotile, made of eight kites, and its mirror image"
      >
        <defs>
          <linearGradient id="cx14-hat-fade" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#fff" stopOpacity="0" />
            <stop offset="0.12" stopColor="#fff" stopOpacity="1" />
            <stop offset="0.88" stopColor="#fff" stopOpacity="1" />
            <stop offset="1" stopColor="#fff" stopOpacity="0" />
          </linearGradient>
          <mask id="cx14-hat-mask">
            <rect x={0} y={0} width={W} height={H} fill="url(#cx14-hat-fade)" />
          </mask>
        </defs>
        {[OUTLINE, MIRROR].map((o, i) => (
          <path key={i} d={P(o)} style={{ fill: i ? GT.bSoft : GT.aSoft, stroke: 'none' }} />
        ))}
        {/* the kite grid on top of the fills, so the eight kites show inside each hat */}
        <g mask="url(#cx14-hat-mask)">
          {grid.map((d, i) => (
            <path key={i} d={d} style={{ fill: 'none', stroke: GT.lineStrong, strokeWidth: 1 }} />
          ))}
        </g>
        {[OUTLINE, MIRROR].map((o, i) => (
          <path
            key={i}
            d={P(o)}
            style={{
              fill: 'none',
              stroke: i ? GT.b : GT.a,
              strokeWidth: 3,
              strokeLinejoin: 'round',
            }}
          />
        ))}
        <T x={X(1.5)} y={Y(-2.75)} size={13} color={GT.a} weight={600}>
          the hat
        </T>
        <T x={X(7.5)} y={Y(-2.75 + R3)} size={13} color={GT.b} weight={600}>
          its mirror image
        </T>
      </svg>
    </Specimen>
  );
}
