/**
 * Plate 1 geometry, pure (no React) so a sweep script can check it. Screen coordinates,
 * y down; angles in degrees, 0° = +x, counter-clockwise on screen (like shared `polar`).
 *
 * The walk between the cities is drawn at a FIXED length, and the Earth's radius follows from
 * the angle: R = arc / angle. So a bigger shadow angle draws a smaller Earth, exactly as the
 * numbers say. Angles are drawn EXAG times larger so they can be seen.
 */
export const DIST = 5000; // stadia, Syene to Alexandria (Cleomedes)
export const EARTH_KM = 40008; // meridional circumference, km
export const EXAG = 2.5;
export const THETA_MIN = 6;
export const THETA_MAX = 9;

const RAD = Math.PI / 180;
export type P = { x: number; y: number };
export const polar = (cx: number, cy: number, r: number, deg: number): P => ({
  x: cx + r * Math.cos(deg * RAD),
  y: cy - r * Math.sin(deg * RAD),
});

export function eratosthenes(theta: number, W: number) {
  const narrow = W < 520;
  const H = narrow ? 384 : 410;
  const walk = narrow ? 64 : 72; // drawn length of the 5,000 stadia, px
  const a = theta * EXAG; // drawn angle
  const R = walk / (a * RAD);
  const top = 90; // Syene straight up
  const aAlex = top + a; // Alexandria to the left (north)
  const ys = 122; // Syene's ground level, fixed
  const cx = W * (narrow ? 0.62 : 0.58);
  const cy = ys + R;
  const syene = polar(cx, cy, R, top);
  const alex = polar(cx, cy, R, aAlex);
  const g = narrow ? 40 : 46; // gnomon length
  const tipS = polar(cx, cy, R + g, top);
  const tipA = polar(cx, cy, R + g, aAlex);
  // vertical sunlight past Alexandria's tip meets the ground tangent at Alexandria
  const away = { x: Math.cos((aAlex + 90) * RAD), y: -Math.sin((aAlex + 90) * RAD) };
  const s = (tipA.x - alex.x) / away.x;
  const shadowEnd = { x: alex.x + away.x * s, y: alex.y + away.y * s };
  // the drawn shadow angle at the tip, measured back from the geometry
  const vx = alex.x - tipA.x;
  const vy = alex.y - tipA.y;
  const drawnTipAngle = Math.atan2(Math.abs(vx), vy) / RAD; // vs straight down
  // Earth's surface: as much of the circle as fits
  const room = Math.min(cx, W - cx) - 10;
  const span = Math.min(62, Math.asin(Math.min(1, room / R)) / RAD);
  const surfFrom = top - span;
  const surfTo = top + span;
  const rc = Math.min(64, R * 0.34); // radius of the angle arc at the centre
  const walkR = R - 14;
  const walkLabel = polar(cx, cy, R - 30, (top + aAlex) / 2);
  const centreLabel = polar(cx, cy, rc + 14, aAlex + 3);
  const tipArcR = 24;
  const tipLabel = { x: tipA.x + 8, y: tipA.y + tipArcR + 18 };
  const rays = (narrow ? [0.07, 0.93] : [0.1, 0.2, 0.84, 0.94]).map((f) => W * f);
  return {
    narrow, H, W, a, R, top, aAlex, ys, cx, cy, syene, alex, g, tipS, tipA, shadowEnd,
    drawnTipAngle, surfFrom, surfTo, rc, walkR, walkLabel, centreLabel, tipArcR, tipLabel, rays,
    labels: {
      sun: { x: rays[0] + 8, y: 18, size: 12, text: 'sunlight, parallel', anchor: 'start' as const },
      syene: { x: syene.x + 10, y: tipS.y + 16, size: 13, text: 'Syene', anchor: 'start' as const },
      syeneSub: { x: syene.x + 10, y: tipS.y + 30, size: 11, text: 'no shadow', anchor: 'start' as const },
      alex: { x: tipA.x - 10, y: tipA.y + 6, size: 13, text: 'Alexandria', anchor: 'end' as const },
      alexSub: { x: tipA.x - 10, y: tipA.y + 20, size: 11, text: 'a short shadow', anchor: 'end' as const },
      walk: { x: walkLabel.x, y: walkLabel.y, size: 11, text: 'd = 5,000 stadia', anchor: 'middle' as const, mono: true },
      centre: { x: centreLabel.x, y: centreLabel.y, size: 13, text: `θ = ${theta.toFixed(1)}°`, anchor: 'end' as const },
      tip: { x: tipLabel.x, y: tipLabel.y, size: 13, text: 'θ', anchor: 'start' as const },
      earth: { x: cx + 8, y: cy - 6, size: 11, text: "Earth's centre", anchor: 'start' as const },
    },
  };
}

export function circumference(theta: number, stadionM: number) {
  const stadia = (360 / theta) * DIST;
  const km = (stadia * stadionM) / 1000;
  return { stadia, km, errPct: ((km - EARTH_KM) / EARTH_KM) * 100 };
}
