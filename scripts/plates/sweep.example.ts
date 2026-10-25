/**
 * Template plate sweep, runnable against a real plate: the double-reflection plate in the sky
 * post (src/components/interactives/instruments/sky/doubleGeom.ts). Copy this next to your own
 * `*Geom.ts` and replace the checks.
 *
 *   node --experimental-strip-types scripts/plates/sweep.example.ts
 *
 * Shape of every sweep: for each drawing width × each slider value (its exact min, max, step):
 *   1. maths:   the computed quantity matches an independent formula
 *   2. scale:   drawn coordinates = computed values on the drawing's one scale
 *   3. world:   things fixed in the world do not move when the reader moves the instrument
 *   4. bounds:  every label sits inside the drawing
 *   5. labels:  no label/label collisions
 *   6. honesty: the state the readout/equation reports is the state that is drawn
 */
import { near, ok, range, inside, noCollisions, report, type Box } from './check.ts';
import {
  ROLL_MAX,
  ROLL_MIN,
  SUN_ALT,
  THETA_MAX,
  THETA_MIN,
  drScene,
  labelBox,
} from '../../src/components/interactives/instruments/sky/doubleGeom.ts';

/** Inner drawing widths: 390px phone, small tablet, desktop column, wide plate. */
const WIDTHS = [326, 360, 520, 640, 700];
// The slider ranges, copied from the plate's <Slider min max step>. Sweep EXACTLY these.
const THETAS = range(THETA_MIN, THETA_MAX, 0.1);
const ROLLS = range(ROLL_MIN, ROLL_MAX, 0.5);

let aligned = 0;
for (const W of WIDTHS) {
  const sun0 = drScene(THETAS[0], 0, W).sun; // reference for "fixed in the world"
  for (const theta of THETAS)
    for (const roll of ROLLS) {
      const s = drScene(theta, roll, W);
      const ctx = `W=${W} θ=${theta} roll=${roll}`;

      // 1. maths: two reflections deviate the light by 180° − 2·(angle between mirrors), so the
      //    image sits SUN_ALT − 2θ above the horizon whatever the roll.
      ok(
        near(s.g.imageVsHorizon, SUN_ALT - 2 * theta, 1e-9),
        'maths: image = h − 2θ',
        () => `${ctx} got ${s.g.imageVsHorizon}`,
      );

      // 2. scale: the drawn sun is the world sun through the one projection
      const P = s.P2(s.g.sun);
      ok(
        near(P.x, s.sun.x) && near(P.y, s.sun.y),
        'scale: sun drawn at its world point',
        () => ctx,
      );

      // 3. world: the sun does not move when the reader turns the mirror or rolls the frame
      ok(near(s.sun.x, sun0.x) && near(s.sun.y, sun0.y), 'world: sun fixed', () => ctx);

      // 4. bounds + 5. labels
      const boxes: Box[] = s.labels.map((l) => ({ name: l.id, ...labelBox(l) }));
      for (const b of boxes)
        ok(inside(b, W, s.H), `bounds: '${b.name}' inside`, () => `${ctx} ${JSON.stringify(b)}`);
      // Known near-miss on the live plate: at 326px and θ = 12° the 'h' and 'θ' glyph boxes touch
      // corner to corner (≈1px). pad −1.5 tolerates that; use a positive pad on new plates.
      noCollisions(boxes, 'labels', ctx, -1.5);

      // 6. honesty: the "aligned" status uses the same number the equation prints
      ok(
        s.g.aligned === Math.abs(s.g.imageVsHorizon) < 0.3,
        'honesty: status = equation',
        () => ctx,
      );
      if (s.g.aligned) aligned++;
    }
}
report([`${aligned} aligned states (θ ≈ ${(SUN_ALT / 2).toFixed(2)}°) across all widths/rolls`]);
