---
name: verify-interactives
description: >-
  Verify interactive plates, charts and 3D models in this Astro site across their whole parameter
  range before reporting them done. Use after building or changing any widget under
  src/components/interactives whose drawing depends on sliders, toggles, width or time: move its
  geometry into a pure module, sweep every parameter with an assertion script, then screenshot it
  (real WebGL, 390px, dark) in an Upstash Box and look at the images. Catches drawings that
  disagree with their numbers, labels off-plate or colliding, sliders that allow impossible
  states, and animations that run backwards at some settings.
---

# Verifying interactive plates

The default state of a plate always looks right because that is the state you designed it in.
The bugs live at the other settings: the minaret's error band at a short baseline, the escapement
at a 4 m rod and 20 fps, the label at 330px. A plate is done only when (1) a script has checked
every parameter combination numerically and (2) you have looked at screenshots of the extremes.

Read `.claude/skills/interactive-articles/SKILL.md` §14 first (the rules these checks enforce).

## 1. Pure geometry module (do this while building, not after)

Put every meaningful drawn coordinate in a pure TypeScript module next to the plate:
`<topic>/<plate>Geom.ts` (or one `geometry.ts` per post). No React, no DOM, no CSS.

- Signature: `fooScene(params…, W)` returns the drawing in viewBox units: points, segment ends,
  arcs, band polygons, label anchors and texts, plus the computed numbers the readouts and the
  worked equation print. The `.tsx` only maps that object to SVG/three.js.
- ONE function feeds every view of the same state: main drawing, inset (eyepiece, zoom), status
  text, worked equation, SVG poster, 3D rays. A second hand-written formula for an inset always
  looks right and is therefore never checked (the sextant eyepiece bug).
- Constants that bound sliders (`THETA_MIN`, `ROLL_MAX`, `maxAltitude(day)`) live in the module and
  the `.tsx` imports them, so the sweep and the slider cannot drift apart.
- State the frame in the header comment: units, origin, y up or down, angle convention, what is
  fixed in the world and what moves with the instrument.
- Relative imports inside geometry modules carry the `.ts` extension when a sweep imports them
  directly (Node strips types but does not resolve extensionless paths).

Worked examples (copy their shape):

| Module | What it holds |
| --- | --- |
| `instruments/earth/geometry.ts` | `minaretLayout`, `straitLayout`, `aframeLayout`, `cordLayout`, `chartLayout`; `textBox`/`overlap`/`segHitsBox` helpers; exact error region `wedgeRegion` |
| `instruments/earth/geometry.sweep.ts` | the full sweep for all earth plates (about 4.5M checks, all passing) |
| `instruments/sky/doubleGeom.ts` | `doubleReflection(θ, roll)` world optics + `drScene(θ, roll, W)` screen layout + `labelBox` |
| `instruments/sky/sextantView.ts` | `sextantView(reading, tilt)`: one function for the 3D rays, eyepiece inset, status and equation (on branch `rev-sky-3d-wip`, pending merge) |
| `instruments/time/quadrantGeom.ts` | `quadrant(h, day)`, `quadrantLabels`, `maxAltitude(day)` (the slider's clamp) |
| `instruments/time/escapement.ts` | contact solver + `stepMovement` (substepped, frame-rate independent) |

## 2. Numeric sweep

Template: `scripts/plates/sweep.example.ts` (runs against the real double-reflection plate) and
the helper kit `scripts/plates/check.ts` (`ok`, `near`, `range`, `textBox`, `overlap`, `inside`,
`segHitsBox`, `noCollisions`, `report`). Run:

```bash
node --experimental-strip-types scripts/plates/sweep.example.ts   # Node 22; plain `node` on 23+
```

Loop over widths × every slider value (its exact `min`, `max` and `step`, from the module's
constants) × the toggles × the combinations that matter. Widths: `[326, 360, 520, 640, 700]`
(inner drawing width at a 390px phone up to a wide plate; add 1400 for `wide` plates). Assert:

| Check | Catches |
| --- | --- |
| maths: computed value = independent formula or brute force | wrong formula, wrong sign, lopsided error range printed as ± |
| scale: drawn point = value × the drawing's one scale | the minaret tip drawn at 78.8 m for a 76 m result |
| world: fixed objects identical across instrument params | the sun moving when the mirror turns |
| bounds: every label/mark inside `0..W × 0..H` (or flagged clipped) | zenith off the top near the pole, off-plate content at 330px |
| labels: no label/label overlap, no label on the line it names | δ/"sun" collisions, angle labels on sight lines |
| honesty: readout, status, slider value, drawn state agree | slider at 38.0° while the thread is clamped at 37.6°; status tolerance 0.12° vs eyepiece |
| validity: every slider value gives a physical state | telescope into the plate, ray missing the mirror |
| formatting: readout strings match `/^[^e]*$/` etc. | `1.5e+2 nm` leaking from `toPrecision` |

Group failures by tag (the kit does) so millions of checks print one line per kind of bug, with
the first failing context. Fix the module, not the test, unless the test was wrong; when you
loosen a tolerance, say why in a comment.

**Independent model.** For optics and mechanics, also check the module against a second model
written differently (e.g. full 3D vector reflection vs the 2D formula, or brute force over the
four sign combinations of an error bound). The sextant was checked this way (63,091 assertions).

**Animations.** Sample time: step the same `update`/`step` function the plate runs at several
frame rates (20, 30, 60, 144 fps) and several params, and assert the motion's physics: direction
(a bob lags the apex it hangs from), rate (one escape tooth per pendulum period from 0.1 m to
4 m), and things that should not move (a cord's string when its hook only moves vertically).

## 3. Visual sweep in an Upstash Box

The local workspace cannot reach npm, so build, preview and screenshot in a box (see the
`upstash:upstash-box-remote-work` skill). The box's built-in browser has no WebGL (3D plates
show only their poster), so run your own headless Chromium:

```bash
export npm_config_manage_package_manager_versions=false
pnpm build && setsid pnpm preview --host 0.0.0.0 --port 4321 >/tmp/preview.log 2>&1 &
mkdir -p ~/pw && cd ~/pw && npm i playwright-core     # scratch folder, not a site dependency
node <repo>/scripts/plates/snap.mjs http://localhost:4321/articles/<slug> --out shots
node <repo>/scripts/plates/snap.mjs http://localhost:4321/articles/<slug> --widths 390 --dark
node <repo>/scripts/plates/snap.mjs http://localhost:4321/articles/<slug> --plate 3 \
  --set "index mirror turned θ=12" --set "roll the instrument=-5" --out shots/p3-min
```

`scripts/plates/snap.mjs` already handles the traps:

- Chromium flags `--use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist
  --no-sandbox` (software WebGL). Default binary `/usr/bin/chromium`; `--chromium <path>` else.
- Scrolls in steps with `behavior: 'instant'` (the site sets `scroll-behavior: smooth`, so a
  plain `scrollTo` never reaches the islands and `client:visible` never hydrates).
- Dark mode = add the `dark` class on `<html>` after load.
- `--set "<slider label>=<value>"` drives a shared `<Slider>` by its visible label (native value
  setter + `input` event, which React listens to) and prints when the browser clamped the value.
- One PNG per `.in-plate`, named `<width>[-dark]-p<plate no>.png`; logs page errors.

Shots per plate: at least 5 settings, including both extremes of every slider, the clamp point,
zero/negative values, at 1280px; plus 390px and dark at least once. Copy PNGs out (base64 via
`box_exec`, decode into your scratchpad) and **Read every image**. A screenshot you did not look
at verified nothing.

Animated plates: screenshot a few times with different `--wait` values, or sample the update
function numerically (§2), since one frame cannot show direction.

## 4. Before reporting

- `pnpm check` 0 errors, `pnpm build` passes, the sweep exits 0.
- In the report, list per plate: params swept, number of checks, what the sweep or the
  screenshots caught and how it was fixed. Keep the sweep script in the repo next to the module
  so the next change can rerun it.

## 5. Log: what the sweeps caught (instruments series, Oct 2026)

- Minaret triangulation: band was a min..max rectangle centred at 78.8 m (true 76 m), clamped at
  y=8, labelled `±18.1 m`. Now the exact overlap of the two error wedges, clipped with an arrow
  when cut off, labelled `63.5 to 94.1 m (12.5 low, 18.1 high)`; sweep asserts range = brute force.
- Sextant 3D: horizon in the rolled instrument frame; eyepiece from a separate formula; ray drawn
  into the telescope when it missed; status tolerance disagreed with the eyepiece. Fixed with one
  `sextantView` for every view, checked against an independent 3D optics model.
- Double reflection: the sun moved with the mirror. World now fixed; sweep asserts it.
- Sine quadrant: slider 38.0° vs thread clamped at 37.6° (slider max now `maxAltitude(day)`);
  negative `sin φ sin δ` drawn outside the grid (now drawn as an addition, the muvakkit's rule).
- Escapement: wheel stalled or ran backwards at short rods and 20 to 30 fps. Substepped solver.
- A-frame level: bob swung toward the apex's motion instead of lagging. Cord level: string swayed
  though the hook moves only vertically.
- Labels: angle labels on sight lines, range label off the drawing, Shortt/quartz and sundial
  labels colliding on the ladder, period line through the axis title at 4 m, δ/"sun" collision,
  zenith off the top near the pole.
- `1.5e+2 nm` in a readout: explicit number formatting.
- Theodolite: telescope could tilt into the plate; slider narrowed to the clear range.
- Alaturka time: early-morning hours counted from the wrong sunset (use the previous one).
- Plates with `width={W}` overflowed at 330px before hydration (SSR default 640): `width="100%"`.
