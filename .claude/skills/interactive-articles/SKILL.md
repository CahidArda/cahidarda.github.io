---
name: interactive-articles
description: >-
  Author interactive blog posts for this Astro site — MDX articles with KaTeX math and React-island
  widgets (charts, diagrams, animations). Use whenever creating or editing a post under
  src/content/articles, building a widget under src/components/interactives, or touching KaTeX,
  SVG/canvas diagrams, or widget animations. Encodes hard-won pitfalls (display-math fences,
  SVG/HTML alignment, arrow direction, node fill state, box text overflow, z-order, CSS vars in
  SVG/canvas) so we don't re-learn them.
---

# Interactive articles, charts & diagrams

How to build posts like `src/content/articles/sakana-fugu.mdx` and the widgets in
`src/components/interactives/fugu/`. Read this **before** writing a math-heavy or diagram-heavy post —
most of it is failures we already hit and fixed.

## 0. Ground truth (already set up — don't redo)

- Math pipeline: `remark-math` + `rehype-katex` are in `astro.config.mjs`; KaTeX CSS is imported once
  in `src/layouts/ArticleLayout.astro`. **Do not** add the plugins or CSS again.
- OG image per post is generated at build from the slug (`src/lib/og.ts`, `src/pages/og/[slug].png.ts`).
  Just pick a good slug; never hand-create `og/*.png`.
- Frontmatter must satisfy `src/content.config.ts` (`title`, `description`, `date`, `tags` — the enum
  there is the source of truth; default `['blog']`). `draft: true` excludes a post from the build.
- React islands don't share state across mounts. Cross-widget messaging → a `window` `CustomEvent`.
- Reusable blocks already in `src/components/`: `EqTip.astro` (hover tooltip explaining a display
  equation, modeled on `TagBadge.astro`'s CSS tooltip), `TweetEmbed.astro` (X/tweet embed with a text
  `fallback`; use the `twitter.com/…/status/…` URL form). Prefer these over re-inventing.

## 1. Article shell

```mdx
---
title: 'Title'
description: 'One-sentence summary (also the meta/OG description).'
date: 2026-06-22
tags: ['blog']
---

import MyWidget from '../../components/interactives/<topic>/MyWidget.tsx';

Prose…

<div class="fx-figure">
  <MyWidget client:visible />
</div>
```

- Every widget gets `client:visible` (lazy-hydrate on scroll; never block first paint) and is wrapped
  in a full-bleed figure div (`.fx-figure`, defined in `global.css`).
- `##` headings become the in-sidebar table of contents (scroll-spy). Keep them meaningful.
- Cite sources generously, and link a quote to the **exact document its words are in** (e.g. the
  technical-report PDF on GitHub), not a generic landing/product page. House citation style is a
  parenthetical after the closing quote, `"…quote." ([Source](url))`, **not** the `— [Source]` dash
  form. Mark estimated/illustrative data as such ("illustrative of the trends…"), never as exact figures.
- **House style: no em-dashes in article prose or widget text.** Use `.`, `:`, `,`, or parentheses,
  and do a quick [signs-of-AI-writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) pass
  (drop intensifiers like "brutally"/"striking", avoid the `not X; it's Y` antithesis, thin out
  tricolons). The **one** exception: never alter an em-dash that sits **inside a verbatim quotation**,
  fixing prose must not corrupt a quote.
- **Markdown link URLs with literal parentheses must be percent-encoded**, or the `[text](url)` breaks:
  `Fine-tuning_(deep_learning)` → `Fine-tuning_%28deep_learning%29`.
- **Internal article links carry NO trailing slash.** The build emits `dist/articles/<slug>.html`
  (a file, not a `<slug>/index.html` directory), so the canonical path is `/articles/<slug>` with no
  trailing `/`. Write `[text](/articles/my-post)`, never `/articles/my-post/` (the trailing-slash form
  does not match the emitted page and reads as a broken/redirecting link).
- "This gets technical, skip ahead" notes: link to the **generated** heading id, which is
  github-slugger style (lowercase, spaces→`-`, punctuation dropped). `## Fugu-Ultra: conducting an
  orchestra` → `#fugu-ultra-conducting-an-orchestra`. **Verify** the id in the built HTML.

## 2. KaTeX in MDX — the traps

1. **Display math needs the `$$` fences on their OWN lines.** A single-line `$$E=mc^2$$` renders as
   *inline* KaTeX (no centering). This is the #1 surprise here.

   ```
   ✅  $$            ❌  $$E = mc^2$$   (renders inline)
       E = mc^2
       $$
   ```

2. **Never split the *content* across lines with the fences attached.** `$$ a \n b $$` throws a KaTeX
   parse error that **cascades and breaks every later equation** on the page. Keep the body on one line
   between own-line fences; use `\qquad`/`\;` for spacing, not newlines.

3. **Wrapping math in a JSX component needs blank lines.** To attach a tooltip/figure wrapper (e.g.
   `<EqTip note="…">`) around display math, leave a blank line between the JSX tags and the `$$` fences,
   or MDX treats the `$$` as literal text and KaTeX never runs (you get raw `$$` in the output, no
   `katex-display`):

   ```mdx
   <EqTip note="what it means">

   $$
   E = mc^2
   $$

   </EqTip>
   ```

Inline `$...$` is fine anywhere. **Verify** after building (see §6): `katex-error` must be `0`,
`katex-display` must equal the number of block equations (the post-KaTeX class is `katex-display`, not
`math math-display`).

## 3. Widgets: the shared design system

Define a `shared.tsx` per topic ONCE and reuse it everywhere so the post reads as authored:

- **One fixed color per entity**, as theme-aware CSS vars in `global.css` (`:root` + `.dark`), exposed
  to Tailwind via `@theme` and referenced as `var(--color-…)`. The reader learns the cast once.
- A shared **frame** component (bordered figure + title), shared **hooks** (`useReducedMotion`,
  `useTick`, `useGlide`), shared chips/legends.
- **Prose insulation.** Widgets render *inside* `.prose`, so editorial rules (link underlines, list
  markers, code chrome) leak in. Give every widget root a class (we use `.fx` / `not-prose`) and reset
  those in `global.css` under `.prose .fx { … }`.
- **`prefers-reduced-motion`:** every animation needs an instant/static fallback. Use `useReducedMotion`
  and render the final state (a "poster") instead of animating. Also kill transitions in CSS under the
  media query.
- **Mobile:** must work at ~380px and reflow (not horizontally scroll). Test narrow.

## 4. Charts & diagrams — the alignment rules (read this twice)

These are the exact things we had to fix in this repo.

### 4.1 Use ONE coordinate system per diagram

Do **not** mix an SVG drawn in a `0 0 100 100` viewBox with `preserveAspectRatio="none"` over HTML
nodes positioned in `%`. `none` stretches the SVG non-uniformly while the HTML uses real percentages,
so lines and boxes drift apart. Instead: draw the **whole** diagram as one SVG with a fixed viewBox
(e.g. `0 0 400 240`) and the **default** `preserveAspectRatio` (uniform `xMidYMid meet`). Nodes are
`<rect>`+`<text>`, connectors are `<line>`. Everything shares the same units → always aligned.

### 4.2 Anchor connectors to box EDGES, not centers

A line from node-center to node-center emerges from a box corner and runs *under* the boxes. Compute
edge anchors from node geometry:

```ts
const rightOf = (n) => ({ x: n.cx + n.w / 2, y: n.cy });
const leftOf  = (n) => ({ x: n.cx - n.w / 2, y: n.cy });
// edge: from rightOf(source) to leftOf(target)
```

### 4.3 Arrows must follow the ACTIVE direction

A connector's arrowhead has to point the way data is currently flowing. On a return leg, **swap the
line's endpoints** (keep the arrow marker on the `to` end) so it flips — don't leave it pointing the
original way. Verify per phase that the arrow reverses (e.g. `Fugu→worker` on dispatch,
`worker→Fugu` on return).

### 4.4 A node is "active" only while the token is AT it

If you fill a node on a *range* of phases, it keeps its background after the action has moved on. Model
three states — `off` (grayed, neutral stroke), `wait` (in the path, colored border, no fill),
`active` (colored fill) — and make `active` true only for the single current step. Things light up as
the flow reaches them and de-fill as it leaves.

### 4.5 SVG `<text>` does not wrap — fit it

Keep node labels short or boxes wide enough. Mono ≈ `0.6×fontSize` px per char (≈7px at 13px). A
112-unit box fits ~9 chars at size 12. The right-most node must stay inside the viewBox width. We had
"Claude Opus 4.8" overflow its box → switched to short titles ("Opus 4.8") + provider subtitle.

### 4.6 Z-order: moving dots go BEHIND the boxes

Render a travelling token/marker **before** the node `<g>`s in source order so it passes behind boxes
and text, not over the labels.

### 4.7 Colors in SVG and canvas

- In SVG, a CSS variable as a **presentation attribute** doesn't resolve: `fill="var(--x)"` fails. Use
  `style={{ fill: 'var(--x)' }}` instead.
- On `<canvas>`, `getComputedStyle` won't reliably resolve a `var()`-chained `@theme` alias
  (`--color-accent` → `var(--accent)`). Read the **base literal** token (`--accent`, `--paper`, …).
  Handle `devicePixelRatio` (size the backing store `W*dpr × H*dpr`, then `ctx.setTransform(dpr,…)`)
  or the canvas is blurry.

### 4.8 Smooth motion

For a gliding marker use an rAF lerp in **SVG user space** (see `useGlide`) — not a CSS transform with
`px`, which is wrong under viewBox scaling. Snap instantly when reduced-motion is on.

### 4.9 Heavy data

Hand-roll SVG/`<canvas>` to match the brutalist look and keep bundles small. Only reach for a chart lib
if a widget genuinely needs 10k+ points. Use a seeded PRNG (e.g. mulberry32) for reproducible
"random" visuals.

## 5. Animation pacing

Purposeful and fast: 150–300ms transitions, ~1.1–1.4s per narrative phase. Captions, node states,
arrows, and the token must all agree about what phase it is — desync ("caption says the worker is
solving, but the token is back at the hub") reads as a bug. Auto-play on scroll-into-view via the
island's mount effect (it fires when `client:visible` hydrates).

## 6. Verify before committing

```bash
pnpm check          # 0 errors
pnpm build          # succeeds; note the page count
```

Then grep the built page (`dist/articles/<slug>/index.html`):

- `katex-error` → **0**; `katex-display` → equals your block-equation count.
- `client="visible"` → equals your widget count.
- Islands are SSR'd, so the **initial** SVG is in the HTML — extract `<line>`/`<rect>` coords and
  confirm line endpoints land exactly on box edges (catches §4.2 regressions without a browser).
- Optionally `pnpm preview` and `curl` the page + each `/_astro/<Widget>.*.js` for `200`.

Quick logic checks: Node 22 runs `.ts` directly with `node --experimental-strip-types file.ts`
(type-only imports are stripped, so pure logic modules run without their React deps). Add `.ts`
extensions to relative imports when doing this.

Heads-up: `pnpm format` (Prettier) reorders/reflows files. **Re-read a file after formatting** before
your next Edit, or `old_string` won't match.

## 7. Fact-check sourced claims before publishing

When a post leans on papers/reports, verify **every quote, equation, and number** against the primary
source, not memory or a secondary summary.

- Extract PDF text once: `pip install pypdf`, then dump pages to `.txt`. **Gotcha:** extracted text is
  often not clean UTF-8, so `grep` flags the file as binary and silently matches nothing — use
  `grep -a`, or a Python `needle.lower() in text.lower()` check.
- Match quotes word-for-word. A sentence can appear **verbatim in one document but reworded in
  another** — attribute it to the document the exact words are in. (In the Fugu post a line credited to
  "the Conductor" was actually verbatim from the technical report; the paper's own wording differed.)
- Re-derive each equation and re-check table numbers cell-by-cell against the source figure/table.
- If a sentence is the author's **gloss** on a quote, confirm it's actually true. (We had "no
  retraining" sitting next to a quote that only supported "no weight access"; the product FAQ says they
  retrain ~2 weeks to add a model.)

## 8. Log: things we actually fixed (Sakana Fugu post)

- Display `$$` math rendered inline → fences must be on their own lines (§2.1).
- A multi-line `$$` block threw a KaTeX parse error that broke all later equations (§2.2).
- Overview & Conductor diagrams: lines drifted because of `preserveAspectRatio="none"` + `%` HTML →
  rebuilt as one pure-SVG coordinate system; anchored lines to edges; added arrowheads (§4.1–4.2).
- Return-trip arrow kept pointing the wrong way; worker box kept its fill after the token left →
  active-only-at-current-step + endpoint swap (§4.3–4.4).
- Worker labels overflowed their boxes → short titles + provider subtitle (§4.5).
- Travelling token rendered over the labels → moved behind the nodes (§4.6).
- Canvas (evolution sim) needed base tokens + dpr scaling (§4.7).
- Prose: dropped filler ("the obvious next question"), broke a dense two-item paragraph into a list,
  attributed every quote with a link, added a "skip ahead" note with a verified anchor.
- Built `EqTip.astro` for per-equation hover tooltips; learned display `$$` inside a JSX wrapper needs
  blank lines around it or KaTeX never runs (§2.3).
- Proofread pass: scrubbed every em-dash from prose **and** widget labels (house style), kept the one
  inside a verbatim quote, switched citations from `— [Source]` to `([Source])`, and dropped
  editorializing intensifiers.
- Verified all quotes/equations/benchmark numbers against the source PDFs (§7): caught a quote
  misattributed to the wrong paper and a false "no retraining" gloss; re-pointed "technical report"
  links to the actual PDF and landing links to the product page.
- A parenthesized Wikipedia URL needed percent-encoding (`%28…%29`) to survive markdown link syntax.
- Embedded the launch tweet via `TweetEmbed.astro` (placement matters: a tweet wedged between a
  paragraph and a blockquote read badly; moving it to the top with a one-line lead-in fixed it).
