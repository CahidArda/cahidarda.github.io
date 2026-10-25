// Screenshot every plate on a page, with real WebGL, at chosen widths, theme and slider values.
// See .claude/skills/verify-interactives/SKILL.md §3.
//
//   node scripts/plates/snap.mjs <url> [--out dir] [--widths 1280,390] [--dark]
//        [--plate 3] [--set "index mirror turned θ=12"] [--set "roll the instrument=-5"]
//        [--wait 1500] [--chromium /usr/bin/chromium]
//
// playwright-core is resolved from the CURRENT directory, so install it in a scratch folder
// (`npm i playwright-core`) and run the script from there. It is not a site dependency.
// Writes <out>/<w>[-dark][-pN].png per plate, and prints console errors from the page.
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const require = createRequire(join(process.cwd(), 'noop.js'));
const { chromium } = require('playwright-core');

const argv = process.argv.slice(2);
const url = argv[0]?.startsWith('--') ? undefined : argv[0];
const opt = (k, d) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : d);
const sets = argv.flatMap((a, i) => (a === '--set' ? [argv[i + 1]] : []));
const out = opt('out', 'shots');
const widths = opt('widths', '1280,390').split(',').map(Number);
const dark = argv.includes('--dark');
const only = opt('plate', '');
const wait = +opt('wait', '1500');
if (!url) throw new Error('usage: node snap.mjs <url> [--out dir] [--widths ..] [--dark] ...');
mkdirSync(out, { recursive: true });

const browser = await chromium.launch({
  executablePath: opt('chromium', '/usr/bin/chromium'),
  // Software WebGL: without these the BlueprintCanvas shows only its SVG poster.
  args: [
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--no-sandbox',
  ],
});
for (const w of widths) {
  const page = await browser.newPage({ viewport: { width: w, height: 900 }, deviceScaleFactor: 2 });
  page.on(
    'console',
    (m) => m.type() === 'error' && console.log(`[${w}] console:`, m.text().slice(0, 200)),
  );
  page.on('pageerror', (e) => console.log(`[${w}] pageerror:`, e.message.slice(0, 200)));
  await page.goto(url, { waitUntil: 'networkidle' });
  if (dark) await page.evaluate(() => document.documentElement.classList.add('dark'));
  // The site has `scroll-behavior: smooth`; step with behavior 'instant' so every
  // client:visible island actually enters the viewport and hydrates.
  await page.evaluate(async () => {
    const step = innerHeight * 0.6;
    for (let y = 0; y < document.body.scrollHeight; y += step) {
      scrollTo({ top: y, behavior: 'instant' });
      await new Promise((r) => setTimeout(r, 120));
    }
    scrollTo({ top: 0, behavior: 'instant' });
  });
  const plates = page.locator('.in-plate');
  const n = await plates.count();
  for (let i = 0; i < n; i++) {
    const plate = plates.nth(i);
    const no =
      (
        (await plate
          .locator('.in-plate-no')
          .first()
          .textContent()
          .catch(() => '')) ?? ''
      ).replace(/\D+/g, '') || String(i + 1);
    if (only && only !== no) continue;
    await plate.evaluate((el) => el.scrollIntoView({ block: 'center', behavior: 'instant' }));
    // --set "<slider label>=<value>": drive a shared <Slider> the way a reader does (React
    // listens to 'input'; the native setter bypasses React's value tracking).
    for (const s of sets) {
      const [label, value] = [s.slice(0, s.lastIndexOf('=')), s.slice(s.lastIndexOf('=') + 1)];
      const done = await plate.evaluate(
        (el, [label, value]) => {
          const lab = [...el.querySelectorAll('label')].find((l) => l.textContent.trim() === label);
          const input = lab && el.querySelector(`#${CSS.escape(lab.htmlFor)}`);
          if (!input) return false;
          const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
          set.call(input, value);
          input.dispatchEvent(new Event('input', { bubbles: true }));
          return input.value; // the browser clamps to min/max: compare with what you asked for
        },
        [label, value],
      );
      if (done === false) continue;
      if (done !== value) console.log(`plate ${no}: '${label}' clamped ${value} -> ${done}`);
    }
    await page.waitForTimeout(wait); // let WebGL render and animations settle
    const file = join(out, `${w}${dark ? '-dark' : ''}-p${no}.png`);
    await plate.screenshot({ path: file });
    console.log(file);
  }
  await page.close();
}
await browser.close();
