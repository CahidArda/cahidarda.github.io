// Step-plate screenshot driver. node stepshot.mjs <url> <scenario.json> <outdir> [--widths 1280,390] [--dark]
// scenario: [{ plate: 3, name: 'p3-a', do: [ ['set','date',172], ['click','Next'], ['step',2], ['button','Fill the bowl'], ['seg','constant head'], ['wait',800] ] }]
import { createRequire } from 'node:module';
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
const require = createRequire(join(process.cwd(), 'noop.js'));
const { chromium } = require('playwright-core');
const [url, scen, out] = process.argv.slice(2);
const argv = process.argv;
const opt = (k, d) => (argv.includes(`--${k}`) ? argv[argv.indexOf(`--${k}`) + 1] : d);
const widths = opt('widths', '1280').split(',').map(Number);
const dark = argv.includes('--dark');
const only = opt('only', '');
mkdirSync(out, { recursive: true });
const S = JSON.parse(readFileSync(scen, 'utf8'));
const browser = await chromium.launch({
  executablePath: '/usr/bin/chromium',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
});
for (const w of widths) {
  const page = await browser.newPage({ viewport: { width: w, height: 1000 }, deviceScaleFactor: 1.5 });
  page.on('pageerror', (e) => console.log(`[${w}] pageerror:`, e.message.slice(0, 300)));
  page.on('console', (m) => m.type() === 'error' && console.log(`[${w}] console:`, m.text().slice(0, 200)));
  await page.goto(url, { waitUntil: 'networkidle' });
  if (dark) await page.evaluate(() => document.documentElement.classList.add('dark'));
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += innerHeight * 0.6) {
      scrollTo({ top: y, behavior: 'instant' });
      await new Promise((r) => setTimeout(r, 100));
    }
  });
  for (const sc of S) {
    if (only && !sc.name.startsWith(only)) continue;
    const plate = page.locator('.in-plate').filter({ has: page.locator('.in-plate-no', { hasText: new RegExp(`\\b${sc.plate}$`) }) }).first();
    await plate.evaluate((el) => el.scrollIntoView({ block: 'start', behavior: 'instant' }));
    // wait for the island to hydrate (Astro drops the `ssr` attribute when it has)
    await plate.evaluate(async (el) => {
      for (let i = 0; i < 100; i++) {
        const isl = el.closest('astro-island');
        if (!isl || !isl.hasAttribute('ssr')) return;
        await new Promise((r) => setTimeout(r, 100));
      }
    });
    await page.waitForTimeout(400);
    for (const [op, a, b] of sc.do ?? []) {
      if (op === 'set') {
        const r = await plate.evaluate((el, [label, value]) => {
          const lab = [...el.querySelectorAll('label')].find((l) => l.textContent.trim() === label);
          const input = lab && el.querySelector(`#${CSS.escape(lab.htmlFor)}`);
          if (!input) return 'NOLABEL';
          const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
          set.call(input, String(value));
          input.dispatchEvent(new Event('input', { bubbles: true }));
          return input.value;
        }, [a, b]);
        if (r === 'NOLABEL') console.log(`${sc.name}: no slider '${a}'`);
        else if (Number(r) !== Number(b)) console.log(`${sc.name}: '${a}' clamped ${b} -> ${r}`);
      } else if (op === 'step') {
        await plate.locator('.in-step-rail button').nth(a).click();
      } else if (op === 'next') {
        for (let k = 0; k < (a ?? 1); k++) {
          const nx = plate.locator('.in-btn-next');
          if (await nx.isDisabled()) { console.log(`${sc.name}: Next disabled (${await plate.locator('.in-step-hint').textContent().catch(() => '')})`); break; }
          await nx.click();
        }
      } else if (op === 'button' || op === 'seg') {
        const btn = plate.locator('.in-step-ctrl button, .in-step-settings button', { hasText: a }).first();
        if (await btn.count()) await btn.click(); else console.log(`${sc.name}: no button '${a}'`);
      } else if (op === 'wait') await page.waitForTimeout(a);
      await page.waitForTimeout(80);
    }
    await page.mouse.move(0, 0);
    await page.waitForTimeout(sc.wait ?? 700);
    const file = join(out, `${w}${dark ? '-dark' : ''}-${sc.name}.jpg`);
    await plate.screenshot({ path: file, quality: 72 });
    console.log(file);
  }
  await page.close();
}
await browser.close();
