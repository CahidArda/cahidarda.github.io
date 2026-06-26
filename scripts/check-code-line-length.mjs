#!/usr/bin/env node
// Verifies that every line inside a fenced code block (``` ... ```) in the article
// content is at most MAX characters, so code snippets never need horizontal
// scrolling. Reports every offending line and exits non-zero if there is one or
// more.
//
//   node scripts/check-code-line-length.mjs
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';

const MAX = 75;
const ROOT = 'src/content/articles';

function markdownFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return markdownFiles(p);
    return ['.md', '.mdx'].includes(extname(p)) ? [p] : [];
  });
}

const offenders = [];
for (const file of markdownFiles(ROOT)) {
  let inFence = false;
  readFileSync(file, 'utf8')
    .split('\n')
    .forEach((line, idx) => {
      if (line.trimStart().startsWith('```')) {
        inFence = !inFence;
        return;
      }
      if (inFence && line.length > MAX) {
        offenders.push({ file, line: idx + 1, len: line.length, text: line });
      }
    });
}

if (offenders.length > 0) {
  console.error(`Code-snippet lines longer than ${MAX} characters:\n`);
  for (const o of offenders) {
    console.error(`  ${o.file}:${o.line}  (${o.len} chars)\n    ${o.text}`);
  }
  console.error(`\n${offenders.length} line(s) exceed ${MAX} characters.`);
  process.exit(1);
}

console.log(`OK: every code-snippet line is within ${MAX} characters.`);
