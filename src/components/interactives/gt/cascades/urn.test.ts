import { play, drawBalls, crowdAnswer, pooledAnswer, type Colour } from './urn.ts';
import { mulberry32 } from './rng.ts';
const B: Colour = 'blue', R: Colour = 'red';
console.log(play([R, R, B, B, R, B, B, B, B, R, B, B]).map((t) => `${t.ball[0]}->${t.guess[0]}${t.following ? '*' : ''}`).join(' '));
console.log(play([B, R, R, B, B, B]).map((t) => `${t.ball[0]}->${t.guess[0]}${t.following ? '*' : ''}`).join(' '));
for (const N of [12, 1000]) {
  const rnd = mulberry32(7);
  let wrong = 0, pooledWrong = 0; const runs = 200000;
  for (let i = 0; i < runs; i++) {
    const urn: Colour = rnd() < 0.5 ? B : R;
    const balls = drawBalls(urn, 12, rnd);
    if (crowdAnswer(play(balls)) !== urn) wrong++;
    if (pooledAnswer(balls, rnd) !== urn) pooledWrong++;
  }
  console.log(N, 'crowd wrong', wrong / runs, 'pooled wrong', pooledWrong / runs);
}
// exact
let pw = 0; const C = (n: number, k: number) => { let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; };
for (let k = 0; k <= 12; k++) { const p = C(12, k) * (2/3)**k * (1/3)**(12-k); if (k < 6) pw += p; if (k === 6) pw += p/2; }
console.log('exact pooled wrong', pw);
// exact crowd wrong for 12: last guess wrong. walk absorbing at +-2; if no cascade, last person follows own ball
let st: Record<number, number> = { 0: 1 }; let wrongAbs = 0;
for (let i = 0; i < 12; i++) {
  const ns: Record<number, number> = {};
  for (const [k, p] of Object.entries(st)) { const d = +k;
    for (const [s, q] of [[1, 2/3], [-1, 1/3]] as const) { const nd = d + s; if (nd === -2) { if (i < 11) wrongAbs += p*q; else wrongAbs += p*q; } else if (nd === 2) {} else ns[nd] = (ns[nd]||0) + p*q; } }
  st = ns;
}
// non-cascaded at end: last person guessed own ball; last ball wrong prob: state after 12 is d=0 (even)... last guess = own ball
console.log('exact crowd wrong (12)', wrongAbs, 'leftover', st);
