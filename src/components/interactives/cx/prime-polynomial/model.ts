/**
 * The prime-representing polynomial of Jones, Sato, Wada and Wiens (Amer. Math. Monthly 83,
 * 1976, p. 449), transcribed from the scan of the printed page and checked term by term against
 * Wikipedia ("Formula for primes") and MathWorld ("Prime Diophantine Equations"):
 *
 *   (k+2){1 − [term 1]² − [term 2]² − … − [term 14]²}
 *
 * The 14 brackets are kept in the paper's printed order. All arithmetic is BigInt, because
 * even small inputs give values far past 2^53.
 */

export type Vars = Record<string, bigint>;

export const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('');

export interface Term {
  /** formula in a tiny markup: letters italic, ^n superscript, '-' becomes a minus sign */
  src: string;
  /** letters the bracket uses */
  uses: string;
  f: (v: Vars) => bigint;
}

const sq = (x: bigint) => x * x;

export const TERMS: Term[] = [
  { src: 'wz + h + j - q', uses: 'hjqwz', f: ({ w, z, h, j, q }) => w * z + h + j - q },
  {
    src: '(gk + 2g + k + 1)(h + j) + h - z',
    uses: 'ghjkz',
    f: ({ g, k, h, j, z }) => (g * k + 2n * g + k + 1n) * (h + j) + h - z,
  },
  { src: '2n + p + q + z - e', uses: 'enpqz', f: ({ n, p, q, z, e }) => 2n * n + p + q + z - e },
  {
    src: '16(k + 1)^3(k + 2)(n + 1)^2 + 1 - f^2',
    uses: 'fkn',
    f: ({ k, n, f }) => 16n * (k + 1n) ** 3n * (k + 2n) * sq(n + 1n) + 1n - sq(f),
  },
  {
    src: 'e^3(e + 2)(a + 1)^2 + 1 - o^2',
    uses: 'aeo',
    f: ({ e, a, o }) => e ** 3n * (e + 2n) * sq(a + 1n) + 1n - sq(o),
  },
  {
    src: '(a^2 - 1)y^2 + 1 - x^2',
    uses: 'axy',
    f: ({ a, y, x }) => (sq(a) - 1n) * sq(y) + 1n - sq(x),
  },
  {
    src: '16r^2y^4(a^2 - 1) + 1 - u^2',
    uses: 'aruy',
    f: ({ r, y, a, u }) => 16n * sq(r) * y ** 4n * (sq(a) - 1n) + 1n - sq(u),
  },
  {
    src: '((a + u^2(u^2 - a))^2 - 1)(n + 4dy)^2 + 1 - (x + cu)^2',
    uses: 'acdnuxy',
    f: ({ a, u, n, d, y, x, c }) =>
      (sq(a + sq(u) * (sq(u) - a)) - 1n) * sq(n + 4n * d * y) + 1n - sq(x + c * u),
  },
  { src: 'n + l + v - y', uses: 'lnvy', f: ({ n, l, v, y }) => n + l + v - y },
  {
    src: '(a^2 - 1)l^2 + 1 - m^2',
    uses: 'alm',
    f: ({ a, l, m }) => (sq(a) - 1n) * sq(l) + 1n - sq(m),
  },
  { src: 'ai + k + 1 - l - i', uses: 'aikl', f: ({ a, i, k, l }) => a * i + k + 1n - l - i },
  {
    src: 'p + l(a - n - 1) + b(2an + 2a - n^2 - 2n - 2) - m',
    uses: 'ablmnp',
    f: ({ p, l, a, n, b, m }) =>
      p + l * (a - n - 1n) + b * (2n * a * n + 2n * a - sq(n) - 2n * n - 2n) - m,
  },
  {
    src: 'q + y(a - p - 1) + s(2ap + 2a - p^2 - 2p - 2) - x',
    uses: 'apqsxy',
    f: ({ q, y, a, p, s, x }) =>
      q + y * (a - p - 1n) + s * (2n * a * p + 2n * a - sq(p) - 2n * p - 2n) - x,
  },
  {
    src: 'z + pl(a - p) + t(2ap - p^2 - 1) - pm',
    uses: 'almptz',
    f: ({ z, p, l, a, t, m }) => z + p * l * (a - p) + t * (2n * a * p - sq(p) - 1n) - p * m,
  },
];

export function evaluate(v: Vars) {
  const terms = TERMS.map((t) => t.f(v));
  const sum = terms.reduce((s, x) => s + x * x, 0n);
  return { terms, out: (v.k + 2n) * (1n - sum) };
}

/* ───────────── the witness that 2 is prime (k = 0) ───────────── */

/**
 * Smallest a with 32^3·34·(a+1)^2 + 1 a perfect square: walk the solutions of
 * X^2 − 17Y^2 = 1 (fundamental 33 + 8√17) until 256 | Y, then a + 1 = Y/256 and o = X,
 * since 32^3·34 = 17·256^2.
 */
function pellA() {
  let X = 33n;
  let Y = 8n;
  while (Y % 256n !== 0n) [X, Y] = [33n * X + 136n * Y, 8n * X + 33n * Y];
  return { a: Y / 256n - 1n, o: X };
}
const P = pellA();

/**
 * Every letter of the witness for the prime 2 that can be written down. The paper's proof
 * (Theorem 2.12, with Wilson's theorem 1! + 1 = 2) fixes them: n = 2, p = 3, q = 16, z = 9,
 * e = 32, then a from the Pell equation above, y = ψ_a(2) = 2a, x = X_a(2) = 2a² − 1,
 * l = 1, m = a. The other four (r, u, c, d) are far too long to write (see SIZES).
 */
export const WITNESS2: Record<string, bigint | null> = {
  k: 0n,
  g: 0n,
  h: 2n,
  j: 5n,
  w: 1n,
  q: 16n,
  z: 9n,
  n: 2n,
  p: 3n,
  e: 32n,
  f: 17n,
  a: P.a,
  o: P.o,
  y: 2n * P.a,
  x: 2n * P.a * P.a - 1n,
  l: 1n,
  m: P.a,
  i: 0n,
  v: 2n * P.a - 3n,
  b: 0n,
  s: 1n,
  t: 0n,
  r: null,
  u: null,
  c: null,
  d: null,
};

/** Which brackets can be checked with the writable letters (all but terms 7 and 8). */
export const CHECKABLE = TERMS.map((t) => [...t.uses].every((ch) => WITNESS2[ch] !== null));

/* ───────────── sizes (number of decimal digits) ───────────── */

const digits = (x: bigint) => (x < 0n ? -x : x).toString().length;
const aF = Number(P.a);
/** u = X_a(16a): 16a is the first step where 4y² = 16a² divides ψ_a; log10 u ≈ 16a·log10(2a). */
const uDigits = 16 * aF * Math.log10(2 * aF);

export interface SizeRow {
  label: string;
  letters: string;
  digits: number;
  exact: boolean;
}

export const SIZES: SizeRow[] = [
  { label: '16 letters', letters: 'k g b i t s l w h n p j z q f e', digits: 2, exact: true },
  { label: 'a, m, y, v', letters: 'amyv', digits: digits(P.a), exact: true },
  { label: 'o', letters: 'o', digits: digits(P.o), exact: true },
  { label: 'x', letters: 'x', digits: digits(2n * P.a * P.a - 1n), exact: true },
  { label: 'r, u', letters: 'ru', digits: uDigits, exact: false },
  { label: 'd', letters: 'd', digits: 4 * uDigits, exact: false },
  { label: 'c', letters: 'c', digits: 7 * uDigits, exact: false },
];

export const A_STR = P.a.toString();
export const U_DIGITS = uDigits;

/* ───────────── random inputs ───────────── */

export function randomVars(rand: () => number, max = 9): Vars {
  const v: Vars = {};
  for (const ch of LETTERS) v[ch] = BigInt(Math.floor(rand() * (max + 1)));
  return v;
}

/* ───────────── formatting ───────────── */

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
export const sup = (n: number | string) =>
  String(n)
    .split('')
    .map((c) => SUP[Number(c)] ?? c)
    .join('');

/** Short, fixed-width-ish rendering: full below a million, else "−3.1×10¹²". */
export function short(x: bigint): string {
  const neg = x < 0n;
  const s = (neg ? -x : x).toString();
  const m = neg ? '−' : '';
  if (s.length <= 6) return m + Number(s).toLocaleString('en-US');
  return `${m}${s[0]}.${s[1]}×10${sup(s.length - 1)}`;
}

/** "7.0×10⁵⁷" for a float. */
export function sci(x: number, sig = 2): string {
  const e = Math.floor(Math.log10(x));
  const mant = x / 10 ** e;
  return `${mant.toFixed(sig - 1)}×10${sup(e)}`;
}
