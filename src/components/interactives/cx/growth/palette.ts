import { GT } from '../../gt/shared';
import type { CurveId } from './model';

/** One colour per growth rate, shared by every figure on the page. Cool = polynomial, hot = not. */
export const COLOR: Record<CurveId, string> = {
  n: GT.opt,
  nlogn: GT.a,
  n2: GT.alt,
  n3: GT.warn,
  exp: GT.b,
  fact: GT.ink,
};
