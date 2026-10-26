// Node resolve hook for sweeps: lets extensionless relative imports (e.g. './sextantGeom')
// resolve to the .ts file, as Vite does. Used by noonProcGeom.sweep.ts.
export async function resolve(spec, ctx, next) {
  try {
    return await next(spec, ctx);
  } catch (e) {
    if (spec.startsWith('.') && !/\.[cm]?[jt]sx?$/.test(spec)) return next(spec + '.ts', ctx);
    throw e;
  }
}
