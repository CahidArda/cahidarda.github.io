/** Read prefers-reduced-motion synchronously (the shared hook is false until its effect runs). */
export const prefersReduced = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
