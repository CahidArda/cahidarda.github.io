import { useEffect, useState } from 'react';

/** The tolerance t shared by every widget on the page (islands talk via a window event). */
export const T0 = 0.34;
const EVT = 'gt-seg-t';

export function useSharedT(init = T0) {
  const [t, setT] = useState(init);
  useEffect(() => {
    const h = (e: Event) => setT((e as CustomEvent<number>).detail);
    window.addEventListener(EVT, h);
    return () => window.removeEventListener(EVT, h);
  }, []);
  const set = (v: number) => {
    setT(v);
    window.dispatchEvent(new CustomEvent(EVT, { detail: v }));
  };
  return [t, set] as const;
}

export const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Read prefers-reduced-motion synchronously (the shared hook is false until its effect runs). */
export const prefersReduced = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
