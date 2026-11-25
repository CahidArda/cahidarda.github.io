import { useEffect, useState, type RefObject } from 'react';

/**
 * True when the element is narrower than `px`. Charts use it to switch to a narrower
 * viewBox on phones so SVG text stays legible instead of shrinking with the figure.
 * SSR renders the wide layout; phones switch right after hydration.
 */
export function useNarrow(ref: RefObject<Element | null>, px = 520) {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setNarrow(e.contentRect.width < px));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, px]);
  return narrow;
}
