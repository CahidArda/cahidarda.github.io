import { useEffect, useRef, useState } from 'react';

/**
 * True when the element is narrower than `below` px. SVG widgets switch to a narrower
 * viewBox on phones so marks and labels stay legible instead of shrinking uniformly.
 * SSR renders the wide layout.
 */
export function useCompact<T extends Element = HTMLDivElement>(below = 520) {
  const ref = useRef<T | null>(null);
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const obs = new ResizeObserver(([e]) => setCompact(e.contentRect.width < below));
    obs.observe(el);
    return () => obs.disconnect();
  }, [below]);
  return [ref, compact] as const;
}
