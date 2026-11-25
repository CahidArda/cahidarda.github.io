import { useEffect, useRef, useState } from 'react';

/**
 * SVG text scales down with the viewBox on phones. This returns a font multiplier (>= 1)
 * so a label of `base` user units never renders smaller than `minPx` screen pixels.
 */
export function useBoost(viewW: number, base = 9, minPx = 8.5) {
  const ref = useRef<SVGSVGElement | null>(null);
  const [k, setK] = useState(1);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => {
      const scale = e.contentRect.width / viewW;
      setK(scale > 0 ? Math.max(1, Math.min(1.8, minPx / (base * scale))) : 1);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [viewW, base, minPx]);
  return [ref, k] as const;
}
