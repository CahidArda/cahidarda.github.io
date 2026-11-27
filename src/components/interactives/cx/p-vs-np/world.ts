import { useEffect, useRef, useState } from 'react';

/** The two candidate worlds. `ne` = P ≠ NP (what most experts expect), `eq` = P = NP. */
export type World = 'ne' | 'eq';

/**
 * Glides a 0..1 "merge" value toward the active world at a constant rate (no easing
 * overshoot, no pause), driven by requestAnimationFrame. Returns the eased value to draw.
 */
export function useMerge(world: World, reduced: boolean, seconds = 1.1): number {
  const target = world === 'eq' ? 1 : 0;
  const [m, setM] = useState(target);
  const mRef = useRef(target);
  useEffect(() => {
    if (reduced) {
      mRef.current = target;
      setM(target);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.max(0, Math.min(0.05, (now - last) / 1000));
      last = now;
      const d = target - mRef.current;
      const stepAbs = dt / seconds;
      if (Math.abs(d) <= stepAbs) {
        mRef.current = target;
        setM(target);
        return;
      }
      mRef.current += Math.sign(d) * stepAbs;
      setM(mRef.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, reduced, seconds]);
  // smoothstep: starts and ends gently, never stops in between
  return m * m * (3 - 2 * m);
}

/**
 * Auto-demo shared by both world widgets: once the figure is on screen, wait `delay` ms and
 * commit to P = NP, unless the reader got there first. Reduced motion skips the demo (the
 * figure stays in the P ≠ NP world). Returns a function to call on any reader input.
 */
export function useDemoFlip(
  inView: boolean,
  reduced: boolean,
  setSel: (w: World) => void,
  delay = 1200,
) {
  const done = useRef(false);
  const touched = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!inView || done.current || reduced) return;
    done.current = true;
    timer.current = setTimeout(() => {
      // the reduced-motion hook settles one render late, so ask the media query directly
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
      if (!touched.current) setSel('eq');
    }, delay);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  return () => {
    touched.current = true;
  };
}
