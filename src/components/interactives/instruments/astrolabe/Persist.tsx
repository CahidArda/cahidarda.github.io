/**
 * StepPlate re-mounts its view for every step (and every hover preview). A WebGL canvas must not
 * be rebuilt each time, so the 3D model lives in a portal on a detached <div> that the step's
 * <Slot> adopts while it is shown. Before hydration (and on the server) the Slot shows `fallback`.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

export function usePersistentHost() {
  const [host, setHost] = useState<HTMLDivElement | null>(null);
  useEffect(() => {
    const d = document.createElement('div');
    d.style.width = '100%';
    setHost(d);
  }, []);
  return host;
}

export function PersistentPortal({ host, children }: { host: HTMLDivElement | null; children: ReactNode }) {
  return host ? createPortal(children, host) : null;
}

export function Slot({ host, fallback, onShow }: { host: HTMLDivElement | null; fallback: ReactNode; onShow?: () => void }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || !host) return;
    el.appendChild(host);
    onShow?.();
    return () => {
      if (host.parentNode === el) el.removeChild(host);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [host]);
  return <div ref={ref}>{host ? null : fallback}</div>;
}
