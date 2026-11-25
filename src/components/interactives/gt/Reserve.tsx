import type { ReactNode } from 'react';

/**
 * Stacks every variant of a changing text in one grid cell and shows only the active one,
 * so the block is always as tall as its tallest variant at any width (no layout shift
 * when the text changes). Used by the GT-01/02/03 widgets for captions and readout subs.
 */
export function Reserve({
  items,
  active,
  className = '',
}: {
  items: ReactNode[];
  active: number;
  className?: string;
}) {
  return (
    <div className={`grid ${className}`}>
      {items.map((t, i) => (
        <div
          key={i}
          aria-hidden={i !== active}
          style={{ gridArea: '1 / 1', visibility: i === active ? 'visible' : 'hidden' }}
        >
          {t}
        </div>
      ))}
    </div>
  );
}
