import type { ReactNode } from 'react';

/**
 * A readout whose label is a formula (n², 2ⁿ, n log n), so it keeps its case: the shared
 * BigStat label is uppercased, which would turn n into N. Serif lining numerals, mono unit.
 */
export function Stat({
  label,
  value,
  unit,
  color,
  sub,
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: string;
  color: string;
  sub?: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="font-mono text-[0.8rem] font-semibold" style={{ color }}>
        {label}
      </div>
      <div
        className="mt-0.5 whitespace-nowrap font-display text-2xl leading-none tabular-nums lining-nums"
        style={{ color }}
      >
        {value}
        {unit && <span className="ml-1.5 font-mono text-[0.72rem] text-muted">{unit}</span>}
      </div>
      {sub && (
        <div className="mt-1 whitespace-nowrap font-mono text-[0.68rem] text-muted tabular-nums">
          {sub}
        </div>
      )}
    </div>
  );
}
