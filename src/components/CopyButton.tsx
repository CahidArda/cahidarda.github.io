import { useState } from 'react';

interface Props {
  /** Text to copy. */
  value: string;
  label?: string;
}

export default function CopyButton({ value, label = 'Copy' }: Props) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard may be blocked */
    }
  }

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? 'Copied' : label}
      className="inline-flex items-center gap-1.5 border border-line-strong px-2.5 py-1 font-mono text-[0.7rem] tracking-[0.18em] text-ink-soft uppercase transition-colors hover:border-accent hover:text-accent"
    >
      {copied ? '✓ Copied' : label}
    </button>
  );
}
