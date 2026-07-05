import { useEffect, useMemo, useRef, useState } from 'react';

export interface PaletteItem {
  title: string;
  subtitle: string;
  href: string;
  external: boolean;
  /** Right-aligned label: the item's tag (Blog, Publications, Repositories…) or "Tag". */
  badge: string;
}

interface Props {
  items: PaletteItem[];
}

export default function CommandPalette({ items }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  // ⌘K / Ctrl-K to open, Esc to close.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === 'Escape') {
        setOpen(false);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery('');
      setActive(0);
      // focus after the dialog paints
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (it) => it.title.toLowerCase().includes(q) || it.subtitle.toLowerCase().includes(q),
    );
  }, [query, items]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  function go(item: PaletteItem | undefined) {
    if (!item) return;
    if (item.external) {
      window.open(item.href, '_blank', 'noopener');
    } else {
      window.location.href = item.href;
    }
    setOpen(false);
  }

  function onInputKey(e: React.KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      go(results[active]);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 pt-[12vh] backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-xl border border-line-strong bg-paper shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-line px-4">
          <span aria-hidden="true" className="font-mono text-sm text-muted">
            ⌘K
          </span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKey}
            placeholder="Jump to an article or tag…"
            aria-label="Search articles and tags"
            className="w-full bg-transparent py-4 text-base text-ink outline-none placeholder:text-muted"
          />
        </div>

        <ul ref={listRef} className="max-h-[50vh] overflow-y-auto py-2">
          {results.length === 0 && (
            <li className="px-4 py-6 text-center text-sm text-muted">No matches.</li>
          )}
          {results.map((item, i) => (
            <li key={`${item.badge}-${item.href}`}>
              <button
                type="button"
                onMouseEnter={() => setActive(i)}
                onClick={() => go(item)}
                className={`flex w-full items-center justify-between gap-4 px-4 py-2.5 text-left transition-colors ${
                  i === active ? 'bg-paper-raised' : ''
                }`}
              >
                <span className="min-w-0">
                  <span className="block truncate text-ink">{item.title}</span>
                  <span className="block truncate text-xs text-muted">{item.subtitle}</span>
                </span>
                <span
                  className="shrink-0 font-mono text-[0.6rem] tracking-[0.18em] text-muted uppercase"
                  aria-hidden="true"
                >
                  {item.external && '↗ '}
                  {item.badge}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
