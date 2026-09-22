/**
 * Wrap the first letter of an article's first paragraph in <span class="dropcap">
 * and mark that paragraph `.has-dropcap`. A real element (rather than ::first-letter)
 * lets the initial sit in a fixed square tile in every browser, whatever the glyph.
 * Styling only applies under `.prose-drop` (blog posts); elsewhere the span is inert.
 */
export default function rehypeDropcap() {
  return (tree) => {
    const p = (tree.children ?? []).find((n) => n.type === 'element' && n.tagName === 'p');
    if (!p) return;
    const text = firstText(p);
    if (!text) return;
    // Keep leading punctuation (quotes, brackets) with the letter.
    const m = text.node.value.match(/^(\s*)([\p{Ps}\p{Pi}"'‘“]*\p{L})/u);
    if (!m) return;
    const [, lead, initial] = m;
    const rest = text.node.value.slice(lead.length + initial.length);
    // A plain inline span: copy/paste, reader modes and screen readers still get
    // the whole word, since nothing is duplicated or hidden.
    const span = {
      type: 'element',
      tagName: 'span',
      properties: { className: ['dropcap'] },
      children: [{ type: 'text', value: initial }],
    };
    text.parent.children.splice(text.index, 1, span, { type: 'text', value: rest });
    p.properties = p.properties ?? {};
    const cls = p.properties.className ?? [];
    p.properties.className = [...(Array.isArray(cls) ? cls : [cls]), 'has-dropcap'];
  };
}

/** Depth-first: the first non-empty text node, following first children only. */
function firstText(node) {
  for (let i = 0; i < (node.children ?? []).length; i++) {
    const child = node.children[i];
    if (child.type === 'text') {
      if (child.value.trim() === '') continue;
      return { node: child, parent: node, index: i };
    }
    if (child.type === 'element') return firstText(child);
    return null;
  }
  return null;
}
