import { useEffect, useMemo, useRef, useState } from 'react';
import { nodeCategory } from '../lib/parser';
import { hloOpName, lastOpNameSegment } from '../lib/metadata';
import type { HloModule } from '../lib/types';

interface SearchOverlayProps {
  module: HloModule;
  useOpName: boolean;
  showLastNameOnly: boolean;
  onClose: () => void;
  onSelect: (computation: string, node: string) => void;
}

export function SearchOverlay({ module, useOpName, showLastNameOnly, onClose, onSelect }: SearchOverlayProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => { inputRef.current?.focus(); }, []);
  const results = useMemo(() => module.computations.flatMap(computation => computation.nodes.flatMap(node => {
    const opName = hloOpName(node.raw);
    return `${node.name} ${node.op} ${computation.name} ${opName || ''}`.toLowerCase().includes(query.trim().toLowerCase())
      ? [{ computation, node, opName }] : [];
  })).slice(0, 30), [module, query]);
  return <div className="search-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="search-panel max-h-[80dvh]" role="dialog" aria-modal="true" aria-label="Search nodes">
      <div className="search-field"><span>⌕</span><input ref={inputRef} type="search" placeholder="Search nodes or operations…" autoComplete="off" autoCapitalize="none" autoCorrect="off" enterKeyHint="search" value={query}
        onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && results[0]) onSelect(results[0].computation.name, results[0].node.name); }} /><kbd>esc</kbd></div>
      <div className="search-results">{results.length ? results.map(({ computation, node, opName }) => <button type="button" className="search-result" key={node.id} onClick={() => onSelect(computation.name, node.name)}>
        <span className={`type-badge ${nodeCategory(node)}`}>{node.op}</span><span><strong>{useOpName && opName ? (showLastNameOnly ? lastOpNameSegment(opName) : opName) : `%${node.name}`}</strong><small>{useOpName && opName ? `%${node.name} · ` : ''}%{computation.name}</small></span><span>↗</span>
      </button>) : <div className="search-empty">No matching nodes</div>}</div>
    </div>
  </div>;
}
