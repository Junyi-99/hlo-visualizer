import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { nodeCategory } from '../lib/parser';
import { hloOpName, lastOpNameSegment } from '../lib/metadata';
import type { HloModule } from '../lib/types';
import { Icon } from './Icon';

interface SearchOverlayProps {
  module: HloModule;
  useOpName: boolean;
  showLastNameOnly: boolean;
  onClose: () => void;
  onSelect: (computation: string, node: string) => void;
}

const MAX_RESULTS = 30;

export function SearchOverlay({ module, useOpName, showLastNameOnly, onClose, onSelect }: SearchOverlayProps) {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return module.computations
      .flatMap(computation =>
        computation.nodes.flatMap(node => {
          const opName = hloOpName(node.raw);
          const haystack = `${node.name} ${node.op} ${computation.name} ${opName || ''}`.toLowerCase();
          return haystack.includes(needle) ? [{ computation, node, opName }] : [];
        })
      )
      .slice(0, MAX_RESULTS);
  }, [module, query]);

  const selectFirst = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && results[0]) onSelect(results[0].computation.name, results[0].node.name);
  };

  const closeOnBackdrop = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) onClose();
  };

  return (
    <div className="search-overlay" onMouseDown={closeOnBackdrop}>
      <div className="search-panel max-h-[80dvh]" role="dialog" aria-modal="true" aria-label="Search nodes">
        <div className="search-field">
          <Icon name="search" size={18} />
          <input
            ref={inputRef}
            type="search"
            placeholder="Search nodes or operations…"
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            enterKeyHint="search"
            value={query}
            onChange={event => setQuery(event.target.value)}
            onKeyDown={selectFirst}
          />
          <kbd>esc</kbd>
        </div>

        <div className="search-results">
          {results.length ? (
            results.map(({ computation, node, opName }) => {
              const metadataName = useOpName ? opName : null;
              const label = metadataName ? (showLastNameOnly ? lastOpNameSegment(metadataName) : metadataName) : `%${node.name}`;
              const location = metadataName ? `%${node.name} · %${computation.name}` : `%${computation.name}`;
              return (
                <button type="button" className="search-result" key={node.id} onClick={() => onSelect(computation.name, node.name)}>
                  <span className={`type-badge ${nodeCategory(node)}`}>{node.op}</span>
                  <span>
                    <strong>{label}</strong>
                    <small>{location}</small>
                  </span>
                  <span>↗</span>
                </button>
              );
            })
          ) : (
            <div className="search-empty">No matching nodes</div>
          )}
        </div>
      </div>
    </div>
  );
}
