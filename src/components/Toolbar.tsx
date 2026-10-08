import { useEffect, useRef, useState } from 'react';
import { computationLinks } from '../lib/parser';
import { computationRole } from '../lib/graph-layout';
import type { HloModule } from '../lib/types';

interface ToolbarProps {
  module: HloModule;
  current: string | null;
  canGoBack: boolean;
  zoom: number;
  useOpName: boolean;
  showLastNameOnly: boolean;
  showMemoryLocation: boolean;
  onUseOpName: (value: boolean) => void;
  onShowLastNameOnly: (value: boolean) => void;
  onShowMemoryLocation: (value: boolean) => void;
  onBack: () => void;
  onComputation: (name: string | null) => void;
  onFit: () => void;
  onZoom: (delta: number) => void;
  onSearch: () => void;
  onImport: () => void;
}

export function Toolbar({ module, current, canGoBack, zoom, useOpName, showLastNameOnly, showMemoryLocation, onUseOpName, onShowLastNameOnly, onShowMemoryLocation, onBack, onComputation, onFit, onZoom, onSearch, onImport }: ToolbarProps) {
  const [displayOpen, setDisplayOpen] = useState(false);
  const displayRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!displayOpen) return;
    const dismiss = (event: PointerEvent) => {
      if (!displayRef.current?.contains(event.target as Node)) setDisplayOpen(false);
    };
    const onEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDisplayOpen(false);
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', onEscape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', onEscape);
    };
  }, [displayOpen]);
  const computation = current ? module.byName.get(current) : null;
  const links = computationLinks(module);
  const role = computation && computationRole(computation, links);
  const hasLoopLinks = computation && links.some(link => link.from === computation.name && (link.role === 'body' || link.role === 'condition'));
  return <>
    <header className="topbar"><div className="crumb"><span>Workspace</span><span className="slash">/</span><strong>{!computation ? 'Overview' : computation.entry ? 'Entry' : computation.name}</strong></div>
      <div className="top-actions"><button className="search-button" type="button" title="Search nodes (/)" onClick={onSearch}><span>⌕</span> Search nodes <kbd>/</kbd></button><button className="primary-button" type="button" onClick={onImport}>＋ Open HLO</button></div>
    </header>
    <section className="heading"><div>
      <div className="eyebrow">{!computation ? 'MODULE MAP' : computation.entry ? 'ENTRY COMPUTATION' : role}</div>
      <h2>{computation ? `%${computation.name}` : 'Computation overview'}</h2>
      <p>{!computation ? 'Arrows show which computation an instruction invokes. Click a computation to inspect its instruction dependencies.' : hasLoopLinks ? 'Follow the inputs into the loop, then open its body or condition.' : `The ${computation.nodes.length} instructions in this computation, shown in data dependency order.`}</p>
      <select id="mobile-computations" aria-label="Choose computation" value={current || ''} onChange={event => onComputation(event.target.value || null)}>
        <option value="">Overview · all computations</option>
        {module.computations.map(c => <option key={c.name} value={c.name}>%{c.name}{c.entry ? ' · entry' : ''}</option>)}
      </select>
    </div><div className="heading-controls">
      <div className="view-actions">
      {canGoBack && <button className="subtle-button" type="button" onClick={onBack}>← Back</button>}
      <div className="display-menu" ref={displayRef}>
        <button className={`subtle-button display-trigger${displayOpen ? ' active' : ''}`} type="button" aria-expanded={displayOpen} aria-controls="display-options" onClick={() => setDisplayOpen(value => !value)}>Display <span aria-hidden="true">⌄</span></button>
        {displayOpen && <div className="display-popover" id="display-options" role="group" aria-label="Node display options">
          <strong>Node labels</strong>
          <label className="display-option"><input type="checkbox" checked={useOpName} onChange={event => onUseOpName(event.target.checked)} />Use op_name in metadata</label>
          <label className="display-option"><input type="checkbox" checked={showLastNameOnly} disabled={!useOpName} onChange={event => onShowLastNameOnly(event.target.checked)} />Show last name only</label>
          <label className="display-option"><input type="checkbox" checked={showMemoryLocation} onChange={event => onShowMemoryLocation(event.target.checked)} />展示内存位置</label>
        </div>}
      </div>
      <button className="subtle-button" type="button" onClick={onFit}>Fit view</button>
      <div className="zoom-controls">
        <button className="icon-button" type="button" aria-label="Zoom out" onClick={() => onZoom(-0.15)}>−</button>
        <span className="zoom-value">{Math.round(zoom * 100)}%</span>
        <button className="icon-button" type="button" aria-label="Zoom in" onClick={() => onZoom(0.15)}>＋</button>
      </div>
      </div>
    </div></section>
  </>;
}
