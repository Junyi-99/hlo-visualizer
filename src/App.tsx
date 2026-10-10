import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import sampleHlo from '../sample.hlo?raw';
import { GraphCanvas, type GraphHandle } from './components/GraphCanvas';
import { Icon } from './components/Icon';
import { ImportDialog } from './components/ImportDialog';
import { Inspector } from './components/Inspector';
import { SearchOverlay } from './components/SearchOverlay';
import { Sidebar } from './components/Sidebar';
import { Toolbar } from './components/Toolbar';
import { useStoredState } from './hooks/useStoredState';
import { groupCopyPairs } from './lib/copy-grouping';
import { findExample } from './lib/examples';
import { lloExamples, type LloExample } from './llo/examples';
import { clampZoom, type LayoutMode } from './lib/graph-layout';
import { parseHlo, reachable } from './lib/parser';
import type { HloModule } from './lib/types';

interface HistoryEntry {
  current: string | null;
  selected: string | null;
}

const readAutoGroup = (saved: string | null) => saved === 'true';
const readLayoutMode = (saved: string | null): LayoutMode => (saved === 'horizontal' || saved === 'vertical' ? saved : 'auto');

const lloLink = (program: LloExample | undefined) => (program ? `?view=llo&example=${program.id}` : null);

const isTextField = (element: Element | null) => element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement;

// Mirrors the loaded example into ?example=<id> so a reviewed module can be linked directly.
function syncExampleParam(example: string | null) {
  const url = new URL(location.href);
  if (example) url.searchParams.set('example', example);
  else url.searchParams.delete('example');
  url.searchParams.delete('node');
  window.history.replaceState(null, '', url);
}

export default function App() {
  const [module, setModule] = useState<HloModule>(() => parseHlo(sampleHlo));
  const [exampleId, setExampleId] = useState<string | null>(null);

  const [current, setCurrent] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const [autoGroup, setAutoGroup] = useStoredState('auto-group', readAutoGroup);
  const [layoutMode, setLayoutMode] = useStoredState('graph-layout', readLayoutMode);
  const [sidebarCollapsed, setSidebarCollapsed] = useStoredState('hlo-sidebar-collapsed', readAutoGroup);
  const [zoom, setZoom] = useState(1);
  const [useOpName, setUseOpName] = useState(false);
  const [showLastNameOnly, setShowLastNameOnly] = useState(false);
  const [showMemoryLocation, setShowMemoryLocation] = useState(false);

  const [searchOpen, setSearchOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);

  const graphRef = useRef<GraphHandle>(null);
  // ?node=<name> from a cross link opens that instruction once its example has loaded.
  const pendingNode = useRef(new URLSearchParams(location.search).get('node'));

  const computation = current ? module.byName.get(current) || null : null;
  const node = selected && computation ? computation.byName.get(selected) || null : null;
  const copyGrouping = useMemo(() => (computation ? groupCopyPairs(computation, autoGroup) : null), [computation, autoGroup]);
  const graphComputation = copyGrouping?.computation;
  const graphSelected = selected ? (copyGrouping?.aliases.get(selected) ?? selected) : null;
  const selectedCopyGroup = graphSelected ? (copyGrouping?.groups.get(graphSelected) ?? null) : null;
  const upstreamCount = graphSelected && graphComputation ? reachable(graphComputation, graphSelected, 'up').size : 0;
  const downstreamCount = graphSelected && graphComputation ? reachable(graphComputation, graphSelected, 'down').size : 0;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSearchOpen(false);
        return;
      }
      if (event.key === '/' && !searchOpen && !importOpen && !isTextField(document.activeElement)) {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [searchOpen, importOpen]);

  const loadText = useCallback((source: string, example: string | null = null) => {
    const parsed = parseHlo(source);
    if (!parsed.computations.length) throw new Error('No computations found. Paste a textual HLO module with %computation { … } blocks.');

    setModule(parsed);
    setCurrent(parsed.computations.length === 1 ? parsed.computations[0].name : null);
    setSelected(null);
    setHistory([]);
    setImportOpen(false);
    setNotesOpen(false);
    setZoom(1);
    setExampleId(example);
    syncExampleParam(example);
  }, []);

  const loadExample = useCallback(
    (id: string) => {
      const example = findExample(id);
      return example?.load().then(source => loadText(source, example.id));
    },
    [loadText]
  );

  useEffect(() => {
    const id = new URLSearchParams(location.search).get('example');
    if (id) void loadExample(id);
  }, [loadExample]);

  useEffect(() => {
    const name = pendingNode.current;
    if (!name || !exampleId || exampleId !== new URLSearchParams(location.search).get('example')) return;
    pendingNode.current = null;
    const owner = module.computations.find(item => item.byName.has(name));
    if (!owner) return;
    setCurrent(owner.name);
    setSelected(name);
    requestAnimationFrame(() => graphRef.current?.centerNode(name));
  }, [module, exampleId]);

  // Wait a frame so the graph has rendered the node before centering on it.
  const centerSoon = (name: string) => requestAnimationFrame(() => graphRef.current?.centerNode(name));

  const openComputation = (name: string | null, focus: string | null = null) => {
    if (name && !module.byName.has(name)) return;
    if (current !== name) setHistory(prior => [...prior, { current, selected }]);
    setCurrent(name);
    setSelected(focus);
    setZoom(1);
    if (focus) centerSoon(focus);
  };

  // One-argument form for callbacks, so no extra argument is taken as a focus node.
  const showComputation = (name: string | null) => openComputation(name);

  const goBack = () => {
    const previous = history.at(-1);
    if (!previous) return;
    setHistory(prior => prior.slice(0, -1));
    setCurrent(previous.current);
    setSelected(previous.selected);
  };

  const openReference = (name: string) => {
    setSelected(name);
    centerSoon(name);
  };

  const onUseOpName = (value: boolean) => {
    setUseOpName(value);
    if (!value) setShowLastNameOnly(false);
  };

  const viewProps = { module, current, zoom, autoGroup, layoutMode, useOpName, showLastNameOnly, showMemoryLocation };

  const toolbarActions = {
    onAutoGroup: setAutoGroup,
    onLayoutMode: setLayoutMode,
    onUseOpName,
    onShowLastNameOnly: setShowLastNameOnly,
    onShowMemoryLocation: setShowMemoryLocation,
    onBack: goBack,
    onComputation: showComputation,
    onFit: () => graphRef.current?.fit(),
    onArrange: () => graphRef.current?.arrange(),
    onZoom: (delta: number) => setZoom(value => clampZoom(value + delta)),
    onSearch: () => setSearchOpen(true),
    onImport: () => setImportOpen(true),
    onExample: (id: string) => void loadExample(id)
  };

  // LLO programs compiled from this HLO example, one per instruction. A before-optimization
  // module has no programs of its own, so its sidebar link falls back to the compiled module's default.
  const lloPrograms = useMemo(() => lloExamples.filter(item => item.hloExample === exampleId), [exampleId]);
  const nodeLloHref = node ? lloLink(lloPrograms.find(item => item.instruction === node.name)) : null;
  const compiledId = exampleId?.replace(/\.before$/, '.after');
  const lloHref = nodeLloHref ?? lloLink(lloExamples.find(item => item.default && item.hloExample === compiledId)) ?? '?view=llo';

  const status = node
    ? `%${node.name} · ${upstreamCount} upstream · ${downstreamCount} downstream`
    : computation
      ? ''
      : 'Computation links · instruction data edges are inside each computation';

  return (
    <>
      <div className={`app antialiased${node ? ' inspector-open' : ''}${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}>
        <Sidebar
          module={module}
          current={current}
          collapsed={sidebarCollapsed}
          lloHref={lloHref}
          onOverview={() => showComputation(null)}
          onComputation={showComputation}
        />
        <main className="main min-w-0">
          <Toolbar
            {...viewProps}
            {...toolbarActions}
            canGoBack={history.length > 0}
            exampleId={exampleId}
            sidebarCollapsed={sidebarCollapsed}
            onToggleSidebar={() => setSidebarCollapsed(value => !value)}
            lloHref={lloHref}
          />
          <GraphCanvas
            {...viewProps}
            ref={graphRef}
            copyGrouping={copyGrouping}
            selected={selected}
            onZoom={setZoom}
            onSelect={setSelected}
            onComputation={showComputation}
          />
          <div className="bottom-bar">
            <div role="status">{status}</div>
            <div>
              {module.warnings.length ? (
                <button type="button" className="parse-notes-button" onClick={() => setNotesOpen(true)}>
                  {module.warnings.length} parse notes ↗
                </button>
              ) : null}
            </div>
          </div>
        </main>
        <Inspector
          copyGroup={selectedCopyGroup}
          module={module}
          computation={computation}
          node={node}
          upstreamCount={upstreamCount}
          downstreamCount={downstreamCount}
          lloHref={nodeLloHref}
          onClose={() => setSelected(null)}
          onNode={openReference}
          onComputation={showComputation}
        />
      </div>

      {searchOpen && (
        <SearchOverlay
          module={module}
          useOpName={useOpName}
          showLastNameOnly={showLastNameOnly}
          onClose={() => setSearchOpen(false)}
          onSelect={(computationName, nodeName) => {
            setSearchOpen(false);
            openComputation(computationName, nodeName);
          }}
        />
      )}
      {importOpen && <ImportDialog onClose={() => setImportOpen(false)} onLoad={source => loadText(source)} />}
      {notesOpen && <ParseNotesDialog warnings={module.warnings} onClose={() => setNotesOpen(false)} />}
    </>
  );
}

function ParseNotesDialog({ warnings, onClose }: { warnings: string[]; onClose: () => void }) {
  return (
    <div
      className="notes-overlay"
      role="presentation"
      onMouseDown={event => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="notes-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Parse notes"
        onKeyDown={event => {
          if (event.key === 'Escape') onClose();
        }}
      >
        <header>
          <strong>Parse notes</strong>
          <button className="icon-button" type="button" aria-label="Close parse notes" autoFocus onClick={onClose}>
            <Icon name="close" />
          </button>
        </header>
        <p>Some input could not be represented exactly. Check these lines before relying on the graph.</p>
        <ul>
          {warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
