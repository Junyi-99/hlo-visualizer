import { useEffect, useRef, useState } from 'react';
import sampleHlo from '../sample.hlo?raw';
import { GraphCanvas, type GraphHandle } from './components/GraphCanvas';
import { ImportDialog } from './components/ImportDialog';
import { Inspector } from './components/Inspector';
import { SearchOverlay } from './components/SearchOverlay';
import { Sidebar } from './components/Sidebar';
import { Icon } from './components/Icon';
import { Toolbar } from './components/Toolbar';
import { examples } from './lib/examples';
import { parseHlo, reachable } from './lib/parser';
import type { HloModule } from './lib/types';

interface HistoryEntry { current: string | null; selected: string | null }

export default function App() {
  const [module, setModule] = useState<HloModule>(() => parseHlo(sampleHlo));
  const [current, setCurrent] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [zoom, setZoom] = useState(1);
  const [useOpName, setUseOpName] = useState(false);
  const [showLastNameOnly, setShowLastNameOnly] = useState(false);
  const [showMemoryLocation, setShowMemoryLocation] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  const [exampleId, setExampleId] = useState<string | null>(null);
  const graphRef = useRef<GraphHandle>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setSearchOpen(false); return; }
      if (event.key === '/' && !searchOpen && !importOpen && !(document.activeElement instanceof HTMLInputElement) && !(document.activeElement instanceof HTMLTextAreaElement)) {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [searchOpen, importOpen]);

  const openComputation = (name: string | null, focus: string | null = null) => {
    if (name && !module.byName.has(name)) return;
    if (current !== name) setHistory(prior => [...prior, { current, selected }]);
    setCurrent(name);
    setSelected(focus);
    setZoom(1);
    if (focus) requestAnimationFrame(() => graphRef.current?.centerNode(focus));
  };
  const goBack = () => {
    const previous = history.at(-1);
    if (!previous) return;
    setHistory(prior => prior.slice(0, -1));
    setCurrent(previous.current);
    setSelected(previous.selected);
  };
  const loadExample = async (id: string) => {
    const example = examples.find(item => item.id === id);
    if (!example) return;
    loadText(await example.load(), id);
  };
  // ?example=<id> loads a bundled example, so a reviewed module can be linked directly.
  useEffect(() => {
    const id = new URLSearchParams(location.search).get('example');
    if (id) void loadExample(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const loadText = (source: string, example: string | null = null) => {
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
    const url = new URL(location.href);
    if (example) url.searchParams.set('example', example); else url.searchParams.delete('example');
    window.history.replaceState(null, '', url);
  };
  const openReference = (name: string) => {
    setSelected(name);
    requestAnimationFrame(() => graphRef.current?.centerNode(name));
  };

  const computation = current ? module.byName.get(current) || null : null;
  const node = selected && computation ? computation.byName.get(selected) || null : null;
  const upstream = node && computation ? reachable(computation, node.name, 'up') : new Set<string>();
  const downstream = node && computation ? reachable(computation, node.name, 'down') : new Set<string>();

  return <>
    <div className={`app antialiased${node ? ' inspector-open' : ''}`}>
      <Sidebar module={module} current={current} onOverview={() => openComputation(null)} onComputation={name => openComputation(name)} />
      <main className="main min-w-0">
        <Toolbar module={module} current={current} canGoBack={history.length > 0} zoom={zoom} useOpName={useOpName} showLastNameOnly={showLastNameOnly} showMemoryLocation={showMemoryLocation}
          onUseOpName={value => { setUseOpName(value); if (!value) setShowLastNameOnly(false); }} onShowLastNameOnly={setShowLastNameOnly}
          onShowMemoryLocation={setShowMemoryLocation}
          onBack={goBack} onComputation={name => openComputation(name)} onFit={() => graphRef.current?.fit()}
          onZoom={delta => setZoom(value => Math.min(1.5, Math.max(0.45, value + delta)))} onSearch={() => setSearchOpen(true)} onImport={() => setImportOpen(true)} exampleId={exampleId} onExample={id => void loadExample(id)} />
        <GraphCanvas ref={graphRef} module={module} current={current} selected={selected} zoom={zoom} useOpName={useOpName} showLastNameOnly={showLastNameOnly} showMemoryLocation={showMemoryLocation} onZoom={setZoom} onSelect={setSelected} onComputation={name => openComputation(name)} />
        <div className="bottom-bar"><div>{node ? `%${node.name} · ${upstream.size} upstream · ${downstream.size} downstream` : computation ? '' : 'Computation links · instruction data edges are inside each computation'}</div>
          <div>{module.warnings.length ? <button type="button" className="parse-notes-button" onClick={() => setNotesOpen(true)}>{module.warnings.length} parse notes ↗</button> : null}</div></div>
      </main>
      <Inspector module={module} computation={computation} node={node} upstreamCount={upstream.size} downstreamCount={downstream.size} onClose={() => setSelected(null)} onNode={openReference} onComputation={name => openComputation(name)} />
    </div>
    {searchOpen && <SearchOverlay module={module} useOpName={useOpName} showLastNameOnly={showLastNameOnly} onClose={() => setSearchOpen(false)} onSelect={(computationName, nodeName) => { setSearchOpen(false); openComputation(computationName, nodeName); }} />}
    {importOpen && <ImportDialog onClose={() => setImportOpen(false)} onLoad={source => loadText(source)} />}
    {notesOpen && <div className="notes-overlay" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setNotesOpen(false); }}><div className="notes-panel" role="dialog" aria-modal="true" aria-label="Parse notes"><header><strong>Parse notes</strong><button className="icon-button" type="button" aria-label="Close parse notes" onClick={() => setNotesOpen(false)}><Icon name="close" /></button></header><p>Some input could not be represented exactly. Check these lines before relying on the graph.</p><ul>{module.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></div></div>}
  </>;
}
