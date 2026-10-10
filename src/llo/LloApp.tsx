import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { Drawer } from 'vaul';
import sample from './sample.llo?raw';
import { AppSidebar, SidebarToggle } from '../components/AppSidebar';
import { GuideRow } from '../components/Inspector';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { useStoredState } from '../hooks/useStoredState';
import { findLloExample, lloExamples, lloTopologies, type LloExample } from './examples';
import { lloGuide } from './instruction-guide';
import { definitionOf, parseLlo, units, type LloInstruction, type LloProgram, type Unit } from './parser';
import './llo.css';

const descriptions: Record<Unit, string> = {
  MXU: 'Matrix unit · systolic matrix multiply and accumulator operations',
  XLU: 'Cross-lane unit · transpose and lane rearrangement',
  DMA: 'Asynchronous transfer between HBM and on-chip memory, or its wait',
  VLOAD: 'Load a vector from VMEM',
  VSTORE: 'Store a vector to VMEM',
  VPU: 'Vector arithmetic, conversion, masking, or lane operation',
  SCALAR: 'Scalar arithmetic, address calculation, or control',
  OTHER: 'Compiler operation or instruction without a recognized unit'
};

const partKey = (target: EventTarget) =>
  target instanceof Element ? (target.closest<HTMLElement>('[data-part]')?.dataset.part ?? null) : null;

// The instruction text with each part wrapped in a span; hovering or clicking a part highlights its explanation row.
function LloCode({ html, activePart, activate }: { html: string; activePart: string | null; activate: (key: string | null) => void }) {
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    ref.current?.querySelectorAll<HTMLElement>('[data-part]').forEach(element => {
      element.classList.toggle('active-part', element.dataset.part === activePart);
    });
    // oxlint-disable-next-line react/exhaustive-effect-dependencies -- new html replaces the spans, so re-apply highlights
  }, [html, activePart]);

  return (
    <pre
      ref={ref}
      className="llo-code"
      dangerouslySetInnerHTML={{ __html: html }}
      onMouseOver={event => activate(partKey(event.target))}
      onMouseOut={() => activate(null)}
      onFocus={event => activate(partKey(event.target))}
      onBlur={() => activate(null)}
      onClick={event => {
        const key = partKey(event.target);
        activate(key);
        if (key)
          ref.current?.parentElement
            ?.querySelector(`.guide-row[data-part="${key}"]`)
            ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }}
    />
  );
}

function InstructionChip({
  instruction,
  selected,
  onSelect
}: {
  instruction: LloInstruction;
  selected: boolean;
  onSelect: (instruction: LloInstruction) => void;
}) {
  return (
    <button
      className={`llo-chip unit-${instruction.unit.toLowerCase()}${selected ? ' selected' : ''}`}
      type="button"
      onClick={() => onSelect(instruction)}
      title={instruction.raw}
    >
      <span>{instruction.opcode}</span>
      {instruction.output && <small>{instruction.output}</small>}
    </button>
  );
}

function InstructionDetails({
  program,
  selected,
  onSelect,
  onClose
}: {
  program: LloProgram;
  selected: LloInstruction;
  onSelect: (instruction: LloInstruction) => void;
  onClose?: () => void;
}) {
  const [activePart, setActivePart] = useState<string | null>(null);
  const guide = lloGuide(selected);
  const region = program.regions.find(item => item.name === selected.region);
  const consumers = selected.output
    ? program.instructions.filter(
        item =>
          item.region === selected.region &&
          item !== selected &&
          item.references.includes(selected.output!) &&
          definitionOf(program, item, selected.output!) === selected
      )
    : [];
  return (
    <>
      <div className="llo-inspector-head">
        <div>
          <div className="eyebrow">
            {selected.unit} · LINE {selected.line}
          </div>
          <h2>{selected.opcode}</h2>
        </div>
        {onClose && (
          <button className="llo-close" type="button" aria-label="Close inspector" onClick={onClose}>
            ×
          </button>
        )}
      </div>
      <p>{guide.summary}</p>
      <div className="llo-inspector-section">
        <div className="eyebrow">LOCATION</div>
        <div className="llo-detail">
          {region?.label || selected.region}
          {selected.bundle ? ` · bundle ${selected.bundle}` : ' · unscheduled'}
          {selected.depth > 0 && ` · loop depth ${selected.depth}`}
        </div>
      </div>
      <div className="llo-inspector-section">
        <div className="eyebrow">EXACT INSTRUCTION</div>
        <LloCode html={guide.html} activePart={activePart} activate={setActivePart} />
        <div className="guide-heading">Explanation</div>
        <div className="guide-list">
          {guide.parts.map(part => (
            <GuideRow key={part.key} part={part} active={activePart === part.key} activate={key => setActivePart(key)} />
          ))}
        </div>
        <div className="guide-sources">
          Opcode and modifier meanings are inferred from TPU dumps and Google’s published TPU material; the ISA itself is not public.
        </div>
      </div>
      {selected.references.length > 0 && (
        <div className="llo-inspector-section">
          <div className="eyebrow">REGISTER INPUTS</div>
          <div className="llo-ref-list">
            {selected.references.map(ref => {
              const definition = definitionOf(program, selected, ref);
              return (
                <button
                  type="button"
                  key={ref}
                  disabled={!definition}
                  onClick={() => definition && onSelect(definition)}
                  title={definition ? `Jump to line ${definition.line}` : 'Definition not present in this dump'}
                >
                  {ref}
                  <span>{definition ? `← ${definition.opcode}` : 'external / omitted'}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      {selected.allocations.length > 0 && (
        <div className="llo-inspector-section">
          <div className="eyebrow">ALLOCATIONS</div>
          {selected.allocations.map(name => {
            const allocation = region?.allocations.find(item => item.name === name);
            return (
              <div className="llo-allocation-detail" key={name}>
                <strong>{name}</strong>
                <span>
                  {allocation
                    ? `${allocation.space.toUpperCase()} · ${allocation.shape} · ${allocation.size}`
                    : 'Definition not present in this dump'}
                </span>
              </div>
            );
          })}
        </div>
      )}
      {consumers.length > 0 && (
        <div className="llo-inspector-section">
          <div className="eyebrow">USERS IN THIS REGION</div>
          <div className="llo-ref-list">
            {consumers.slice(0, 30).map(item => (
              <button type="button" key={item.id} onClick={() => onSelect(item)}>
                {item.opcode}
                <span>line {item.line}</span>
              </button>
            ))}
          </div>
          {consumers.length > 30 && <small>Showing first 30 of {consumers.length}</small>}
        </div>
      )}
    </>
  );
}

function Inspector({
  program,
  selected,
  onSelect,
  onClose
}: {
  program: LloProgram;
  selected: LloInstruction | null;
  onSelect: (instruction: LloInstruction) => void;
  onClose: () => void;
}) {
  const isPhone = useMediaQuery('(max-width: 700px)');
  const [lastSelected, setLastSelected] = useState(selected);
  if (selected && selected !== lastSelected) setLastSelected(selected);

  if (isPhone) {
    const shown = selected ?? lastSelected;
    return (
      <Drawer.Root
        open={!!selected}
        onOpenChange={open => {
          if (!open) onClose();
        }}
      >
        <Drawer.Portal>
          <Drawer.Overlay className="drawer-overlay" />
          <Drawer.Content className="inspector-drawer llo-inspector-drawer" aria-describedby={undefined}>
            <div className="drawer-handle" aria-hidden="true" />
            <div className="llo-sheet-header">
              <Drawer.Title>Instruction details</Drawer.Title>
              <button className="llo-close" type="button" aria-label="Close instruction details" onClick={onClose}>
                ×
              </button>
            </div>
            <div className="llo-inspector llo-sheet-content overscroll-contain">
              {shown && <InstructionDetails key={shown.id} program={program} selected={shown} onSelect={onSelect} />}
            </div>
          </Drawer.Content>
        </Drawer.Portal>
      </Drawer.Root>
    );
  }

  return (
    <aside className={`llo-inspector${selected ? '' : ' llo-inspector-empty'}`}>
      {selected ? (
        <InstructionDetails program={program} selected={selected} onSelect={onSelect} onClose={onClose} />
      ) : (
        <>
          <div className="eyebrow">INSPECTOR</div>
          <h2>Select an instruction</h2>
          <p>Inspect the exact LLO text, register producers, memory allocations, and its place in a bundle.</p>
        </>
      )}
    </aside>
  );
}

export default function LloApp() {
  const [program, setProgram] = useState(() => parseLlo(sample));
  const [fileName, setFileName] = useState('Illustrative LLO excerpt');
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [regionName, setRegionName] = useState('region0');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [activeUnits, setActiveUnits] = useState<Set<Unit>>(() => new Set(units));
  const [importOpen, setImportOpen] = useState(false);
  const [paste, setPaste] = useState('');
  const [rowLimit, setRowLimit] = useState(200);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'));
  const [sidebarCollapsed, setSidebarCollapsed] = useStoredState('llo-sidebar-collapsed', saved => saved === 'true');
  const fileRef = useRef<HTMLInputElement>(null);
  const rowRefs = useRef(new Map<string, HTMLDivElement>());
  const searchRef = useRef<HTMLInputElement>(null);

  const region = program.regions.find(item => item.name === regionName) ?? program.regions[0];
  const example = exampleId ? (lloExamples.find(item => item.id === exampleId) ?? null) : null;
  // Other programs compiled from the same JAX example, one per HLO instruction.
  const siblings = example ? lloExamples.filter(item => item.topology === example.topology && item.example === example.example) : [];
  const exampleDefault = siblings.find(item => item.default);
  // The same instruction in the HLO graph when the shipped HLO example contains it.
  const hloHref = example?.hloExample ? `?example=${example.hloExample}&node=${example.instruction}` : '?';
  const selected = program.instructions.find(item => item.id === selectedId) ?? null;
  const scheduled = !!region?.bundles.length;
  const normalizedQuery = query.trim().toLowerCase();
  const visible = (instruction: LloInstruction) =>
    activeUnits.has(instruction.unit) &&
    (!normalizedQuery || `${instruction.raw} ${instruction.bundle ?? ''}`.toLowerCase().includes(normalizedQuery));
  const matchingBundles = region?.bundles.filter(bundle => bundle.instructions.some(visible)) ?? [];
  const matchingInstructions = region?.instructions.filter(visible) ?? [];
  const bundles = matchingBundles.slice(0, rowLimit);
  const instructions = matchingInstructions.slice(0, rowLimit);
  const counts = Object.fromEntries(
    units.map(unit => [unit, region?.instructions.filter(item => item.unit === unit).length ?? 0])
  ) as Record<Unit, number>;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === '/' && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement)) {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === 'Escape') {
        setImportOpen(false);
        setSelectedId(null);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const select = (instruction: LloInstruction) => {
    if (!visible(instruction)) {
      setQuery('');
      setActiveUnits(new Set(units));
    }
    if (instruction.region !== regionName) setRegionName(instruction.region);
    const target = program.regions.find(item => item.name === instruction.region);
    const position = instruction.bundle
      ? (target?.bundles.findIndex(item => item.address === instruction.bundle) ?? -1)
      : (target?.instructions.indexOf(instruction) ?? -1);
    if (position >= 0) setRowLimit(limit => Math.max(limit, position + 1));
    setSelectedId(instruction.id);
    requestAnimationFrame(() => rowRefs.current.get(instruction.id)?.scrollIntoView({ block: 'center', behavior: 'smooth' }));
  };
  const load = useCallback((source: string, name: string, id: string | null = null) => {
    const parsed = parseLlo(source);
    if (!parsed.instructions.length) throw new Error(parsed.warnings[0]);
    setProgram(parsed);
    setFileName(name);
    setExampleId(id);
    setRegionName(parsed.regions[0].name);
    setSelectedId(null);
    setQuery('');
    setRowLimit(200);
    setActiveUnits(new Set(units));
    setImportOpen(false);
    setError('');
    const url = new URL(location.href);
    if (id) url.searchParams.set('example', id);
    else url.searchParams.delete('example');
    history.replaceState(null, '', url);
  }, []);
  const loadExample = useCallback(
    (entry: LloExample) =>
      void entry.load().then(source => load(source, `${entry.example} · %${entry.instruction} · ${entry.topology}`, entry.id)),
    [load]
  );
  useEffect(() => {
    const entry = findLloExample(new URLSearchParams(location.search).get('example'));
    if (entry) loadExample(entry);
  }, [loadExample]);
  const openFile = async (file: File) => {
    try {
      load(await file.text(), file.name);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setImportOpen(true);
    }
  };
  const chooseFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await openFile(file);
    event.target.value = '';
  };
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void openFile(file);
  };
  const toggleUnit = (unit: Unit) =>
    setActiveUnits(previous => {
      const next = new Set(previous);
      if (next.has(unit)) next.delete(unit);
      else next.add(unit);
      return next;
    });
  const chooseRegion = (name: string) => {
    setRegionName(name);
    setSelectedId(null);
    setRowLimit(200);
  };
  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
  };

  return (
    <div
      className={`llo-app${dragging ? ' llo-dragging' : ''}${sidebarCollapsed ? ' sidebar-collapsed' : ''}`}
      onDragOver={event => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={event => {
        if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <AppSidebar
        id="llo-sidebar"
        collapsed={sidebarCollapsed}
        icon="graph"
        title="LLO Visualizer"
        subtitle="Explore the TPU instruction stream"
        switchTo={{ href: hloHref, label: '← HLO Visualizer' }}
      >
        <div className="llo-side-section">
          <div className="eyebrow">PROGRAM</div>
          <h1 title={fileName}>{fileName}</h1>
          <p>
            {program.regions.length} {program.regions.length === 1 ? 'region' : 'regions'} · {program.instructions.length} instructions ·{' '}
            {program.regions.reduce((sum, item) => sum + item.bundles.length, 0)} bundles
          </p>
          {program.hlo && (
            <div className="llo-hlo">
              <div className="eyebrow">HLO INSTRUCTION</div>
              <p>
                %{program.hlo.name} = {program.hlo.opcode}({program.hlo.operands.join(', ')})
              </p>
              {siblings.length > 1 && (
                <select
                  className="llo-example-select"
                  aria-label="Choose another program from this example"
                  value={exampleId ?? ''}
                  onChange={event => {
                    const next = lloExamples.find(item => item.id === event.target.value);
                    if (next) loadExample(next);
                  }}
                >
                  {siblings.map(item => (
                    <option key={item.id} value={item.id}>
                      %{item.instruction} · {item.opcode} · {item.bundles} bundles
                    </option>
                  ))}
                </select>
              )}
              {example?.hloExample && <a href={hloHref}>Open in HLO graph →</a>}
            </div>
          )}
        </div>
        <div className="llo-side-section llo-region-section">
          <div className="llo-section-title">
            Regions <span>{program.regions.length}</span>
          </div>
          <nav aria-label="LLO regions">
            {program.regions.map(item => (
              <button
                key={item.name}
                type="button"
                className={item.name === region?.name ? 'active' : ''}
                onClick={() => chooseRegion(item.name)}
              >
                <strong>{item.label}</strong>
                <small>{item.bundles.length ? `${item.bundles.length} bundles` : `${item.instructions.length} instructions`}</small>
              </button>
            ))}
          </nav>
        </div>
        <div className="llo-side-bottom">
          <div className="eyebrow">ABOUT THIS VIEW</div>
          <p>
            LLO is the TPU backend’s lower-level instruction representation. Final bundle dumps show operations issued together; earlier
            pass dumps show unscheduled instructions.
          </p>
        </div>
      </AppSidebar>
      <main className="llo-main">
        <header className="llo-topbar">
          <div className="llo-topbar-leading">
            <SidebarToggle controls="llo-sidebar" collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(value => !value)} />
            <div className="llo-crumb">
              Workspace <span>/</span> <strong>LLO</strong> <span>/</span> {region?.label}
            </div>
          </div>
          <div className="llo-top-actions">
            <select
              className="llo-example-select"
              aria-label="Load an LLO example"
              value={exampleDefault?.id ?? ''}
              onChange={event => {
                const next = lloExamples.find(item => item.id === event.target.value);
                if (next) loadExample(next);
              }}
            >
              <option value="" disabled>
                Examples
              </option>
              {lloTopologies.map(topology => (
                <optgroup key={topology} label={topology}>
                  {lloExamples
                    .filter(item => item.topology === topology && item.default)
                    .map(item => (
                      <option key={item.id} value={item.id}>
                        {item.example}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
            <button
              type="button"
              className="llo-icon-button"
              onClick={toggleTheme}
              title="Toggle light or dark theme"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? '◐' : '☀'}
            </button>
            <button type="button" className="llo-button" onClick={() => fileRef.current?.click()}>
              Choose file
            </button>
            <button type="button" className="llo-primary" onClick={() => setImportOpen(true)}>
              Open LLO
            </button>
            <input ref={fileRef} type="file" accept=".txt,.llo,.log,text/plain" onChange={chooseFile} hidden />
          </div>
        </header>
        <section className="llo-heading">
          <div>
            <div className="eyebrow">{scheduled ? 'SCHEDULED BUNDLES' : 'INSTRUCTION STREAM'}</div>
            <h2>{region?.label}</h2>
            <p>
              {scheduled
                ? 'Each row is one VLIW bundle. Columns group its instructions by hardware unit.'
                : 'This pass has no bundle addresses. Instructions appear in source order.'}
            </p>
            <div className="llo-mobile-navigation">
              {program.regions.length > 1 && (
                <select
                  className="llo-mobile-region"
                  aria-label="Choose LLO region"
                  value={region?.name ?? ''}
                  onChange={event => chooseRegion(event.target.value)}
                >
                  {program.regions.map(item => (
                    <option key={item.name} value={item.name}>
                      {item.label}
                    </option>
                  ))}
                </select>
              )}
              <a href={hloHref}>← HLO Visualizer</a>
            </div>
          </div>
          <div className="llo-summary">
            <span>
              <strong>{region?.instructions.length ?? 0}</strong> instructions
            </span>
            <span>
              <strong>{region?.bundles.length ?? 0}</strong> bundles
            </span>
            <span>
              <strong>{region?.allocations.length ?? 0}</strong> allocation definitions
            </span>
          </div>
        </section>
        <section className="llo-controls">
          <label className="llo-search">
            <span>⌕</span>
            <input
              ref={searchRef}
              value={query}
              onChange={event => {
                setQuery(event.target.value);
                setRowLimit(200);
              }}
              placeholder="Search opcode, register, address…"
              aria-label="Search instructions"
            />
            <kbd>/</kbd>
          </label>
          <div className="llo-unit-filters" aria-label="Filter hardware units">
            {units.map(unit => (
              <button
                type="button"
                key={unit}
                className={`llo-unit-filter unit-${unit.toLowerCase()}${activeUnits.has(unit) ? ' active' : ''}`}
                onClick={() => {
                  toggleUnit(unit);
                  setRowLimit(200);
                }}
                title={descriptions[unit]}
              >
                {unit} <span>{counts[unit]}</span>
              </button>
            ))}
          </div>
        </section>
        <div className="llo-content">
          {scheduled ? (
            <div className="llo-timeline">
              <div className="llo-timeline-head">
                <span>BUNDLE</span>
                {units.map(unit => (
                  <span key={unit} title={descriptions[unit]}>
                    {unit}
                  </span>
                ))}
              </div>
              {bundles.map(bundle => (
                <div className="llo-timeline-row" key={`${bundle.region}/${bundle.address}`}>
                  <div className="llo-address">
                    <strong>{bundle.address}</strong>
                    <small>{bundle.instructions.length} ops</small>
                  </div>
                  {units.map(unit => (
                    <div className="llo-lane" key={unit}>
                      {bundle.instructions
                        .filter(item => item.unit === unit && visible(item))
                        .map(item => (
                          <div
                            key={item.id}
                            ref={element => {
                              if (element) rowRefs.current.set(item.id, element);
                            }}
                          >
                            <InstructionChip instruction={item} selected={selectedId === item.id} onSelect={select} />
                          </div>
                        ))}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <div className="llo-stream">
              {instructions.map(item => (
                <div
                  key={item.id}
                  className="llo-stream-row"
                  ref={element => {
                    if (element) rowRefs.current.set(item.id, element);
                  }}
                >
                  <span className="llo-line">{item.line}</span>
                  <span className={`llo-unit-label unit-${item.unit.toLowerCase()}`}>{item.unit}</span>
                  <InstructionChip instruction={item} selected={selectedId === item.id} onSelect={select} />
                  <code>{item.raw}</code>
                </div>
              ))}
            </div>
          )}
          {(scheduled ? bundles.length : instructions.length) === 0 && (
            <div className="llo-empty">No instructions match the current search and unit filters.</div>
          )}
          {(scheduled ? matchingBundles.length : matchingInstructions.length) > rowLimit && (
            <button className="llo-show-more" type="button" onClick={() => setRowLimit(limit => limit + 200)}>
              Show next 200 · {scheduled ? matchingBundles.length - rowLimit : matchingInstructions.length - rowLimit} remaining
            </button>
          )}
        </div>
        <footer className="llo-status">
          {fileName === 'Illustrative LLO excerpt'
            ? 'Illustrative excerpt · open your own LLO dump to inspect a full program'
            : `${fileName} · parsed locally in your browser`}
        </footer>
      </main>
      <Inspector program={program} selected={selected} onSelect={select} onClose={() => setSelectedId(null)} />
      {dragging && <div className="llo-drop-hint">Drop LLO text file to open</div>}
      {importOpen && (
        <div
          className="llo-modal-backdrop"
          onMouseDown={event => {
            if (event.target === event.currentTarget) setImportOpen(false);
          }}
        >
          <div className="llo-modal" role="dialog" aria-modal="true" aria-label="Open LLO text">
            <div className="llo-modal-head">
              <div>
                <div className="eyebrow">IMPORT PROGRAM</div>
                <h2>Open LLO text</h2>
              </div>
              <button type="button" className="llo-close" aria-label="Close" onClick={() => setImportOpen(false)}>
                ×
              </button>
            </div>
            <p>
              Paste a textual LLO pass dump or <code>final_bundles.txt</code>. Your text stays in this browser.
            </p>
            <textarea
              autoFocus
              spellCheck={false}
              value={paste}
              onChange={event => setPaste(event.target.value)}
              placeholder={'$region0: #{fusion}\n  0x0 : { %v0 = vld [vmem:[%s0]] }'}
            />
            <div className="llo-modal-actions">
              <button type="button" className="llo-button" onClick={() => fileRef.current?.click()}>
                Choose file
              </button>
              <span role="alert">{error}</span>
              <button
                type="button"
                className="llo-primary"
                onClick={() => {
                  try {
                    load(paste, 'Pasted LLO');
                  } catch (cause) {
                    setError(cause instanceof Error ? cause.message : String(cause));
                  }
                }}
              >
                Visualize →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
